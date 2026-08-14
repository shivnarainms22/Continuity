"""Task 1: measure the query performance envelope before designing the drill-down.

An investigation issues many queries. Whether the walker can afford one query per
dimension per level, or must batch splits into a single query, is a decision that should
come from measurement rather than intuition -- and the answer also determines whether a
live demo stalls.

Every query goes through the MCP gateway, i.e. the real agent-runtime path.

Run:  uv run python scripts/benchmark_queries.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from continuity.config import ClickHouseConfig
from continuity.data.topology import DIMENSION_HIERARCHY
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

GROUND_TRUTH = Path("data/ground_truth.json")
ROLLUP_DIMS = [d for d in DIMENSION_HIERARCHY if d != "title_id"]


def incident_window() -> tuple[str, str]:
    """Derive the benchmark window from ground truth rather than hardcoding dates.

    This script previously pinned literal January dates. When incident placement moved
    to be relative to the end of the window, those literals silently pointed at a period
    containing no incident, and every GROUP BY returned zero rows -- while the latencies
    still looked entirely plausible. Deriving means a stale artifact fails loudly instead.
    """
    if not GROUND_TRUTH.exists():
        raise SystemExit(f"{GROUND_TRUTH} not found. Run the loader first.")
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    incident = next(i for i in truth["incidents"] if not i["is_decoy"])

    def fmt(value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    return fmt(incident["start"]), fmt(incident["end"])


WINDOW = ("", "")  # populated in main() once ground truth is read

REBUFFER = "sum(rebuffer_ms) / nullIf(sum(watched_ms), 0)"


async def timed(gw, label: str, sql: str, repeats: int = 3) -> tuple[str, float, int]:
    """Run a query several times and report the median, discarding the first."""
    durations: list[float] = []
    rows = 0
    for _ in range(repeats):
        before = len(gw.query_log)
        result = await gw.query(sql)
        rows = len(result.rows)
        durations.append(gw.query_log[before].duration_ms)
    steady = statistics.median(durations[1:]) if len(durations) > 1 else durations[0]
    return label, steady, rows


async def main() -> int:
    load_dotenv(override=False)
    global WINDOW
    WINDOW = incident_window()
    print(f"benchmark window (from ground truth): {WINDOW[0]} -> {WINDOW[1]}")
    results: list[tuple[str, float, int]] = []

    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        # Warm the connection so the one-time server/TLS startup is not charged to the
        # first measurement.
        await gw.query("SELECT 1")

        print("\n--- single-slice metric over the incident window ---")
        results.append(await timed(
            gw, "rollup: whole-population rebuffer, 8h window",
            f"SELECT {REBUFFER} AS v FROM qoe_rollup_5m "
            f"WHERE bucket >= '{WINDOW[0]}' AND bucket < '{WINDOW[1]}'",
        ))
        results.append(await timed(
            gw, "rollup: 2-dim slice rebuffer, 8h window",
            f"SELECT {REBUFFER} AS v FROM qoe_rollup_5m "
            f"WHERE bucket >= '{WINDOW[0]}' AND bucket < '{WINDOW[1]}' "
            "AND device_type = 'roku' AND app_version = '8.2.0'",
        ))
        results.append(await timed(
            gw, "raw events: 2-dim slice rebuffer, 8h window",
            f"SELECT {REBUFFER} AS v FROM playback_events "
            f"WHERE event_time >= '{WINDOW[0]}' AND event_time < '{WINDOW[1]}' "
            "AND device_type = 'roku' AND app_version = '8.2.0'",
        ))

        print("\n--- baseline: trailing 7 days, same time of day ---")
        results.append(await timed(
            gw, "rollup: 7-day trailing per-bucket series",
            "SELECT toStartOfFiveMinute(bucket) AS b, "
            f"{REBUFFER} AS v FROM qoe_rollup_5m "
            "WHERE bucket >= '2026-01-06 18:00:00' AND bucket < '2026-01-13 18:00:00' "
            "GROUP BY b ORDER BY b",
        ))

        print("\n--- one split per dimension (the naive walker: 1 query per dim per level) ---")
        for dim in ROLLUP_DIMS:
            results.append(await timed(
                gw, f"split on {dim}",
                f"SELECT {dim} AS value, {REBUFFER} AS v, sum(watched_ms) AS w "
                "FROM qoe_rollup_5m "
                f"WHERE bucket >= '{WINDOW[0]}' AND bucket < '{WINDOW[1]}' "
                f"GROUP BY {dim} ORDER BY w DESC",
                repeats=2,
            ))

        print("\n--- all dimensions in ONE query (the batched alternative) ---")
        union = " UNION ALL ".join(
            f"SELECT '{d}' AS dim, {d} AS value, {REBUFFER} AS v, sum(watched_ms) AS w "
            "FROM qoe_rollup_5m "
            f"WHERE bucket >= '{WINDOW[0]}' AND bucket < '{WINDOW[1]}' GROUP BY {d}"
            for d in ROLLUP_DIMS
        )
        results.append(await timed(gw, "batched: all dims in one UNION ALL", union, repeats=2))

        print("\n--- title-scoped (raw events, no rollup available) ---")
        results.append(await timed(
            gw, "raw: split on title_id, 8h window",
            f"SELECT title_id AS value, {REBUFFER} AS v, sum(watched_ms) AS w "
            "FROM playback_events "
            f"WHERE event_time >= '{WINDOW[0]}' AND event_time < '{WINDOW[1]}' "
            "GROUP BY title_id ORDER BY w DESC LIMIT 20",
            repeats=2,
        ))

    print(f"\n{'=' * 78}")
    print(f"{'query':<52}{'median ms':>12}{'rows':>10}")
    print("-" * 78)
    for label, ms, rows in results:
        print(f"{label:<52}{ms:>12,.0f}{rows:>10,}")

    per_dim = [ms for label, ms, _ in results if label.startswith("split on ")]
    batched = next(ms for label, ms, _ in results if label.startswith("batched:"))
    print("-" * 78)
    print(f"{'sum of per-dimension splits (one level)':<52}{sum(per_dim):>12,.0f}")
    print(f"{'same work batched into one query':<52}{batched:>12,.0f}")
    speedup = sum(per_dim) / batched if batched else 0
    print(f"\nbatching is {speedup:.1f}x {'faster' if speedup > 1 else 'SLOWER'} for one level")
    print(f"a naive 8-level walk would cost roughly {sum(per_dim) * 8 / 1000:.1f}s in splits alone")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
