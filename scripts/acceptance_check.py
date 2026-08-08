"""Sub-project 1 acceptance gate: read the loaded dataset the way an analyst would.

A green unit suite proves the generator does what it was told. It does not prove the
resulting dataset can support the product. This script asks the only questions that
matter before sub-project 2 starts:

  1. Is each planted incident actually VISIBLE in its true blast radius?
  2. Is the decoy visible as a VOLUME spike with healthy QoE?
  3. Does a naive fixed-threshold detector fire on nightly peaks?
     (If not, the seasonality problem is asserted rather than real, and the
      seasonality-aware baseline in sub-project 2 has nothing to justify it.)
  4. Is incident ground truth absent from every ClickHouse table?

Reads go through the MCP gateway -- the agent-runtime path -- so this doubles as a
check that mcp-clickhouse holds up against a full-size dataset.

Run:  uv run python scripts/acceptance_check.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

GROUND_TRUTH = Path("data/ground_truth.json")

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str) -> None:
    _results.append((status, name, detail))
    marker = "+" if status == PASS else "X"
    print(f"  [{marker}] {name}\n      {detail}")


def _q(value: str) -> str:
    """Single-quote a literal for SQL. Topology values are validated free of quotes."""
    if "'" in value or "\\" in value:
        raise ValueError(f"refusing to interpolate {value!r}")
    return f"'{value}'"


def _predicate_sql(predicate: dict[str, str]) -> str:
    parts = []
    for key, value in predicate.items():
        if key == "title_id":
            parts.append(f"title_id = {int(value)}")
        else:
            parts.append(f"{key} = {_q(value)}")
    return " AND ".join(parts)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


async def check_incident_visible(gw, incident: dict) -> None:
    """An incident must show a measurable deviation inside its true blast radius.

    The control is the SAME dimension slice at the SAME hours on OTHER days. That
    holds both the audience segment and the time of day fixed, so any difference is
    attributable to the incident rather than to who was watching or to peak-hour load.
    """
    name = incident["incident_id"]
    start, end = _parse(incident["start"]), _parse(incident["end"])
    predicate = incident["predicate"]
    where = _predicate_sql(predicate)

    # title-scoped incidents cannot use the rollup: title_id is deliberately excluded
    # from it to keep group cardinality sane, so those query raw events.
    title_scoped = "title_id" in predicate
    table = "playback_events" if title_scoped else "qoe_rollup_5m"
    time_col = "event_time" if title_scoped else "bucket"

    # Metric names as written by incidents.py: "rebuffer", "startup", "bitrate".
    effects = {e["metric"]: e["multiplier"] for e in incident["effects"]}
    if "startup" in effects:
        metric = "startup"
        expr = (
            "quantileTDigest(0.95)(startup_ms)"
            if title_scoped
            else "quantilesTDigestMerge(0.95)(startup_q)[1]"
        )
        label = "p95 startup ms"
    elif "bitrate" in effects:
        metric = "bitrate"
        expr = "avg(bitrate_kbps)" if title_scoped else "avgMerge(bitrate_avg)"
        label = "avg bitrate kbps"
    elif "rebuffer" in effects:
        metric = "rebuffer"
        expr, label = "sum(rebuffer_ms) / nullIf(sum(watched_ms), 0)", "rebuffer ratio"
    else:
        record(FAIL, f"{name} visible", f"no recognised effect metric in {list(effects)}")
        return

    hours = sorted(
        {
            (start + timedelta(hours=h)).hour
            for h in range(int((end - start).total_seconds() // 3600) + 1)
        }
    )
    hour_list = ",".join(str(h) for h in hours)

    inside = (
        await gw.query(
            f"SELECT {expr} AS v FROM {table} "
            f"WHERE {time_col} >= '{_fmt(start)}' AND {time_col} < '{_fmt(end)}' AND {where}"
        )
    ).rows[0]["v"]

    control = (
        await gw.query(
            f"SELECT {expr} AS v FROM {table} "
            f"WHERE NOT ({time_col} >= '{_fmt(start)}' AND {time_col} < '{_fmt(end)}') "
            f"AND toHour({time_col}) IN ({hour_list}) AND {where}"
        )
    ).rows[0]["v"]

    if inside is None or control is None or control == 0:
        record(FAIL, f"{name} visible", f"insufficient data (inside={inside}, control={control})")
        return

    ratio = inside / control
    target = effects[metric]
    # bitrate regressions are a reduction, so the ratio moves below 1.0
    deviation_ok = (ratio > 1.5) if target > 1 else (ratio < 0.75)

    record(
        PASS if deviation_ok else FAIL,
        f"{name} visible in its blast radius",
        f"{label}: {inside:,.4g} inside vs {control:,.4g} control "
        f"(ratio {ratio:.2f}x, planted {target}x) on {len(predicate)} predicate dim(s)",
    )


async def check_decoy(gw, incident: dict) -> None:
    """The decoy must look like a traffic spike and NOT like a fault.

    A system that flags this is producing exactly the false positive that makes real
    ops teams stop reading alerts.
    """
    name = incident["incident_id"]
    start, end = _parse(incident["start"]), _parse(incident["end"])
    where = _predicate_sql(incident["predicate"])
    hours = sorted(
        {
            (start + timedelta(hours=h)).hour
            for h in range(int((end - start).total_seconds() // 3600) + 1)
        }
    )
    hour_list = ",".join(str(h) for h in hours)

    async def sample(in_window: bool) -> dict:
        negate = "" if in_window else "NOT "
        return (
            await gw.query(
                "SELECT uniq(session_id) AS sessions, "
                "sum(rebuffer_ms) / nullIf(sum(watched_ms), 0) AS rebuffer_ratio "
                "FROM playback_events "
                f"WHERE {negate}(event_time >= '{_fmt(start)}' AND event_time < '{_fmt(end)}') "
                f"AND toHour(event_time) IN ({hour_list}) AND {where}"
            )
        ).rows[0]

    inside, control = await sample(True), await sample(False)
    days_outside = 20  # control spans the rest of the window; normalise per-day
    vol_ratio = inside["sessions"] / max(control["sessions"] / days_outside, 1)
    qoe_ratio = (inside["rebuffer_ratio"] or 0) / max(control["rebuffer_ratio"] or 1e-9, 1e-9)

    volume_elevated = vol_ratio > 2.0
    qoe_healthy = qoe_ratio < 1.5

    record(
        PASS if (volume_elevated and qoe_healthy) else FAIL,
        f"{name} is a volume spike with healthy QoE",
        f"volume {vol_ratio:.1f}x per-day control, rebuffer {qoe_ratio:.2f}x "
        f"(want volume > 2.0x and rebuffer < 1.5x)",
    )


async def check_naive_detector_fires_at_night(gw) -> None:
    """A fixed threshold must produce nightly false positives.

    This is what justifies the seasonality-aware baseline in sub-project 2. If a naive
    detector were already quiet, that design choice would be decoration.
    """
    result = await gw.query(
        """
        WITH per_bucket AS (
            SELECT bucket,
                   sum(rebuffer_ms) / nullIf(sum(watched_ms), 0) AS ratio
            FROM qoe_rollup_5m
            GROUP BY bucket
        ),
        stats AS (SELECT avg(ratio) AS mu, stddevPop(ratio) AS sd FROM per_bucket)
        SELECT toHour(bucket) AS hour, count() AS alerts
        FROM per_bucket, stats
        WHERE ratio > mu + 2 * sd
        GROUP BY hour ORDER BY alerts DESC LIMIT 6
        """
    )
    if not result.rows:
        record(FAIL, "naive threshold detector fires on nightly peaks", "no alerts at all")
        return

    top = [(int(r["hour"]), int(r["alerts"])) for r in result.rows]
    total = sum(a for _, a in top)
    evening = sum(a for h, a in top if 18 <= h <= 23)
    share = evening / total if total else 0

    record(
        PASS if share > 0.5 else FAIL,
        "naive threshold detector fires on nightly peaks",
        f"{total} alerts in top hours; {share:.0%} fall in 18:00-23:00 "
        f"(top hours: {', '.join(f'{h:02d}h x{a}' for h, a in top)})",
    )


async def check_ground_truth_absent(gw, truth: dict) -> None:
    """Hard constraint: the agent must not be able to read the answers."""
    ids = [i["incident_id"] for i in truth["incidents"]]
    tables = (
        await gw.query(
            "SELECT name FROM system.tables WHERE database = currentDatabase() "
            "AND engine NOT LIKE '%View%'"
        )
    ).rows

    leaks: list[str] = []
    for row in tables:
        table = row["name"]
        cols = (
            await gw.query(
                "SELECT name FROM system.columns WHERE database = currentDatabase() "
                f"AND table = {_q(table)} AND type LIKE '%String%'"
            )
        ).rows
        for col in cols:
            column = col["name"]
            conditions = " OR ".join(f"{column} = {_q(i)}" for i in ids)
            hit = (await gw.query(f"SELECT count() AS c FROM {table} WHERE {conditions}")).rows[0][
                "c"
            ]
            if hit:
                leaks.append(f"{table}.{column} ({hit} rows)")

    record(
        PASS if not leaks else FAIL,
        "no ClickHouse table contains incident ground truth",
        "checked every String column in every table" if not leaks else f"LEAKED: {leaks}",
    )


async def main() -> int:
    load_dotenv(override=False)
    if not GROUND_TRUTH.exists():
        print(f"{GROUND_TRUTH} not found. Run the loader first.")
        return 2
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        counts = (
            await gw.query(
                "SELECT (SELECT count() FROM playback_events) AS events, "
                "(SELECT uniq(session_id) FROM playback_events) AS sessions, "
                "(SELECT count() FROM change_log) AS changes"
            )
        ).rows[0]
        print(
            f"\nDataset: {counts['events']:,} events, {counts['sessions']:,} sessions, "
            f"{counts['changes']} change-log entries\n"
        )

        print("Planted incidents must be visible:")
        for incident in truth["incidents"]:
            if incident["is_decoy"]:
                continue
            await check_incident_visible(gw, incident)

        print("\nDecoy must NOT look like a fault:")
        for incident in truth["incidents"]:
            if incident["is_decoy"]:
                await check_decoy(gw, incident)

        print("\nSeasonality must be a real problem:")
        await check_naive_detector_fires_at_night(gw)

        print("\nGround truth must be unreachable from the database:")
        await check_ground_truth_absent(gw, truth)

        print(
            f"\nMCP queries executed: {len(gw.query_log)}, "
            f"slowest {max(q.duration_ms for q in gw.query_log):.0f} ms"
        )

    failed = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 70}")
    print(f"{len(_results) - len(failed)}/{len(_results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(name for _, name, _ in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
