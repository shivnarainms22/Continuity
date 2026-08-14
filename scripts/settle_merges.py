"""Force ClickHouse to finish its background merges, one partition at a time.

Why this exists
---------------
The dataset is written once and read forever. But ClickHouse leaves the loaded data in
many parts (measured: 114 on playback_events, 124 on qoe_rollup_5m) and merges them
whenever it decides to. Those merges were measured holding ~3.9 GiB of a 5.97 GiB server
ceiling, after which OvercommitTracker kills unrelated, cheap foreground queries with
`Code 241 MEMORY_LIMIT_EXCEEDED`.

That is why the failure looked random: it depended on merge timing, not query shape. It
corrupted two arms of the agent-vs-walker comparison and several integration runs, and it
would be far worse during a live demo.

Settling the merges deliberately removes the pressure instead of hoping it does not fire.
Per-partition rather than whole-table, so each merge stays small enough to complete on a
container with a 5.97 GiB ceiling -- an `OPTIMIZE TABLE ... FINAL` across 56 days at once
is exactly the giant merge we are trying to avoid.

    uv run python scripts/settle_merges.py
"""

from __future__ import annotations

import sys
import time

import clickhouse_connect
from dotenv import load_dotenv

from continuity.config import ClickHouseConfig

# Only tables large enough for part count to matter. The tiny dimension tables are
# already single-part.
TABLES = ("playback_events", "qoe_rollup_5m")


def main() -> int:
    load_dotenv(override=False)
    config = ClickHouseConfig.from_env()
    client = clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.user,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )
    try:
        for table in TABLES:
            rows = client.query(
                "SELECT partition, count() AS parts FROM system.parts "
                "WHERE database = {db:String} AND table = {tbl:String} AND active "
                "GROUP BY partition HAVING parts > 1 ORDER BY partition",
                parameters={"db": config.database, "tbl": table},
            ).result_rows
            if not rows:
                print(f"{table}: already one part per partition, nothing to settle")
                continue

            print(f"{table}: {len(rows)} partition(s) with more than one part")
            started = time.perf_counter()
            for index, (partition, _parts) in enumerate(rows, start=1):
                # One partition at a time keeps each merge's memory bounded. FINAL forces
                # the merge now rather than leaving it to the background scheduler.
                client.command(
                    f"OPTIMIZE TABLE {table} PARTITION '{partition}' FINAL",
                    settings={"optimize_throw_if_noop": 0, "max_threads": 4},
                )
                if index % 10 == 0 or index == len(rows):
                    print(
                        f"  {index}/{len(rows)} partitions settled "
                        f"({time.perf_counter() - started:.0f}s elapsed)"
                    )
            print(f"{table}: settled in {time.perf_counter() - started:.0f}s")

        print("\n--- after ---")
        for row in client.query(
            "SELECT table, count() AS parts, formatReadableSize(sum(bytes_on_disk)) AS size "
            "FROM system.parts WHERE database = {db:String} AND active "
            "GROUP BY table ORDER BY parts DESC",
            parameters={"db": config.database},
        ).result_rows:
            print(f"  {row[0]:<20}{row[1]:>6} parts  {row[2]}")

        merge_mem = client.query(
            "SELECT value FROM system.metrics WHERE metric = 'MergesMutationsMemoryTracking'"
        ).result_rows[0][0]
        print(f"\nmerge memory now held: {merge_mem / (1024**2):.0f} MiB")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
