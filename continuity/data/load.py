"""Bulk-load CLI: schema + catalog + telemetry into ClickHouse, ground truth to disk.

Uses `clickhouse-connect` directly, never the MCP gateway: this is build-time ops, not
agent runtime, and `mcp-clickhouse` is read-only by design so it could not run inserts
even if we tried. See CLAUDE.md hard constraint 2.

Ground truth (`data/ground_truth.json`) is the only artifact of this module that carries
incident truth, and it never reaches ClickHouse (hard constraint 3): the catalog and
telemetry tables only ever receive the same columns their DDL defines.

Usage: `uv run python -m continuity.data.load --days 56 [--sessions-per-day N] [--truncate]
[--seed N]`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import clickhouse_connect
import numpy as np
import typer
from dotenv import load_dotenv

from continuity.config import ClickHouseConfig
from continuity.data.catalog import Subscriber, Title, generate_subscribers, generate_titles
from continuity.data.generator import (
    CHANGE_LOG_COLUMNS,
    PLAYBACK_EVENTS_COLUMNS,
    change_log_rows,
    generate,
)
from continuity.data.incidents import PlantedIncident, build_incidents, write_ground_truth
from continuity.data.schema import apply_schema

# Fixed calendar anchor, not "today": the eval harness requires regeneration from the
# same seed to be byte-identical, and a wall-clock default would break that.
WINDOW_START = datetime(2026, 1, 1, tzinfo=UTC)

# 56 days (8 weeks): continuity.analysis.baseline's default week-over-week comparison
# needs 4 prior weeks of the same weekday for every planted incident (see
# continuity.data.incidents.build_incidents, which anchors incident placement to the
# END of the window for exactly this reason). --days remains configurable.
DEFAULT_DAYS = 56
DEFAULT_SESSIONS_PER_DAY = 250_000
DEFAULT_SEED = 20260908
DEFAULT_BATCH_SIZE = 50_000
DEFAULT_TITLE_COUNT = 500
DEFAULT_SUBSCRIBER_COUNT = 20_000
DEFAULT_GROUND_TRUTH_PATH = Path("data/ground_truth.json")

_TITLES_COLUMNS: tuple[str, ...] = (
    "title_id",
    "name",
    "genre",
    "content_type",
    "release_date",
    "is_premiere",
)
_SUBSCRIBERS_COLUMNS: tuple[str, ...] = (
    "subscriber_id",
    "plan",
    "monthly_arpu",
    "signup_date",
    "tenure_days",
    "country",
    "region",
)

# Order matters for --truncate: playback_events first is fine, but qoe_rollup_5m MUST be
# included explicitly. A materialized view is an insert trigger, not a live join --
# deleting/truncating its source table does NOT cascade to the target table it writes
# into, so omitting qoe_rollup_5m here would let repeated loads silently accumulate
# garbage in the drill-down data.
TRUNCATE_TABLES: tuple[str, ...] = (
    "playback_events",
    "qoe_rollup_5m",
    "titles",
    "subscribers",
    "change_log",
)

REPORTED_TABLES: tuple[str, ...] = TRUNCATE_TABLES

app = typer.Typer(add_completion=False)


class LoadError(RuntimeError):
    """A load step failed. Never swallowed, and the message names what failed and how
    far the load got, so a partial load is never mistaken for a successful one."""


@dataclass(frozen=True)
class LoadReport:
    row_counts: dict[str, int]
    elapsed_s: float
    ground_truth_path: Path


def _connect(config: ClickHouseConfig):
    return clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.user,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )


def truncate_all(config: ClickHouseConfig) -> None:
    """Clear playback_events, its rollup MV target, and the catalog tables.

    Exposed standalone (not just as a step inside `run_load`) so its effect -- the
    rollup actually ending up empty -- can be verified directly.
    """
    client = _connect(config)
    try:
        for table in TRUNCATE_TABLES:
            try:
                client.command(f"TRUNCATE TABLE IF EXISTS {table}")
            except Exception as exc:
                raise LoadError(f"Truncate of {table!r} failed: {exc}") from exc
    finally:
        client.close()


def _pick_incident_titles(titles: Sequence[Title]) -> tuple[int, int]:
    """Pick a premiere title for the decoy and a distinct title for the encode fault,
    deterministically from the generated catalog."""
    if not titles:
        raise LoadError("Cannot plant incidents: the generated title catalog is empty.")
    premiere = next((t for t in titles if t.is_premiere), titles[0])
    encode = next((t for t in titles if t.title_id != premiere.title_id), titles[0])
    return premiere.title_id, encode.title_id


def _title_rows(titles: Sequence[Title]) -> list[tuple]:
    return [
        (t.title_id, t.name, t.genre, t.content_type, t.release_date, int(t.is_premiere))
        for t in titles
    ]


def _subscriber_rows(subscribers: Sequence[Subscriber]) -> list[tuple]:
    return [
        (s.subscriber_id, s.plan, s.monthly_arpu, s.signup_date, s.tenure_days, s.country, s.region)
        for s in subscribers
    ]


def _change_log_row_tuples(incidents: Sequence[PlantedIncident]) -> list[tuple]:
    rows = change_log_rows(incidents)
    return [tuple(row[c] for c in CHANGE_LOG_COLUMNS) for row in rows]


def _column_values(batch: dict, column: str) -> list:
    """Convert one generated column to values clickhouse-connect stores correctly.

    Datetime columns are passed as integer epoch milliseconds, NOT as datetime objects.
    `.tolist()` on a numpy datetime64 yields a NAIVE datetime, and clickhouse-connect
    interprets a naive datetime as LOCAL time before converting it into a
    DateTime64(3, 'UTC') column. On a Pacific machine that silently shifted every
    timestamp by +8 hours, which broke alignment between the data and the incident
    windows recorded in ground truth -- and, being derived from the host timezone, would
    have produced a different dataset on a UTC host such as Cloud Run.

    Integer epoch milliseconds are unambiguous, and the cast is a free vectorised
    reinterpretation rather than a per-row Python conversion. Timezone-aware datetimes
    are equally correct but would cost a Python loop over tens of millions of rows.
    """
    values = batch[column]
    if values.dtype.kind == "M":
        return values.astype("datetime64[ms]").astype("int64").tolist()
    return values.tolist()


def _insert(client, table: str, data: list[tuple], column_names: Sequence[str]) -> int:
    try:
        client.insert(table, data, column_names=list(column_names))
    except Exception as exc:
        raise LoadError(f"Insert into {table!r} failed after 0 of {len(data)} rows: {exc}") from exc
    return len(data)


def _load_playback_events(
    client,
    batches: Iterable[dict[str, np.ndarray]],
    progress: Callable[[str], None],
) -> int:
    rows_so_far = 0
    batches_done = 0
    started = time.perf_counter()
    try:
        for batch in batches:
            batch_rows = len(batch["event_time"])
            # clickhouse-connect's column-oriented insert expects native Python values
            # (datetime.datetime, int, str, uuid.UUID), not numpy scalars/datetime64 --
            # .tolist() converts every column's numpy dtype to the matching native type.
            columns = [_column_values(batch, c) for c in PLAYBACK_EVENTS_COLUMNS]
            client.insert(
                "playback_events",
                columns,
                column_names=list(PLAYBACK_EVENTS_COLUMNS),
                column_oriented=True,
            )
            rows_so_far += batch_rows
            batches_done += 1
            elapsed = time.perf_counter() - started
            progress(
                f"playback_events: batch {batches_done} done, {rows_so_far} rows so far, "
                f"{elapsed:.1f}s elapsed"
            )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        raise LoadError(
            f"Loading playback_events failed on batch {batches_done + 1} "
            f"after {rows_so_far} rows loaded ({elapsed:.1f}s elapsed): {exc}"
        ) from exc
    return rows_so_far


def _row_counts(client, tables: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in tables:
        try:
            result = client.query(f"SELECT count() FROM {table}")
        except Exception as exc:
            raise LoadError(f"Row-count check failed for {table!r}: {exc}") from exc
        counts[table] = int(result.result_rows[0][0])
    return counts


def run_load(
    config: ClickHouseConfig,
    *,
    days: int,
    sessions_per_day: int = DEFAULT_SESSIONS_PER_DAY,
    seed: int = DEFAULT_SEED,
    truncate: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    title_count: int = DEFAULT_TITLE_COUNT,
    subscriber_count: int = DEFAULT_SUBSCRIBER_COUNT,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_PATH,
    progress: Callable[[str], None] = print,
) -> LoadReport:
    """Apply the schema, (optionally) truncate, generate + load, write ground truth.

    Raises `LoadError` naming the failed step and how far it got. Never reports success
    on a partial load.
    """
    started = time.perf_counter()

    try:
        apply_schema(config)
    except Exception as exc:
        raise LoadError(
            f"Schema application failed against {config.host}:{config.port}: {exc}"
        ) from exc

    try:
        client = _connect(config)
    except Exception as exc:
        raise LoadError(f"Could not connect to {config.host}:{config.port}: {exc}") from exc

    try:
        if truncate:
            progress("truncating existing tables...")
            for table in TRUNCATE_TABLES:
                try:
                    client.command(f"TRUNCATE TABLE IF EXISTS {table}")
                except Exception as exc:
                    raise LoadError(f"Truncate of {table!r} failed: {exc}") from exc

        seed_seq = np.random.SeedSequence(seed)
        titles_seed, subscribers_seed = seed_seq.spawn(2)
        titles = generate_titles(
            np.random.default_rng(titles_seed), title_count, as_of=WINDOW_START.date()
        )
        subscribers = generate_subscribers(
            np.random.default_rng(subscribers_seed), subscriber_count, as_of=WINDOW_START.date()
        )
        premiere_title_id, encode_title_id = _pick_incident_titles(titles)
        incidents = build_incidents(
            WINDOW_START,
            days,
            premiere_title_id=premiere_title_id,
            encode_title_id=encode_title_id,
        )

        progress(f"loading {len(titles)} titles...")
        _insert(client, "titles", _title_rows(titles), _TITLES_COLUMNS)

        progress(f"loading {len(subscribers)} subscribers...")
        _insert(client, "subscribers", _subscriber_rows(subscribers), _SUBSCRIBERS_COLUMNS)

        change_rows = _change_log_row_tuples(incidents)
        progress(f"loading {len(change_rows)} change_log rows...")
        _insert(client, "change_log", change_rows, CHANGE_LOG_COLUMNS)

        progress(
            f"generating and loading playback_events "
            f"(days={days}, sessions_per_day={sessions_per_day})..."
        )
        batches = generate(
            seed=seed,
            window_start=WINDOW_START,
            days=days,
            sessions_per_day=sessions_per_day,
            titles=titles,
            subscribers=subscribers,
            incidents=incidents,
            batch_size=batch_size,
        )
        _load_playback_events(client, batches, progress)

        progress(f"writing ground truth to {ground_truth_path}...")
        write_ground_truth(incidents, ground_truth_path, seed=seed, days=days)

        row_counts = _row_counts(client, REPORTED_TABLES)
    finally:
        client.close()

    elapsed_s = time.perf_counter() - started
    progress(
        f"done in {elapsed_s:.1f}s: " + ", ".join(f"{t}={n}" for t, n in row_counts.items())
    )
    return LoadReport(
        row_counts=row_counts, elapsed_s=elapsed_s, ground_truth_path=ground_truth_path
    )


@app.command()
def main(
    days: int = typer.Option(DEFAULT_DAYS, "--days", help="Days of telemetry to generate."),
    sessions_per_day: int = typer.Option(
        DEFAULT_SESSIONS_PER_DAY, "--sessions-per-day", help="Sessions per nominal weekday."
    ),
    truncate: bool = typer.Option(
        False, "--truncate", help="Clear playback_events, qoe_rollup_5m, and the catalog first."
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed", help="Generation seed."),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, "--batch-size", help="playback_events rows per insert batch."
    ),
    ground_truth_path: Path = typer.Option(  # noqa: B008 -- Path is mutable-by-annotation to
        # ruff's B008 check, but this typer.Option() singleton is only ever read, not mutated.
        DEFAULT_GROUND_TRUTH_PATH,
        "--ground-truth-path",
        help="Where to write ground truth. Never a ClickHouse destination.",
    ),
) -> None:
    load_dotenv(override=False)  # never clobbers vars already exported in the shell
    config = ClickHouseConfig.from_env()
    try:
        report = run_load(
            config,
            days=days,
            sessions_per_day=sessions_per_day,
            seed=seed,
            truncate=truncate,
            batch_size=batch_size,
            ground_truth_path=ground_truth_path,
            progress=typer.echo,
        )
    except LoadError as exc:
        typer.echo(f"LOAD FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Load complete in {report.elapsed_s:.1f}s. Row counts: {report.row_counts}")


if __name__ == "__main__":
    app()
