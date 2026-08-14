"""Integration tests for the bulk-load CLI. Require ClickHouse on localhost:8123.

Uses `clickhouse-connect` directly for setup/verification, matching the build-time /
agent-runtime split: loading is build-time ops, the MCP gateway is the runtime read path.

Every test drives a small dataset (tens/hundreds of sessions, not millions) so the suite
stays fast, and every test that asserts on row counts truncates first so ordering between
tests never matters.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import clickhouse_connect
import pytest

from continuity.config import ClickHouseConfig
from continuity.data import load as load_module

pytestmark = pytest.mark.integration

# Small catalog sizes so every run_load() call in this file finishes in a couple of
# seconds, not the tens of seconds a full-size (500 title / 20k subscriber) catalog costs.
_TITLE_COUNT = 30
_SUBSCRIBER_COUNT = 200

_REPORTED_TABLES = ("playback_events", "qoe_rollup_5m", "titles", "subscribers", "change_log")

# These tests truncate and reload, so they must never touch the working database.
_TEST_DATABASE = "continuity_test"


@pytest.fixture(scope="module")
def config() -> ClickHouseConfig:
    """Point every test in this file at a DEDICATED database.

    These tests call the loader with truncate=True. Run against the working database
    they silently destroy whatever is loaded there and replace it with a 1-day toy
    dataset -- which is exactly what happened once: a full `pytest` run wiped a 59.8M-row
    dataset, and the damage only surfaced later as a benchmark returning zero rows.

    Isolation belongs here rather than in a convention about when it is safe to run the
    suite, because a test suite that destroys real data is a trap that will be sprung
    again.
    """
    return dataclasses.replace(ClickHouseConfig.from_env(), database=_TEST_DATABASE)


@pytest.fixture(scope="module", autouse=True)
def _working_database_must_be_untouched():
    """Fail loudly if anything in this file writes to the working database.

    Every test here truncates and reloads. Twice now a path escaped the isolation (first
    the in-process config, then a CLI subprocess inheriting os.environ) and destroyed a
    fully loaded dataset, and BOTH times the suite reported all-green -- the assertions
    were satisfied by the toy data the leak had just written.

    Comparing the working database's row count across the module turns that silent
    destruction into a failure that names itself.
    """
    working = ClickHouseConfig.from_env()
    if working.database == _TEST_DATABASE:
        pytest.fail(
            f"CLICKHOUSE_DATABASE is set to {_TEST_DATABASE!r}, so this guard cannot "
            "distinguish the working database from the test one."
        )

    def count() -> int | None:
        client = clickhouse_connect.get_client(
            host=working.host, port=working.port, username=working.user,
            password=working.password, database=working.database, secure=working.secure,
        )
        try:
            return int(client.query("SELECT count() FROM playback_events").result_rows[0][0])
        except Exception:
            return None  # table absent is fine; there is simply nothing to protect
        finally:
            client.close()

    before = count()
    yield
    after = count()
    if before is not None and after != before:
        pytest.fail(
            f"tests in this file changed the WORKING database {working.database!r}: "
            f"playback_events went from {before:,} to {after:,} rows. Something bypassed "
            f"the {_TEST_DATABASE!r} isolation -- check for a subprocess inheriting "
            "os.environ, or a config built from the environment rather than the fixture."
        )


@pytest.fixture
def ch_client(config: ClickHouseConfig):
    client = clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.user,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )
    try:
        yield client
    finally:
        client.close()


def _subprocess_env(**overrides: str) -> dict[str, str]:
    """Environment for a CLI subprocess, pinned to the dedicated test database.

    The in-process fixture isolates via dataclasses.replace on the config object, which a
    SUBPROCESS never sees: it builds its own config from .env and would target the working
    database. That gap destroyed a fully loaded 63.85M-row dataset twice, and both times
    the suite reported all-green because every assertion in this file was satisfied by the
    toy dataset it had just written.

    Passing os.environ straight through to a subprocess that runs with --truncate is the
    bug. Always build the environment here.
    """
    env = dict(os.environ)
    env["CLICKHOUSE_DATABASE"] = _TEST_DATABASE
    env.update(overrides)
    return env


def _count(ch_client, table: str) -> int:
    return int(ch_client.query(f"SELECT count() FROM {table}").result_rows[0][0])


def _run(config: ClickHouseConfig, ground_truth_path: Path, **overrides):
    kwargs = dict(
        days=1,
        sessions_per_day=500,
        seed=20260908,
        truncate=True,
        batch_size=500,
        title_count=_TITLE_COUNT,
        subscriber_count=_SUBSCRIBER_COUNT,
        ground_truth_path=ground_truth_path,
        progress=lambda _msg: None,
    )
    kwargs.update(overrides)
    return load_module.run_load(config, **kwargs)


# --- end-to-end load ---------------------------------------------------------------


def test_load_populates_every_table_and_writes_ground_truth(config, tmp_path):
    ground_truth_path = tmp_path / "ground_truth.json"
    report = _run(config, ground_truth_path)

    for table in _REPORTED_TABLES:
        assert report.row_counts[table] > 0, f"{table} has no rows after load"

    assert ground_truth_path.exists()
    payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    assert len(payload["incidents"]) == 4


def test_mv_actually_fires_on_insert(config, ch_client, tmp_path):
    """The rollup is a materialized view (an insert trigger); this proves it actually
    populates rather than silently staying empty while playback_events fills up."""
    _run(config, tmp_path / "ground_truth.json")
    assert _count(ch_client, "playback_events") > 0
    assert _count(ch_client, "qoe_rollup_5m") > 0


def test_stored_timestamps_are_utc_and_not_shifted_by_the_host_timezone(
    config, ch_client, tmp_path
):
    """Stored event_time must equal the generated instant exactly.

    Regression guard. `.tolist()` on a numpy datetime64 yields a NAIVE datetime, and
    clickhouse-connect interprets a naive datetime as LOCAL time before writing it into
    a DateTime64(3, 'UTC') column. On a Pacific host that shifted every timestamp by
    +8 hours, so the incident windows recorded in ground truth no longer pointed at the
    events they described -- and because the offset comes from the host timezone, the
    dataset differed between a developer laptop and a UTC host such as Cloud Run.

    Asserting against a fixed expected instant rather than "close to now" is the point:
    a timezone bug reproduces only where the offset is non-zero.
    """
    _run(config, tmp_path / "ground_truth.json")

    lo, hi = ch_client.query(
        "SELECT min(event_time), max(event_time) FROM playback_events"
    ).result_rows[0]

    window_start = load_module.WINDOW_START.replace(tzinfo=None)
    offset_hours = (lo.replace(tzinfo=None) - window_start).total_seconds() / 3600
    assert -1 < offset_hours < 1, (
        f"earliest stored event {lo} is {offset_hours:+.1f}h from the generation window "
        f"start {window_start}; expected under an hour. A non-zero whole-hour offset "
        f"means naive datetimes were reinterpreted through the host timezone on insert."
    )
    # One day of data plus the tail of sessions that began just before midnight.
    assert (hi.replace(tzinfo=None) - window_start).total_seconds() < 26 * 3600, (
        f"latest stored event {hi} is more than 26h after the window start "
        f"{window_start} for a 1-day load -- timestamps were shifted forward on insert"
    )


# --- truncate clears the rollup too -------------------------------------------------


def test_truncate_all_empties_playback_events_and_the_rollup(config, ch_client, tmp_path):
    """Deleting playback_events does NOT cascade to the MV's target table -- it must be
    truncated explicitly, or repeated loads silently accumulate garbage in qoe_rollup_5m."""
    _run(config, tmp_path / "ground_truth.json")
    assert _count(ch_client, "qoe_rollup_5m") > 0  # sanity: there was something to clear

    load_module.truncate_all(config)

    for table in _REPORTED_TABLES:
        assert _count(ch_client, table) == 0, f"{table} not empty after truncate_all"


def _rollup_totals(ch_client) -> tuple:
    """Merge-invariant totals for qoe_rollup_5m.

    Raw count() on an AggregatingMergeTree reflects how many unmerged parts happen to
    exist at query time (a background-merge-timing artifact), not the logical data, so
    it is the wrong thing to compare for idempotency. Summed/merged aggregates are
    invariant regardless of merge state and are what actually proves "no accumulation".
    """
    row = ch_client.query(
        "SELECT sum(starts), sum(errors), sum(watched_ms), sum(rebuffer_ms), uniqMerge(sessions) "
        "FROM qoe_rollup_5m"
    ).result_rows[0]
    return tuple(row)


def test_rerunning_with_truncate_produces_identical_row_counts(config, ch_client, tmp_path):
    """Idempotency: loading the same window twice with --truncate must not accumulate."""
    first = _run(config, tmp_path / "gt1.json")
    first_rollup = _rollup_totals(ch_client)
    second = _run(config, tmp_path / "gt2.json")
    second_rollup = _rollup_totals(ch_client)

    first_non_rollup = {t: n for t, n in first.row_counts.items() if t != "qoe_rollup_5m"}
    second_non_rollup = {t: n for t, n in second.row_counts.items() if t != "qoe_rollup_5m"}
    assert first_non_rollup == second_non_rollup
    assert first_rollup == second_rollup


# --- ground truth never enters ClickHouse -------------------------------------------


def test_no_clickhouse_table_contains_incident_ground_truth(config, ch_client, tmp_path):
    ground_truth_path = tmp_path / "ground_truth.json"
    _run(config, ground_truth_path)
    payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    incident_ids = [inc["incident_id"] for inc in payload["incidents"]]
    assert incident_ids  # sanity: ground truth actually has incidents to check for

    # No column anywhere is named after ground-truth-only concepts.
    forbidden_columns = {
        "incident_id",
        "affected_fraction",
        "volume_multiplier",
        "is_decoy",
        "effects",
        "predicate",
    }
    columns = ch_client.query(
        "SELECT name FROM system.columns WHERE database = currentDatabase()"
    ).result_rows
    column_names = {row[0] for row in columns}
    assert not (forbidden_columns & column_names)

    # No literal incident id string leaked into any change_log text field.
    rows = ch_client.query(
        "SELECT description, component, dimension_key, dimension_value FROM change_log"
    ).result_rows
    for row in rows:
        for value in row:
            for incident_id in incident_ids:
                assert incident_id not in str(value)


# --- loud errors ---------------------------------------------------------------------


def test_partial_playback_events_insert_failure_raises_with_progress_context(
    config, tmp_path, monkeypatch
):
    """An insert failing partway through playback_events must not be swallowed, and the
    error must name the table and how many rows were loaded before it broke."""
    real_get_client = load_module.clickhouse_connect.get_client
    call_count = {"playback_events": 0}

    def fake_get_client(**kwargs):
        client = real_get_client(**kwargs)
        real_insert = client.insert

        def failing_insert(table, *args, **kw):
            if table == "playback_events":
                call_count["playback_events"] += 1
                if call_count["playback_events"] == 2:
                    raise RuntimeError("simulated ClickHouse insert failure")
            return real_insert(table, *args, **kw)

        client.insert = failing_insert
        return client

    monkeypatch.setattr(load_module.clickhouse_connect, "get_client", fake_get_client)

    with pytest.raises(load_module.LoadError) as exc_info:
        _run(
            config,
            tmp_path / "ground_truth.json",
            sessions_per_day=3000,
            batch_size=300,
        )

    message = str(exc_info.value)
    assert "playback_events" in message
    assert "rows loaded" in message
    assert call_count["playback_events"] == 2  # proves it failed on the 2nd batch, not the 1st


def test_bad_config_fails_loudly_instead_of_reporting_success(tmp_path):
    bad_config = ClickHouseConfig(
        host="localhost", port=8123, user="default", password="not-the-real-password",
        database="continuity", secure=False,
    )
    with pytest.raises(load_module.LoadError):
        _run(bad_config, tmp_path / "ground_truth.json")


def test_cli_exits_nonzero_and_prints_failure_on_bad_credentials(tmp_path):
    env = _subprocess_env(CLICKHOUSE_PASSWORD="definitely-wrong")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "continuity.data.load",
            "--days",
            "1",
            "--sessions-per-day",
            "50",
            "--ground-truth-path",
            str(tmp_path / "ground_truth.json"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "LOAD FAILED" in result.stderr


# --- progress reporting ---------------------------------------------------------------


def test_progress_callback_reports_batches_rows_and_elapsed(config, tmp_path):
    messages: list[str] = []
    _run(
        config,
        tmp_path / "ground_truth.json",
        sessions_per_day=3000,
        batch_size=300,
        progress=messages.append,
    )
    batch_messages = [m for m in messages if "playback_events" in m and "batch" in m]
    assert len(batch_messages) > 1, "expected multiple batches to be reported individually"
    assert any("rows so far" in m for m in batch_messages)
    assert any("elapsed" in m for m in batch_messages)


# --- CLI wiring end-to-end -------------------------------------------------------------


def test_cli_runs_end_to_end_with_a_small_dataset(config, ch_client, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "continuity.data.load",
            "--days",
            "1",
            "--sessions-per-day",
            "500",
            "--truncate",
            "--seed",
            "20260908",
            "--ground-truth-path",
            str(tmp_path / "ground_truth.json"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "Load complete" in result.stdout
    assert (tmp_path / "ground_truth.json").exists()
    assert _count(ch_client, "playback_events") > 0
