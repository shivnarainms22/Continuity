"""Integration tests for the ClickHouse schema. Require ClickHouse on localhost:8123.

Schema application and data insertion go through clickhouse-connect directly, matching
the project's build-time/runtime split: DDL and bulk loading are build-time ops via
clickhouse-connect, while the MCP gateway is the sole agent-runtime read path. It is
still used here to verify `system.tables`, since that is a genuine runtime-shaped read.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import clickhouse_connect
import pytest

from continuity.config import ClickHouseConfig
from continuity.data.schema import TABLES, apply_schema

pytestmark = pytest.mark.integration

_EVENT_COLUMNS = [
    "event_time",
    "session_id",
    "subscriber_id",
    "title_id",
    "device_type",
    "os_version",
    "app_version",
    "cdn",
    "pop",
    "isp",
    "country",
    "region",
    "event_type",
    "watched_ms",
    "rebuffer_ms",
    "startup_ms",
    "bitrate_kbps",
    "error_code",
]

_BUCKET = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def config() -> ClickHouseConfig:
    return ClickHouseConfig.from_env()


@pytest.fixture(scope="module", autouse=True)
def _schema_applied(config: ClickHouseConfig) -> None:
    apply_schema(config)


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


def _event_row(
    marker: str,
    *,
    event_type: str,
    startup_ms: int = 0,
    bitrate_kbps: int = 0,
    watched_ms: int = 0,
    rebuffer_ms: int = 0,
    session_id: uuid.UUID | None = None,
) -> list:
    # `region` carries a unique per-test marker so inserted rows are cheaply
    # identifiable and never collide with another test run's leftovers, even if a
    # prior run's cleanup mutation is still in flight.
    return [
        _BUCKET,
        session_id or uuid.uuid4(),
        1,
        1,
        "roku",
        "roku_os_14.0",
        "8.2.0",
        "cdn_meridian",
        "mer-iad-1",
        "comcast",
        "US",
        marker,
        event_type,
        watched_ms,
        rebuffer_ms,
        startup_ms,
        bitrate_kbps,
        "",
    ]


def _cleanup(ch_client, marker: str) -> None:
    ch_client.command(
        "ALTER TABLE playback_events DELETE WHERE region = {marker:String}",
        parameters={"marker": marker},
    )
    ch_client.command(
        "ALTER TABLE qoe_rollup_5m DELETE WHERE region = {marker:String}",
        parameters={"marker": marker},
    )


def test_apply_schema_creates_all_tables(config: ClickHouseConfig) -> None:
    apply_schema(config)  # module fixture already applied it once; must not raise here


async def test_all_expected_tables_exist(gateway) -> None:
    result = await gateway.query("SELECT name FROM system.tables WHERE database = 'continuity'")
    assert {r["name"] for r in result.rows} >= TABLES


def test_apply_schema_is_idempotent(config: ClickHouseConfig) -> None:
    apply_schema(config)
    apply_schema(config)


def test_rollup_mv_populates_and_merges_readably(ch_client) -> None:
    """The MV is the drill-down performance structure. An MV that never fires would
    invalidate the whole design, so this proves it actually populates on insert and
    that the resulting aggregate states can be read back with the *Merge combinators.
    """
    marker = f"probe_{uuid.uuid4().hex[:12]}"
    session_a, session_b = uuid.uuid4(), uuid.uuid4()

    rows = [
        _event_row(marker, event_type="start", startup_ms=1200, session_id=session_a),
        _event_row(
            marker, event_type="heartbeat", bitrate_kbps=4000, watched_ms=5000, session_id=session_a
        ),
        _event_row(
            marker,
            event_type="heartbeat",
            bitrate_kbps=6000,
            watched_ms=5000,
            rebuffer_ms=100,
            session_id=session_b,
        ),
    ]
    try:
        ch_client.insert("playback_events", rows, column_names=_EVENT_COLUMNS)

        result = ch_client.query(
            """
            SELECT
                uniqMerge(sessions)  AS sessions,
                sum(starts)          AS starts,
                sum(watched_ms)      AS watched_ms,
                sum(rebuffer_ms)     AS rebuffer_ms,
                avgMerge(bitrate_avg) AS bitrate_avg
            FROM qoe_rollup_5m
            WHERE region = {marker:String}
            """,
            parameters={"marker": marker},
        )
        assert result.result_rows, "MV did not populate qoe_rollup_5m for the inserted events"
        sessions, starts, watched_ms, rebuffer_ms, bitrate_avg = result.result_rows[0]
        assert sessions == 2
        assert starts == 1
        assert watched_ms == 10000
        assert rebuffer_ms == 100
        assert bitrate_avg == pytest.approx(5000.0)
    finally:
        _cleanup(ch_client, marker)


def test_startup_ignores_heartbeats_and_bitrate_ignores_starts(ch_client) -> None:
    """startup_ms is only meaningful on 'start' events, bitrate only on 'heartbeat'.

    Heartbeats naturally report startup_ms=0; 'start' events carry no real bitrate. If
    the MV aggregated either unconditionally it would drag the reported values toward
    zero (startup) or an unrelated constant (bitrate) and hide real regressions.
    """
    marker = f"probe_{uuid.uuid4().hex[:12]}"
    start_startups = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]

    rows = [
        _event_row(marker, event_type="start", startup_ms=ms, bitrate_kbps=99_999)
        for ms in start_startups
    ]
    rows += [
        _event_row(marker, event_type="heartbeat", startup_ms=0, bitrate_kbps=5000)
        for _ in range(50)
    ]
    try:
        ch_client.insert("playback_events", rows, column_names=_EVENT_COLUMNS)

        result = ch_client.query(
            """
            SELECT
                quantilesTDigestMerge(0.5, 0.95)(startup_q) AS startup_q,
                avgMerge(bitrate_avg)                       AS bitrate_avg
            FROM qoe_rollup_5m
            WHERE region = {marker:String}
            """,
            parameters={"marker": marker},
        )
        (p50, p95), bitrate_avg = result.result_rows[0]

        # If the 50 zero-startup heartbeats had leaked into the aggregate, the median
        # of 60 mostly-zero values would itself be zero.
        assert p50 == pytest.approx(1450, rel=0.15)
        assert p95 > 1000

        # If the 10 start events' bitrate_kbps=99_999 had leaked in, the average would
        # be far above the real heartbeat bitrate of 5000.
        assert bitrate_avg == pytest.approx(5000.0)
    finally:
        _cleanup(ch_client, marker)
