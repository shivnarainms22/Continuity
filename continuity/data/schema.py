"""ClickHouse DDL. Ordering keys are chosen for the drill-down access pattern.

Applied via clickhouse-connect directly: schema management is build-time ops, not
agent runtime, and mcp-clickhouse is read-only by design so it cannot run DDL anyway.
"""

from __future__ import annotations

import clickhouse_connect

from continuity.config import ClickHouseConfig

TABLES = {
    "playback_events",
    "qoe_rollup_5m",
    "qoe_rollup_5m_mv",
    "titles",
    "subscribers",
    "change_log",
}

# Every drill-down query filters a narrow time window first, so event_time leads the
# ordering key; the delivery dimensions follow in descending order of how often the
# investigation splits on them.
PLAYBACK_EVENTS = """
CREATE TABLE IF NOT EXISTS playback_events
(
    event_time    DateTime64(3, 'UTC'),
    session_id    UUID,
    subscriber_id UInt32,
    title_id      UInt32,
    device_type   LowCardinality(String),
    os_version    LowCardinality(String),
    app_version   LowCardinality(String),
    cdn           LowCardinality(String),
    pop           LowCardinality(String),
    isp           LowCardinality(String),
    country       LowCardinality(String),
    region        LowCardinality(String),
    event_type    Enum8('start'=1,'heartbeat'=2,'rebuffer'=3,'error'=4,'end'=5),
    watched_ms    UInt32,
    rebuffer_ms   UInt32,
    startup_ms    UInt32,
    bitrate_kbps  UInt32,
    error_code    LowCardinality(String)
)
ENGINE = MergeTree
PARTITION BY toDate(event_time)
ORDER BY (event_time, cdn, device_type, app_version)
"""

# title_id is deliberately EXCLUDED from the rollup. With ~500 titles it would multiply
# the group cardinality by two orders of magnitude and make the rollup larger than the
# raw table for no benefit. Title-level analysis queries playback_events directly over a
# narrow time window, which the partition + ordering key already make cheap.
QOE_ROLLUP = """
CREATE TABLE IF NOT EXISTS qoe_rollup_5m
(
    bucket       DateTime('UTC'),
    cdn          LowCardinality(String),
    pop          LowCardinality(String),
    isp          LowCardinality(String),
    device_type  LowCardinality(String),
    os_version   LowCardinality(String),
    app_version  LowCardinality(String),
    country      LowCardinality(String),
    region       LowCardinality(String),
    sessions     AggregateFunction(uniq, UUID),
    starts       SimpleAggregateFunction(sum, UInt64),
    errors       SimpleAggregateFunction(sum, UInt64),
    watched_ms   SimpleAggregateFunction(sum, UInt64),
    rebuffer_ms  SimpleAggregateFunction(sum, UInt64),
    startup_q    AggregateFunction(quantilesTDigest(0.5, 0.95), UInt32),
    bitrate_avg  AggregateFunction(avg, UInt32)
)
ENGINE = AggregatingMergeTree
PARTITION BY toDate(bucket)
ORDER BY (bucket, cdn, pop, device_type, app_version, isp, os_version, country, region)
"""

# startup_ms is only meaningful on 'start' events and bitrate only on 'heartbeat';
# aggregating unconditionally would drag both toward zero and hide real regressions.
QOE_ROLLUP_MV = """
CREATE MATERIALIZED VIEW IF NOT EXISTS qoe_rollup_5m_mv TO qoe_rollup_5m AS
SELECT
    toStartOfFiveMinute(event_time) AS bucket,
    cdn, pop, isp, device_type, os_version, app_version, country, region,
    uniqState(session_id)                                        AS sessions,
    sumSimpleState(toUInt64(event_type = 'start'))               AS starts,
    sumSimpleState(toUInt64(event_type = 'error'))               AS errors,
    sumSimpleState(toUInt64(watched_ms))                         AS watched_ms,
    sumSimpleState(toUInt64(rebuffer_ms))                        AS rebuffer_ms,
    quantilesTDigestStateIf(0.5, 0.95)(startup_ms, event_type = 'start') AS startup_q,
    avgStateIf(bitrate_kbps, event_type = 'heartbeat')           AS bitrate_avg
FROM playback_events
GROUP BY bucket, cdn, pop, isp, device_type, os_version, app_version, country, region
"""

TITLES = """
CREATE TABLE IF NOT EXISTS titles
(
    title_id     UInt32,
    name         String,
    genre        LowCardinality(String),
    content_type LowCardinality(String),
    release_date Date,
    is_premiere  UInt8
)
ENGINE = MergeTree ORDER BY title_id
"""

SUBSCRIBERS = """
CREATE TABLE IF NOT EXISTS subscribers
(
    subscriber_id UInt32,
    plan          LowCardinality(String),
    monthly_arpu  Decimal(8, 2),
    signup_date   Date,
    tenure_days   UInt16,
    country       LowCardinality(String),
    region        LowCardinality(String)
)
ENGINE = MergeTree ORDER BY subscriber_id
"""

CHANGE_LOG = """
CREATE TABLE IF NOT EXISTS change_log
(
    change_id       UInt32,
    changed_at      DateTime('UTC'),
    change_type     LowCardinality(String),
    component       String,
    description     String,
    dimension_key   LowCardinality(String),
    dimension_value LowCardinality(String)
)
ENGINE = MergeTree ORDER BY changed_at
"""

STATEMENTS = [PLAYBACK_EVENTS, QOE_ROLLUP, QOE_ROLLUP_MV, TITLES, SUBSCRIBERS, CHANGE_LOG]


def apply_schema(config: ClickHouseConfig) -> None:
    """Create the database and every table. Idempotent.

    clickhouse-connect raises on a failed statement rather than swallowing it, so a
    broken DDL statement here surfaces immediately instead of degrading into a schema
    that silently lacks a table.
    """
    bootstrap = clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.user,
        password=config.password,
        secure=config.secure,
    )
    try:
        bootstrap.command(f"CREATE DATABASE IF NOT EXISTS {config.database}")
    finally:
        bootstrap.close()

    client = clickhouse_connect.get_client(
        host=config.host,
        port=config.port,
        username=config.user,
        password=config.password,
        database=config.database,
        secure=config.secure,
    )
    try:
        for statement in STATEMENTS:
            client.command(statement)
    finally:
        client.close()
