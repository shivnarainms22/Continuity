"""Integration tests for the MCP gateway. Require ClickHouse on localhost:8123.

These deliberately do NOT mock the MCP server. The ClickHouse hackathon track is graded
on runtime use of mcp-clickhouse, so mocking the one component under judgement would
verify nothing.
"""

import pytest

from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway, QueryError

pytestmark = pytest.mark.integration


@pytest.fixture
def config() -> ClickHouseConfig:
    return ClickHouseConfig.from_env()


async def test_returns_rows_as_dicts_keyed_by_column(gateway):
    result = await gateway.query("SELECT 1 AS n, 'a' AS s")

    assert result.rows == [{"n": 1, "s": "a"}]
    assert result.columns == ["n", "s"]


async def test_multiple_rows_preserve_order(gateway):
    result = await gateway.query("SELECT number AS n FROM numbers(3) ORDER BY n")

    assert result.rows == [{"n": 0}, {"n": 1}, {"n": 2}]


async def test_empty_result_is_empty_not_error(gateway):
    """mcp-clickhouse returns columns=[] for an empty set even though the query names one."""
    result = await gateway.query("SELECT 1 AS n WHERE 0")

    assert result.rows == []


async def test_null_and_numeric_types_survive(gateway):
    result = await gateway.query("SELECT 1.5 AS f, NULL AS nul, toUInt64(9) AS u")

    assert result.rows == [{"f": 1.5, "nul": None, "u": 9}]


async def test_sql_error_raises_and_does_not_masquerade_as_empty(gateway):
    """The critical case. A broken query must never look like 'no data'.

    If it did, every downstream stage would report "no anomaly found" and appear
    perfectly healthy while being blind.
    """
    with pytest.raises(QueryError) as excinfo:
        await gateway.query("SELECT * FROM table_that_does_not_exist")

    assert "table_that_does_not_exist" in str(excinfo.value)


async def test_syntax_error_raises(gateway):
    with pytest.raises(QueryError):
        await gateway.query("SELEKT nonsense")


async def test_failed_query_is_not_recorded_as_successful(gateway):
    with pytest.raises(QueryError):
        await gateway.query("SELECT * FROM nope_not_here")

    assert gateway.query_log == []


async def test_query_log_records_provenance(gateway):
    """Every claim in a generated brief links back to the query behind it."""
    await gateway.query("SELECT 1")
    await gateway.query("SELECT 2")

    assert [q.sql for q in gateway.query_log] == ["SELECT 1", "SELECT 2"]
    assert all(q.duration_ms >= 0 for q in gateway.query_log)
    assert [q.row_count for q in gateway.query_log] == [1, 1]


async def test_scalar_returns_single_value(gateway):
    result = await gateway.query("SELECT 42 AS answer")

    assert result.scalar() == 42


async def test_scalar_rejects_multi_row(gateway):
    result = await gateway.query("SELECT number FROM numbers(3)")

    with pytest.raises(QueryError, match="exactly 1 row"):
        result.scalar()


async def test_scalar_rejects_multi_column(gateway):
    result = await gateway.query("SELECT 1 AS a, 2 AS b")

    with pytest.raises(QueryError, match="exactly 1 column"):
        result.scalar()


async def test_use_outside_context_manager_raises(config):
    gateway = ClickHouseMCPGateway(config)

    with pytest.raises(QueryError, match="context manager"):
        await gateway.query("SELECT 1")


async def test_list_tables_reaches_the_database(gateway):
    """Proves the connection targets the configured database, not a default."""
    result = await gateway.query("SELECT name FROM system.databases WHERE name = 'continuity'")

    assert result.rows == [{"name": "continuity"}]
