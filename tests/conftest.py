import pytest
import pytest_asyncio
from dotenv import load_dotenv

from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

# Integration tests read connection settings from .env. Unit tests that exercise config
# parsing clear these vars themselves, so loading here does not leak into them.
load_dotenv(override=False)


@pytest_asyncio.fixture(scope="module")
async def _mcp_session():
    """One mcp-clickhouse session per test module.

    Starting the server costs roughly twenty seconds (subprocess spawn plus the first
    ClickHouse connection), so a session per test made the suite unusable. The gateway
    owns its session in a dedicated task, which is what makes sharing it safe.
    """
    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        yield gw


@pytest.fixture
def gateway(_mcp_session):
    """The shared session, with provenance state reset so tests stay independent."""
    _mcp_session.query_log.clear()
    return _mcp_session
