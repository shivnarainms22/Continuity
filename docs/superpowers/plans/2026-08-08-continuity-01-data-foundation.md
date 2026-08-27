# Continuity Sub-project 1: Data Foundation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a ClickHouse cluster holding realistic streaming-telemetry with deliberately planted incidents, reachable through the official `mcp-clickhouse` server, with ground truth recorded outside the database.

**Architecture:** A synthetic telemetry generator produces heartbeat-level playback events exhibiting real diurnal/weekly seasonality and load-correlated QoE degradation. Four incidents are injected with known blast radii, one of which is a decoy that looks like an incident but has healthy QoE. Events land in a `MergeTree` table with an `AggregatingMergeTree` rollup materialized view for interactive drill-down. All agent-runtime reads go through `mcp-clickhouse`; only bulk loading uses `clickhouse-connect` directly.

**Tech Stack:** Python 3.13, uv, ClickHouse 25.x (Docker), `clickhouse-connect` (loading), `mcp-clickhouse` + `mcp` (runtime reads), pytest, NumPy.

---

## Why the seasonality work is not optional

The generator makes QoE *genuinely worse at peak hours* — rebuffering rises with concurrency, as it does in reality. This is deliberate and load-bearing:

- A naive threshold detector fires every single night at 21:00. That is the false-positive problem real ops teams actually have.
- It forces the seasonality-aware baseline in sub-project 2 to be real work rather than decoration.
- It gives the demo a strong 20 seconds: show the naive detector screaming nightly, then show Continuity silent on those and loud on the real incident.

If the generator produces flat baseline QoE, sub-project 2 becomes trivial and the project loses its most credible claim.

---

## File structure

```
D:\Hackathon\
├── CLAUDE.md                       # project conventions, exact commands (REQUIRED by global rules)
├── LICENSE                         # Apache-2.0
├── pyproject.toml                  # uv-managed, deps pinned
├── docker-compose.yml              # local ClickHouse
├── .env.example                    # documented, no secrets
├── .gitignore
├── continuity/
│   ├── __init__.py
│   ├── config.py                   # ClickHouseConfig, GenerationConfig — env parsing + validation
│   ├── gateway/
│   │   ├── __init__.py
│   │   └── mcp_gateway.py          # ClickHouseMCPGateway — the ONLY runtime read path
│   └── data/
│       ├── __init__.py
│       ├── schema.py               # DDL as constants + apply_schema()
│       ├── topology.py             # CDN/PoP/ISP/device/geo dimension universe
│       ├── catalog.py              # titles + subscribers generation
│       ├── seasonality.py          # diurnal/weekly/noise curves — PURE FUNCTIONS
│       ├── incidents.py            # PlantedIncident defs + ground-truth serialization
│       ├── generator.py            # event stream assembly
│       └── load.py                 # CLI entrypoint
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_seasonality.py
│   ├── test_topology.py
│   ├── test_catalog.py
│   ├── test_incidents.py
│   ├── test_generator.py
│   ├── integration/
│   │   ├── test_schema.py          # needs Docker ClickHouse
│   │   └── test_mcp_gateway.py     # needs Docker ClickHouse + mcp-clickhouse
└── data/
    └── ground_truth.json           # generated; gitignored; NEVER loaded into ClickHouse
```

`seasonality.py`, `incidents.py` and `topology.py` are pure and dependency-free by design — they carry the subtle logic and must be testable without Docker.

---

## Chunk 1: Scaffold and de-risk the mandated integration

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, `CLAUDE.md`, `continuity/__init__.py`

- [ ] **Step 1: Initialise the repo and Python project**

```bash
cd /d/Hackathon
git init
git branch -m main
git checkout -b feat/data-foundation
uv init --python 3.13 --no-workspace
```

Never work on `main` — the branch is created before any code exists.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "continuity"
version = "0.1.0"
description = "Agentic incident investigation for streaming video QoE"
requires-python = ">=3.13"
dependencies = [
    "clickhouse-connect>=0.8",
    "mcp-clickhouse>=0.1.10",
    "mcp>=1.9",
    "numpy>=2.1",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "python-dotenv>=1.0",
    "typer>=0.15",
]

[dependency-groups]
dev = ["pytest>=8.3", "pytest-asyncio>=0.24", "ruff>=0.8"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["integration: requires a running ClickHouse (deselect with '-m \"not integration\"')"]

[tool.ruff]
line-length = 100
target-version = "py313"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**No AI package appears here.** Google AI packages arrive in sub-project 3 and will be `google-adk` / `google-genai` only. Any other vendor's AI SDK entering this file disqualifies the submission.

- [ ] **Step 3: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.env
data/*.json
data/*.parquet
.pytest_cache/
.ruff_cache/
node_modules/
dist/
```

`data/ground_truth.json` is gitignored — it is regenerated deterministically from a seed, and keeping it out of the repo makes it obvious the agent never reads it.

- [ ] **Step 4: Write `.env.example`**

```bash
# Local Docker ClickHouse (development)
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=continuity_dev
CLICKHOUSE_DATABASE=continuity
CLICKHOUSE_SECURE=false

# Generation
CONTINUITY_SEED=20260908
CONTINUITY_DAYS=21
CONTINUITY_SESSIONS_PER_DAY=250000
```

- [ ] **Step 5: Add the Apache-2.0 LICENSE**

```bash
curl -sL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

Confirm the file is ~11KB and begins with "Apache License". The rules require the license be *detectable in the repo About section*; if GitHub does not show it, replace with the canonical text from `github.com/licenses`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold continuity project"
```

Expected: one commit, no `.env` in the tree.

---

### Task 2: Local ClickHouse

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  clickhouse:
    image: clickhouse/clickhouse-server:25.8
    container_name: continuity-ch
    ports:
      - "8123:8123"   # HTTP — this is what mcp-clickhouse uses
      - "9000:9000"   # native
    environment:
      CLICKHOUSE_DB: continuity
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: continuity_dev
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: 1
    ulimits:
      nofile: { soft: 262144, hard: 262144 }
    volumes:
      - ch_data:/var/lib/clickhouse
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8123/ping"]
      interval: 5s
      timeout: 3s
      retries: 20

volumes:
  ch_data:
```

`mcp-clickhouse` connects over the **HTTP interface (8123/8443), not native TCP** — verified against the upstream README. Port 8123 must be the one exposed.

- [ ] **Step 2: Start it and wait for health**

```bash
docker compose up -d
docker compose ps
```

Expected: `continuity-ch` with status `healthy` within ~30s.

- [ ] **Step 3: Verify connectivity over HTTP**

```bash
curl -s "http://localhost:8123/?user=default&password=continuity_dev" --data-binary "SELECT version()"
```

Expected: a version string like `25.8.x.x`. If this fails, nothing downstream can work — stop and fix here.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml && git commit -m "feat: local clickhouse via docker compose"
```

---

### Task 3: Config with validation

**Files:**
- Create: `continuity/config.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest
from continuity.config import ClickHouseConfig

def test_from_env_reads_all_fields(monkeypatch):
    for k, v in {
        "CLICKHOUSE_HOST": "ch.example.com", "CLICKHOUSE_PORT": "8443",
        "CLICKHOUSE_USER": "reader", "CLICKHOUSE_PASSWORD": "pw",
        "CLICKHOUSE_DATABASE": "continuity", "CLICKHOUSE_SECURE": "true",
    }.items():
        monkeypatch.setenv(k, v)
    cfg = ClickHouseConfig.from_env()
    assert cfg.host == "ch.example.com"
    assert cfg.port == 8443
    assert cfg.secure is True

def test_missing_password_raises_with_actionable_message(monkeypatch):
    monkeypatch.delenv("CLICKHOUSE_PASSWORD", raising=False)
    monkeypatch.setenv("CLICKHOUSE_HOST", "localhost")
    with pytest.raises(ValueError, match="CLICKHOUSE_PASSWORD"):
        ClickHouseConfig.from_env()

@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True),
    ("false", False), ("False", False), ("0", False), ("no", False),
])
def test_secure_parses_common_boolean_spellings(monkeypatch, raw, expected):
    for k, v in {"CLICKHOUSE_HOST": "h", "CLICKHOUSE_PASSWORD": "p"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CLICKHOUSE_SECURE", raw)
    assert ClickHouseConfig.from_env().secure is expected

def test_repr_does_not_leak_password(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_HOST", "h")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "hunter2")
    assert "hunter2" not in repr(ClickHouseConfig.from_env())
```

The password-redaction test is not ceremony: this object gets logged during agent runs and screenshotted in a demo video.

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'continuity.config'`

- [ ] **Step 3: Implement**

```python
"""Environment-backed configuration. Fails loudly and early on missing values."""
from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or export {name} in your shell."
        )
    return value


def _parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    lowered = raw.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise ValueError(f"{name}={raw!r} is not a recognised boolean.")


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    secure: bool

    @classmethod
    def from_env(cls) -> "ClickHouseConfig":
        port_raw = os.environ.get("CLICKHOUSE_PORT", "8123").strip()
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(f"CLICKHOUSE_PORT={port_raw!r} is not an integer.") from exc
        if not 1 <= port <= 65535:
            raise ValueError(f"CLICKHOUSE_PORT={port} is outside 1-65535.")
        return cls(
            host=_require("CLICKHOUSE_HOST"),
            port=port,
            user=os.environ.get("CLICKHOUSE_USER", "default").strip() or "default",
            password=_require("CLICKHOUSE_PASSWORD"),
            database=os.environ.get("CLICKHOUSE_DATABASE", "continuity").strip() or "continuity",
            secure=_parse_bool("CLICKHOUSE_SECURE", default=False),
        )

    def __repr__(self) -> str:
        return (
            f"ClickHouseConfig(host={self.host!r}, port={self.port}, user={self.user!r}, "
            f"password='***', database={self.database!r}, secure={self.secure})"
        )
```

- [ ] **Step 4: Verify passing**

Run: `uv run pytest tests/test_config.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add continuity/config.py tests/test_config.py
git commit -m "feat: clickhouse config with validation and password redaction"
```

---

### Task 4: Discover the mcp-clickhouse response shape

**This task writes no production code. It exists to stop us guessing.**

Lesson `measure-the-sim-dont-derive-it` applies directly: the exact structure `run_query` returns over MCP is *readable*, so read it rather than inferring it from documentation and then debugging a parser against a wrong assumption.

**Files:**
- Create: `scripts/probe_mcp.py` (throwaway; deleted at the end of the task)

- [ ] **Step 1: Write the probe**

```python
"""Throwaway: print the raw shape of every mcp-clickhouse response. Delete after Task 5."""
import asyncio, json, os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main() -> None:
    params = StdioServerParameters(
        command="uv",
        args=["run", "--with", "mcp-clickhouse", "--python", "3.13", "mcp-clickhouse"],
        env={**os.environ,
             "CLICKHOUSE_HOST": "localhost", "CLICKHOUSE_PORT": "8123",
             "CLICKHOUSE_USER": "default", "CLICKHOUSE_PASSWORD": "continuity_dev",
             "CLICKHOUSE_SECURE": "false"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("=== TOOLS ===")
            for t in tools.tools:
                print(f"  {t.name}: {json.dumps(t.inputSchema)}")

            for label, name, args in [
                ("SIMPLE", "run_query", {"query": "SELECT 1 AS n, 'a' AS s"}),
                ("EMPTY", "run_query", {"query": "SELECT 1 WHERE 0"}),
                ("TYPES", "run_query", {"query": "SELECT toDateTime('2026-08-08 12:00:00') AS d, 1.5 AS f, NULL AS n"}),
                ("ERROR", "run_query", {"query": "SELECT * FROM does_not_exist"}),
                ("DATABASES", "list_databases", {}),
            ]:
                print(f"\n=== {label} ===")
                try:
                    res = await session.call_tool(name, args)
                    print(f"isError={res.isError}")
                    for c in res.content:
                        print(f"  type={type(c).__name__} {getattr(c, 'text', c)!r}")
                except Exception as exc:
                    print(f"  RAISED {type(exc).__name__}: {exc}")

asyncio.run(main())
```

- [ ] **Step 2: Run it and record the output**

Run: `uv run python scripts/probe_mcp.py`

Record verbatim in the commit message:
1. Exact tool names and their input schemas (is the parameter `query` or `sql`?)
2. Result content shape — JSON string in `TextContent`? What key holds rows?
3. **How an empty result set is represented** — this is where parsers silently break
4. **How a SQL error surfaces** — `isError=True`, or a raised exception, or an error string in content?

Item 4 matters most. Lesson `lenient-parser-hides-dead-fetch`: if a failed query returns something that parses as "no rows", every downstream stage reports "no anomaly found" and looks healthy while being blind.

- [ ] **Step 3: Commit the findings**

```bash
git add scripts/probe_mcp.py
git commit -m "chore: probe mcp-clickhouse response shapes

<paste the recorded output here>"
```

---

### Task 5: The MCP gateway

**Files:**
- Create: `continuity/gateway/__init__.py`, `continuity/gateway/mcp_gateway.py`, `tests/integration/test_mcp_gateway.py`
- Delete: `scripts/probe_mcp.py`

**Write the parser against the recorded probe output, not against expectations.**

- [ ] **Step 1: Write the failing integration test**

```python
import pytest
from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway, QueryError

pytestmark = pytest.mark.integration

async def test_returns_typed_rows():
    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        result = await gw.query("SELECT 1 AS n, 'a' AS s")
    assert result.rows == [{"n": 1, "s": "a"}]
    assert result.sql == "SELECT 1 AS n, 'a' AS s"

async def test_empty_result_is_empty_not_error():
    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        result = await gw.query("SELECT 1 AS n WHERE 0")
    assert result.rows == []

async def test_sql_error_raises_and_does_not_masquerade_as_empty():
    """The critical case: a broken query must NOT look like 'no data'."""
    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        with pytest.raises(QueryError) as exc:
            await gw.query("SELECT * FROM table_that_does_not_exist")
    assert "table_that_does_not_exist" in str(exc.value)

async def test_query_is_recorded_for_provenance():
    """Every brief claim must link to its SQL, so the gateway records it."""
    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        await gw.query("SELECT 1")
        await gw.query("SELECT 2")
        assert [q.sql for q in gw.query_log] == ["SELECT 1", "SELECT 2"]
        assert all(q.duration_ms >= 0 for q in gw.query_log)
```

The provenance test is what makes "click any number to see the SQL" possible later. Building it into the gateway now costs nothing; retrofitting it in sub-project 4 would be painful.

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/integration/test_mcp_gateway.py -v -m integration`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Shape the response parsing to the **probe output from Task 4**. Skeleton — fill `_parse_result` from real observed data:

```python
"""The single runtime read path to ClickHouse, via the official mcp-clickhouse server.

The ClickHouse hackathon track requires runtime access to go through mcp-clickhouse.
Bulk loading (continuity/data/load.py) uses clickhouse-connect directly, which is
build-time ops rather than agent runtime.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import TracebackType

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from continuity.config import ClickHouseConfig


class QueryError(RuntimeError):
    """A query failed. Never swallowed, never degraded into an empty result."""


@dataclass(frozen=True)
class ExecutedQuery:
    sql: str
    duration_ms: float
    row_count: int


@dataclass
class QueryResult:
    sql: str
    rows: list[dict]

    def scalar(self):
        """Single value from a single-row single-column result."""
        if len(self.rows) != 1:
            raise QueryError(f"scalar() needs exactly 1 row, got {len(self.rows)}: {self.sql}")
        values = list(self.rows[0].values())
        if len(values) != 1:
            raise QueryError(f"scalar() needs exactly 1 column, got {len(values)}: {self.sql}")
        return values[0]


class ClickHouseMCPGateway:
    def __init__(self, config: ClickHouseConfig) -> None:
        self._config = config
        self._session: ClientSession | None = None
        self._stack: list = []
        self.query_log: list[ExecutedQuery] = []

    def _server_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command="uv",
            args=["run", "--with", "mcp-clickhouse", "--python", "3.13", "mcp-clickhouse"],
            env={
                "CLICKHOUSE_HOST": self._config.host,
                "CLICKHOUSE_PORT": str(self._config.port),
                "CLICKHOUSE_USER": self._config.user,
                "CLICKHOUSE_PASSWORD": self._config.password,
                "CLICKHOUSE_DATABASE": self._config.database,
                "CLICKHOUSE_SECURE": "true" if self._config.secure else "false",
            },
        )

    async def __aenter__(self) -> "ClickHouseMCPGateway":
        client = stdio_client(self._server_params())
        read, write = await client.__aenter__()
        self._stack.append(client)
        session = ClientSession(read, write)
        await session.__aenter__()
        self._stack.append(session)
        await session.initialize()
        self._session = session
        return self

    async def __aexit__(self, exc_type, exc, tb: TracebackType | None) -> None:
        self._session = None
        for ctx in reversed(self._stack):
            try:
                await ctx.__aexit__(exc_type, exc, tb)
            except Exception:
                pass  # teardown failures must not mask the original error
        self._stack.clear()

    async def query(self, sql: str) -> QueryResult:
        if self._session is None:
            raise QueryError("Gateway used outside its async context manager.")
        started = time.perf_counter()
        response = await self._session.call_tool("run_query", {"query": sql})
        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.isError:
            raise QueryError(f"ClickHouse rejected query: {_content_text(response)}\nSQL: {sql}")

        rows = _parse_rows(_content_text(response), sql)
        self.query_log.append(ExecutedQuery(sql=sql, duration_ms=elapsed_ms, row_count=len(rows)))
        return QueryResult(sql=sql, rows=rows)


def _content_text(response) -> str:
    return "\n".join(getattr(c, "text", "") for c in response.content)


def _parse_rows(text: str, sql: str) -> list[dict]:
    """Shape this to the probe output from Task 4. Must distinguish error from empty."""
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise QueryError(f"Unparseable MCP response: {stripped[:400]}\nSQL: {sql}") from exc
    if isinstance(payload, dict) and "error" in payload:
        raise QueryError(f"ClickHouse error: {payload['error']}\nSQL: {sql}")
    if isinstance(payload, list):
        return payload
    raise QueryError(f"Unexpected MCP payload type {type(payload).__name__}\nSQL: {sql}")
```

- [ ] **Step 4: Verify passing**

Run: `uv run pytest tests/integration/test_mcp_gateway.py -v -m integration`
Expected: 4 passed.

If `test_sql_error_raises_and_does_not_masquerade_as_empty` fails, **stop and fix before continuing.** A gateway that turns errors into empty results will make every later stage silently wrong.

- [ ] **Step 5: Delete the probe and commit**

```bash
rm scripts/probe_mcp.py
git add -A
git commit -m "feat: clickhouse mcp gateway with provenance logging and loud errors"
```

**Milestone: the rule-critical integration is proven on day 2, not day 15.**

---

## Chunk 2: Schema and dimension universe

### Task 6: Schema DDL

**Files:**
- Create: `continuity/data/__init__.py`, `continuity/data/schema.py`, `tests/integration/test_schema.py`

- [ ] **Step 1: Write the failing integration test**

```python
import pytest
from continuity.config import ClickHouseConfig
from continuity.data.schema import apply_schema, TABLES
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

pytestmark = pytest.mark.integration

def test_apply_schema_creates_all_tables():
    cfg = ClickHouseConfig.from_env()
    apply_schema(cfg)

async def _tables() -> set[str]:
    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gw:
        result = await gw.query(
            "SELECT name FROM system.tables WHERE database = 'continuity'"
        )
    return {r["name"] for r in result.rows}

async def test_all_expected_tables_exist():
    assert TABLES <= await _tables()

def test_apply_schema_is_idempotent():
    cfg = ClickHouseConfig.from_env()
    apply_schema(cfg)
    apply_schema(cfg)  # must not raise

async def test_rollup_mv_populates_on_insert():
    """The MV is the drill-down performance structure. Verify it actually fires."""
    cfg = ClickHouseConfig.from_env()
    apply_schema(cfg)
    async with ClickHouseMCPGateway(cfg) as gw:
        before = await gw.query("SELECT count() AS c FROM qoe_rollup_5m")
        # insert handled by fixture in conftest; see Task 12
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/integration/test_schema.py -v -m integration`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement the DDL**

```python
"""ClickHouse DDL. Ordering keys are chosen for the drill-down access pattern."""
from __future__ import annotations

import clickhouse_connect

from continuity.config import ClickHouseConfig

TABLES = {
    "playback_events", "qoe_rollup_5m", "qoe_rollup_5m_mv",
    "titles", "subscribers", "change_log",
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
    change_id     UInt32,
    changed_at    DateTime('UTC'),
    change_type   LowCardinality(String),
    component     String,
    description   String,
    dimension_key   LowCardinality(String),
    dimension_value LowCardinality(String)
)
ENGINE = MergeTree ORDER BY changed_at
"""

STATEMENTS = [PLAYBACK_EVENTS, QOE_ROLLUP, QOE_ROLLUP_MV, TITLES, SUBSCRIBERS, CHANGE_LOG]


def apply_schema(config: ClickHouseConfig) -> None:
    """Create the database and every table. Idempotent."""
    client = clickhouse_connect.get_client(
        host=config.host, port=config.port, username=config.user,
        password=config.password, secure=config.secure,
    )
    try:
        client.command(f"CREATE DATABASE IF NOT EXISTS {config.database}")
        client.close()
        client = clickhouse_connect.get_client(
            host=config.host, port=config.port, username=config.user,
            password=config.password, database=config.database, secure=config.secure,
        )
        for statement in STATEMENTS:
            client.command(statement)
    finally:
        client.close()
```

- [ ] **Step 4: Verify passing**

Run: `uv run pytest tests/integration/test_schema.py -v -m integration`
Expected: 3 passed, 1 skipped (the MV-population test lands in Task 12).

- [ ] **Step 5: Commit**

```bash
git add continuity/data/ tests/integration/test_schema.py
git commit -m "feat: clickhouse schema with aggregating rollup mv"
```

---

### Task 7: Dimension universe

**Files:**
- Create: `continuity/data/topology.py`, `tests/test_topology.py`

Defines the drill-down hierarchy. Pure data, no I/O.

- [ ] **Step 1: Write the failing test**

```python
from continuity.data.topology import (
    CDNS, DEVICE_TYPES, DIMENSION_HIERARCHY, POPS_BY_CDN, app_versions_for, pops_for,
)

def test_hierarchy_is_ordered_coarse_to_fine():
    assert DIMENSION_HIERARCHY[0] == "cdn"
    assert DIMENSION_HIERARCHY.index("cdn") < DIMENSION_HIERARCHY.index("pop")
    assert DIMENSION_HIERARCHY.index("device_type") < DIMENSION_HIERARCHY.index("app_version")

def test_every_cdn_has_pops():
    assert all(pops_for(c) for c in CDNS)

def test_pops_are_globally_unique_across_cdns():
    """A PoP name must identify its CDN unambiguously or drill-down attributes wrongly."""
    seen = [p for c in CDNS for p in pops_for(c)]
    assert len(seen) == len(set(seen))

def test_app_versions_are_device_appropriate():
    assert app_versions_for("roku") != app_versions_for("ios")
    assert all(v for v in app_versions_for("roku"))

def test_no_dimension_value_contains_sql_quote():
    """Values are interpolated into SQL literals downstream."""
    everything = [*CDNS, *DEVICE_TYPES, *(p for c in CDNS for p in pops_for(c))]
    assert not any("'" in v or "\\" in v for v in everything)
```

The SQL-quote test is a cheap standing guard against injection via generated dimension values.

- [ ] **Step 2: Run to confirm failure**

Run: `uv run pytest tests/test_topology.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""The dimension universe. This IS the drill-down hierarchy."""
from __future__ import annotations

# Coarse to fine. The Scope stage in sub-project 2 walks this order.
DIMENSION_HIERARCHY: tuple[str, ...] = (
    "cdn", "pop", "isp", "country", "region", "device_type", "os_version", "app_version",
)

CDNS: tuple[str, ...] = ("cdn_meridian", "cdn_northwind", "cdn_solstice")

POPS_BY_CDN: dict[str, tuple[str, ...]] = {
    "cdn_meridian":  ("mer-iad-1", "mer-ord-1", "mer-dfw-1", "mer-lax-1", "mer-lhr-1"),
    "cdn_northwind": ("nw-atl-2", "nw-sea-1", "nw-jfk-3", "nw-fra-1"),
    "cdn_solstice":  ("sol-den-1", "sol-mia-1", "sol-sjc-2", "sol-yyz-1"),
}

ISPS: tuple[str, ...] = (
    "comcast", "charter", "att", "verizon", "cox", "tmobile", "bt", "deutsche_telekom",
)

GEO: tuple[tuple[str, str], ...] = (
    ("US", "us_northeast"), ("US", "us_southeast"), ("US", "us_midwest"),
    ("US", "us_west"), ("CA", "ca_east"), ("GB", "gb_south"), ("DE", "de_west"),
)

DEVICE_TYPES: tuple[str, ...] = (
    "roku", "firetv", "samsung_tv", "lg_tv", "ios", "android", "web",
)

_OS_VERSIONS: dict[str, tuple[str, ...]] = {
    "roku":       ("roku_os_13.0", "roku_os_14.0", "roku_os_14.1"),
    "firetv":     ("fireos_7", "fireos_8"),
    "samsung_tv": ("tizen_6.5", "tizen_7.0"),
    "lg_tv":      ("webos_23", "webos_24"),
    "ios":        ("ios_17.5", "ios_18.2"),
    "android":    ("android_14", "android_15"),
    "web":        ("chrome_129", "safari_18", "firefox_131"),
}

# The 8.2.0 line exists on every platform so that "app version 8.2.0 is bad" is not
# trivially separable from "roku is bad" -- the drill-down has to do real work.
_APP_VERSIONS: dict[str, tuple[str, ...]] = {
    # Roku carries a legacy 8.0.9 build so its set is not identical to iOS's, which the
    # test above asserts. (Corrected 2026-08-08: the original draft gave roku and ios
    # identical tuples, contradicting that assertion.)
    "roku":       ("8.0.9", "8.1.4", "8.2.0"),
    "firetv":     ("8.1.4", "8.2.0"),
    "samsung_tv": ("8.1.2", "8.1.4"),
    "lg_tv":      ("8.1.2", "8.1.4"),
    "ios":        ("8.1.4", "8.2.0"),
    "android":    ("8.1.4", "8.2.0"),
    "web":        ("web_2026.7", "web_2026.8"),
}

# Device population weights -- TV platforms dominate watch time in real streaming data.
DEVICE_WEIGHTS: dict[str, float] = {
    "roku": 0.24, "firetv": 0.18, "samsung_tv": 0.15, "lg_tv": 0.09,
    "ios": 0.13, "android": 0.13, "web": 0.08,
}


def pops_for(cdn: str) -> tuple[str, ...]:
    try:
        return POPS_BY_CDN[cdn]
    except KeyError:
        raise ValueError(f"Unknown CDN {cdn!r}. Known: {sorted(POPS_BY_CDN)}") from None


def os_versions_for(device_type: str) -> tuple[str, ...]:
    try:
        return _OS_VERSIONS[device_type]
    except KeyError:
        raise ValueError(f"Unknown device {device_type!r}. Known: {sorted(_OS_VERSIONS)}") from None


def app_versions_for(device_type: str) -> tuple[str, ...]:
    try:
        return _APP_VERSIONS[device_type]
    except KeyError:
        raise ValueError(f"Unknown device {device_type!r}. Known: {sorted(_APP_VERSIONS)}") from None
```

Note the comment on `_APP_VERSIONS`: because `8.2.0` spans several device types, an incident scoped to *Roku on 8.2.0* cannot be found by splitting on one dimension alone. That is what makes the hierarchical drill-down necessary rather than ornamental.

- [ ] **Step 4: Verify passing**

Run: `uv run pytest tests/test_topology.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add continuity/data/topology.py tests/test_topology.py
git commit -m "feat: dimension topology for drill-down hierarchy"
```

---

## Chunk 3: Seasonality, incidents, generation

Tasks 8, 9 and 10 are pure and mutually independent — they can be built in parallel.
Task 11 consumes all three. Task 12 consumes 11. Task 13 gates the sub-project.

### Task 8: `continuity/data/seasonality.py`

Pure functions, no I/O, no randomness except through an injected `numpy.random.Generator`.

```python
DIURNAL_WEIGHTS: tuple[float, ...]          # 24 entries, sum == 1.0, argmax in 20..22
def diurnal_weight(hour: int) -> float      # ValueError outside 0..23
def weekday_factor(weekday: int) -> float   # Mon=0..Sun=6; weekend > weekday
def expected_sessions(bucket: datetime, sessions_per_day: int) -> float
def load_factor(bucket: datetime) -> float  # 0.0..1.0, normalised concurrency
def degrade(base: float, load: float, *, alpha: float, beta: float) -> float
```

`degrade` implements the load→QoE coupling: `base * (1 + alpha * load**beta)`.
Defaults `alpha=1.2, beta=2.0` give roughly 2.2x worse rebuffering at peak than at trough.

**This coupling is the point of the module.** It is what makes a naive fixed-threshold
detector fire every night at 21:00, which is the false-positive problem real ops teams
have, and therefore what forces the seasonality-aware baseline in sub-project 2 to be
real work. Do not flatten it.

Tests: weights all positive; sum ≈ 1.0; argmax in the 20:00–22:00 band; `diurnal_weight`
raises outside 0..23; weekend factor strictly greater than midweek; `load_factor` stays
within [0, 1] across a full week sampled every 5 minutes; `degrade` is monotonic in load,
equals `base` at load 0, and never returns less than `base`.

### Task 9: `continuity/data/catalog.py`

```python
@dataclass(frozen=True)
class Title:      title_id, name, genre, content_type, release_date, is_premiere
@dataclass(frozen=True)
class Subscriber: subscriber_id, plan, monthly_arpu, signup_date, tenure_days, country, region

def generate_titles(rng, count: int) -> list[Title]
def generate_subscribers(rng, count: int, *, as_of: date) -> list[Subscriber]
```

Plans and ARPU: `basic` 8.99, `standard` 15.99, `premium` 22.99, mixed roughly 35/45/20.
Tenure is skewed toward newer subscribers (an exponential-shaped draw), because the churn
heuristic in sub-project 2 weights low-tenure subscribers as higher risk — a uniform
tenure distribution would make that heuristic meaningless.

Countries and regions must be drawn from `topology.GEO` so joins line up.

Tests: determinism under a fixed seed; ARPU always matches the plan; tenure never
negative and consistent with `signup_date` against `as_of`; every country/region pair
appears in `topology.GEO`; exactly the requested counts; ids are unique and stable.

### Task 10: `continuity/data/incidents.py`

```python
@dataclass(frozen=True)
class Effect:            metric: str; multiplier: float
@dataclass(frozen=True)
class ChangeLogEntry:    change_id, changed_at, change_type, component, description,
                         dimension_key, dimension_value
@dataclass(frozen=True)
class PlantedIncident:
    incident_id: str
    kind: str
    start: datetime
    end: datetime
    predicate: dict[str, str]        # the TRUE blast radius
    affected_fraction: float         # share of matching sessions actually hit
    effects: tuple[Effect, ...]
    volume_multiplier: float = 1.0
    change: ChangeLogEntry | None = None
    is_decoy: bool = False

    def matches(self, dims: dict[str, str], when: datetime) -> bool

def build_incidents(window_start: datetime, days: int, *, premiere_title_id: int,
                    encode_title_id: int) -> tuple[PlantedIncident, ...]
def write_ground_truth(incidents, path: Path, *, seed: int, days: int) -> None
```

Four incidents, three real and one decoy:

| id | predicate | effect | window |
|---|---|---|---|
| `INC-APP-ROKU-820` | `device_type=roku, app_version=8.2.0` | rebuffer ×4.5 | day 12, 18:00, 8h |
| `INC-POP-NW-ATL-2` | `cdn=cdn_northwind, pop=nw-atl-2` | startup ×3.2, rebuffer ×2.0 | day 15, 02:00, 6h |
| `INC-ENCODE-<title>` | `title_id=<encode_title_id>` | bitrate ×0.45, rebuffer ×2.5 | day 18, 09:00, 30h |
| `DECOY-PREMIERE-<title>` | `title_id=<premiere_title_id>` | **none** — `volume_multiplier=6.0` | day 20, 20:00, 5h |

The first is the one that justifies the whole drill-down: `8.2.0` also ships on firetv,
ios and android, and Roku also runs `8.0.9` and `8.1.4`, so neither `device_type=roku`
nor `app_version=8.2.0` alone identifies it.

Each real incident carries a `ChangeLogEntry` timed shortly before its start, so the
Correlate stage has something true to find. The decoy carries **no** change entry.

Tests: the decoy has elevated volume and an empty `effects` tuple; every real incident has
at least one effect and a change entry whose `changed_at` precedes its `start`; `matches`
is false outside the time window and false when any predicate key differs; predicates only
reference keys in `topology.DIMENSION_HIERARCHY` plus `title_id`; ground truth round-trips
through JSON unchanged.

**`write_ground_truth` writes to `data/ground_truth.json` and nothing writes incident
truth to ClickHouse.** A test asserts the serialized payload contains the true predicates,
and a separate check in Task 13 asserts no ClickHouse table holds them.

### Task 11: `continuity/data/generator.py`

Assembles sessions into heartbeat events. NumPy-vectorised per 5-minute bucket, yielding
column-oriented batches ready for `clickhouse-connect`.

**Required: multiply `expected_sessions(...)` by `weekday_factor(bucket.weekday())`.**
`expected_sessions` applies only the intra-day shape so its 288-bucket sum lands exactly
on `sessions_per_day`; the weekday term is deliberately left to the caller. Omitting it
makes the model incoherent, because `load_factor` *does* include weekday — a Saturday
would show 25% higher load, and therefore worse QoE, on identical session volume to a
Tuesday. Load represents concurrency and concurrency comes from volume; they must move
together. A test must assert weekend volume exceeds midweek volume.

Per bucket: draw a session count from `expected_sessions` with noise, sample dimensions
(device by `DEVICE_WEIGHTS`, then CDN/PoP/ISP/geo/title/subscriber), compute baseline QoE
per dimension, apply `degrade` for the bucket's load, then apply any matching incident's
effects to the affected share. Emit `start`, N `heartbeat`, occasional `rebuffer`/`error`,
and `end` per session.

Tests: identical output for the same seed (regeneration must be byte-identical — the eval
harness depends on it); a session matching no incident predicate has baseline QoE; a
session inside `INC-APP-ROKU-820` shows roughly 4.5× the rebuffer of an equivalent session
outside it; `startup_ms` is non-zero only on `start` events; bitrate is non-zero only on
`heartbeat`; the decoy window shows elevated session volume with QoE inside normal bounds.

### Task 12: `continuity/data/load.py`

Typer CLI: `uv run python -m continuity.data.load --days 21 [--sessions-per-day N] [--truncate]`.
Applies the schema, generates, inserts in batches via `clickhouse-connect`, writes
`data/ground_truth.json`, and reports row counts per table plus elapsed time.

Must be re-runnable: `--truncate` clears `playback_events` **and** `qoe_rollup_5m` (a
materialized view does not cascade deletes — see Task 6).

### Task 13: End-to-end acceptance

Not a unit test. Load a full window, then read the data as an analyst would:

1. Each of the three real incidents is **visible in the rollup** — a measurable metric
   deviation inside its true blast radius during its true window.
2. The decoy shows elevated session volume with QoE within normal bounds.
3. A naive fixed-threshold detector fires on nightly peaks — proving the seasonality
   problem is real and not asserted.
4. No ClickHouse table contains incident ground truth.
5. Regenerating with the same seed is byte-identical.

Applying lesson `one-live-run-before-declaring-done`: a green unit suite proves the
generator does what it was told, not that the dataset supports the product. The dataset
gets inspected before this sub-project is called complete.

---

## Acceptance criteria for sub-project 1

- [ ] `docker compose up -d` yields a healthy ClickHouse
- [ ] `uv run pytest -m "not integration"` — all pass, no Docker needed
- [ ] `uv run pytest -m integration` — all pass against local Docker
- [ ] `uv run python -m continuity.data.load --days 21` completes and reports row counts
- [ ] A SQL query over `qoe_rollup_5m` shows each of the 3 real incidents as a visible deviation within its true blast radius
- [ ] The decoy shows elevated session volume with QoE within normal bounds
- [ ] A naive fixed-threshold detector fires on nightly peaks — demonstrating why seasonality-awareness is required
- [ ] `data/ground_truth.json` exists, is complete, and **no ClickHouse table contains incident truth**
- [ ] Regeneration with the same seed is byte-identical
