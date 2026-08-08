"""The single agent-runtime read path to ClickHouse, via the official mcp-clickhouse server.

The ClickHouse hackathon track requires runtime access to go through mcp-clickhouse, so
every read the agent performs lands here. Bulk loading (continuity/data/load.py) uses
clickhouse-connect directly; that is build-time ops rather than agent runtime, and
mcp-clickhouse is read-only by design.

Response shapes below were observed from a live server (mcp-clickhouse 0.4.1), not
inferred from documentation:

    success  isError=False  {"columns": ["n", "s"], "rows": [[1, "a"]]}
    empty    isError=False  {"columns": [], "rows": []}
    failure  isError=True   plain text, not JSON

Note that ``columns`` comes back empty for an empty result even when the query names
columns, so column metadata cannot be relied on when there are no rows.

Concurrency design
------------------
The MCP session lives in a dedicated asyncio task and callers submit queries over a
queue. This is not incidental:

* ``stdio_client`` builds anyio cancel scopes bound to the task that entered them, so
  entering and exiting the session from different tasks raises "Attempted to exit cancel
  scope in a different task". Owning the session in one task makes the gateway safe to
  hold across requests, background jobs and test fixtures alike.
* Starting the server costs seconds (subprocess spawn plus first-connection truststore
  initialisation). A long-lived session pays that once instead of per query.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from continuity.config import ClickHouseConfig

_QUERY_TOOL = "run_query"
_SHUTDOWN_TIMEOUT_S = 10.0


class QueryError(RuntimeError):
    """A query or connection failed.

    Never swallowed and never degraded into an empty result: a silent partial failure
    is invisible by construction and would make every downstream stage confidently wrong.
    """


@dataclass(frozen=True)
class ExecutedQuery:
    """A record of one successful query, for provenance in generated briefs."""

    sql: str
    duration_ms: float
    row_count: int


@dataclass(frozen=True)
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]

    def scalar(self) -> Any:
        """The single value of a single-row, single-column result."""
        if len(self.rows) != 1:
            raise QueryError(f"scalar() needs exactly 1 row, got {len(self.rows)}. SQL: {self.sql}")
        values = list(self.rows[0].values())
        if len(values) != 1:
            raise QueryError(f"scalar() needs exactly 1 column, got {len(values)}. SQL: {self.sql}")
        return values[0]


def _server_executable() -> str:
    """Locate the mcp-clickhouse console script inside the active environment."""
    bindir = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    for candidate in (bindir / "mcp-clickhouse.exe", bindir / "mcp-clickhouse"):
        if candidate.exists():
            return str(candidate)
    raise QueryError(
        f"The mcp-clickhouse console script was not found under {bindir}. "
        "Run 'uv sync' to install project dependencies."
    )


class ClickHouseMCPGateway:
    """Async context manager owning one long-lived mcp-clickhouse stdio session."""

    def __init__(self, config: ClickHouseConfig) -> None:
        self._config = config
        self.query_log: list[ExecutedQuery] = []

        self._task: asyncio.Task[None] | None = None
        self._requests: asyncio.Queue[tuple[str, asyncio.Future[Any]] | None] | None = None
        self._ready: asyncio.Event | None = None
        self._started = False
        self._failure: BaseException | None = None
        self._stderr_path: Path | None = None

    # -- lifecycle ---------------------------------------------------------------

    async def __aenter__(self) -> ClickHouseMCPGateway:
        self._requests = asyncio.Queue()
        self._ready = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="mcp-clickhouse-session")

        await self._ready.wait()
        if not self._started:
            failure = self._failure
            await self._shutdown()
            raise QueryError(
                f"Could not start the mcp-clickhouse server against "
                f"{self._config.host}:{self._config.port}.\n"
                f"Server output:\n{self._read_stderr() or '(none captured)'}"
            ) from failure
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._shutdown()

    async def _shutdown(self) -> None:
        if self._task is not None and not self._task.done():
            if self._requests is not None:
                with contextlib.suppress(Exception):
                    self._requests.put_nowait(None)
            try:
                await asyncio.wait_for(self._task, timeout=_SHUTDOWN_TIMEOUT_S)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._task
            except Exception:
                pass  # teardown failures must not mask the caller's own exception

        self._task = None
        self._requests = None
        self._started = False
        self._cleanup_stderr()

    # -- the session task --------------------------------------------------------

    async def _run(self) -> None:
        """Own the MCP session for its whole lifetime. Never awaited by callers."""
        # A crashed MCP server surfaces to the client as a bare "Connection closed",
        # with the real cause (an ImportError, a bad credential) only on the server's
        # stderr. Capture it so failures are diagnosable instead of opaque.
        # Not a `with` block: the handle must stay open for the whole session lifetime
        # and is closed in the `finally` below.
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+", suffix=".mcp-stderr.log", delete=False, encoding="utf-8"
        )
        self._stderr_path = Path(handle.name)
        try:
            async with (
                stdio_client(self._server_params(), errlog=handle) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                self._started = True
                self._ready.set()  # type: ignore[union-attr]
                await self._serve(session)
        except BaseException as exc:  # noqa: BLE001 - recorded and re-surfaced to callers
            self._failure = exc
            raise
        finally:
            self._ready.set()  # type: ignore[union-attr]
            with contextlib.suppress(Exception):
                handle.close()

    async def _serve(self, session: ClientSession) -> None:
        """Pump queued queries through the session until asked to stop."""
        assert self._requests is not None
        while True:
            item = await self._requests.get()
            if item is None:
                return
            sql, future = item
            if future.done():
                continue  # caller gave up
            try:
                response = await session.call_tool(_QUERY_TOOL, {"query": sql})
            except BaseException as exc:  # noqa: BLE001 - handed to the waiting caller
                if not future.done():
                    future.set_exception(exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise
            else:
                if not future.done():
                    future.set_result(response)

    def _server_params(self) -> StdioServerParameters:
        return StdioServerParameters(
            command=_server_executable(),
            args=[],
            env={
                **os.environ,
                "CLICKHOUSE_HOST": self._config.host,
                "CLICKHOUSE_PORT": str(self._config.port),
                "CLICKHOUSE_USER": self._config.user,
                "CLICKHOUSE_PASSWORD": self._config.password,
                "CLICKHOUSE_DATABASE": self._config.database,
                "CLICKHOUSE_SECURE": "true" if self._config.secure else "false",
            },
        )

    # -- stderr capture ----------------------------------------------------------

    def _read_stderr(self) -> str:
        if self._stderr_path is None or not self._stderr_path.exists():
            return ""
        try:
            return self._stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _cleanup_stderr(self) -> None:
        if self._stderr_path is not None:
            with contextlib.suppress(OSError):
                self._stderr_path.unlink(missing_ok=True)
            self._stderr_path = None

    # -- queries -----------------------------------------------------------------

    async def query(self, sql: str) -> QueryResult:
        """Run a read query. Raises QueryError on any failure."""
        if self._requests is None or self._task is None:
            raise QueryError(
                "Gateway used outside its async context manager. "
                "Use 'async with ClickHouseMCPGateway(config) as gateway:'."
            )
        if self._task.done():
            raise QueryError(
                f"The mcp-clickhouse session has stopped.\n"
                f"Server output:\n{self._read_stderr() or '(none captured)'}"
            )

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        started = time.perf_counter()
        await self._requests.put((sql, future))
        try:
            response = await future
        except Exception as exc:
            raise QueryError(
                f"MCP transport failed while running query.\nSQL: {sql}\n"
                f"Server output:\n{self._read_stderr() or '(none captured)'}"
            ) from exc
        duration_ms = (time.perf_counter() - started) * 1000

        body = "\n".join(
            text for item in response.content if (text := getattr(item, "text", "")) is not None
        )

        # Checked BEFORE parsing: an error body is plain text, and letting it fall
        # through to the JSON parser would report a confusing failure.
        if response.isError:
            raise QueryError(f"ClickHouse rejected the query.\nSQL: {sql}\n{body}")

        columns, rows = _parse_body(body, sql)
        self.query_log.append(ExecutedQuery(sql=sql, duration_ms=duration_ms, row_count=len(rows)))
        return QueryResult(sql=sql, columns=columns, rows=rows)


def _parse_body(body: str, sql: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Turn the {"columns": [...], "rows": [[...]]} payload into dict rows."""
    stripped = body.strip()
    if not stripped:
        raise QueryError(f"mcp-clickhouse returned an empty body.\nSQL: {sql}")

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise QueryError(
            f"mcp-clickhouse returned an unparseable body.\nSQL: {sql}\n{stripped[:500]}"
        ) from exc

    if not isinstance(payload, dict) or "rows" not in payload:
        raise QueryError(
            f"Unexpected mcp-clickhouse payload shape: {type(payload).__name__}.\n"
            f"SQL: {sql}\n{stripped[:500]}"
        )

    columns: list[str] = payload.get("columns") or []
    raw_rows: list[list[Any]] = payload.get("rows") or []

    if raw_rows and not columns:
        raise QueryError(
            f"mcp-clickhouse returned {len(raw_rows)} rows but no column names.\nSQL: {sql}"
        )

    rows: list[dict[str, Any]] = []
    for index, values in enumerate(raw_rows):
        if len(values) != len(columns):
            raise QueryError(
                f"Row {index} has {len(values)} values but there are "
                f"{len(columns)} columns.\nSQL: {sql}"
            )
        rows.append(dict(zip(columns, values, strict=True)))

    return columns, rows
