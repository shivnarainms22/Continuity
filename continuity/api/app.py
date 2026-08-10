"""FastAPI app for the Continuity walking skeleton (see ADR-001).

Owns exactly ONE `ClickHouseMCPGateway` for the whole process lifetime, opened at
startup and closed at shutdown. That single, long-lived gateway is the piece this
skeleton exists to prove: it holds a subprocess and a dedicated asyncio task, and
reusing it across every request is what avoids paying the ~21s mcp-clickhouse startup
cost per investigation (see mcp_gateway.py's own module docstring and ADR-001's
"Cloud Run throttles CPU between requests" section).

Serves the built frontend (`web/dist/`, produced by `npm run build`) as static files
from this same app, with an SPA fallback so client-side routes resolve to `index.html`
instead of 404ing -- except under `/api/`, which must 404 normally so a missing API
route is never mistaken for a valid page.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from continuity.api.ground_truth import (
    DEFAULT_GROUND_TRUTH_PATH,
    GroundTruthError,
    load_incident_summaries,
)
from continuity.api.incidents_severity import router as incidents_severity_router
from continuity.api.investigate_stream import router as investigate_router
from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway, QueryError

load_dotenv(override=False)

# continuity/api/app.py -> continuity/api -> continuity -> repo root -> web/dist
_REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = _REPO_ROOT / "web" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = ClickHouseConfig.from_env()
    async with ClickHouseMCPGateway(config) as gateway:
        app.state.gateway = gateway
        app.state.ground_truth_path = DEFAULT_GROUND_TRUTH_PATH
        yield
    app.state.gateway = None


app = FastAPI(title="Continuity walking skeleton", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict:
    """`ok` plus whether the long-lived gateway session is actually answering queries.

    Runs a trivial query rather than only checking task liveness: a hung session that
    never crashed would otherwise report healthy. This doubles as the evidence that the
    session survives across requests -- `queries_run` only ever grows, never resets,
    because the same gateway (and its query_log) is reused for the app's whole lifetime.
    """
    gateway: ClickHouseMCPGateway | None = getattr(app.state, "gateway", None)
    if gateway is None:
        return {"status": "ok", "gateway_live": False, "queries_run": 0}
    try:
        await gateway.query("SELECT 1")
        gateway_live = True
    except QueryError:
        gateway_live = False
    return {"status": "ok", "gateway_live": gateway_live, "queries_run": len(gateway.query_log)}


@app.get("/api/incidents")
async def list_incidents() -> list[dict]:
    ground_truth_path: Path = getattr(app.state, "ground_truth_path", DEFAULT_GROUND_TRUTH_PATH)
    try:
        return load_incident_summaries(ground_truth_path)
    except GroundTruthError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.include_router(investigate_router)
app.include_router(incidents_severity_router)

# `check_dir=False`: mounting must not crash the app before `npm run build` has ever
# run (e.g. in tests, or a fresh checkout) -- a request under a missing directory 404s
# at request time instead. STATIC_DIR is read from the module namespace on every
# request (not captured at mount time), so tests can monkeypatch it directly.
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets", check_dir=False), name="assets")


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str) -> FileResponse:
    """Any non-API path resolves to `index.html` so client-side routes (deep links)
    work; a real static file at that path (favicon, manifest, ...) is served directly
    instead. `/api/...` is excluded so a missing API route 404s normally rather than
    silently returning the SPA shell.
    """
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    if not STATIC_DIR.is_dir():
        raise HTTPException(
            status_code=404, detail="Frontend not built. Run 'npm run build' in web/."
        )
    candidate = STATIC_DIR / full_path
    if full_path and candidate.is_file():
        return FileResponse(candidate)
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=404, detail="Frontend not built. Run 'npm run build' in web/."
        )
    return FileResponse(index)
