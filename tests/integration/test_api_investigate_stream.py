"""Integration proof for continuity/api/investigate_stream.py and the gateway lifespan
in continuity/api/app.py, against the real ClickHouse instance every other file under
tests/integration/ uses. No mocking: the ClickHouse hackathon track is graded on runtime
use of mcp-clickhouse, so mocking the one thing under judgement would prove nothing.

Runs the app as a REAL uvicorn subprocess rather than Starlette's `TestClient` --
`TestClient`'s in-process ASGI transport was measured to buffer an entire
`StreamingResponse` before handing any of it to a synchronous caller (all "incremental"
SSE frames arrived within microseconds of each other under `TestClient`, while the same
endpoint against a real uvicorn process delivered them spread over ~8 real seconds). That
is a test-harness artifact, not a property of the app, so exercising a live server is the
only way to honestly test the ADR-001 claim that SSE actually streams.

Covers the two riskiest claims in ADR-001:
    1. SSE streams incrementally -- events arrive spread over the investigation, not all
       at once at the end (buffering would make SSE pointless).
    2. The MCP session survives across separate requests -- query_log only ever grows,
       and only the very first request against a freshly started server pays anything
       close to the ~21s mcp-clickhouse subprocess-spawn cost; every later request is fast.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PORT = 8971
_BASE_URL = f"http://127.0.0.1:{_PORT}"
_READY_TIMEOUT_S = 60.0  # generous: mcp-clickhouse's own first-connection cost is ~21s


def _subprocess_env() -> dict[str, str]:
    """Explicit environment for the server subprocess -- see test_load.py's own
    `_subprocess_env` for the incident this project-wide pattern guards against."""
    return dict(os.environ)


@pytest.fixture(scope="module")
def live_server() -> Iterator[httpx.Client]:
    """A real `uvicorn` process running the app, plus one shared client to talk to it.

    One shared `httpx.Client` (not a fresh one per call) so this sandbox's own
    measured ~21s "first outbound connection in a new process" tax is paid exactly once,
    during the readiness poll below -- not attributed to the gateway inside the tests
    that actually measure per-request latency.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "continuity.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(_PORT),
        ],
        cwd=_REPO_ROOT,
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client = httpx.Client(timeout=10.0)
    deadline = time.time() + _READY_TIMEOUT_S
    last_error: Exception | None = None
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"uvicorn exited early:\n{proc.stdout.read()}")
            try:
                response = client.get(f"{_BASE_URL}/api/health")
                if response.status_code == 200 and response.json().get("gateway_live"):
                    break
            except httpx.HTTPError as exc:
                last_error = exc
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"server did not become healthy within {_READY_TIMEOUT_S}s: {last_error}"
            )

        yield client
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _split_frames(raw: str) -> list[str]:
    return [frame for frame in raw.strip().split("\n\n") if frame.strip()]


def _frame_data(frame: str) -> dict:
    data_line = next(line for line in frame.splitlines() if line.startswith("data: "))
    return json.loads(data_line[len("data: ") :])


def test_investigate_stream_delivers_sse_events_incrementally_not_all_at_once(live_server):
    """Consumes the endpoint with a timestamping client and shows the events arriving
    spread over time -- the single most likely thing to silently not work (ADR-001)."""
    arrivals: list[tuple[float, str]] = []
    buffer = ""
    start = time.perf_counter()
    with live_server.stream(
        "GET", f"{_BASE_URL}/api/investigate/INC-APP-ROKU-820/stream"
    ) as response:
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        assert response.headers["content-type"].startswith("text/event-stream")

        for chunk in response.iter_text():
            buffer += chunk
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                if frame.strip():
                    arrivals.append((time.perf_counter() - start, frame))

    assert len(arrivals) >= 6, f"expected >= 6 SSE frames (5 stages + done), got {len(arrivals)}"

    timestamps = [t for t, _ in arrivals]
    span = timestamps[-1] - timestamps[0]
    assert span > 1.0, (
        f"all {len(arrivals)} SSE frames arrived within {span:.3f}s of each other -- "
        "this looks like buffering, not incremental streaming"
    )

    gaps = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
    assert any(gap > 0.3 for gap in gaps), (
        f"no gap between consecutive frames exceeded 300ms (arrival times={timestamps}) -- "
        "events are arriving in a single burst, not incrementally"
    )

    frames = [_frame_data(frame) for _, frame in arrivals]
    stage_names = [frame["stage"] for frame in frames if "stage" in frame]
    assert "session_startup" in stage_names
    assert "detect" in stage_names
    assert "brief" in frames[-1]
    assert "roku" in frames[-1]["brief"].lower()


def test_gateway_session_persists_across_separate_requests_without_restarting(live_server):
    """3 requests to an endpoint that queries ClickHouse, against one live server
    process: query_log must keep growing and none may pay the ~21s mcp-clickhouse
    subprocess-spawn cost -- that cost is paid once, at server startup (already spent
    by the `live_server` fixture above before this test ever runs)."""
    queries_run_by_call: list[int] = []
    elapsed_by_call: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        response = live_server.get(f"{_BASE_URL}/api/health")
        elapsed_by_call.append(time.perf_counter() - started)

        assert response.status_code == 200
        body = response.json()
        assert body["gateway_live"] is True
        queries_run_by_call.append(body["queries_run"])

    assert queries_run_by_call == sorted(queries_run_by_call), (
        f"queries_run decreased across requests -- looks like the session_log (and "
        f"therefore the session) was reset mid-run: {queries_run_by_call}"
    )
    assert queries_run_by_call[-1] > queries_run_by_call[0], (
        f"query_log did not grow across 3 separate requests: {queries_run_by_call}"
    )
    assert all(elapsed < 5.0 for elapsed in elapsed_by_call), (
        f"a /api/health call took {max(elapsed_by_call):.2f}s -- looks like a fresh "
        f"mcp-clickhouse subprocess was spawned for this request instead of reusing the "
        f"one from server startup: {elapsed_by_call}"
    )
