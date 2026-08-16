"""Unit tests for the FastAPI app's routes that do not need a live ClickHouse gateway.

`TestClient(app)` used WITHOUT the `with` block deliberately never runs the app's
`lifespan` (Starlette only triggers startup/shutdown when the client is entered as a
context manager) -- so `/api/incidents` and the SPA fallback are exercised exactly as
they behave before any gateway exists, with no Docker dependency. `/api/health`'s
gateway-live branch and the SSE endpoint's real ClickHouse behaviour are covered by
tests/integration/test_api_investigate_stream.py instead.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from continuity.api import app as app_module

client = TestClient(app_module.app)


def test_health_fails_the_probe_when_the_gateway_cannot_answer():
    """A health endpoint that returns 200 while the database is unreachable is worse
    than none: every probe, load balancer and uptime check believes it.

    This was live. On a cold Cloud Run instance /api/health returned
    {"status": "ok", "gateway_live": false} with HTTP 200 while the mcp-clickhouse
    session was still connecting, so the platform saw a healthy instance and routed a
    visitor to an app that could not answer a single question. It self-healed within
    seconds, which is precisely what made it easy to miss.
    """
    app_module.app.state.gateway = None

    response = client.get("/api/health")

    assert response.status_code == 503, "an unreachable gateway must fail the probe"
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["gateway_live"] is False


def test_health_reports_ok_when_the_gateway_answers():
    class _LiveGateway:
        query_log = [object(), object()]

        async def query(self, sql):
            return None

    app_module.app.state.gateway = _LiveGateway()
    try:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "gateway_live": True, "queries_run": 2}
    finally:
        app_module.app.state.gateway = None


def test_incidents_reads_the_configured_ground_truth_file(tmp_path, monkeypatch):
    payload = {
        "incidents": [
            {
                "incident_id": "INC-FAKE-1",
                "kind": "pop_fault",
                "start": "2026-05-01T00:00:00+00:00",
                "end": "2026-05-01T03:00:00+00:00",
                "predicate": {"cdn": "cdn_northwind"},
                "is_decoy": False,
            }
        ]
    }
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(app_module.app.state, "ground_truth_path", path, raising=False)

    response = client.get("/api/incidents")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "INC-FAKE-1",
            "window": {"start": "2026-05-01T00:00:00+00:00", "end": "2026-05-01T03:00:00+00:00"},
            "predicate": {"cdn": "cdn_northwind"},
            "kind": "pop_fault",
            "is_decoy": False,
        }
    ]


def test_incidents_returns_500_when_ground_truth_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module.app.state, "ground_truth_path", tmp_path / "missing.json", raising=False
    )

    response = client.get("/api/incidents")

    assert response.status_code == 500


def test_deep_link_returns_index_html_not_404(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html>skeleton</html>", encoding="utf-8")
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path)

    response = client.get("/some/deep/client/route")

    assert response.status_code == 200
    assert "skeleton" in response.text


def test_real_static_file_is_served_directly_not_the_spa_shell(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html>shell</html>", encoding="utf-8")
    (tmp_path / "favicon.ico").write_bytes(b"\x00\x01")
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path)

    response = client.get("/favicon.ico")

    assert response.status_code == 200
    assert response.content == b"\x00\x01"


def test_unknown_api_route_404s_instead_of_falling_back_to_the_spa_shell(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html>shell</html>", encoding="utf-8")
    monkeypatch.setattr(app_module, "STATIC_DIR", tmp_path)

    response = client.get("/api/not-a-real-endpoint")

    assert response.status_code == 404
    assert "shell" not in response.text


def test_deep_link_404s_before_the_frontend_has_ever_been_built(monkeypatch):
    monkeypatch.setattr(app_module, "STATIC_DIR", app_module._REPO_ROOT / "definitely-not-built")

    response = client.get("/some/route")

    assert response.status_code == 404
