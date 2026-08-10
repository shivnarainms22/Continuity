"""Unit tests for continuity/api/incidents_severity.py's route wiring and its pure
input-loading helper. No ClickHouse involved -- `TestClient(app)` used without the
`with` block never runs the lifespan (see tests/api/test_app.py's own docstring), so
`/api/incidents/{id}/severity` is exercised exactly as it behaves before any gateway
exists. The real, queried impact numbers are covered by
tests/integration/test_incidents_severity_real.py instead.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from continuity.api import app as app_module
from continuity.api.ground_truth import GroundTruthError
from continuity.api.incidents_severity import load_raw_incident

client = TestClient(app_module.app)


def test_severity_returns_503_when_no_gateway_is_attached():
    app_module.app.state.gateway = None

    response = client.get("/api/incidents/INC-APP-ROKU-820/severity")

    assert response.status_code == 503


def test_load_raw_incident_returns_the_matching_row(tmp_path):
    payload = {
        "incidents": [
            {
                "incident_id": "INC-FAKE-1",
                "start": "2026-05-01T00:00:00+00:00",
                "end": "2026-05-01T03:00:00+00:00",
                "predicate": {"cdn": "cdn_northwind"},
                "effects": [{"metric": "rebuffer", "multiplier": 4.5}],
                "is_decoy": False,
            }
        ]
    }
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    row = load_raw_incident(path, "INC-FAKE-1")

    assert row["effects"] == [{"metric": "rebuffer", "multiplier": 4.5}]


def test_load_raw_incident_raises_on_unknown_id(tmp_path):
    payload = {"incidents": []}
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GroundTruthError, match="Unknown incident"):
        load_raw_incident(path, "INC-DOES-NOT-EXIST")


def test_load_raw_incident_raises_on_missing_file(tmp_path):
    with pytest.raises(GroundTruthError, match="not found"):
        load_raw_incident(tmp_path / "missing.json", "INC-FAKE-1")


def test_severity_404s_for_an_unknown_incident_id(monkeypatch, tmp_path):
    payload = {"incidents": []}
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(app_module.app.state, "ground_truth_path", path, raising=False)
    app_module.app.state.gateway = object()  # any truthy sentinel; must fail before use

    response = client.get("/api/incidents/INC-DOES-NOT-EXIST/severity")

    assert response.status_code == 404
    app_module.app.state.gateway = None
