"""Integration test for continuity/api/incidents_severity.py against the real dataset.

Read only, through the `gateway` fixture (tests/conftest.py) -- calls
`compute_incident_severity` directly rather than spinning up a live uvicorn process
(unlike tests/integration/test_api_investigate_stream.py, this endpoint is not SSE, so
there is no streaming/buffering risk that would require a real server process; see that
file's own module docstring for why SSE specifically does need one).

The incident and its window/predicate/effect are read straight from
data/ground_truth.json, never hardcoded -- hardcoded incident dates are what has broken
this project's tests twice before (see tests/integration/test_detect_real.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from continuity.api.incidents_severity import compute_incident_severity, load_raw_incident

pytestmark = pytest.mark.integration

_GROUND_TRUTH_PATH = Path(__file__).resolve().parents[2] / "data" / "ground_truth.json"


async def test_severity_for_a_real_planted_incident_is_a_positive_arr_band(gateway):
    row = load_raw_incident(_GROUND_TRUTH_PATH, "INC-APP-ROKU-820")

    result = await compute_incident_severity(gateway, row)

    assert result["id"] == "INC-APP-ROKU-820"
    assert result["affected_subscribers"] > 0
    assert (
        0
        < result["arr_at_risk_low"]
        <= result["arr_at_risk_expected"]
        <= result["arr_at_risk_high"]
    )
    assert "subscribers" in result["sql"]


async def test_severity_for_a_decoy_with_no_effects_is_a_real_honest_zero(gateway):
    row = load_raw_incident(_GROUND_TRUTH_PATH, "DECOY-PREMIERE-3")
    assert row["effects"] == [], "this test assumes the decoy fixture has no planted effects"

    result = await compute_incident_severity(gateway, row)

    assert result == {
        "id": "DECOY-PREMIERE-3",
        "affected_subscribers": 0,
        "arr_at_risk_low": 0.0,
        "arr_at_risk_expected": 0.0,
        "arr_at_risk_high": 0.0,
        "sql": None,
    }
