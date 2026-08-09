"""Integration test for continuity.analysis.impact against the real dataset.

Read only, through the `gateway` fixture (tests/conftest.py), which is built from
ClickHouseConfig.from_env() -- the DEFAULT database holding the full 63.85M-event
dataset, not the separate continuity_test database test_load.py uses.

The incident window and predicate are DERIVED from data/ground_truth.json, never
hardcoded -- hardcoded incident dates are what has broken this project's tests twice
before (see tests/integration/test_detect_real.py, test_split_real.py).

`qoe_delta_ratio` -- (actual - baseline) / baseline for the metric that flagged the
incident -- is a caller-supplied severity input (impact.py depends only on slices.py,
not on detect.py/baseline.py, so it never computes this itself). Ground truth records
INC-APP-ROKU-820's rebuffer effect as a 4.5x multiplier, i.e. actual = 4.5 * baseline,
so qoe_delta_ratio = 4.5 - 1 = 3.5.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from continuity.analysis.impact import compute_impact
from continuity.analysis.slices import Slice

pytestmark = pytest.mark.integration

_GROUND_TRUTH_PATH = Path(__file__).resolve().parents[2] / "data" / "ground_truth.json"


def _roku_820_incident() -> dict:
    payload = json.loads(_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    matches = [inc for inc in payload["incidents"] if inc["kind"] == "device_app_fault"]
    assert len(matches) == 1, "expected exactly one device_app_fault incident in ground truth"
    return matches[0]


def _window(incident: dict) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(incident["start"]).replace(tzinfo=None)
    end = datetime.fromisoformat(incident["end"]).replace(tzinfo=None)
    return start, end


def _slice_for(incident: dict) -> Slice:
    slice_ = Slice()
    for key, value in incident["predicate"].items():
        slice_ = slice_.refine(key, value)
    return slice_


def _qoe_delta_ratio(incident: dict, metric: str) -> float:
    effect = next(e for e in incident["effects"] if e["metric"] == metric)
    return effect["multiplier"] - 1.0


async def test_impact_for_roku_820_blast_radius_is_scoped_and_positive(gateway):
    incident = _roku_820_incident()
    window = _window(incident)
    slice_ = _slice_for(incident)
    qoe_delta_ratio = _qoe_delta_ratio(incident, "rebuffer")

    result = await compute_impact(
        gateway, slice_=slice_, window=window, qoe_delta_ratio=qoe_delta_ratio
    )

    total_subscribers_result = await gateway.query("SELECT count() FROM subscribers")
    total_subscribers = int(total_subscribers_result.scalar())

    # A scoped fault, not everyone: some subscribers were hit, but nowhere near the
    # whole 20,000-subscriber base -- device_type=roku AND app_version=8.2.0 is one
    # combination among seven device types and (on roku) three app versions.
    assert result.affected_subscribers > 0
    assert result.affected_subscribers < total_subscribers * 0.5, (
        f"expected a materially smaller blast radius than the {total_subscribers}-"
        f"subscriber base, got {result.affected_subscribers}"
    )

    assert isinstance(result.arr_at_risk_expected, Decimal)
    assert result.arr_at_risk_expected > Decimal("0")
    assert result.arr_at_risk_low <= result.arr_at_risk_expected <= result.arr_at_risk_high

    assert result.methodology is not None
    assert result.methodology.affected_subscriber_count == result.affected_subscribers
    assert result.methodology.window == window
    assert result.methodology.slice == slice_
    assert result.sql, "expected the query that produced this result to be recorded"


async def test_impact_for_a_slice_matching_nothing_is_zero_not_a_crash(gateway):
    incident = _roku_820_incident()
    window = _window(incident)
    qoe_delta_ratio = _qoe_delta_ratio(incident, "rebuffer")
    # web devices do not ship app_version 8.2.0 as roku's faulty build -- pin an
    # app_version/device_type pairing that does not occur in the catalog
    # (continuity/data/topology.py) inside this narrow 8-hour window.
    nothing_slice = Slice().refine("device_type", "samsung_tv").refine("app_version", "8.0.9")

    result = await compute_impact(
        gateway, slice_=nothing_slice, window=window, qoe_delta_ratio=qoe_delta_ratio
    )

    assert result.affected_subscribers == 0
    assert result.arr_at_risk_low == Decimal("0.00")
    assert result.arr_at_risk_expected == Decimal("0.00")
    assert result.arr_at_risk_high == Decimal("0.00")
    assert result.methodology is not None
