"""Integration tests for continuity/analysis/correlate.py against the real dataset, read
through the MCP gateway.

Read-only: no writes, no truncation. Uses the `gateway` fixture from tests/conftest.py,
built from ClickHouseConfig.from_env() -- the DEFAULT database holding the full
63.85M-event dataset, not the separate continuity_test database test_load.py uses.

Incident windows and predicates are DERIVED from data/ground_truth.json, never
hardcoded -- see tests/integration/test_detect_real.py's own docstring for why a
hardcoded date has already broken this project twice (incident placement moved to be
relative to the end of the dataset window, see continuity/data/incidents.py).

The anomaly window passed to correlate_changes() is the ground-truth incident window
itself, exactly as tests/integration/test_split_real.py uses it directly rather than
re-deriving it from detect() -- correlate.py's own dependency is slices.py + metrics.py,
not detect.py, so this file does not run detection first.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from continuity.analysis.correlate import correlate_changes
from continuity.analysis.slices import Slice
from continuity.data.topology import pops_for

pytestmark = pytest.mark.integration

_GROUND_TRUTH_PATH = Path(__file__).resolve().parents[2] / "data" / "ground_truth.json"


def _ground_truth_incidents() -> list[dict]:
    payload = json.loads(_GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return payload["incidents"]


def _by_kind(kind: str) -> dict:
    matches = [inc for inc in _ground_truth_incidents() if inc["kind"] == kind]
    assert len(matches) == 1, f"expected exactly one {kind!r} incident in ground truth"
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


# --- (a) the three real incidents: the true change_log entry ranks first -----------


async def test_roku_820_incident_ranks_the_true_change_first(gateway):
    incident = _by_kind("device_app_fault")
    true_change = incident["change"]
    slice_ = _slice_for(incident)
    window = _window(incident)

    result = await correlate_changes(gateway, blast_radius=slice_, anomaly_window=window)

    assert result.candidates, "expected at least one candidate for the roku/8.2.0 incident"
    top = result.candidates[0]
    assert top.change_id == true_change["change_id"]
    assert top.dimension_key == true_change["dimension_key"]
    assert top.dimension_value == true_change["dimension_value"]
    assert top.dimensional_overlap is True
    assert top.temporal_delta.total_seconds() > 0, "the true change must precede onset"
    assert top.score > 0
    # app_version=8.2.0 also ships on firetv/ios/android -- the disconfirming-evidence
    # check must actually run against them, not report "nothing to check".
    evidence = top.disconfirming_evidence
    assert evidence.sibling_dimension == "device_type"


async def test_pop_nw_atl_2_incident_ranks_the_true_change_first(gateway):
    incident = _by_kind("pop_fault")
    true_change = incident["change"]
    slice_ = _slice_for(incident)
    window = _window(incident)

    result = await correlate_changes(gateway, blast_radius=slice_, anomaly_window=window)

    assert result.candidates, "expected at least one candidate for the nw-atl-2 pop incident"
    top = result.candidates[0]
    assert top.change_id == true_change["change_id"]
    assert top.dimensional_overlap is True
    assert top.temporal_delta.total_seconds() > 0
    assert top.score > 0


async def test_encode_incident_ranks_the_true_change_first(gateway):
    incident = _by_kind("encode_fault")
    true_change = incident["change"]
    slice_ = _slice_for(incident)
    window = _window(incident)

    result = await correlate_changes(gateway, blast_radius=slice_, anomaly_window=window)

    assert result.candidates, "expected at least one candidate for the encode incident"
    top = result.candidates[0]
    assert top.change_id == true_change["change_id"]
    assert top.dimensional_overlap is True
    assert top.temporal_delta.total_seconds() > 0
    # title_id is the blast radius's only predicate -- there is no other dimension to
    # disconfirm against, and that must be recorded rather than silently skipped.
    assert top.disconfirming_evidence.sibling_dimension is None


# --- (b) the decoy has no change_log entry: nothing should rank highly -------------


async def test_decoy_incident_yields_no_high_scoring_candidate(gateway):
    incident = _by_kind("decoy_premiere")
    slice_ = _slice_for(incident)
    window = _window(incident)

    result = await correlate_changes(gateway, blast_radius=slice_, anomaly_window=window)

    assert all(c.score < 0.5 for c in result.candidates), (
        f"the decoy has no planted change_log entry; got candidates: {result.candidates}"
    )


# --- (c) rejections are populated, not just debug logging -------------------------


async def test_rejected_candidates_populated_when_a_real_change_contradicts_the_blast_radius(
    gateway,
):
    """A different pop under the SAME cdn as the pop-fault incident, checked over the
    incident's own window: the real logged change (pop=nw-atl-2) targets a disjoint
    population and must be explicitly rejected, not silently absent from the results."""
    incident = _by_kind("pop_fault")
    true_cdn = incident["predicate"]["cdn"]
    true_pop = incident["predicate"]["pop"]
    other_pop = next(p for p in pops_for(true_cdn) if p != true_pop)
    slice_ = Slice().refine("cdn", true_cdn).refine("pop", other_pop)
    window = _window(incident)

    result = await correlate_changes(gateway, blast_radius=slice_, anomaly_window=window)

    assert result.candidates == (), "the real change contradicts this blast radius"
    assert len(result.rejected) == 1
    assert result.rejected[0].change_id == incident["change"]["change_id"]
    assert "no dimensional overlap" in result.rejected[0].reason


async def test_no_changes_in_window_is_not_an_error(gateway):
    """A quiet slice/window with no change_log rows at all: empty, not an error."""
    decoy = _by_kind("decoy_premiere")
    onset, end = _window(decoy)

    result = await correlate_changes(
        gateway,
        blast_radius=Slice().refine("title_id", decoy["predicate"]["title_id"]),
        anomaly_window=(onset, end),
    )

    assert result.candidates == ()
    assert result.rejected == ()
