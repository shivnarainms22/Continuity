"""Integration test for continuity/analysis/walk.py against the real, live dataset.

Proves the project's core technical claim end-to-end, with no LLM involved: walking
from the WHOLE POPULATION over an incident's true window (from data/ground_truth.json
-- never hardcoded, per CLAUDE.md) arrives, unassisted, at the slice that isolates the
fault.

* INC-APP-ROKU-820: the walk must reach BOTH device_type=roku AND app_version=8.2.0.
  Neither dimension alone identifies the fault -- 8.2.0 also ships on
  firetv/ios/android, and roku also runs 8.0.9/8.1.4 -- so reaching both is the point.
* INC-POP-NW-ATL-2: the walk must reach BOTH cdn=cdn_northwind AND pop=nw-atl-2.

Read only. Uses the DEFAULT database from ClickHouseConfig.from_env() via the
`gateway` fixture in tests/conftest.py -- the full 63.85M-event dataset, not the
separate continuity_test database test_load.py uses.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from continuity.analysis.walk import StopReason, walk

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


async def test_walk_isolates_roku_820_by_both_device_type_and_app_version(gateway):
    incident = _by_kind("device_app_fault")
    window = _window(incident)

    result = await walk(gateway, metric_name="rebuffer", window=window)

    predicates = dict(result.final_slice.predicates)
    assert predicates.get("device_type") == "roku", (
        f"expected device_type=roku in the final slice, got {predicates}. "
        f"Path: {[(s.dimension, s.value, s.share_of_deviation) for s in result.path]}"
    )
    assert predicates.get("app_version") == "8.2.0", (
        f"expected app_version=8.2.0 in the final slice, got {predicates}. "
        f"Path: {[(s.dimension, s.value, s.share_of_deviation) for s in result.path]}"
    )
    # Every refinement is auditable: dimension, value, share, and the SQL behind it.
    assert result.path
    assert all(step.sql and step.baseline_sql for step in result.path)
    assert all(step.share_of_deviation > 0.0 for step in result.path)
    assert result.elapsed_ms > 0.0
    assert result.query_log, "every number must be traceable to a logged query"


async def test_walk_isolates_pop_fault_by_both_cdn_and_pop(gateway):
    incident = _by_kind("pop_fault")
    window = _window(incident)

    result = await walk(gateway, metric_name="startup", window=window)

    predicates = dict(result.final_slice.predicates)
    assert predicates.get("cdn") == "cdn_northwind", (
        f"expected cdn=cdn_northwind in the final slice, got {predicates}. "
        f"Path: {[(s.dimension, s.value, s.share_of_deviation) for s in result.path]}"
    )
    assert predicates.get("pop") == "nw-atl-2", (
        f"expected pop=nw-atl-2 in the final slice, got {predicates}. "
        f"Path: {[(s.dimension, s.value, s.share_of_deviation) for s in result.path]}"
    )
    assert result.path
    assert all(step.sql and step.baseline_sql for step in result.path)


async def test_walk_completes_quickly_enough_for_a_live_demo(gateway):
    """The full 8-level query benchmark measured ~350ms batched (see the Task 8 plan
    doc) -- the walker itself must not add material overhead on top of that."""
    incident = _by_kind("device_app_fault")
    window = _window(incident)

    result = await walk(gateway, metric_name="rebuffer", window=window)

    assert result.elapsed_ms < 15_000, (
        f"walk took {result.elapsed_ms:.0f}ms, too slow for a live demo turn"
    )


async def test_walk_records_a_stop_reason_on_every_result(gateway):
    """"Why did it stop here" must always be answerable, whatever the outcome."""
    incident = _by_kind("device_app_fault")
    window = _window(incident)

    result = await walk(gateway, metric_name="rebuffer", window=window)

    assert isinstance(result.stop_reason, StopReason)
