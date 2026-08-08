"""Integration test for continuity.analysis.split against the real 59.8M-event dataset.

Proves the project's core technical claim: a two-step drill-down (split the whole
population on device_type, then split the winning slice on app_version) isolates the
planted incident INC-APP-ROKU-820 (device_type=roku AND app_version=8.2.0), even though
app_version 8.2.0 also ships on firetv/ios/android and roku also runs 8.0.9/8.1.4 -- so
neither dimension alone identifies the fault; only the two-step composition does.

Read only. Uses the DEFAULT database from ClickHouseConfig.from_env() via the `gateway`
fixture in tests/conftest.py -- NOT the separate continuity_test database used by
test_load.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from continuity.analysis.metrics import METRICS
from continuity.analysis.slices import Slice
from continuity.analysis.split import split_dimension

pytestmark = pytest.mark.integration

# The INC-APP-ROKU-820 window, from data/ground_truth.json.
_WINDOW_START = datetime(2026, 1, 13, 18, 0, 0)
_WINDOW_END = datetime(2026, 1, 14, 2, 0, 0)
_WINDOW = (_WINDOW_START, _WINDOW_END)
# Same time-of-day, 7 days earlier: a normal period unaffected by the incident.
_BASELINE_WINDOW = (_WINDOW_START - timedelta(days=7), _WINDOW_END - timedelta(days=7))


async def test_splitting_whole_population_on_device_type_ranks_roku_first(gateway):
    result = await split_dimension(
        gateway,
        slice_=Slice(),
        metric=METRICS["rebuffer"],
        dimension="device_type",
        window=_WINDOW,
        baseline_window=_BASELINE_WINDOW,
    )

    assert result.informative is True
    assert result.values, "split returned no device_type values at all"
    top = result.values[0]
    assert top.value == "roku", (
        f"expected 'roku' to rank first by contribution, got {top.value!r}. "
        f"Full ranking: {[(c.value, c.contribution) for c in result.values]}"
    )
    assert top.contribution is not None and top.contribution > 0
    assert top.share_of_deviation is not None and top.share_of_deviation > 0


async def test_splitting_roku_slice_on_app_version_then_ranks_820_first(gateway):
    """The second step of the drill-down: roku alone doesn't isolate the fault (roku
    also runs 8.0.9 and 8.1.4 unaffected), so this must go one dimension deeper."""
    roku = Slice().refine("device_type", "roku")

    result = await split_dimension(
        gateway,
        slice_=roku,
        metric=METRICS["rebuffer"],
        dimension="app_version",
        window=_WINDOW,
        baseline_window=_BASELINE_WINDOW,
    )

    assert result.informative is True
    assert result.values, "split returned no app_version values at all"
    top = result.values[0]
    assert top.value == "8.2.0", (
        f"expected '8.2.0' to rank first by contribution, got {top.value!r}. "
        f"Full ranking: {[(c.value, c.contribution) for c in result.values]}"
    )
    assert top.contribution is not None and top.contribution > 0
    assert top.share_of_deviation is not None and top.share_of_deviation > 0
