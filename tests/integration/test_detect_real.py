"""Integration tests for continuity/analysis/detect.py against the real dataset, read
through the MCP gateway.

Read-only: no truncation, no writes. Uses the `gateway` fixture from tests/conftest.py,
which is built from ClickHouseConfig.from_env() -- the DEFAULT database holding the
full dataset, NOT the `continuity_test` database tests/integration/test_load.py uses.

Incident windows are DERIVED from data/ground_truth.json rather than hardcoded --
hardcoded incident dates are what made the week-over-week baseline change (moving
incident placement to the end of the window, see continuity/data/incidents.py) painful
to keep in sync. Look up each incident by `kind`, since its `incident_id` embeds a
dynamically-picked title_id (encode/decoy) that can change between reloads.

REQUIRES A RELOAD before these tests can pass: `detect()` now defaults to
`ComparisonMode.WEEK_OVER_WEEK`, which needs `DEFAULT_LOOKBACK_WEEKS` (4) weeks of
prior history for every incident. The dataset currently loaded (and the committed
data/ground_truth.json) still reflects the old 21-day window with start-relative
incident offsets, which does not provide that history. Once the 56-day reload (with
incidents.py's new end-relative placement) runs, this file needs no further changes --
every window is derived from the regenerated ground truth.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from continuity.analysis.baseline import DEFAULT_LOOKBACK_WEEKS
from continuity.analysis.detect import detect
from continuity.analysis.slices import Slice
from continuity.data.load import WINDOW_START as DATASET_START

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


def _overlaps(window, true_start: datetime, true_end: datetime) -> bool:
    return window.start < true_end and window.end > true_start


# --- (a) INC-APP-ROKU-820: two dimensions, neither alone identifies it ------------


async def test_detects_roku_820_rebuffer_incident(gateway):
    incident = _by_kind("device_app_fault")
    true_start, true_end = _window(incident)
    slice_ = _slice_for(incident)

    result = await detect(
        gateway,
        slice_,
        "rebuffer",
        true_start - timedelta(hours=6),
        true_end + timedelta(hours=6),
    )

    assert result.windows, "expected at least one anomaly window for the roku/8.2.0 incident"
    assert any(_overlaps(w, true_start, true_end) for w in result.windows)
    # higher-is-worse metric: the incident is a rebuffer SPIKE, so peak_z is positive.
    assert any(w.peak_z > 3.0 for w in result.windows)
    assert all(w.sql == result.sql for w in result.windows)


# --- (b) the encode incident: direction handling. bitrate is lower-is-worse -------


async def test_detects_encode_incident_on_bitrate_with_correct_direction(gateway):
    """A detector that only looked for increases would miss this incident entirely --
    the encode fault is a bitrate DROP (x0.45), so every reported window must carry a
    negative z, not a positive one."""
    incident = _by_kind("encode_fault")
    true_start, true_end = _window(incident)
    slice_ = _slice_for(incident)

    result = await detect(
        gateway,
        slice_,
        "bitrate",
        true_start - timedelta(hours=3),
        true_end + timedelta(hours=3),
    )

    assert result.windows, "expected at least one anomaly window for the encode bitrate crash"
    assert any(_overlaps(w, true_start, true_end) for w in result.windows)
    assert all(w.peak_z < 0 for w in result.windows), (
        "bitrate is lower-is-worse; a direction-blind detector would report positive z"
    )


# --- (c) the decoy is a volume spike with no QoE effect: must find NOTHING -------


async def test_stays_silent_on_the_volume_only_decoy(gateway):
    """The false-positive test that matters most. The decoy incident is a 6x volume
    spike with effects=[] in ground truth -- rebuffer, the primary QoE health signal,
    must show no anomaly windows across it."""
    incident = _by_kind("decoy_premiere")
    true_start, true_end = _window(incident)
    slice_ = _slice_for(incident)

    result = await detect(
        gateway,
        slice_,
        "rebuffer",
        true_start - timedelta(hours=6),
        true_end + timedelta(hours=6),
    )

    assert result.windows == []
    assert result.total_buckets > 0


# --- (d) whole population over a quiet period: replaces the 353 false positives ---


async def test_whole_population_quiet_period_produces_zero_anomaly_windows(gateway):
    """Direct replacement for the naive mean+2sigma detector's 353 alerts, all false,
    all in 18:00-23:00 (see scripts/acceptance_check.py).

    The quiet window is derived, not hardcoded: 3 days of buffer before the earliest
    planted incident, 5 days long, and asserted to still have `DEFAULT_LOOKBACK_WEEKS`
    weeks of prior history inside the loaded dataset -- otherwise every bucket would
    read UNKNOWN rather than genuinely NORMAL, and the `windows == []` assertion below
    would pass for the wrong reason. `unknown_fraction == 0.0` is the guard against
    exactly that silent false pass.
    """
    ground_truth = _ground_truth_incidents()
    earliest_start = min(_window(inc)[0] for inc in ground_truth)
    dataset_start = DATASET_START.replace(tzinfo=None)

    end = (earliest_start - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=5)
    assert start - timedelta(weeks=DEFAULT_LOOKBACK_WEEKS) >= dataset_start, (
        "derived quiet window does not have enough prior history in the loaded dataset "
        "for a week-over-week baseline -- check incident placement / --days"
    )

    result = await detect(gateway, Slice(), "rebuffer", start, end)

    assert result.windows == []
    assert result.total_buckets == 1440  # 5 days * 288 five-minute buckets/day
    assert result.unknown_fraction == pytest.approx(0.0)
