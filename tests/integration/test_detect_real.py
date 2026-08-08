"""Integration tests for continuity/analysis/detect.py against the real 59.8M-event
dataset (2026-01-01..2026-01-22), read through the MCP gateway.

Read-only: no truncation, no writes. Uses the `gateway` fixture from tests/conftest.py,
which is built from ClickHouseConfig.from_env() -- the DEFAULT database holding the
full dataset, NOT the `continuity_test` database tests/integration/test_load.py uses.

Ground truth windows (data/ground_truth.json), for reference only -- never read by the
detector itself:
  INC-APP-ROKU-820:   device_type=roku AND app_version=8.2.0, rebuffer x4.5,
                       2026-01-13 18:00 -> 2026-01-14 02:00
  INC-POP-NW-ATL-2:   cdn=cdn_northwind AND pop=nw-atl-2, startup x3.2,
                       2026-01-16 02:00 -> 08:00
  INC-ENCODE-1:       title_id=1, bitrate x0.45, 2026-01-19 09:00 -> 2026-01-20 15:00
  DECOY-PREMIERE-3:   title_id=3, volume x6, NO QoE effect,
                       2026-01-21 20:00 -> 2026-01-22 01:00
"""

from __future__ import annotations

from datetime import datetime

import pytest

from continuity.analysis.detect import detect
from continuity.analysis.slices import Slice

pytestmark = pytest.mark.integration


def _overlaps(window, true_start: datetime, true_end: datetime) -> bool:
    return window.start < true_end and window.end > true_start


# --- (a) INC-APP-ROKU-820: two dimensions, neither alone identifies it ------------


async def test_detects_roku_820_rebuffer_incident(gateway):
    true_start, true_end = datetime(2026, 1, 13, 18, 0), datetime(2026, 1, 14, 2, 0)
    slice_ = Slice().refine("device_type", "roku").refine("app_version", "8.2.0")

    result = await detect(
        gateway, slice_, "rebuffer", datetime(2026, 1, 13, 12, 0), datetime(2026, 1, 14, 8, 0)
    )

    assert result.windows, "expected at least one anomaly window for the roku/8.2.0 incident"
    assert any(_overlaps(w, true_start, true_end) for w in result.windows)
    # higher-is-worse metric: the incident is a rebuffer SPIKE, so peak_z is positive.
    assert any(w.peak_z > 3.0 for w in result.windows)
    assert all(w.sql == result.sql for w in result.windows)


# --- (b) INC-ENCODE-1: direction handling. bitrate is lower-is-worse -------------


async def test_detects_encode_incident_on_bitrate_with_correct_direction(gateway):
    """A detector that only looked for increases would miss this incident entirely --
    the encode fault is a bitrate DROP (x0.45), so every reported window must carry a
    negative z, not a positive one."""
    true_start, true_end = datetime(2026, 1, 19, 9, 0), datetime(2026, 1, 20, 15, 0)
    slice_ = Slice().refine("title_id", "1")

    result = await detect(
        gateway, slice_, "bitrate", datetime(2026, 1, 19, 6, 0), datetime(2026, 1, 20, 18, 0)
    )

    assert result.windows, "expected at least one anomaly window for the title_id=1 bitrate crash"
    assert any(_overlaps(w, true_start, true_end) for w in result.windows)
    assert all(w.peak_z < 0 for w in result.windows), (
        "bitrate is lower-is-worse; a direction-blind detector would report positive z"
    )


# --- (c) the decoy is a volume spike with no QoE effect: must find NOTHING -------


async def test_stays_silent_on_the_volume_only_decoy(gateway):
    """The false-positive test that matters most. DECOY-PREMIERE-3 is a 6x volume
    spike with effects=[] in ground truth -- rebuffer, the primary QoE health signal,
    must show no anomaly windows across it."""
    slice_ = Slice().refine("title_id", "3")

    result = await detect(
        gateway, slice_, "rebuffer", datetime(2026, 1, 21, 14, 0), datetime(2026, 1, 22, 6, 0)
    )

    assert result.windows == []
    assert result.total_buckets > 0


# --- (d) whole population over a quiet week: replaces the 353 false positives ---


async def test_whole_population_quiet_period_produces_zero_anomaly_windows(gateway):
    """Direct replacement for the naive mean+2sigma detector's 353 alerts, all false,
    all in 18:00-23:00 (see scripts/acceptance_check.py). 2026-01-05..2026-01-10
    contains five nightly traffic peaks and zero planted incidents.

    Individual buckets crossing the robust z threshold are still allowed -- some noise
    at that level is expected even with a seasonality-aware baseline -- but none of it
    sustains for min_run_length buckets, so zero WINDOWS are reported. That's the
    point: run-length + gap-tolerant grouping is what turns per-bucket noise into a
    silent detector rather than the individual z-score threshold alone.
    """
    result = await detect(gateway, Slice(), "rebuffer", datetime(2026, 1, 5), datetime(2026, 1, 10))

    assert result.windows == []
    assert result.total_buckets == 1440  # 5 days * 288 five-minute buckets/day
    assert result.unknown_fraction == pytest.approx(0.0)
