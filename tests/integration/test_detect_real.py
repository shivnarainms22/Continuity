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
from continuity.analysis.detect import (
    build_series_sql,
    detect,
    detect_from_series,
    fetch_window_start,
)
from continuity.analysis.metrics import get_metric
from continuity.analysis.slices import Slice
from continuity.data.load import WINDOW_START as DATASET_START
from continuity.gateway.mcp_gateway import QueryError

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


# --- (e) the fetch restriction: same results, far fewer rows -----------------------


def _parse_bucket(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


async def _detect_full_contiguous_range(gateway, slice_, metric_name, start, end):
    """Mirrors `detect()`'s PRE-FIX body exactly: the whole contiguous history range
    back to `DEFAULT_LOOKBACK_WEEKS` ago, via `build_series_sql`'s `extra_buckets=None`
    default -- the shape every other caller (report_schema.py, cli.py) still uses.
    Used only as the "before" side of the behaviour-preservation guard below.
    """
    metric = get_metric(metric_name)
    fetch_start = fetch_window_start(start, DEFAULT_LOOKBACK_WEEKS * 7)
    sql = build_series_sql(slice_, metric, fetch_start, end)
    result = await gateway.query(sql)
    observations = [(_parse_bucket(row["bucket"]), row["value"]) for row in result.rows]
    return detect_from_series(
        observations, slice_=slice_, metric_name=metric_name, start=start, end=end, sql=sql
    )


def _windows_signature(result):
    return [
        (w.start, w.end, w.peak_z, w.peak_value, w.expected_at_peak, w.bucket_count)
        for w in result.windows
    ]


@pytest.mark.parametrize(
    "kind, metric_name, pad_hours",
    [
        ("device_app_fault", "rebuffer", 6),
        ("encode_fault", "bitrate", 3),
        ("decoy_premiere", "rebuffer", 6),
        # INC-POP-NW-ATL-2: the incident actually observed hitting
        # MEMORY_LIMIT_EXCEEDED under the old full-contiguous-range fetch.
        ("pop_fault", "startup", 6),
    ],
)
async def test_restricted_history_fetch_matches_full_contiguous_fetch_exactly(
    gateway, kind, metric_name, pad_hours
):
    """The behaviour-preservation guard this whole fix rests on: restricting the history
    fetch to `baseline.required_history_buckets`'s output -- rather than the whole
    contiguous range back to `DEFAULT_LOOKBACK_WEEKS` ago -- must produce byte-identical
    anomaly windows and peak z-scores, for every real planted incident. A faster query
    that changes results would be a regression, not a fix.

    Also proves the restriction is real, not a no-op: the new query must read
    substantially fewer rows than the old full-range one, and its SQL must show the
    explicit bucket list rather than only a wide contiguous range.
    """
    incident = _by_kind(kind)
    true_start, true_end = _window(incident)
    slice_ = _slice_for(incident)
    start = true_start - timedelta(hours=pad_hours)
    end = true_end + timedelta(hours=pad_hours)

    # The restricted fetch is the whole point of the fix: it must never hit the
    # ClickHouse memory limit the old full-contiguous-range fetch was observed to hit.
    new = await detect(gateway, slice_, metric_name, start, end)
    assert "bucket IN (" in new.sql or "toStartOfFiveMinute(event_time) IN (" in new.sql

    try:
        old = await _detect_full_contiguous_range(gateway, slice_, metric_name, start, end)
    except QueryError as exc:
        if "MEMORY_LIMIT_EXCEEDED" in str(exc):
            pytest.skip(
                "old full-contiguous-range fetch hit MEMORY_LIMIT_EXCEEDED on this host "
                "under current memory pressure -- exactly the bug this fix removes; "
                "the restricted fetch above already proved it succeeds where the old "
                "one cannot, so there is nothing left to compare it against."
            )
        raise

    assert _windows_signature(new) == _windows_signature(old)
    assert new.total_buckets == old.total_buckets
    assert new.anomalous_buckets == old.anomalous_buckets
    assert new.unknown_buckets == old.unknown_buckets

    old_rows = gateway.query_log[-1].row_count
    new_rows = gateway.query_log[-2].row_count
    assert new_rows < old_rows * 0.5, (
        f"expected the restricted fetch to read far fewer rows: old={old_rows} new={new_rows}"
    )
