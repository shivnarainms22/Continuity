"""Unit tests for continuity/analysis/detect.py. Pure maths, no ClickHouse.

Synthetic series are built with a uniform robust baseline (median ~10.1, MAD ~0.3)
across a 7-day trailing window with small per-day jitter, so MAD > 0 and z-scores are
well defined. Overrides inject specific test-day bucket values to simulate blips,
sustained incidents, and gaps.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from continuity.analysis.baseline import Baseline, BaselineStatus
from continuity.analysis.detect import (
    BucketLabel,
    BucketStatus,
    build_series_sql,
    detect_from_series,
    fetch_window_start,
    group_windows,
    label_buckets,
)
from continuity.analysis.metrics import get_metric
from continuity.analysis.slices import Slice

WINDOW_START = datetime(2026, 6, 8, 20, 0)  # a Monday, arbitrary but fixed
_JITTER = [-0.4, 0.3, -0.2, 0.4, -0.3, 0.2, 0.1]  # 7 values -> median 0.1, MAD 0.3


def _series(
    window_start: datetime,
    n_buckets: int,
    *,
    baseline_level: float = 10.0,
    trailing_days: int = 7,
    overrides: dict[int, float] | None = None,
    missing: set[int] | None = None,
) -> tuple[list[tuple[datetime, float]], datetime, datetime]:
    """Trailing history + test day for `n_buckets` consecutive 5-minute buckets.

    `overrides` sets a specific actual value for the test day at a given bucket index
    (0-based into the window); `missing` skips adding a test-day observation entirely,
    simulating a bucket with no rows (e.g. a thin slice with zero traffic).
    """
    overrides = overrides or {}
    missing = missing or set()
    observations: list[tuple[datetime, float]] = []
    for i in range(n_buckets):
        bucket_time = window_start + i * timedelta(minutes=5)
        for day_offset in range(1, trailing_days + 1):
            day = bucket_time.date() - timedelta(days=day_offset)
            jitter = _JITTER[(day_offset - 1) % len(_JITTER)]
            observations.append(
                (datetime.combine(day, bucket_time.time()), baseline_level + jitter)
            )
        if i in missing:
            continue
        actual = overrides.get(i, baseline_level)
        observations.append((datetime.combine(bucket_time.date(), bucket_time.time()), actual))
    end = window_start + n_buckets * timedelta(minutes=5)
    return observations, window_start, end


# ---------------------------------------------------------------------------
# label_buckets: per-bucket classification
# ---------------------------------------------------------------------------


def test_label_buckets_flags_a_sustained_deviation_as_anomalous():
    obs, start, end = _series(WINDOW_START, 6, overrides={2: 15.0, 3: 15.0, 4: 15.0})
    labels = label_buckets(obs, start=start, end=end, metric=get_metric("rebuffer"))
    assert [label.status for label in labels] == [
        BucketStatus.NORMAL,
        BucketStatus.NORMAL,
        BucketStatus.ANOMALOUS,
        BucketStatus.ANOMALOUS,
        BucketStatus.ANOMALOUS,
        BucketStatus.NORMAL,
    ]


def test_label_buckets_marks_a_bucket_with_no_rows_as_unknown_not_normal():
    """A missing actual must never default to 'normal' -- that would make thin
    slices with zero traffic in a bucket silently invisible to the detector."""
    obs, start, end = _series(WINDOW_START, 3, missing={1})
    labels = label_buckets(obs, start=start, end=end, metric=get_metric("rebuffer"))
    assert labels[0].status is BucketStatus.NORMAL
    assert labels[1].status is BucketStatus.UNKNOWN
    assert labels[1].value is None
    assert labels[2].status is BucketStatus.NORMAL


def test_label_buckets_is_insufficient_data_when_trailing_history_is_too_thin():
    """Fewer than min_observations clean comparison days -> UNKNOWN for every bucket,
    regardless of the actual value -- never a confident 'normal'."""
    obs, start, end = _series(WINDOW_START, 4, trailing_days=2, overrides={2: 999.0})
    labels = label_buckets(obs, start=start, end=end, metric=get_metric("rebuffer"))
    assert all(label.status is BucketStatus.UNKNOWN for label in labels)


def test_label_buckets_rejects_end_not_after_start():
    with pytest.raises(ValueError, match="after start"):
        label_buckets([], start=WINDOW_START, end=WINDOW_START, metric=get_metric("rebuffer"))


# ---------------------------------------------------------------------------
# group_windows: run-length + gap tolerance
# ---------------------------------------------------------------------------


def _labels_from_statuses(statuses: list[BucketStatus]) -> list[BucketLabel]:
    ok = Baseline(expected=10.0, spread=1.0, z=5.0, sample_size=7, status=BaselineStatus.OK)
    return [
        BucketLabel(
            bucket=WINDOW_START + i * timedelta(minutes=5), value=15.0, baseline=ok, status=status
        )
        for i, status in enumerate(statuses)
    ]


A, N, U = BucketStatus.ANOMALOUS, BucketStatus.NORMAL, BucketStatus.UNKNOWN


def test_group_windows_ignores_a_single_bucket_blip():
    labels = _labels_from_statuses([N, N, A, N, N])
    assert group_windows(labels) == []


def test_group_windows_reports_a_run_at_exactly_the_minimum_length():
    labels = _labels_from_statuses([N, A, A, A, N])
    assert group_windows(labels, min_run_length=3) == [(1, 3)]


def test_group_windows_drops_a_run_one_short_of_the_minimum_length():
    labels = _labels_from_statuses([N, A, A, N])
    assert group_windows(labels, min_run_length=3) == []


def test_group_windows_merges_across_a_single_recovered_bucket():
    """One normal bucket mid-incident must not split one incident into two windows."""
    labels = _labels_from_statuses([N, A, A, N, A, A, N])
    windows = group_windows(labels, min_run_length=3, max_gap=1)
    assert windows == [(1, 5)]


def test_group_windows_splits_when_the_gap_exceeds_tolerance():
    labels = _labels_from_statuses([A, A, A, N, N, A, A, A])
    windows = group_windows(labels, min_run_length=3, max_gap=1)
    assert windows == [(0, 2), (5, 7)]


def test_group_windows_unknown_buckets_act_as_a_gap_not_as_anomalous():
    labels = _labels_from_statuses([N, A, A, U, A, A, N])
    windows = group_windows(labels, min_run_length=3, max_gap=1)
    assert windows == [(1, 5)]


def test_group_windows_rejects_non_positive_min_run_length():
    with pytest.raises(ValueError, match="min_run_length"):
        group_windows([], min_run_length=0)


def test_group_windows_rejects_negative_max_gap():
    with pytest.raises(ValueError, match="max_gap"):
        group_windows([], max_gap=-1)


# ---------------------------------------------------------------------------
# detect_from_series: end-to-end pure detection
# ---------------------------------------------------------------------------


def test_detect_from_series_finds_nothing_for_a_single_bucket_blip():
    obs, start, end = _series(WINDOW_START, 6, overrides={2: 20.0})
    result = detect_from_series(
        obs, slice_=Slice(), metric_name="rebuffer", start=start, end=end, sql="SELECT 1"
    )
    assert result.windows == []
    assert result.anomalous_buckets == 1
    assert result.total_buckets == 6


def test_detect_from_series_reports_one_window_for_a_sustained_incident():
    obs, start, end = _series(WINDOW_START, 6, overrides={2: 20.0, 3: 20.0, 4: 20.0})
    result = detect_from_series(
        obs, slice_=Slice(), metric_name="rebuffer", start=start, end=end, sql="SELECT 1"
    )
    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.start == WINDOW_START + timedelta(minutes=10)
    assert window.end == WINDOW_START + timedelta(minutes=25)
    assert window.bucket_count == 3
    assert window.peak_value == pytest.approx(20.0)
    assert window.peak_z > 3.0
    assert window.expected_at_peak == pytest.approx(10.1)
    assert window.sql == "SELECT 1"
    assert window.slice == Slice()
    assert window.metric == "rebuffer"


def test_detect_from_series_keeps_one_window_through_a_single_recovered_bucket():
    obs, start, end = _series(
        WINDOW_START, 7, overrides={1: 20.0, 2: 20.0, 3: 10.0, 4: 20.0, 5: 20.0}
    )
    result = detect_from_series(
        obs, slice_=Slice(), metric_name="rebuffer", start=start, end=end, sql="SELECT 1"
    )
    assert len(result.windows) == 1
    assert result.windows[0].bucket_count == 5


def test_detect_from_series_is_direction_aware_for_a_bitrate_drop():
    """bitrate is lower-is-worse: a sustained DROP below baseline must be flagged."""
    obs, start, end = _series(
        WINDOW_START, 6, baseline_level=5000.0, overrides={2: 2000.0, 3: 2000.0, 4: 2000.0}
    )
    result = detect_from_series(
        obs, slice_=Slice(), metric_name="bitrate", start=start, end=end, sql="SELECT 1"
    )
    assert len(result.windows) == 1
    assert result.windows[0].peak_z < -3.0
    assert result.windows[0].peak_value == pytest.approx(2000.0)


def test_detect_from_series_does_not_flag_a_bitrate_rise_as_anomalous():
    """A rise in bitrate is an improvement, not degradation -- must not be flagged
    for a metric where lower is worse. A detector that only checks magnitude, not
    direction, would miss the real bitrate-crash incident and false-positive here."""
    obs, start, end = _series(
        WINDOW_START, 6, baseline_level=5000.0, overrides={2: 9000.0, 3: 9000.0, 4: 9000.0}
    )
    result = detect_from_series(
        obs, slice_=Slice(), metric_name="bitrate", start=start, end=end, sql="SELECT 1"
    )
    assert result.windows == []


def test_detect_from_series_counts_unknown_buckets_and_never_treats_them_as_windows():
    obs, start, end = _series(WINDOW_START, 5, trailing_days=2)
    result = detect_from_series(
        obs, slice_=Slice(), metric_name="rebuffer", start=start, end=end, sql="SELECT 1"
    )
    assert result.windows == []
    assert result.unknown_buckets == 5
    assert result.total_buckets == 5
    assert result.unknown_fraction == pytest.approx(1.0)


def test_detection_result_unknown_fraction_reflects_a_mixed_series():
    obs, start, end = _series(WINDOW_START, 4, missing={0, 1})
    result = detect_from_series(
        obs, slice_=Slice(), metric_name="rebuffer", start=start, end=end, sql="SELECT 1"
    )
    assert result.unknown_buckets == 2
    assert result.total_buckets == 4
    assert result.unknown_fraction == pytest.approx(0.5)


def test_detection_result_unknown_fraction_is_zero_for_an_empty_series_range_guard():
    """Defensive: unknown_fraction must not divide by zero if total_buckets is 0."""
    from continuity.analysis.detect import DetectionResult

    result = DetectionResult(
        slice=Slice(), metric="rebuffer", windows=[], total_buckets=0, anomalous_buckets=0,
        unknown_buckets=0, sql="SELECT 1",
    )
    assert result.unknown_fraction == 0.0


# ---------------------------------------------------------------------------
# SQL construction
# ---------------------------------------------------------------------------


def test_build_series_sql_uses_the_rollup_for_a_dimension_only_slice():
    slice_ = Slice().refine("device_type", "roku")
    sql = build_series_sql(
        slice_, get_metric("rebuffer"), datetime(2026, 1, 1), datetime(2026, 1, 8)
    )
    assert "qoe_rollup_5m" in sql
    assert "device_type = 'roku'" in sql
    assert "GROUP BY bucket" in sql
    assert "2026-01-01 00:00:00" in sql
    assert "2026-01-08 00:00:00" in sql


def test_build_series_sql_uses_raw_events_for_a_title_id_slice():
    slice_ = Slice().refine("title_id", "1")
    sql = build_series_sql(
        slice_, get_metric("bitrate"), datetime(2026, 1, 1), datetime(2026, 1, 2)
    )
    assert "playback_events" in sql
    assert "toStartOfFiveMinute(event_time) AS bucket" in sql
    assert "title_id = 1" in sql


def test_build_series_sql_never_emits_a_bare_count_against_the_rollup():
    slice_ = Slice()
    for metric_name in ("rebuffer", "startup", "bitrate", "errors"):
        sql = build_series_sql(
            slice_, get_metric(metric_name), datetime(2026, 1, 1), datetime(2026, 1, 2)
        )
        assert "count(" not in sql.lower()


def test_fetch_window_start_floors_to_midnight_n_days_before():
    start = fetch_window_start(datetime(2026, 1, 13, 18, 0), trailing_days=7)
    assert start == datetime(2026, 1, 6, 0, 0)
