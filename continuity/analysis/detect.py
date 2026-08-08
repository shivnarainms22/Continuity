"""Seasonality-aware anomaly detection over a per-5-minute-bucket time series.

Fetches one series per (slice, metric, range) through the MCP gateway, computes a
robust median/MAD baseline (`baseline.py`) for every bucket, and groups the anomalous
buckets into contiguous ANOMALY WINDOWS. A lone 5-minute blip is noise; a sustained run
is an incident -- see `group_windows` for the run-length + gap-tolerance logic that
makes that distinction.

The DEFAULT comparison strategy is `ComparisonMode.WEEK_OVER_WEEK` (baseline.py): every
bucket is measured against the same weekday's same time-of-day bucket over the
preceding `lookback_weeks` weeks, which removes weekly seasonality as well as diurnal.
`ComparisonMode.TRAILING_DAYS` (the original strategy) remains available for metrics
with no weekly structure -- see baseline.py's module docstring for why trailing-N-days
alone produces false positives on metrics that do vary by weekday.

The maths (`label_buckets`, `group_windows`, `detect_from_series`) is pure and testable
without Docker. Only `detect()` touches the gateway, and it does so with exactly one
query: the whole comparison history and the test window are pulled together, per Task
1's benchmark (a 7-day 5-minute series costs ~41ms).

Buckets whose baseline could not be computed (`BaselineStatus.INSUFFICIENT_DATA`) are
UNKNOWN, never silently folded into "normal" -- `DetectionResult.unknown_fraction` lets a
caller see when too much of a slice is unmeasurable to trust a quiet result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from continuity.analysis.baseline import (
    DEFAULT_LOOKBACK_WEEKS,
    DEFAULT_MIN_OBSERVATIONS,
    DEFAULT_TRAILING_DAYS,
    Baseline,
    BaselineStatus,
    ComparisonMode,
    Direction,
    compute_baseline,
    is_anomalous,
    select_comparison_window,
    select_week_over_week_window,
)
from continuity.analysis.metrics import Metric, get_metric
from continuity.analysis.slices import Slice
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

BUCKET_WIDTH = timedelta(minutes=5)

# Robust z threshold. Matches baseline.is_anomalous's own default: 3 robust standard
# deviations under median/MAD. Combined with the run-length requirement below, this is
# the pairing that was silent on nightly peaks and loud on all three planted incidents
# (see tests/integration/test_detect_real.py) -- a naive mean+2sigma detector with no
# run-length requirement produced 353 false alerts, all in 18:00-23:00.
DEFAULT_THRESHOLD = 3.0

# Week-over-week is the default comparison strategy -- see the module docstring for why
# trailing-days alone is wrong for metrics with weekly structure (bitrate, errors).
DEFAULT_MODE = ComparisonMode.WEEK_OVER_WEEK

# A single 5-minute blip is noise. Three consecutive anomalous buckets (15 minutes) is
# the shortest run that separated the planted incidents from noise on this dataset.
DEFAULT_MIN_RUN_LENGTH = 3

# One recovered bucket mid-incident must not split one incident into two windows, but
# two consecutive quiet buckets should end it.
DEFAULT_MAX_GAP = 1


class BucketStatus(Enum):
    """Per-bucket classification. UNKNOWN is never merged into NORMAL or ANOMALOUS --
    see baseline.py's own INSUFFICIENT_DATA guard for why that distinction matters."""

    ANOMALOUS = "anomalous"
    NORMAL = "normal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BucketLabel:
    """One bucket's actual value, computed baseline, and resulting classification."""

    bucket: datetime
    value: float | None
    baseline: Baseline
    status: BucketStatus


@dataclass(frozen=True)
class AnomalyWindow:
    """A contiguous run of anomalous buckets, with the query that produced it.

    `peak_z`, `peak_value`, `expected_at_peak` describe the single worst bucket in the
    window (largest deviation in the metric's bad direction) so a caller has one
    headline number without losing the full span.
    """

    slice: Slice
    metric: str
    start: datetime
    end: datetime
    peak_z: float
    peak_value: float
    expected_at_peak: float
    bucket_count: int
    sql: str


@dataclass(frozen=True)
class DetectionResult:
    """Every anomaly window found in [start, end), plus enough bucket accounting for a
    caller to tell "measured and quiet" apart from "mostly unmeasurable"."""

    slice: Slice
    metric: str
    windows: list[AnomalyWindow]
    total_buckets: int
    anomalous_buckets: int
    unknown_buckets: int
    sql: str

    @property
    def unknown_fraction(self) -> float:
        if self.total_buckets == 0:
            return 0.0
        return self.unknown_buckets / self.total_buckets


def _direction_for(metric: Metric) -> Direction:
    return Direction.HIGHER_IS_WORSE if metric.higher_is_worse else Direction.LOWER_IS_WORSE


def _floor_to_bucket(dt: datetime) -> datetime:
    minute = (dt.minute // 5) * 5
    return dt.replace(minute=minute, second=0, microsecond=0)


def _bucket_range(start: datetime, end: datetime) -> list[datetime]:
    start, end = _floor_to_bucket(start), _floor_to_bucket(end)
    if end <= start:
        raise ValueError(f"end ({end}) must be after start ({start})")
    buckets = []
    current = start
    while current < end:
        buckets.append(current)
        current += BUCKET_WIDTH
    return buckets


def fetch_window_start(start: datetime, trailing_days: int = DEFAULT_TRAILING_DAYS) -> datetime:
    """Start of the range to fetch: midnight `trailing_days` days before `start`'s date.

    Flooring to midnight (rather than subtracting `trailing_days` from `start` directly)
    guarantees every comparison day is fetched in full, regardless of what time of day
    `start` itself falls on -- otherwise the first fetched day would be missing its
    early hours and every early-morning target bucket would lose one comparison day.

    Despite the name, `trailing_days` here just means "how many calendar days of
    history to fetch" -- `detect()` passes `trailing_days` itself under
    `ComparisonMode.TRAILING_DAYS`, or `lookback_weeks * 7` under
    `ComparisonMode.WEEK_OVER_WEEK`, so the fetched range covers whichever comparison
    window `mode` needs.
    """
    return datetime.combine(start.date() - timedelta(days=trailing_days), datetime.min.time())


def build_series_sql(
    slice_: Slice, metric: Metric, fetch_start: datetime, fetch_end: datetime
) -> str:
    """The single query that fetches the whole (trailing baseline + test window) series.

    One `GROUP BY bucket` query, not one query per bucket -- see the module docstring.
    """
    raw_events = slice_.requires_raw_events
    expr = metric.sql_for(raw_events=raw_events)
    where = slice_.where_sql()
    start_literal = fetch_start.strftime("%Y-%m-%d %H:%M:%S")
    end_literal = fetch_end.strftime("%Y-%m-%d %H:%M:%S")
    if raw_events:
        return (
            f"SELECT toStartOfFiveMinute(event_time) AS bucket, {expr} AS value "
            f"FROM playback_events "
            f"WHERE event_time >= '{start_literal}' AND event_time < '{end_literal}' "
            f"AND {where} "
            f"GROUP BY bucket ORDER BY bucket"
        )
    return (
        f"SELECT bucket, {expr} AS value FROM qoe_rollup_5m "
        f"WHERE bucket >= '{start_literal}' AND bucket < '{end_literal}' "
        f"AND {where} "
        f"GROUP BY bucket ORDER BY bucket"
    )


def _select_comparison(
    observations: Sequence[tuple[datetime, float | None]],
    bucket: datetime,
    *,
    mode: ComparisonMode,
    trailing_days: int,
    lookback_weeks: int,
) -> list[float]:
    if mode is ComparisonMode.WEEK_OVER_WEEK:
        return select_week_over_week_window(observations, bucket, lookback_weeks=lookback_weeks)
    return select_comparison_window(observations, bucket, trailing_days=trailing_days)


def label_buckets(
    observations: Sequence[tuple[datetime, float | None]],
    *,
    start: datetime,
    end: datetime,
    metric: Metric,
    mode: ComparisonMode = DEFAULT_MODE,
    trailing_days: int = DEFAULT_TRAILING_DAYS,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    threshold: float = DEFAULT_THRESHOLD,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> list[BucketLabel]:
    """Classify every 5-minute bucket in [start, end) as ANOMALOUS, NORMAL or UNKNOWN.

    A bucket absent from `observations` (no rows for that 5-minute interval, e.g. a
    thin slice with no traffic) is treated as a missing actual, which `compute_baseline`
    already turns into INSUFFICIENT_DATA -> UNKNOWN rather than a false "normal".
    """
    by_bucket = dict(observations)
    direction = _direction_for(metric)
    labels: list[BucketLabel] = []
    for bucket in _bucket_range(start, end):
        actual = by_bucket.get(bucket)
        comparison = _select_comparison(
            observations,
            bucket,
            mode=mode,
            trailing_days=trailing_days,
            lookback_weeks=lookback_weeks,
        )
        result = compute_baseline(actual, comparison, min_observations=min_observations)
        if result.status is BaselineStatus.INSUFFICIENT_DATA:
            status = BucketStatus.UNKNOWN
        elif is_anomalous(result, direction=direction, threshold=threshold):
            status = BucketStatus.ANOMALOUS
        else:
            status = BucketStatus.NORMAL
        labels.append(BucketLabel(bucket=bucket, value=actual, baseline=result, status=status))
    return labels


def _find_runs(labels: Sequence[BucketLabel]) -> list[tuple[int, int]]:
    """Maximal contiguous runs of ANOMALOUS buckets, as inclusive (start, end) indices."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, label in enumerate(labels):
        if label.status is BucketStatus.ANOMALOUS:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(labels) - 1))
    return runs


def _merge_runs(runs: Sequence[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by at most `max_gap` non-anomalous buckets."""
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def group_windows(
    labels: Sequence[BucketLabel],
    *,
    min_run_length: int = DEFAULT_MIN_RUN_LENGTH,
    max_gap: int = DEFAULT_MAX_GAP,
) -> list[tuple[int, int]]:
    """Contiguous (allowing small gaps) anomalous runs long enough to report.

    Returns inclusive (start_idx, end_idx) index pairs into `labels`. A run merged
    across a gap must still contain at least `min_run_length` ANOMALOUS buckets in
    total (the gap buckets themselves do not count) -- otherwise two isolated blips a
    few buckets apart would pass the length bar without ever sustaining an incident.
    """
    if min_run_length < 1:
        raise ValueError(f"min_run_length must be >= 1, got {min_run_length}")
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")
    runs = _find_runs(labels)
    merged = _merge_runs(runs, max_gap)
    kept = []
    for start, end in merged:
        anomalous_count = sum(
            1 for label in labels[start : end + 1] if label.status is BucketStatus.ANOMALOUS
        )
        if anomalous_count >= min_run_length:
            kept.append((start, end))
    return kept


def _build_window(
    labels: Sequence[BucketLabel],
    start_idx: int,
    end_idx: int,
    *,
    slice_: Slice,
    metric_name: str,
    sql: str,
    direction: Direction,
) -> AnomalyWindow:
    span = labels[start_idx : end_idx + 1]
    anomalous = [label for label in span if label.status is BucketStatus.ANOMALOUS]
    if direction is Direction.HIGHER_IS_WORSE:
        peak = max(anomalous, key=lambda label: label.baseline.z)
    else:
        peak = min(anomalous, key=lambda label: label.baseline.z)
    return AnomalyWindow(
        slice=slice_,
        metric=metric_name,
        start=span[0].bucket,
        end=span[-1].bucket + BUCKET_WIDTH,
        peak_z=peak.baseline.z,
        peak_value=peak.value,
        expected_at_peak=peak.baseline.expected,
        bucket_count=len(span),
        sql=sql,
    )


def detect_from_series(
    observations: Sequence[tuple[datetime, float | None]],
    *,
    slice_: Slice,
    metric_name: str,
    start: datetime,
    end: datetime,
    sql: str,
    mode: ComparisonMode = DEFAULT_MODE,
    trailing_days: int = DEFAULT_TRAILING_DAYS,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    threshold: float = DEFAULT_THRESHOLD,
    min_run_length: int = DEFAULT_MIN_RUN_LENGTH,
    max_gap: int = DEFAULT_MAX_GAP,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> DetectionResult:
    """Pure detection over an already-fetched series. No I/O -- `detect()` below is the
    only function that touches the gateway; this is what tests/analysis/test_detect.py
    exercises directly with synthetic observations."""
    metric = get_metric(metric_name)
    labels = label_buckets(
        observations,
        start=start,
        end=end,
        metric=metric,
        mode=mode,
        trailing_days=trailing_days,
        lookback_weeks=lookback_weeks,
        threshold=threshold,
        min_observations=min_observations,
    )
    direction = _direction_for(metric)
    groups = group_windows(labels, min_run_length=min_run_length, max_gap=max_gap)
    windows = [
        _build_window(
            labels, s, e, slice_=slice_, metric_name=metric_name, sql=sql, direction=direction
        )
        for s, e in groups
    ]
    anomalous_buckets = sum(1 for label in labels if label.status is BucketStatus.ANOMALOUS)
    unknown_buckets = sum(1 for label in labels if label.status is BucketStatus.UNKNOWN)
    return DetectionResult(
        slice=slice_,
        metric=metric_name,
        windows=windows,
        total_buckets=len(labels),
        anomalous_buckets=anomalous_buckets,
        unknown_buckets=unknown_buckets,
        sql=sql,
    )


def _parse_bucket(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


async def detect(
    gateway: ClickHouseMCPGateway,
    slice_: Slice,
    metric_name: str,
    start: datetime,
    end: datetime,
    *,
    mode: ComparisonMode = DEFAULT_MODE,
    trailing_days: int = DEFAULT_TRAILING_DAYS,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    threshold: float = DEFAULT_THRESHOLD,
    min_run_length: int = DEFAULT_MIN_RUN_LENGTH,
    max_gap: int = DEFAULT_MAX_GAP,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> DetectionResult:
    """Fetch the series through the MCP gateway (one query) and detect anomaly windows
    over [start, end). This is the only function in this module that performs I/O."""
    metric = get_metric(metric_name)
    days_of_history = trailing_days if mode is ComparisonMode.TRAILING_DAYS else lookback_weeks * 7
    fetch_start = fetch_window_start(start, days_of_history)
    sql = build_series_sql(slice_, metric, fetch_start, end)
    result = await gateway.query(sql)
    observations = [(_parse_bucket(row["bucket"]), row["value"]) for row in result.rows]
    return detect_from_series(
        observations,
        slice_=slice_,
        metric_name=metric_name,
        start=start,
        end=end,
        sql=sql,
        mode=mode,
        trailing_days=trailing_days,
        lookback_weeks=lookback_weeks,
        threshold=threshold,
        min_run_length=min_run_length,
        max_gap=max_gap,
        min_observations=min_observations,
    )
