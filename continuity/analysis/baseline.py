"""Seasonality-aware, outlier-robust baselines.

Pure maths only -- no SQL, no ClickHouse, no I/O. Operates on plain
`(timestamp, value)` observations and plain numbers so it is fully testable
without Docker. Integration with `slices.py` / `metrics.py` happens in a later
task; this module knows nothing about either.

Two comparison-window strategies select which observations a bucket's actual value is
measured against:

- `select_week_over_week_window` (the DEFAULT used by `detect.py`): the same (weekday,
  time-of-day) bucket over the preceding `lookback_weeks` weeks (default 4), excluding
  the bucket's own day. This removes diurnal AND weekly seasonality by construction.
- `select_comparison_window`: the same time-of-day bucket over the trailing N days
  (default 7), excluding the day under test. This removes only diurnal seasonality --
  correct for metrics with no weekly structure, but on a metric that DOES vary by
  weekday (bitrate, errors: this dataset averages ~3100 kbps Mon-Thu, ~2900 Fri, ~2525
  Sat/Sun) a trailing 7-day window contains exactly one instance of the target
  bucket's own weekday and six of other weekdays. It does not neutralise a weekly
  pattern, it amplifies it: on a weekend bucket the tight six-day midweek cluster
  produces a tiny MAD, so the legitimate weekend value reads as a huge |z| -- a false
  positive every weekend. See
  `test_trailing_days_mode_false_positives_on_a_legitimate_weekend_value_but_week_over_week_does_not`.

Either way, baseline = the MEDIAN of the selected comparison values. Spread = MAD
(median absolute deviation), scaled by 1.4826 so it estimates sigma under normality.
Robust z = (actual - median) / (1.4826 * MAD).

The LEVEL (median) is always the same-(weekday, time-of-day) comparison window above --
that is what removes seasonality and it works. The SPREAD, if computed from those same
`lookback_weeks` (4) points alone, is too few: when they happen to cluster tightly, MAD
is tiny and a trivial absolute move produces a huge z (measured on the real dataset: a
quiet 5-day period with rebuffer moving from ~0.04% to ~0.06% of watch time produced
z as high as 14). `select_neighbourhood_residuals` pools the spread sample over a
time-of-day NEIGHBOURHOOD (+/- `radius` buckets) on the same weekdays instead -- up to
`(2 * radius + 1) * lookback_weeks` points rather than `lookback_weeks`. Because
QoE genuinely varies across that neighbourhood (the shoulders of the evening peak are
not the same level as its centre), each neighbourhood slot's OWN week-over-week median
is subtracted before pooling, so the pooled sample measures dispersion around the
seasonal curve, not the diurnal slope itself. See
`test_select_neighbourhood_residuals_reflects_noise_not_the_diurnal_slope` for the proof
(and the naive-MAD failure it guards against).

Median and MAD rather than mean and standard deviation, deliberately: a real
incident sitting in the comparison window inflates a mean/sigma baseline and can
mask the next incident. A detector that goes blind after one incident is
self-defeating. See `test_median_and_mad_barely_move_when_trailing_window_...`
for the proof.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum

import numpy as np

# Converts MAD to a sigma-equivalent under a normal distribution.
_MAD_TO_SIGMA = 1.4826

DEFAULT_TRAILING_DAYS = 7
DEFAULT_LOOKBACK_WEEKS = 4
DEFAULT_MIN_OBSERVATIONS = 4
# +/- 24 buckets at 5 minutes/bucket = a +/-2 hour time-of-day neighbourhood. A +/-30
# minute neighbourhood (radius=6) already pools 13 * lookback_weeks points and helps,
# but measured on the real 63.85M-row dataset it was not always enough: with n=4 per
# slot, each slot's own-median centring has an inherent small-sample bias (the two
# middle-ranked values get near-zero residuals, the two extreme ones get large ones),
# and that bias does not average away by pooling more copies of the same 4-point
# shape -- it needs slots spread wide enough that the bulk of the neighbourhood's
# diurnal shape genuinely differs, not just more repeats of the same shape. +/-2 hours
# was the smallest radius that reliably cleared every measured false positive on a
# quiet 5-day period (see tests/integration/test_detect_real.py) while leaving every
# planted incident's peak z far above threshold.
DEFAULT_NEIGHBOURHOOD_RADIUS = 24
# The comparison-window functions above operate on 5-minute buckets throughout this
# codebase (see detect.py's BUCKET_WIDTH); the neighbourhood pooling below needs the
# same width to shift target-time-of-day by whole buckets.
DEFAULT_BUCKET_WIDTH = timedelta(minutes=5)


class ComparisonMode(Enum):
    """Which comparison-window selection strategy a bucket's baseline is built from.

    WEEK_OVER_WEEK is the default (see module docstring): it removes weekly seasonality
    as well as diurnal, which TRAILING_DAYS does not. TRAILING_DAYS remains available
    for metrics with no weekly structure -- it is the original, already-tested strategy.
    """

    WEEK_OVER_WEEK = "week_over_week"
    TRAILING_DAYS = "trailing_days"


class BaselineStatus(Enum):
    """Distinguishes "we measured it" from "we could not tell".

    A caller must branch on this before touching `expected`/`spread`/`z`.
    INSUFFICIENT_DATA never carries a usable z-score -- it is never `0.0`,
    never `inf` -- so it can never be silently read as "not anomalous".
    """

    OK = "ok"
    INSUFFICIENT_DATA = "insufficient_data"


class Direction(Enum):
    """Which sign of deviation is bad for a given metric."""

    HIGHER_IS_WORSE = "higher_is_worse"
    LOWER_IS_WORSE = "lower_is_worse"


@dataclass(frozen=True)
class Baseline:
    """Result of comparing an actual value against its robust time-of-day baseline.

    `expected`, `spread`, `z` are all `None` when `status` is
    `INSUFFICIENT_DATA` -- there is no numeric fallback a caller could
    accidentally treat as a real measurement.
    """

    expected: float | None
    spread: float | None
    z: float | None
    sample_size: int
    status: BaselineStatus


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def select_comparison_window(
    observations: Sequence[tuple[datetime, float | None]],
    target: datetime,
    *,
    trailing_days: int = DEFAULT_TRAILING_DAYS,
) -> list[float]:
    """Select values from the same time-of-day bucket over the trailing window.

    "Same time-of-day bucket" means matching (hour, minute) of `target`.
    "Trailing `trailing_days` days" means calendar dates in
    [target.date() - trailing_days, target.date() - 1], i.e. strictly before
    the day under test -- that day is always excluded, even if an observation
    for it is present in `observations`. NaN/None values are dropped.
    """
    if trailing_days <= 0:
        raise ValueError(f"trailing_days must be > 0, got {trailing_days}")
    target_date = target.date()
    earliest_date = target_date - timedelta(days=trailing_days)
    selected: list[float] = []
    for ts, value in observations:
        if (ts.hour, ts.minute) != (target.hour, target.minute):
            continue
        obs_date = ts.date()
        if not (earliest_date <= obs_date < target_date):
            continue
        if _is_missing(value):
            continue
        selected.append(float(value))
    return selected


def _lookback_dates(target_date: date, *, lookback_weeks: int) -> list[date]:
    """Calendar dates exactly `7 * i` days before `target_date`, for `i` in
    `1..lookback_weeks` -- `target_date`'s own weekday on each of the preceding
    `lookback_weeks` weeks, excluding `target_date` itself.

    Shared by `select_week_over_week_window`, `select_neighbourhood_residuals` and
    `required_history_buckets` so all three enumerate the same historical slots by
    construction -- a second, hand-rolled copy of this arithmetic is exactly what
    would let the SQL-fetch restriction in detect.py silently drift out of sync with
    what the baseline actually reads.
    """
    if lookback_weeks <= 0:
        raise ValueError(f"lookback_weeks must be > 0, got {lookback_weeks}")
    return [target_date - timedelta(weeks=i) for i in range(1, lookback_weeks + 1)]


def select_week_over_week_window(
    observations: Sequence[tuple[datetime, float | None]],
    target: datetime,
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> list[float]:
    """Select values from the same (weekday, time-of-day) bucket over the preceding
    `lookback_weeks` weeks.

    "Same weekday, same time-of-day" means matching (hour, minute) of `target` on a
    calendar date exactly `7 * i` days before `target.date()`, for `i` in
    `1..lookback_weeks` -- i.e. `target`'s own weekday, never any other. Because `i`
    starts at 1, `target`'s own day is excluded by construction, exactly like
    `select_comparison_window`'s trailing window. NaN/None values are dropped.

    This is the fix for the flaw `select_comparison_window` has on metrics with weekly
    structure: it never mixes weekday observations into a weekend baseline (or vice
    versa), so it cannot manufacture the false positives a trailing-N-days window does.
    """
    valid_dates = set(_lookback_dates(target.date(), lookback_weeks=lookback_weeks))
    selected: list[float] = []
    for ts, value in observations:
        if (ts.hour, ts.minute) != (target.hour, target.minute):
            continue
        if ts.date() not in valid_dates:
            continue
        if _is_missing(value):
            continue
        selected.append(float(value))
    return selected


def select_neighbourhood_residuals(
    observations: Sequence[tuple[datetime, float | None]],
    target: datetime,
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    radius: int = DEFAULT_NEIGHBOURHOOD_RADIUS,
    bucket_width: timedelta = DEFAULT_BUCKET_WIDTH,
) -> list[float]:
    """Pooled spread sample for `target`'s bucket: residuals around the seasonal curve,
    gathered from a time-of-day neighbourhood on the same weekday over `lookback_weeks`.

    For each of the `2 * radius + 1` time-of-day slots at `target +/- k * bucket_width`
    (`k` in `0..radius`) -- each selected exactly like `select_week_over_week_window`,
    i.e. same weekday, `lookback_weeks` weeks back, target's own day excluded -- this
    computes THAT SLOT's OWN week-over-week median first, then returns
    `value - slot_median` for every observation in every slot.

    Subtracting each slot's own median before pooling is the whole point: a slot half an
    hour from `target` has a different expected LEVEL (QoE varies across a 30-minute
    span, especially at the shoulders of the evening peak), so pooling raw neighbourhood
    values would measure the diurnal slope, not the noise. Residuals measure dispersion
    around the seasonal curve, independent of where on that curve each slot sits -- see
    `test_select_neighbourhood_residuals_reflects_noise_not_the_diurnal_slope`.

    A slot with no data yet (e.g. early in the dataset, before `lookback_weeks` of
    history exists) contributes nothing rather than raising -- same "return fewer
    values, never a fabricated one" contract as `select_week_over_week_window`. With
    `radius=0` this reduces to exactly `select_week_over_week_window`'s own values
    minus their shared median, i.e. the pre-pooling behaviour.

    Indexes `observations` into a dict once up front rather than re-scanning the full
    sequence for each of the `2 * radius + 1` slots (what repeatedly calling
    `select_week_over_week_window` would cost): O(n) total instead of O(n * radius),
    which matters because this runs once per bucket in `detect.py::label_buckets`.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    by_bucket = {ts: float(value) for ts, value in observations if not _is_missing(value)}
    residuals: list[float] = []
    for offset in range(-radius, radius + 1):
        slot_target = target + offset * bucket_width
        slot_time = slot_target.time()
        slot_values = [
            by_bucket[dt]
            for dt in (
                datetime.combine(d, slot_time)
                for d in _lookback_dates(slot_target.date(), lookback_weeks=lookback_weeks)
            )
            if dt in by_bucket
        ]
        if not slot_values:
            continue
        slot_median = float(np.median(np.asarray(slot_values, dtype=float)))
        residuals.extend(v - slot_median for v in slot_values)
    return residuals


def required_history_buckets(
    targets: Iterable[datetime],
    *,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    radius: int = DEFAULT_NEIGHBOURHOOD_RADIUS,
    bucket_width: timedelta = DEFAULT_BUCKET_WIDTH,
) -> set[datetime]:
    """Every historical timestamp `select_week_over_week_window` or
    `select_neighbourhood_residuals` could read for the given `targets` -- a
    detection window's own buckets under WEEK_OVER_WEEK.

    A SQL caller (`detect.py::build_series_sql`) fetches the test window's own range
    in full plus exactly this set, instead of the whole contiguous history range in
    between -- see detect.py's module docstring for why that range is ~30x bigger
    than what the baseline ever consumes.

    For each `target`, every one of the `2 * radius + 1` neighbourhood slots
    (`target +/- k * bucket_width`, `k` in `0..radius`) contributes its own
    `lookback_weeks` historical dates via `_lookback_dates` -- the `offset == 0` slot
    IS the LEVEL comparison `select_week_over_week_window` reads, so no separate pass
    is needed for it. Built from the exact same `_lookback_dates` helper those two
    selection functions use, so this enumeration cannot silently diverge from what
    they actually select -- the trap called out in the task: under-covering here would
    turn OK baselines into INSUFFICIENT_DATA rather than raising anything visible.
    """
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    needed: set[datetime] = set()
    for target in targets:
        for offset in range(-radius, radius + 1):
            slot_target = target + offset * bucket_width
            slot_time = slot_target.time()
            for d in _lookback_dates(slot_target.date(), lookback_weeks=lookback_weeks):
                needed.add(datetime.combine(d, slot_time))
    return needed


def compute_baseline(
    actual: float | None,
    comparison_values: Sequence[float | None],
    *,
    spread_values: Sequence[float | None] | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> Baseline:
    """Median/MAD baseline and robust z-score for `actual` against `comparison_values`.

    `spread_values`, if given, is a separate sample the SPREAD (MAD) is computed from --
    typically `select_neighbourhood_residuals`'s pooled, per-slot-centred residuals,
    which are a far more stable estimate than the handful of raw `comparison_values`
    (see the module docstring). The LEVEL (median) and the `min_observations` gate are
    always computed from `comparison_values` alone, regardless of how many
    `spread_values` are available -- a wide, stable spread sample must never paper over
    a level we could not actually establish. When `spread_values` is `None` (the
    default), the MAD is computed from `comparison_values` itself, exactly as before --
    this keeps `ComparisonMode.TRAILING_DAYS` and any direct caller byte-identical to
    the pre-pooling behaviour.

    Returns `INSUFFICIENT_DATA` (never a numeric z) when:
    - `actual` is missing (None/NaN),
    - fewer than `min_observations` clean comparison values are available,
    - MAD is 0 (a flat/thin trailing window, or a genuinely flat pooled spread) and
      `actual` differs from the median, so the z-score would otherwise be a division
      by zero.

    When MAD is 0 and `actual` equals the median exactly, the bucket is
    genuinely flat and `actual` fits it: that is `OK` with `z=0.0`, not an
    error.
    """
    if min_observations < 1:
        raise ValueError(f"min_observations must be >= 1, got {min_observations}")
    if _is_missing(actual):
        return Baseline(
            expected=None,
            spread=None,
            z=None,
            sample_size=0,
            status=BaselineStatus.INSUFFICIENT_DATA,
        )
    clean = [v for v in comparison_values if not _is_missing(v)]
    n = len(clean)
    if n < min_observations:
        return Baseline(
            expected=None,
            spread=None,
            z=None,
            sample_size=n,
            status=BaselineStatus.INSUFFICIENT_DATA,
        )

    arr = np.asarray(clean, dtype=float)
    median = float(np.median(arr))

    if spread_values is None:
        spread_sample = arr
    else:
        spread_clean = [v for v in spread_values if not _is_missing(v)]
        spread_sample = np.asarray(spread_clean, dtype=float)

    if spread_sample.size == 0:
        mad = 0.0
    else:
        spread_center = float(np.median(spread_sample))
        mad = float(np.median(np.abs(spread_sample - spread_center)))

    if mad == 0.0:
        if actual == median:
            return Baseline(
                expected=median, spread=0.0, z=0.0, sample_size=n, status=BaselineStatus.OK
            )
        return Baseline(
            expected=median,
            spread=0.0,
            z=None,
            sample_size=n,
            status=BaselineStatus.INSUFFICIENT_DATA,
        )

    spread = _MAD_TO_SIGMA * mad
    z = (float(actual) - median) / spread
    return Baseline(expected=median, spread=spread, z=z, sample_size=n, status=BaselineStatus.OK)


def is_anomalous(baseline: Baseline, *, direction: Direction, threshold: float = 3.0) -> bool:
    """Whether `baseline` crosses `threshold` robust standard deviations, direction-aware.

    Raises on `INSUFFICIENT_DATA` rather than returning `False`: silently
    treating "we could not tell" as "not anomalous" is exactly the failure
    mode this module exists to prevent. Callers must check `baseline.status`
    themselves before calling this.
    """
    if baseline.status is BaselineStatus.INSUFFICIENT_DATA or baseline.z is None:
        raise ValueError(
            "cannot evaluate anomaly on an INSUFFICIENT_DATA baseline; check status first"
        )
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if direction is Direction.HIGHER_IS_WORSE:
        return baseline.z >= threshold
    return baseline.z <= -threshold
