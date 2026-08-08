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

Median and MAD rather than mean and standard deviation, deliberately: a real
incident sitting in the comparison window inflates a mean/sigma baseline and can
mask the next incident. A detector that goes blind after one incident is
self-defeating. See `test_median_and_mad_barely_move_when_trailing_window_...`
for the proof.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import numpy as np

# Converts MAD to a sigma-equivalent under a normal distribution.
_MAD_TO_SIGMA = 1.4826

DEFAULT_TRAILING_DAYS = 7
DEFAULT_LOOKBACK_WEEKS = 4
DEFAULT_MIN_OBSERVATIONS = 4


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
    if lookback_weeks <= 0:
        raise ValueError(f"lookback_weeks must be > 0, got {lookback_weeks}")
    target_date = target.date()
    valid_dates = {target_date - timedelta(weeks=i) for i in range(1, lookback_weeks + 1)}
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


def compute_baseline(
    actual: float | None,
    comparison_values: Sequence[float | None],
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> Baseline:
    """Median/MAD baseline and robust z-score for `actual` against `comparison_values`.

    Returns `INSUFFICIENT_DATA` (never a numeric z) when:
    - `actual` is missing (None/NaN),
    - fewer than `min_observations` clean comparison values are available,
    - MAD is 0 (a flat/thin trailing window) and `actual` differs from the
      median, so the z-score would otherwise be a division by zero.

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
    mad = float(np.median(np.abs(arr - median)))

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
