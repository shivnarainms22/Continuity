from datetime import datetime, timedelta

import numpy as np
import pytest

from continuity.analysis.baseline import (
    Baseline,
    BaselineStatus,
    Direction,
    compute_baseline,
    is_anomalous,
    select_comparison_window,
    select_week_over_week_window,
)

# ---------------------------------------------------------------------------
# select_comparison_window: selection logic tested independently of the maths.
# ---------------------------------------------------------------------------


def _series_at_same_time_of_day(day_offsets: list[int], hour: int = 21, minute: int = 0) -> list[
    tuple[datetime, float]
]:
    """Build observations at `hour:minute` on `day_offsets` days before 2026-08-08."""
    target_day = datetime(2026, 8, 8, hour, minute)
    return [(target_day - timedelta(days=offset), float(10 + offset)) for offset in day_offsets]


def test_select_comparison_window_picks_same_time_of_day_across_trailing_days():
    target = datetime(2026, 8, 8, 21, 0)
    observations = _series_at_same_time_of_day([1, 2, 3, 4, 5, 6, 7])
    selected = select_comparison_window(observations, target, trailing_days=7)
    assert sorted(selected) == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]


def test_select_comparison_window_excludes_the_day_under_test():
    """Off-by-one guard: an observation for the target's own day must never leak in,
    even if it happens to be present in the series and matches the time-of-day."""
    target = datetime(2026, 8, 8, 21, 0)
    same_day_obs = (datetime(2026, 8, 8, 21, 0), 999.0)
    trailing_obs = _series_at_same_time_of_day([1, 2, 3])
    observations = [same_day_obs, *trailing_obs]
    selected = select_comparison_window(observations, target, trailing_days=7)
    assert 999.0 not in selected
    assert sorted(selected) == [11.0, 12.0, 13.0]


def test_select_comparison_window_excludes_days_older_than_the_trailing_window():
    target = datetime(2026, 8, 8, 21, 0)
    observations = _series_at_same_time_of_day([1, 7, 8, 30])
    selected = select_comparison_window(observations, target, trailing_days=7)
    assert sorted(selected) == [11.0, 17.0]


def test_select_comparison_window_ignores_different_time_of_day():
    target = datetime(2026, 8, 8, 21, 0)
    observations = [
        (datetime(2026, 8, 7, 21, 0), 100.0),  # matches
        (datetime(2026, 8, 7, 9, 0), 999.0),  # different hour, must be excluded
    ]
    selected = select_comparison_window(observations, target, trailing_days=7)
    assert selected == [100.0]


def test_select_comparison_window_drops_none_and_nan_values():
    target = datetime(2026, 8, 8, 21, 0)
    observations = [
        (datetime(2026, 8, 7, 21, 0), 10.0),
        (datetime(2026, 8, 6, 21, 0), None),
        (datetime(2026, 8, 5, 21, 0), float("nan")),
    ]
    selected = select_comparison_window(observations, target, trailing_days=7)
    assert selected == [10.0]


def test_select_comparison_window_rejects_non_positive_trailing_days():
    target = datetime(2026, 8, 8, 21, 0)
    with pytest.raises(ValueError, match="trailing_days"):
        select_comparison_window([], target, trailing_days=0)


# ---------------------------------------------------------------------------
# select_week_over_week_window: same weekday, same time-of-day, N weeks back.
# ---------------------------------------------------------------------------


def test_select_week_over_week_window_picks_only_the_same_weekday_same_time_of_day():
    target = datetime(2026, 8, 8, 21, 0)  # a Saturday
    same_weekday = _series_at_same_time_of_day([7, 14, 21, 28])  # prior Saturdays
    other_weekdays = _series_at_same_time_of_day([1, 2, 3, 4, 5, 6])  # every other weekday
    observations = [*same_weekday, *other_weekdays]
    selected = select_week_over_week_window(observations, target, lookback_weeks=4)
    assert sorted(selected) == [17.0, 24.0, 31.0, 38.0]


def test_select_week_over_week_window_excludes_the_target_own_day():
    target = datetime(2026, 8, 8, 21, 0)
    own_day = (datetime(2026, 8, 8, 21, 0), 999.0)
    prior_weeks = _series_at_same_time_of_day([7, 14, 21, 28])
    observations = [own_day, *prior_weeks]
    selected = select_week_over_week_window(observations, target, lookback_weeks=4)
    assert 999.0 not in selected
    assert sorted(selected) == [17.0, 24.0, 31.0, 38.0]


def test_select_week_over_week_window_respects_the_lookback_weeks_limit():
    target = datetime(2026, 8, 8, 21, 0)
    observations = _series_at_same_time_of_day([7, 14, 21, 28, 35])  # 5 weeks available
    selected = select_week_over_week_window(observations, target, lookback_weeks=4)
    assert sorted(selected) == [17.0, 24.0, 31.0, 38.0]  # the 5th week (35 days back) excluded


def test_select_week_over_week_window_ignores_different_time_of_day():
    target = datetime(2026, 8, 8, 21, 0)
    observations = [
        (datetime(2026, 8, 1, 21, 0), 100.0),  # 7 days back, matches
        (datetime(2026, 8, 1, 9, 0), 999.0),  # same day, different hour -- must be excluded
    ]
    selected = select_week_over_week_window(observations, target, lookback_weeks=4)
    assert selected == [100.0]


def test_select_week_over_week_window_drops_none_and_nan_values():
    target = datetime(2026, 8, 8, 21, 0)
    observations = [
        (datetime(2026, 8, 1, 21, 0), 10.0),
        (datetime(2026, 7, 25, 21, 0), None),
        (datetime(2026, 7, 18, 21, 0), float("nan")),
    ]
    selected = select_week_over_week_window(observations, target, lookback_weeks=4)
    assert selected == [10.0]


def test_select_week_over_week_window_returns_fewer_values_when_history_is_short():
    """Early in the dataset only 2 prior same-weekday weeks exist yet -- the caller
    (compute_baseline via min_observations) must see exactly 2 values, never a padded
    or fabricated 4."""
    target = datetime(2026, 8, 8, 21, 0)
    observations = _series_at_same_time_of_day([7, 14])
    selected = select_week_over_week_window(observations, target, lookback_weeks=4)
    assert sorted(selected) == [17.0, 24.0]


def test_select_week_over_week_window_rejects_non_positive_lookback_weeks():
    target = datetime(2026, 8, 8, 21, 0)
    with pytest.raises(ValueError, match="lookback_weeks"):
        select_week_over_week_window([], target, lookback_weeks=0)


# ---------------------------------------------------------------------------
# compute_baseline: the statistics, and every edge case that causes silent failure.
# ---------------------------------------------------------------------------


def test_compute_baseline_reports_ok_with_expected_median_and_robust_z():
    comparison = [8.0, 9.0, 10.0, 10.0, 10.0, 11.0, 12.0]
    result = compute_baseline(10.0, comparison)
    assert result.status is BaselineStatus.OK
    assert result.expected == pytest.approx(10.0)
    assert result.sample_size == 7


def test_compute_baseline_z_is_zero_when_actual_equals_median():
    comparison = [8.0, 9.0, 10.0, 11.0, 12.0]
    result = compute_baseline(10.0, comparison)
    assert result.status is BaselineStatus.OK
    assert result.z == 0.0


def test_compute_baseline_z_is_positive_when_actual_exceeds_median():
    comparison = [8.0, 9.0, 10.0, 11.0, 12.0]
    result = compute_baseline(20.0, comparison)
    assert result.status is BaselineStatus.OK
    assert result.z > 0


def test_compute_baseline_z_is_negative_when_actual_is_below_median():
    comparison = [8.0, 9.0, 10.0, 11.0, 12.0]
    result = compute_baseline(2.0, comparison)
    assert result.status is BaselineStatus.OK
    assert result.z < 0


def test_compute_baseline_is_insufficient_data_below_minimum_observations():
    comparison = [10.0, 10.0, 10.0]  # 3 < default minimum of 4
    result = compute_baseline(50.0, comparison)
    assert result.status is BaselineStatus.INSUFFICIENT_DATA
    assert result.expected is None
    assert result.spread is None
    assert result.z is None


def test_compute_baseline_is_insufficient_data_with_zero_observations():
    result = compute_baseline(5.0, [])
    assert result.status is BaselineStatus.INSUFFICIENT_DATA
    assert result.sample_size == 0
    assert result.z is None


def test_compute_baseline_flat_window_with_matching_actual_is_ok_not_anomalous():
    """MAD == 0 AND actual == median: a genuinely flat slice behaving normally,
    not an error."""
    comparison = [10.0, 10.0, 10.0, 10.0, 10.0]
    result = compute_baseline(10.0, comparison)
    assert result.status is BaselineStatus.OK
    assert result.spread == 0.0
    assert result.z == 0.0


def test_compute_baseline_flat_window_with_differing_actual_is_insufficient_not_infinite():
    """MAD == 0 but actual != median: z would be a division by zero. Must never
    surface as inf and must never surface as a silent 0.0 that looks like 'fine'."""
    comparison = [10.0, 10.0, 10.0, 10.0, 10.0]
    result = compute_baseline(15.0, comparison)
    assert result.status is BaselineStatus.INSUFFICIENT_DATA
    assert result.z is None
    assert result.z != float("inf")
    assert result.z != 0.0


def test_compute_baseline_excludes_nan_and_none_from_comparison_values():
    comparison = [10.0, None, 10.0, float("nan"), 10.0, 10.0]
    result = compute_baseline(10.0, comparison)
    assert result.status is BaselineStatus.OK
    assert result.sample_size == 4


def test_compute_baseline_is_insufficient_data_when_actual_is_none():
    result = compute_baseline(None, [8.0, 9.0, 10.0, 11.0, 12.0])
    assert result.status is BaselineStatus.INSUFFICIENT_DATA
    assert result.z is None


def test_compute_baseline_is_insufficient_data_when_actual_is_nan():
    result = compute_baseline(float("nan"), [8.0, 9.0, 10.0, 11.0, 12.0])
    assert result.status is BaselineStatus.INSUFFICIENT_DATA
    assert result.z is None


def test_compute_baseline_rejects_non_positive_min_observations():
    with pytest.raises(ValueError, match="min_observations"):
        compute_baseline(10.0, [1.0, 2.0], min_observations=0)


def test_compute_baseline_status_cannot_be_mistaken_for_ok_via_type():
    """The dataclass is frozen and status is an explicit enum member, not a bool or
    a magic number -- a caller cannot accidentally treat INSUFFICIENT_DATA as OK by
    truthiness or numeric comparison."""
    result = compute_baseline(5.0, [])
    assert isinstance(result, Baseline)
    assert result.status != BaselineStatus.OK
    assert BaselineStatus.OK.value != BaselineStatus.INSUFFICIENT_DATA.value


# ---------------------------------------------------------------------------
# Robustness proof: one extreme outlier in the trailing window must barely move
# the median/MAD baseline, in contrast to a mean/std baseline.
# ---------------------------------------------------------------------------


def test_median_and_mad_barely_move_when_trailing_window_contains_one_extreme_outlier():
    actual = 10.0
    clean_window = [10.0, 11.0, 9.0, 10.0, 12.0, 10.0, 11.0]
    contaminated_window = [10.0, 11.0, 9.0, 10.0, 12.0, 10.0, 1000.0]  # one outlier planted

    robust_before = compute_baseline(actual, clean_window)
    robust_after = compute_baseline(actual, contaminated_window)

    assert robust_before.status is BaselineStatus.OK
    assert robust_after.status is BaselineStatus.OK
    # The median-based baseline is untouched by a single extreme outlier.
    assert robust_before.expected == pytest.approx(10.0)
    assert robust_after.expected == pytest.approx(10.0)
    assert robust_before.z == pytest.approx(0.0)
    assert robust_after.z == pytest.approx(0.0)

    # Contrast: a naive mean/std baseline is wrecked by the same outlier -- this is
    # the exact failure mode this module exists to avoid (an incident in the
    # trailing window inflating sigma and masking the next one).
    naive_mean_before = float(np.mean(clean_window))
    naive_mean_after = float(np.mean(contaminated_window))
    naive_std_before = float(np.std(clean_window, ddof=1))
    naive_std_after = float(np.std(contaminated_window, ddof=1))

    assert naive_mean_before == pytest.approx(10.428571, rel=1e-4)
    assert naive_mean_after == pytest.approx(151.714286, rel=1e-4)
    assert naive_std_before == pytest.approx(0.975900, rel=1e-4)
    assert naive_std_after == pytest.approx(374.060028, rel=1e-4)

    # The naive baseline moved by >140; the robust baseline moved by exactly 0.
    assert abs(naive_mean_after - naive_mean_before) > 140
    assert robust_after.expected == robust_before.expected


# ---------------------------------------------------------------------------
# Direction-aware anomaly detection.
# ---------------------------------------------------------------------------


def test_is_anomalous_true_for_higher_is_worse_when_z_exceeds_threshold():
    result = compute_baseline(50.0, [8.0, 9.0, 10.0, 11.0, 12.0])
    assert is_anomalous(result, direction=Direction.HIGHER_IS_WORSE, threshold=3.0) is True


def test_is_anomalous_false_for_higher_is_worse_when_z_is_negative():
    """Lower than baseline is not anomalous when higher is the bad direction."""
    result = compute_baseline(2.0, [8.0, 9.0, 10.0, 11.0, 12.0])
    assert is_anomalous(result, direction=Direction.HIGHER_IS_WORSE, threshold=3.0) is False


def test_is_anomalous_true_for_lower_is_worse_when_z_is_very_negative():
    """A bitrate crash (low is bad) must be flagged on a negative z."""
    result = compute_baseline(2.0, [8.0, 9.0, 10.0, 11.0, 12.0])
    assert is_anomalous(result, direction=Direction.LOWER_IS_WORSE, threshold=3.0) is True


def test_is_anomalous_false_for_lower_is_worse_when_z_is_positive():
    result = compute_baseline(50.0, [8.0, 9.0, 10.0, 11.0, 12.0])
    assert is_anomalous(result, direction=Direction.LOWER_IS_WORSE, threshold=3.0) is False


def test_is_anomalous_raises_on_insufficient_data_instead_of_returning_false():
    """The critical safety property: a caller cannot get a silent 'not anomalous'
    out of a baseline we could not actually compute."""
    result = compute_baseline(5.0, [])
    with pytest.raises(ValueError, match="INSUFFICIENT_DATA"):
        is_anomalous(result, direction=Direction.HIGHER_IS_WORSE)


def test_is_anomalous_rejects_negative_threshold():
    result = compute_baseline(10.0, [8.0, 9.0, 10.0, 11.0, 12.0])
    with pytest.raises(ValueError, match="threshold"):
        is_anomalous(result, direction=Direction.HIGHER_IS_WORSE, threshold=-1.0)


# ---------------------------------------------------------------------------
# Regression test for the actual production bug: a trailing-N-days window does not
# neutralise weekly seasonality, it amplifies it.
# ---------------------------------------------------------------------------


def test_trailing_days_flags_legit_weekend_value_but_week_over_week_does_not():
    """Measured on the real dataset: bitrate averages ~3100 kbps Mon-Thu, ~2900 Fri,
    ~2525 Sat/Sun. For a Saturday target, a trailing-7-day window contains exactly ONE
    prior Saturday and SIX midweek-leaning days, so its median sits near the midweek
    level (~3100) with a tiny MAD -- a perfectly ordinary weekend value then reads as a
    huge robust z. Week-over-week (same weekday only) compares the Saturday against
    prior Saturdays alone, so the same value is unremarkable.
    """
    weekday_level = {
        0: 3100.0,  # Monday
        1: 3100.0,
        2: 3100.0,
        3: 3100.0,  # Thursday
        4: 2900.0,  # Friday
        5: 2525.0,  # Saturday
        6: 2525.0,  # Sunday
    }
    jitter_cycle = [-12.0, 8.0, -5.0, 15.0, -8.0, 3.0, -3.0, 10.0]

    target = datetime(2026, 8, 8, 21, 0)  # a Saturday
    assert target.weekday() == 5

    observations: list[tuple[datetime, float]] = []
    for day_offset in range(1, 8 * 7 + 1):  # 8 weeks of daily history
        day = target - timedelta(days=day_offset)
        level = weekday_level[day.weekday()]
        jitter = jitter_cycle[day_offset % len(jitter_cycle)]
        observations.append((day, level + jitter))

    # A legitimate weekend value, consistent with the historical Saturday pattern --
    # there is no incident here.
    actual = 2525.0 + jitter_cycle[0]

    trailing_comparison = select_comparison_window(observations, target, trailing_days=7)
    week_over_week_comparison = select_week_over_week_window(
        observations, target, lookback_weeks=4
    )

    trailing_baseline = compute_baseline(actual, trailing_comparison)
    week_over_week_baseline = compute_baseline(actual, week_over_week_comparison)

    assert trailing_baseline.status is BaselineStatus.OK
    assert week_over_week_baseline.status is BaselineStatus.OK

    # The bug: trailing-7-day amplifies the weekly pattern into a huge false-positive z.
    assert trailing_baseline.z == pytest.approx(-16.98, abs=0.1)
    assert (
        is_anomalous(trailing_baseline, direction=Direction.LOWER_IS_WORSE, threshold=3.0) is True
    )

    # The fix: week-over-week never mixes weekday values into a weekend baseline, so
    # the same legitimate value stays well under threshold.
    assert week_over_week_baseline.z == pytest.approx(-1.47, abs=0.1)
    assert (
        is_anomalous(week_over_week_baseline, direction=Direction.LOWER_IS_WORSE, threshold=3.0)
        is False
    )
