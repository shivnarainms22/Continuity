from datetime import datetime, timedelta

import numpy as np
import pytest

from continuity.analysis.baseline import (
    Baseline,
    BaselineStatus,
    Direction,
    compute_baseline,
    is_anomalous,
    required_history_buckets,
    select_comparison_window,
    select_neighbourhood_residuals,
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
# select_neighbourhood_residuals: pooled spread sample, centred per time-of-day slot.
# ---------------------------------------------------------------------------


def test_select_neighbourhood_residuals_pools_far_more_points_than_the_level_window():
    """With radius=6 and 4 lookback weeks, up to 13 * 4 = 52 residuals -- not the 4
    raw points the level window alone would give."""
    target = datetime(2026, 8, 8, 21, 0)  # a Saturday
    observations: list[tuple[datetime, float]] = []
    for week in range(1, 5):
        day = target - timedelta(weeks=week)
        for offset in range(-6, 7):
            observations.append((day + timedelta(minutes=5 * offset), 100.0 + offset))
    residuals = select_neighbourhood_residuals(observations, target, lookback_weeks=4, radius=6)
    assert len(residuals) == 13 * 4


def test_select_neighbourhood_residuals_reflects_noise_not_the_diurnal_slope():
    """The trap this function exists to avoid: a steep diurnal slope across the
    +/-30 minute neighbourhood must not leak into the pooled spread -- only the (much
    smaller) noise around each slot's own median should. A naive MAD of the RAW
    (uncentred) neighbourhood values instead measures the slope -- assert that failure
    mode too, so it is documented as the reason per-slot centring is required."""
    target = datetime(2026, 8, 8, 21, 0)  # a Saturday
    slope_per_bucket = 40.0  # steep: the neighbourhood spans +/-240 around its centre
    noise_cycle = [-1.0, 1.0, -0.6, 0.6, -1.4, 1.4, -0.3, 0.3, -0.9, 0.9, -0.2, 0.2, -1.1, 1.1]

    observations: list[tuple[datetime, float]] = []
    for week in range(1, 5):
        day = target - timedelta(weeks=week)
        for offset in range(-6, 7):
            noise = noise_cycle[(offset + 6 + week) % len(noise_cycle)]
            value = 1000.0 + slope_per_bucket * offset + noise
            observations.append((day + timedelta(minutes=5 * offset), value))

    residuals = select_neighbourhood_residuals(observations, target, lookback_weeks=4, radius=6)
    assert len(residuals) == 13 * 4

    pooled_median = float(np.median(np.asarray(residuals)))
    pooled_mad = float(np.median(np.abs(np.asarray(residuals) - pooled_median)))
    pooled_spread = 1.4826 * pooled_mad

    # Reflects the noise (amplitude ~1.4), nowhere near the slope's own scale.
    assert pooled_spread < 5.0

    # The trap: MAD of the raw, uncentred neighbourhood values instead measures the
    # diurnal slope -- wildly inflated relative to the true noise level.
    raw_values = [value for _, value in observations]
    raw_median = float(np.median(np.asarray(raw_values)))
    naive_mad = float(np.median(np.abs(np.asarray(raw_values) - raw_median)))
    naive_spread = 1.4826 * naive_mad
    assert naive_spread > 100.0
    assert naive_spread > pooled_spread * 20


def test_select_neighbourhood_residuals_with_zero_radius_matches_pre_pooling_deviations():
    """radius=0 collapses to a single time-of-day slot -- the pre-pooling behaviour:
    each value's own week-over-week median subtracted, i.e. exactly the deviations a
    raw MAD over `select_week_over_week_window`'s own output would use."""
    target = datetime(2026, 8, 8, 21, 0)
    observations = _series_at_same_time_of_day([7, 14, 21, 28])
    level_values = select_week_over_week_window(observations, target, lookback_weeks=4)
    median = float(np.median(np.asarray(level_values)))
    expected = sorted(v - median for v in level_values)

    residuals = select_neighbourhood_residuals(observations, target, lookback_weeks=4, radius=0)
    assert sorted(residuals) == pytest.approx(expected)


def test_select_neighbourhood_residuals_skips_slots_with_no_history():
    """A slot with no matching observations (e.g. sparse data at some offsets)
    contributes nothing rather than raising or fabricating a value."""
    target = datetime(2026, 8, 8, 21, 0)
    observations = _series_at_same_time_of_day([7, 14, 21, 28])  # only offset 0 populated
    residuals = select_neighbourhood_residuals(observations, target, lookback_weeks=4, radius=6)
    assert len(residuals) == 4  # only the offset-0 slot contributed


# ---------------------------------------------------------------------------
# required_history_buckets: exactly what a SQL fetch needs, not a full range.
# ---------------------------------------------------------------------------


def test_required_history_buckets_is_far_smaller_than_the_full_contiguous_range():
    """The whole point of the fix: for a realistic test window, the history a SQL
    fetch needs to cover is a small fraction of the full contiguous range between
    the earliest comparison week and the test window -- not the ~30x-too-much range
    detect.py used to fetch."""
    targets = [
        datetime(2026, 2, 14, 20, 0) + timedelta(minutes=5 * i) for i in range(216)  # 18h window
    ]
    history = required_history_buckets(targets, lookback_weeks=4, radius=24)

    full_contiguous_range_bucket_count = (28 + 18 / 24) * 288  # 4 weeks + the window, in buckets
    assert len(history) < full_contiguous_range_bucket_count / 5


def test_required_history_buckets_covers_every_slot_the_selection_functions_read():
    """No under-coverage: for every target, the union over `required_history_buckets`
    must include every timestamp `select_week_over_week_window` (the LEVEL) and
    `select_neighbourhood_residuals` (the SPREAD) could read. Builds a dense synthetic
    series with a value at every 5-minute slot for 4 weeks back plus the window, then
    checks that restricting the series to `required_history_buckets` (plus the targets
    themselves) never changes what either selection function returns."""
    targets = [datetime(2026, 8, 8, 21, 0) + timedelta(minutes=5 * i) for i in range(12)]
    lookback_weeks, radius = 4, 24

    dense_observations: list[tuple[datetime, float]] = []
    start = targets[0] - timedelta(weeks=lookback_weeks, hours=3)
    end = targets[-1] + timedelta(hours=3)
    current = start
    value = 0.0
    while current <= end:
        dense_observations.append((current, value))
        current += timedelta(minutes=5)
        value += 1.0

    history = required_history_buckets(targets, lookback_weeks=lookback_weeks, radius=radius)
    restricted = [(ts, v) for ts, v in dense_observations if ts in history or ts in targets]

    for target in targets:
        full_level = select_week_over_week_window(
            dense_observations, target, lookback_weeks=lookback_weeks
        )
        restricted_level = select_week_over_week_window(
            restricted, target, lookback_weeks=lookback_weeks
        )
        assert sorted(restricted_level) == sorted(full_level)

        full_spread = select_neighbourhood_residuals(
            dense_observations, target, lookback_weeks=lookback_weeks, radius=radius
        )
        restricted_spread = select_neighbourhood_residuals(
            restricted, target, lookback_weeks=lookback_weeks, radius=radius
        )
        assert sorted(restricted_spread) == pytest.approx(sorted(full_spread))


def test_required_history_buckets_never_includes_a_target_own_recent_day():
    """Every enumerated timestamp is at least a week before its target -- the history
    set is purely historical, never overlapping the test window itself (which a SQL
    caller fetches separately, via the contiguous range clause)."""
    target = datetime(2026, 8, 8, 21, 0)
    history = required_history_buckets([target], lookback_weeks=4, radius=6)
    assert all(ts.date() <= target.date() - timedelta(weeks=1) for ts in history)


def test_required_history_buckets_rejects_negative_radius():
    with pytest.raises(ValueError, match="radius"):
        required_history_buckets([datetime(2026, 8, 8, 21, 0)], radius=-1)


def test_select_neighbourhood_residuals_rejects_negative_radius():
    target = datetime(2026, 8, 8, 21, 0)
    with pytest.raises(ValueError, match="radius"):
        select_neighbourhood_residuals([], target, radius=-1)


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


# ---------------------------------------------------------------------------
# compute_baseline with spread_values: the pooled-spread fix for tiny 4-point MAD.
# ---------------------------------------------------------------------------


def test_compute_baseline_uses_spread_values_for_mad_instead_of_comparison_values():
    """A tight 4-point comparison window would give a tiny MAD and a huge z on a
    trivial move -- exactly the measured false positive. A wider, pooled
    `spread_values` sample must be what the MAD actually comes from."""
    comparison = [9.99, 10.0, 10.0, 10.01]  # tight cluster -> tiny naive MAD
    pooled_spread_values = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]  # wide pooled residuals
    actual = 10.06

    naive = compute_baseline(actual, comparison)
    pooled = compute_baseline(actual, comparison, spread_values=pooled_spread_values)

    assert naive.status is BaselineStatus.OK
    assert pooled.status is BaselineStatus.OK
    assert is_anomalous(naive, direction=Direction.HIGHER_IS_WORSE, threshold=3.0) is True
    assert is_anomalous(pooled, direction=Direction.HIGHER_IS_WORSE, threshold=3.0) is False
    assert abs(pooled.z) < abs(naive.z)
    # The LEVEL (expected) is unaffected by which sample the spread comes from.
    assert pooled.expected == naive.expected == pytest.approx(10.0)


def test_compute_baseline_min_observations_gates_on_level_even_with_a_large_spread_sample():
    """Requirement: INSUFFICIENT_DATA still applies based on the LEVEL sample size,
    regardless of how many pooled spread_values happen to be available."""
    comparison = [10.0, 10.0]  # 2 < the default minimum of 4
    pooled_spread_values = list(range(50))  # plenty of spread samples
    result = compute_baseline(50.0, comparison, spread_values=pooled_spread_values)
    assert result.status is BaselineStatus.INSUFFICIENT_DATA
    assert result.sample_size == 2
    assert result.z is None


def test_compute_baseline_flat_pooled_spread_is_insufficient_not_infinite_or_zero():
    """A genuinely flat pooled spread (MAD == 0) with a differing actual must remain
    INSUFFICIENT_DATA -- never inf, never a silent 0.0 that reads as 'fine'."""
    comparison = [10.0, 10.0, 10.0, 10.0]
    result = compute_baseline(15.0, comparison, spread_values=[0.0, 0.0, 0.0])
    assert result.status is BaselineStatus.INSUFFICIENT_DATA
    assert result.z is None
    assert result.z != float("inf")
    assert result.z != 0.0


def test_compute_baseline_empty_spread_values_falls_back_to_zero_mad_not_a_crash():
    """If every neighbourhood slot happened to contribute nothing, spread_values is an
    empty list -- must degrade to the same flat-window handling as MAD == 0, never a
    crash or a fabricated spread."""
    comparison = [10.0, 10.0, 10.0, 10.0]
    result = compute_baseline(10.0, comparison, spread_values=[])
    assert result.status is BaselineStatus.OK
    assert result.spread == 0.0
    assert result.z == 0.0


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
