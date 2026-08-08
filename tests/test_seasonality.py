from datetime import datetime, timedelta

import pytest

from continuity.data.seasonality import (
    DIURNAL_WEIGHTS,
    degrade,
    diurnal_weight,
    expected_sessions,
    load_factor,
    weekday_factor,
)


def test_diurnal_weights_are_all_strictly_positive():
    assert len(DIURNAL_WEIGHTS) == 24
    assert all(w > 0 for w in DIURNAL_WEIGHTS)


def test_diurnal_weights_sum_to_one():
    assert sum(DIURNAL_WEIGHTS) == pytest.approx(1.0)


def test_diurnal_weights_peak_in_prime_time_band():
    peak_hour = DIURNAL_WEIGHTS.index(max(DIURNAL_WEIGHTS))
    assert 20 <= peak_hour <= 22


def test_diurnal_weight_matches_the_table_at_valid_edges():
    assert diurnal_weight(0) == DIURNAL_WEIGHTS[0]
    assert diurnal_weight(23) == DIURNAL_WEIGHTS[23]


def test_diurnal_weight_raises_below_zero():
    with pytest.raises(ValueError, match="-1"):
        diurnal_weight(-1)


def test_diurnal_weight_raises_above_23():
    with pytest.raises(ValueError, match="24"):
        diurnal_weight(24)


def test_weekday_factor_weekend_strictly_exceeds_midweek():
    midweek = max(weekday_factor(d) for d in (1, 2, 3))  # Tue, Wed, Thu
    weekend = min(weekday_factor(d) for d in (5, 6))  # Sat, Sun
    assert weekend > midweek


def test_weekday_factor_accepts_full_valid_range():
    assert all(weekday_factor(d) > 0 for d in range(7))


def test_weekday_factor_raises_below_zero():
    with pytest.raises(ValueError, match="-1"):
        weekday_factor(-1)


def test_weekday_factor_raises_above_six():
    with pytest.raises(ValueError, match="7"):
        weekday_factor(7)


def test_expected_sessions_integrates_to_sessions_per_day_over_one_day():
    """Sums over all 288 five-minute buckets of a day; wrong scaling would be off by 288x."""
    day_start = datetime(2026, 8, 4, 0, 0)  # Tuesday
    sessions_per_day = 250_000
    buckets = [day_start + timedelta(minutes=5 * i) for i in range(288)]
    total = sum(expected_sessions(b, sessions_per_day) for b in buckets)
    assert total == pytest.approx(sessions_per_day, rel=1e-9)


def test_expected_sessions_is_zero_when_sessions_per_day_is_zero():
    day_start = datetime(2026, 8, 4, 0, 0)
    buckets = [day_start + timedelta(minutes=5 * i) for i in range(288)]
    assert sum(expected_sessions(b, 0) for b in buckets) == 0.0


def test_expected_sessions_rejects_negative_sessions_per_day():
    with pytest.raises(ValueError, match="-1"):
        expected_sessions(datetime(2026, 8, 4, 12, 0), -1)


def test_load_factor_stays_within_unit_interval_across_a_full_week():
    week_start = datetime(2026, 8, 3, 0, 0)  # Monday
    buckets = [week_start + timedelta(minutes=5 * i) for i in range(7 * 288)]
    values = [load_factor(b) for b in buckets]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_degrade_equals_base_at_zero_load():
    assert degrade(1.0, 0.0) == pytest.approx(1.0)


def test_degrade_is_monotonically_non_decreasing_in_load():
    base = 2.0
    loads = [i / 20 for i in range(21)]  # 0.0 .. 1.0
    values = [degrade(base, load) for load in loads]
    assert all(later >= earlier for earlier, later in zip(values, values[1:], strict=False))


def test_degrade_never_returns_less_than_base():
    base = 5.0
    for load in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert degrade(base, load) >= base


def test_degrade_rejects_negative_base():
    with pytest.raises(ValueError, match="base"):
        degrade(-1.0, 0.5)


def test_degrade_rejects_load_below_zero():
    with pytest.raises(ValueError, match="load"):
        degrade(1.0, -0.1)


def test_degrade_rejects_load_above_one():
    with pytest.raises(ValueError, match="load"):
        degrade(1.0, 1.1)


def test_peak_hour_qoe_is_materially_worse_than_trough_hour_qoe():
    """Anti-flattening guard: this is what makes a naive detector fire every night.

    If this ratio collapses toward 1.0, the seasonality-aware baseline in the next
    sub-project stops being real work and the project's most credible demo claim dies.
    """
    peak_bucket = datetime(2026, 8, 8, 21, 0)  # Saturday, prime time
    trough_bucket = datetime(2026, 8, 4, 9, 0)  # Tuesday, mid-morning
    base_rebuffer = 1.0
    peak_qoe = degrade(base_rebuffer, load_factor(peak_bucket))
    trough_qoe = degrade(base_rebuffer, load_factor(trough_bucket))
    assert peak_qoe / trough_qoe >= 1.8
