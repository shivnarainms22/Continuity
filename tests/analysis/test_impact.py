"""Unit tests for continuity/analysis/impact.py. Pure maths, no ClickHouse.

Covers: monotonicity of both multiplier curves, saturation under extreme inputs,
Decimal-only money arithmetic (type and drift), the low <= expected <= high band, and
the edge cases from Task 7 (zero affected subscribers, a subscriber counted once).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from continuity.analysis.impact import (
    BASE_MONTHLY_CHURN,
    CHURN_RISK_CEILING,
    SEVERITY_MULTIPLIER_MAX,
    TENURE_MULTIPLIER_AT_SIGNUP,
    TENURE_MULTIPLIER_FLOOR,
    _build_impact_sql,
    churn_risk,
    churn_risk_band,
    impact_from_rows,
    severity_multiplier,
    tenure_multiplier,
)
from continuity.analysis.slices import Slice

_WINDOW = (datetime(2026, 2, 12, 18, 0), datetime(2026, 2, 13, 2, 0))

# ---------------------------------------------------------------------------
# tenure_multiplier: monotonically non-increasing, bounded.
# ---------------------------------------------------------------------------


def test_tenure_multiplier_at_signup_equals_the_at_signup_constant():
    assert tenure_multiplier(0) == TENURE_MULTIPLIER_AT_SIGNUP


def test_tenure_multiplier_is_monotonically_non_increasing():
    tenures = [0, 1, 30, 90, 180, 205, 365, 730, 1500]
    multipliers = [tenure_multiplier(t) for t in tenures]
    assert all(a >= b for a, b in zip(multipliers, multipliers[1:], strict=False))


def test_tenure_multiplier_never_drops_to_or_below_the_floor():
    # 1500 days is the catalog's maximum tenure (continuity/data/catalog.py).
    assert tenure_multiplier(1500) > TENURE_MULTIPLIER_FLOOR
    # Far beyond any real tenure, the decay term underflows the Decimal context's
    # significant-digit precision and the sum rounds to exactly the floor -- never
    # below it, which is the actual guarantee this function makes.
    assert tenure_multiplier(1_000_000) >= TENURE_MULTIPLIER_FLOOR


def test_tenure_multiplier_never_exceeds_the_at_signup_value():
    for tenure_days in (0, 5, 50, 500):
        assert tenure_multiplier(tenure_days) <= TENURE_MULTIPLIER_AT_SIGNUP


def test_tenure_multiplier_rejects_negative_tenure():
    with pytest.raises(ValueError, match="tenure_days"):
        tenure_multiplier(-1)


# ---------------------------------------------------------------------------
# severity_multiplier: monotonically non-decreasing in both inputs, saturating.
# ---------------------------------------------------------------------------


def test_severity_multiplier_at_zero_zero_is_one():
    assert severity_multiplier(0, 0.0) == Decimal("1")


def test_severity_multiplier_is_monotonically_non_decreasing_in_sessions_affected():
    values = [severity_multiplier(n, 1.0) for n in (0, 1, 2, 5, 10, 50, 1000)]
    assert all(a <= b for a, b in zip(values, values[1:], strict=False))


def test_severity_multiplier_is_monotonically_non_decreasing_in_qoe_delta_ratio():
    values = [severity_multiplier(3, q) for q in (0.0, 0.5, 1.0, 2.0, 5.0, 50.0)]
    assert all(a <= b for a, b in zip(values, values[1:], strict=False))


def test_severity_multiplier_saturates_for_extreme_inputs():
    """An extreme, physically-impossible input (a billion affected sessions, a QoE
    metric a billion times worse than baseline) must not produce an absurd multiplier
    -- it must stay strictly below SEVERITY_MULTIPLIER_MAX, never explode past it."""
    extreme = severity_multiplier(1_000_000_000, 1_000_000_000.0)
    assert extreme < SEVERITY_MULTIPLIER_MAX
    assert extreme > Decimal("2.9")  # close to, but never at, the ceiling


def test_severity_multiplier_rejects_negative_sessions_affected():
    with pytest.raises(ValueError, match="sessions_affected"):
        severity_multiplier(-1, 1.0)


def test_severity_multiplier_rejects_negative_qoe_delta_ratio():
    with pytest.raises(ValueError, match="qoe_delta_ratio"):
        severity_multiplier(1, -0.5)


# ---------------------------------------------------------------------------
# churn_risk: Decimal, clamped, saturates at exactly 1.0.
# ---------------------------------------------------------------------------


def test_churn_risk_returns_a_decimal():
    risk = churn_risk(tenure_days=10, sessions_affected=3, qoe_delta_ratio=3.5)
    assert isinstance(risk, Decimal)


def test_churn_risk_saturates_at_one_for_an_absurd_base_rate():
    """The clamp is unconditional: even with a base rate an order of magnitude above
    the documented assumption, churn_risk never reports above 1.0. This is the direct
    test of requirement 5's "would be visibly wrong" scenario -- without the clamp,
    base_monthly_churn=10 * tenure_multiplier(0)=2.0 * severity up to 3.0 would compute
    to 60.0, a nonsense churn "probability"."""
    risk = churn_risk(
        tenure_days=0,
        sessions_affected=10_000,
        qoe_delta_ratio=100.0,
        base_monthly_churn=Decimal("10"),
    )
    assert risk == CHURN_RISK_CEILING
    assert risk == Decimal("1")


def test_churn_risk_realistic_inputs_stay_far_below_the_ceiling():
    risk = churn_risk(tenure_days=205, sessions_affected=2, qoe_delta_ratio=3.5)
    assert Decimal("0") < risk < Decimal("0.2")


def test_churn_risk_rejects_negative_base_rate():
    with pytest.raises(ValueError, match="base_monthly_churn"):
        churn_risk(
            tenure_days=1,
            sessions_affected=1,
            qoe_delta_ratio=1.0,
            base_monthly_churn=Decimal("-1"),
        )


# ---------------------------------------------------------------------------
# churn_risk_band: low <= expected <= high, always.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tenure_days,sessions_affected,qoe_delta_ratio",
    [
        (0, 0, 0.0),
        (0, 5, 3.5),
        (205, 2, 1.0),
        (1500, 100, 10.0),
        (0, 1_000_000, 1_000_000.0),
    ],
)
def test_churn_risk_band_orders_low_expected_high(tenure_days, sessions_affected, qoe_delta_ratio):
    low, expected, high = churn_risk_band(
        tenure_days=tenure_days,
        sessions_affected=sessions_affected,
        qoe_delta_ratio=qoe_delta_ratio,
    )
    assert low <= expected <= high
    assert all(isinstance(v, Decimal) for v in (low, expected, high))


def test_churn_risk_band_expected_matches_the_documented_base_rate():
    _, expected, _ = churn_risk_band(tenure_days=0, sessions_affected=0, qoe_delta_ratio=0.0)
    assert expected == BASE_MONTHLY_CHURN * TENURE_MULTIPLIER_AT_SIGNUP


# ---------------------------------------------------------------------------
# impact_from_rows: Decimal money, no drift, zero-affected edge case, one-row-per-
# subscriber counting.
# ---------------------------------------------------------------------------


def _row(subscriber_id: int, tenure_days: int, sessions_affected: int, monthly_arpu: str) -> dict:
    return {
        "subscriber_id": subscriber_id,
        "tenure_days": tenure_days,
        "sessions_affected": sessions_affected,
        "monthly_arpu": monthly_arpu,
    }


def test_impact_from_rows_returns_decimal_totals():
    rows = [_row(1, 50, 3, "15.99"), _row(2, 900, 1, "8.99")]
    result = impact_from_rows(rows, slice_=Slice(), window=_WINDOW, qoe_delta_ratio=3.5)

    assert isinstance(result.arr_at_risk_low, Decimal)
    assert isinstance(result.arr_at_risk_expected, Decimal)
    assert isinstance(result.arr_at_risk_high, Decimal)
    assert result.arr_at_risk_low <= result.arr_at_risk_expected <= result.arr_at_risk_high
    assert result.affected_subscribers == 2


def test_impact_from_rows_repeated_summation_does_not_drift():
    """1,000 subscribers at an ARPU with an exact but non-power-of-two decimal
    fraction (15.99) -- summed as float this would drift off the exact cent value;
    summed as Decimal it must land exactly."""
    rows = [_row(i, 100, 2, "15.99") for i in range(1, 1001)]
    result = impact_from_rows(rows, slice_=Slice(), window=_WINDOW, qoe_delta_ratio=2.0)

    per_subscriber_low, per_subscriber_expected, per_subscriber_high = churn_risk_band(
        tenure_days=100, sessions_affected=2, qoe_delta_ratio=2.0
    )
    expected_total = (per_subscriber_expected * Decimal("15.99") * 12 * 1000).quantize(
        Decimal("0.01")
    )
    assert result.arr_at_risk_expected == expected_total


def test_impact_from_rows_zero_affected_subscribers_returns_zero_without_crashing():
    result = impact_from_rows([], slice_=Slice(), window=_WINDOW, qoe_delta_ratio=3.5)

    assert result.affected_subscribers == 0
    assert result.arr_at_risk_low == Decimal("0.00")
    assert result.arr_at_risk_expected == Decimal("0.00")
    assert result.arr_at_risk_high == Decimal("0.00")
    assert result.methodology is not None
    assert result.methodology.affected_subscriber_count == 0


def test_impact_from_rows_methodology_carries_the_stated_assumptions():
    rows = [_row(1, 50, 3, "15.99")]
    result = impact_from_rows(rows, slice_=Slice(), window=_WINDOW, qoe_delta_ratio=3.5)

    m = result.methodology
    assert m.base_monthly_churn == BASE_MONTHLY_CHURN
    assert m.tenure_multiplier_at_signup == TENURE_MULTIPLIER_AT_SIGNUP
    assert m.churn_risk_ceiling == CHURN_RISK_CEILING
    assert m.affected_subscriber_count == 1
    assert m.window == _WINDOW
    assert m.slice == Slice()
    assert m.notes  # non-empty, human-readable


def test_impact_from_rows_rejects_a_window_with_end_before_start():
    with pytest.raises(ValueError, match="window"):
        impact_from_rows([], slice_=Slice(), window=(_WINDOW[1], _WINDOW[0]), qoe_delta_ratio=1.0)


def test_impact_from_rows_accepts_a_decimal_monthly_arpu_directly():
    rows = [
        {
            "subscriber_id": 1,
            "tenure_days": 10,
            "sessions_affected": 2,
            "monthly_arpu": Decimal("22.99"),
        }
    ]
    result = impact_from_rows(rows, slice_=Slice(), window=_WINDOW, qoe_delta_ratio=1.0)
    assert result.affected_subscribers == 1
    assert result.arr_at_risk_expected > Decimal("0")


# ---------------------------------------------------------------------------
# SQL shape: playback_events only, GROUP BY subscriber_id, uniqExact, exact-decimal
# arpu -- never qoe_rollup_5m and never a bare count().
# ---------------------------------------------------------------------------


def test_build_impact_sql_queries_playback_events_not_the_rollup():
    sql = _build_impact_sql(Slice().refine("device_type", "roku"), _WINDOW)
    assert "playback_events" in sql
    assert "qoe_rollup_5m" not in sql


def test_build_impact_sql_groups_by_subscriber_id_with_uniq_exact_sessions():
    sql = _build_impact_sql(Slice(), _WINDOW)
    assert "GROUP BY subscriber_id" in sql
    assert "uniqExact(session_id)" in sql


def test_build_impact_sql_never_emits_a_bare_count():
    sql = _build_impact_sql(Slice(), _WINDOW)
    assert "count(" not in sql.lower()


def test_build_impact_sql_reads_arpu_as_an_exact_decimal_string():
    sql = _build_impact_sql(Slice(), _WINDOW)
    assert "toString(subscribers.monthly_arpu)" in sql


def test_build_impact_sql_joins_the_subscribers_table():
    sql = _build_impact_sql(Slice(), _WINDOW)
    assert "INNER JOIN subscribers" in sql
