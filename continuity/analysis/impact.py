"""Blast radius -> affected subscribers -> churn risk -> ARR at risk.

The stage that turns an engineering metric into a business decision, so it is written
to be interrogated rather than trusted. No trained model: a transparent, documented
heuristic.

    churn_risk(subscriber) = base_monthly_churn
                            * tenure_multiplier(tenure_days)      # newer churns more
                            * severity_multiplier(sessions_affected, qoe_delta_ratio)
    arr_at_risk = sum over affected subscribers of (churn_risk * monthly_arpu * 12)

Every coefficient below is a named module-level constant with a comment stating what
it represents and that it is an ASSUMPTION, not a measurement -- there is no churn
event anywhere in this synthetic dataset to calibrate against, so pretending otherwise
would be false precision. `Methodology` carries every one of those assumptions plus the
window, the slice and the affected-subscriber count as DATA on `ImpactResult`, not as a
docstring, because sub-project 4 renders it next to the number it explains.

`ImpactResult` reports a low/expected/high BAND, not a point estimate: `churn_risk_band`
recomputes the whole formula at `base_monthly_churn` scaled by
+/-`BASE_CHURN_VARIATION`, so the band is a direct, auditable consequence of the one
stated uncertainty rather than an invented margin. `severity_multiplier` is bounded by
construction (a saturating curve, see below) and `churn_risk` additionally clamps at
`CHURN_RISK_CEILING` -- a monthly churn PROBABILITY, so it can never exceed 1.0 no
matter how extreme `sessions_affected` or `qoe_delta_ratio` get. An unbounded multiplier
that let churn "risk" reach 3.7 would be visibly wrong the moment a judge saw the number.

Money is `Decimal` throughout, never `float`. `subscribers.monthly_arpu` is queried as
`toString(monthly_arpu)` so the exact decimal digits ClickHouse holds are reconstructed
via `Decimal(str(...))` -- never round-tripped through a JSON float, which is the one
place a silent cent of drift could enter. `churn_risk` itself is computed entirely in
`Decimal` (Python's decimal module supports non-integer exponents), so no float ever
touches a quantity that feeds the ARR sum.

Affected subscribers are counted at the SQL layer with `uniqExact(session_id)` GROUPed
BY `subscriber_id` against `playback_events` -- never `qoe_rollup_5m`, which has neither
`subscriber_id` nor `title_id` (see continuity/data/schema.py) and could not answer this
regardless. `GROUP BY subscriber_id` is what makes a subscriber with many affected
sessions count exactly ONCE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from continuity.analysis.slices import RAW_EVENTS_TABLE, Slice
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_CENTS = Decimal("0.01")

# --- money conversion ------------------------------------------------------
# 12 calendar months per year. A unit conversion, not an assumption.
ARR_MONTHS_PER_YEAR = 12

# --- base churn assumption --------------------------------------------------
# The monthly probability an otherwise-unaffected subscriber cancels, with no QoE
# incident in play. Broadly in line with publicly reported SVOD monthly voluntary
# churn (low single-digit percent). ASSUMPTION, not measured -- this synthetic dataset
# contains no cancellation events to calibrate against.
BASE_MONTHLY_CHURN = Decimal("0.025")

# How far the base rate could plausibly be off, as a fraction of itself. Used only to
# build the low/expected/high band (never a statistical confidence interval) -- a
# stated, documented range of assumption uncertainty a reader can argue with rather
# than a false-precision point estimate. ASSUMPTION.
BASE_CHURN_VARIATION = Decimal("0.40")

# --- tenure multiplier -------------------------------------------------------
# tenure_multiplier(tenure_days) decays from TENURE_MULTIPLIER_AT_SIGNUP (tenure=0)
# towards TENURE_MULTIPLIER_FLOOR with an exponential half-life of
# TENURE_HALF_LIFE_DAYS, modelling the well-documented pattern that new subscribers
# churn faster than long-tenured ones. All three constants below are ASSUMPTIONS --
# there is no churn label in this dataset to fit them against.

# A brand-new subscriber (tenure_days=0) is assumed to churn at 2x the base rate.
TENURE_MULTIPLIER_AT_SIGNUP = Decimal("2.0")
# A long-tenured subscriber is assumed to churn at half the base rate -- loyal
# subscribers churn less than average, not zero.
TENURE_MULTIPLIER_FLOOR = Decimal("0.5")
# Days of tenure after which the multiplier has closed half the gap to its floor.
# ~6 months, chosen so the catalog's median tenure (~205 days, see
# continuity/data/catalog.py) sits inside the decay's active range rather than
# already flattened out at the floor.
TENURE_HALF_LIFE_DAYS = Decimal("180")

# --- severity multiplier -----------------------------------------------------
# severity_multiplier(sessions_affected, qoe_delta_ratio) rises from 1.0 (no extra
# severity: zero affected sessions, zero QoE degradation) towards
# SEVERITY_MULTIPLIER_MAX as either input grows, via a saturating (x / (x + k)) curve
# in each input -- never unbounded, so no combination of inputs can drive it past its
# ceiling. All three constants are ASSUMPTIONS.

# The ceiling severity can ever multiply the base rate by. Combined with
# TENURE_MULTIPLIER_AT_SIGNUP and BASE_MONTHLY_CHURN this keeps every realistic
# churn_risk far below CHURN_RISK_CEILING; the ceiling itself is the guarantee for the
# unrealistic case (see test_churn_risk_saturates_at_one_for_an_absurd_base_rate).
SEVERITY_MULTIPLIER_MAX = Decimal("3.0")
# Sessions-affected value at which the sessions factor reaches half its max
# contribution. A subscriber with a handful of affected sessions already reads as
# meaningfully more severe than one with a single affected session.
SEVERITY_SESSIONS_HALF_SATURATION = Decimal("5")
# qoe_delta_ratio value at which the QoE factor reaches half its max contribution.
# qoe_delta_ratio is (actual - baseline) / baseline for the metric driving the
# incident, so 2.0 means "the metric got 3x worse than baseline".
SEVERITY_QOE_HALF_SATURATION = Decimal("2.0")

# A monthly churn PROBABILITY cannot exceed 1.0 -- a mathematical identity, not an
# assumption, but named and applied explicitly so the ceiling is visible, testable,
# and independent of whatever the multiplier curves above compute.
CHURN_RISK_CEILING = Decimal("1.0")

_METHODOLOGY_NOTES = (
    "Heuristic, not a trained model. churn_risk = base_monthly_churn * "
    "tenure_multiplier(tenure_days) * severity_multiplier(sessions_affected, "
    "qoe_delta_ratio), capped at churn_risk_ceiling. arr_at_risk = sum over affected "
    "subscribers of churn_risk * monthly_arpu * 12. The low/expected/high band comes "
    "from scaling base_monthly_churn by +/-base_churn_variation -- every other "
    "coefficient is held fixed across the band. Every coefficient is a documented "
    "assumption, not a measurement from this dataset."
)


@dataclass(frozen=True)
class Methodology:
    """Every assumption `ImpactResult` rests on, as data -- rendered in the UI next to
    the number it explains, not buried in a docstring only a code reader would see."""

    base_monthly_churn: Decimal
    base_churn_variation: Decimal
    tenure_multiplier_at_signup: Decimal
    tenure_multiplier_floor: Decimal
    tenure_half_life_days: Decimal
    severity_multiplier_max: Decimal
    severity_sessions_half_saturation: Decimal
    severity_qoe_half_saturation: Decimal
    churn_risk_ceiling: Decimal
    qoe_delta_ratio: Decimal
    affected_subscriber_count: int
    window: tuple[datetime, datetime]
    slice: Slice
    notes: str


@dataclass(frozen=True)
class SubscriberImpact:
    """One affected subscriber's churn-risk band, before the ARR sum."""

    subscriber_id: int
    tenure_days: int
    sessions_affected: int
    monthly_arpu: Decimal
    churn_risk_low: Decimal
    churn_risk_expected: Decimal
    churn_risk_high: Decimal


@dataclass(frozen=True)
class ImpactResult:
    """Affected-subscriber count and ARR-at-risk band for one (slice, window), plus the
    methodology and the SQL that produced it."""

    slice: Slice
    window: tuple[datetime, datetime]
    affected_subscribers: int
    arr_at_risk_low: Decimal
    arr_at_risk_expected: Decimal
    arr_at_risk_high: Decimal
    methodology: Methodology
    sql: str


# ---------------------------------------------------------------------------
# Pure maths: no SQL, no I/O.
# ---------------------------------------------------------------------------


def tenure_multiplier(tenure_days: int) -> Decimal:
    """Monotonically NON-INCREASING in tenure_days: newer subscribers churn more.

    Exponential decay from TENURE_MULTIPLIER_AT_SIGNUP towards
    TENURE_MULTIPLIER_FLOOR with half-life TENURE_HALF_LIFE_DAYS. Always in
    (TENURE_MULTIPLIER_FLOOR, TENURE_MULTIPLIER_AT_SIGNUP] -- bounded by construction,
    never negative, never below the floor.
    """
    if tenure_days < 0:
        raise ValueError(f"tenure_days must be >= 0, got {tenure_days}")
    decay = Decimal("0.5") ** (Decimal(tenure_days) / TENURE_HALF_LIFE_DAYS)
    return TENURE_MULTIPLIER_FLOOR + (TENURE_MULTIPLIER_AT_SIGNUP - TENURE_MULTIPLIER_FLOOR) * decay


def severity_multiplier(sessions_affected: int, qoe_delta_ratio: Decimal | float) -> Decimal:
    """Monotonically NON-DECREASING in both inputs, saturating at SEVERITY_MULTIPLIER_MAX.

    Each input drives its own `x / (x + k)` factor in `[0, 1)`; the two factors are
    averaged and scaled onto `[1, SEVERITY_MULTIPLIER_MAX)`. Bounded by construction --
    no finite input, however extreme, can push the result to or past
    SEVERITY_MULTIPLIER_MAX (see test_severity_multiplier_saturates_for_extreme_inputs).
    """
    if sessions_affected < 0:
        raise ValueError(f"sessions_affected must be >= 0, got {sessions_affected}")
    qoe = qoe_delta_ratio if isinstance(qoe_delta_ratio, Decimal) else Decimal(str(qoe_delta_ratio))
    if qoe < 0:
        raise ValueError(f"qoe_delta_ratio must be >= 0, got {qoe_delta_ratio}")

    sessions = Decimal(sessions_affected)
    sessions_factor = sessions / (sessions + SEVERITY_SESSIONS_HALF_SATURATION)
    qoe_factor = qoe / (qoe + SEVERITY_QOE_HALF_SATURATION)
    combined = (sessions_factor + qoe_factor) / Decimal("2")
    return Decimal("1") + (SEVERITY_MULTIPLIER_MAX - Decimal("1")) * combined


def churn_risk(
    *,
    tenure_days: int,
    sessions_affected: int,
    qoe_delta_ratio: Decimal | float,
    base_monthly_churn: Decimal = BASE_MONTHLY_CHURN,
) -> Decimal:
    """The full heuristic for one subscriber, clamped at CHURN_RISK_CEILING.

    The clamp is unconditional -- it applies regardless of how base_monthly_churn,
    tenure_multiplier or severity_multiplier are parameterised, so a churn
    "probability" can never be reported above 1.0.
    """
    if base_monthly_churn < 0:
        raise ValueError(f"base_monthly_churn must be >= 0, got {base_monthly_churn}")
    raw = (
        base_monthly_churn
        * tenure_multiplier(tenure_days)
        * severity_multiplier(sessions_affected, qoe_delta_ratio)
    )
    return min(raw, CHURN_RISK_CEILING)


def churn_risk_band(
    *, tenure_days: int, sessions_affected: int, qoe_delta_ratio: Decimal | float
) -> tuple[Decimal, Decimal, Decimal]:
    """(low, expected, high) churn risk, varying only base_monthly_churn by
    +/-BASE_CHURN_VARIATION -- see the module docstring for why the band comes from
    that one stated uncertainty rather than an invented margin. Always
    low <= expected <= high: base_monthly_churn scales the raw product linearly and
    the CHURN_RISK_CEILING clamp (`min`) preserves that ordering."""
    low_rate = BASE_MONTHLY_CHURN * (Decimal("1") - BASE_CHURN_VARIATION)
    high_rate = BASE_MONTHLY_CHURN * (Decimal("1") + BASE_CHURN_VARIATION)
    low = churn_risk(
        tenure_days=tenure_days,
        sessions_affected=sessions_affected,
        qoe_delta_ratio=qoe_delta_ratio,
        base_monthly_churn=low_rate,
    )
    expected = churn_risk(
        tenure_days=tenure_days,
        sessions_affected=sessions_affected,
        qoe_delta_ratio=qoe_delta_ratio,
    )
    high = churn_risk(
        tenure_days=tenure_days,
        sessions_affected=sessions_affected,
        qoe_delta_ratio=qoe_delta_ratio,
        base_monthly_churn=high_rate,
    )
    return low, expected, high


def compute_subscriber_impact(
    *,
    subscriber_id: int,
    tenure_days: int,
    sessions_affected: int,
    monthly_arpu: Decimal,
    qoe_delta_ratio: Decimal | float,
) -> SubscriberImpact:
    low, expected, high = churn_risk_band(
        tenure_days=tenure_days,
        sessions_affected=sessions_affected,
        qoe_delta_ratio=qoe_delta_ratio,
    )
    return SubscriberImpact(
        subscriber_id=subscriber_id,
        tenure_days=tenure_days,
        sessions_affected=sessions_affected,
        monthly_arpu=monthly_arpu,
        churn_risk_low=low,
        churn_risk_expected=expected,
        churn_risk_high=high,
    )


def _validate_window(window: tuple[datetime, datetime]) -> None:
    start, end = window
    if not start < end:
        raise ValueError(f"window start must be before end, got {window!r}")


def _to_decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _row_to_subscriber_impact(
    row: Mapping[str, Any], *, qoe_delta_ratio: Decimal | float
) -> SubscriberImpact:
    return compute_subscriber_impact(
        subscriber_id=int(row["subscriber_id"]),
        tenure_days=int(row["tenure_days"]),
        sessions_affected=int(row["sessions_affected"]),
        monthly_arpu=_to_decimal(row["monthly_arpu"]),
        qoe_delta_ratio=qoe_delta_ratio,
    )


def _sum_arr(impacts: Sequence[SubscriberImpact], attr: str) -> Decimal:
    """Decimal-only accumulation -- never routes through float, so repeated summation
    over many subscribers cannot drift the way float addition would."""
    total = Decimal("0")
    for impact in impacts:
        risk: Decimal = getattr(impact, attr)
        total += risk * impact.monthly_arpu * ARR_MONTHS_PER_YEAR
    return total.quantize(_CENTS)


def summarize_impact(
    impacts: Sequence[SubscriberImpact],
    *,
    slice_: Slice,
    window: tuple[datetime, datetime],
    qoe_delta_ratio: Decimal | float,
    sql: str = "",
) -> ImpactResult:
    """Aggregate already-computed per-subscriber impacts into the final result.

    Never crashes on an empty `impacts` -- zero affected subscribers yields
    `Decimal("0.00")` ARR at every band, not a divide-by-zero or a missing
    methodology; the methodology is populated identically either way.
    """
    methodology = Methodology(
        base_monthly_churn=BASE_MONTHLY_CHURN,
        base_churn_variation=BASE_CHURN_VARIATION,
        tenure_multiplier_at_signup=TENURE_MULTIPLIER_AT_SIGNUP,
        tenure_multiplier_floor=TENURE_MULTIPLIER_FLOOR,
        tenure_half_life_days=TENURE_HALF_LIFE_DAYS,
        severity_multiplier_max=SEVERITY_MULTIPLIER_MAX,
        severity_sessions_half_saturation=SEVERITY_SESSIONS_HALF_SATURATION,
        severity_qoe_half_saturation=SEVERITY_QOE_HALF_SATURATION,
        churn_risk_ceiling=CHURN_RISK_CEILING,
        qoe_delta_ratio=_to_decimal(qoe_delta_ratio),
        affected_subscriber_count=len(impacts),
        window=window,
        slice=slice_,
        notes=_METHODOLOGY_NOTES,
    )
    return ImpactResult(
        slice=slice_,
        window=window,
        affected_subscribers=len(impacts),
        arr_at_risk_low=_sum_arr(impacts, "churn_risk_low"),
        arr_at_risk_expected=_sum_arr(impacts, "churn_risk_expected"),
        arr_at_risk_high=_sum_arr(impacts, "churn_risk_high"),
        methodology=methodology,
        sql=sql,
    )


def impact_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    slice_: Slice,
    window: tuple[datetime, datetime],
    qoe_delta_ratio: Decimal | float,
    sql: str = "",
) -> ImpactResult:
    """Pure end-to-end path from raw gateway-shaped rows to `ImpactResult` -- no I/O,
    what tests/analysis/test_impact.py exercises directly with fabricated rows.

    Each row is expected to carry exactly one distinct subscriber (`_build_impact_sql`
    guarantees this with `GROUP BY subscriber_id`), so a subscriber with many affected
    sessions is counted ONCE here by construction, not by any dedup step in this
    function.
    """
    _validate_window(window)
    impacts = [_row_to_subscriber_impact(row, qoe_delta_ratio=qoe_delta_ratio) for row in rows]
    return summarize_impact(
        impacts, slice_=slice_, window=window, qoe_delta_ratio=qoe_delta_ratio, sql=sql
    )


# ---------------------------------------------------------------------------
# SQL construction and gateway integration.
# ---------------------------------------------------------------------------


def _fmt(dt: datetime) -> str:
    return dt.strftime(_DATETIME_FORMAT)


def _build_impact_sql(slice_: Slice, window: tuple[datetime, datetime]) -> str:
    """Subscriber-level impact always queries playback_events, regardless of the
    slice's own dimensions -- qoe_rollup_5m has neither subscriber_id nor title_id
    (see continuity/data/schema.py) and cannot answer this at any grain.

    `GROUP BY subscriber_id` with `uniqExact(session_id)` collapses every affected
    subscriber to exactly one row, counted once no matter how many of their sessions
    fall inside the slice and window. `toString(monthly_arpu)` hands back the exact
    decimal digits so the Python side reconstructs an exact Decimal, never a float.
    """
    start, end = window
    where = slice_.where_sql()
    return (
        "SELECT subscribers.subscriber_id AS subscriber_id, "
        "subscribers.tenure_days AS tenure_days, "
        "toString(subscribers.monthly_arpu) AS monthly_arpu, "
        "affected.sessions_affected AS sessions_affected "
        "FROM ("
        "SELECT subscriber_id, uniqExact(session_id) AS sessions_affected "
        f"FROM {RAW_EVENTS_TABLE} "
        f"WHERE {where} AND event_time >= '{_fmt(start)}' AND event_time < '{_fmt(end)}' "
        "GROUP BY subscriber_id"
        ") AS affected "
        "INNER JOIN subscribers ON subscribers.subscriber_id = affected.subscriber_id"
    )


async def compute_impact(
    gateway: ClickHouseMCPGateway,
    *,
    slice_: Slice,
    window: tuple[datetime, datetime],
    qoe_delta_ratio: Decimal | float,
) -> ImpactResult:
    """Fetch affected-subscriber rows through the MCP gateway (one query) and turn them
    into an ARR-at-risk band. `qoe_delta_ratio` -- (actual - baseline) / baseline for
    the metric that flagged the incident -- is a caller-supplied severity input, not
    something this module measures itself: detection and baselining are detect.py's
    and baseline.py's job, not this one's.
    """
    _validate_window(window)
    sql = _build_impact_sql(slice_, window)
    result = await gateway.query(sql)
    return impact_from_rows(
        result.rows, slice_=slice_, window=window, qoe_delta_ratio=qoe_delta_ratio, sql=sql
    )
