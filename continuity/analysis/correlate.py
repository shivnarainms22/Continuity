"""Change correlation: rank ``change_log`` entries as plausible causes of an anomaly.

Given a blast-radius ``Slice`` and the anomaly window it was detected over, this module
finds every change_log row in a lookback window before the anomaly's onset and ranks
them by how likely each is to be the cause. Two independent signals combine into the
score, and both matter:

1. TEMPORAL PROXIMITY. A change shortly before onset is a plausible cause; a change
   after onset cannot be one, no matter how good its dimensional match -- causality only
   runs forward in time. This is enforced as a hard filter (see ``classify_change``),
   not just a scoring penalty: a change strictly after ``onset + tolerance`` is REJECTED,
   never merely down-ranked. ``score`` decays linearly from 1.0 (at onset) to 0.0 (at
   ``onset - lookback``) for everything that survives the filter.

2. DIMENSIONAL OVERLAP. change_log rows carry a single (dimension_key, dimension_value)
   pair -- the dimension the change actually touched. Three cases:
   - It matches a predicate the blast radius itself pins (e.g. blast radius has
     app_version=8.2.0 and the change is app_version=8.2.0): strong evidence, full
     weight (``OVERLAP_WEIGHT``).
   - It names a dimension the blast radius pins to a DIFFERENT value (e.g. blast radius
     is device_type=roku and the change is device_type=firetv): the change could not
     have caused THIS blast radius -- REJECTED, not merely down-ranked.
   - It names a dimension the blast radius says nothing about (e.g. blast radius is
     device_type/app_version and the change is isp=comcast): unrelated, not
     contradictory -- weak evidence, reduced weight (``NO_OVERLAP_WEIGHT``), still ranked.

``score = temporal_score * overlap_weight``, multiplicative so neither signal can carry
a candidate alone: a change right before onset with no dimensional relation scores low,
and a dimensionally perfect change from days ago also scores low.

DISCONFIRMING EVIDENCE. Every ranked candidate also carries a check of whether the same
change touched OTHER values of some dimension the blast radius also predicates on, and
whether those other values ALSO degraded. A deploy that shipped to every device type but
only Roku degraded is weaker evidence than one that shipped only to Roku -- this is
recorded (``DisconfirmingEvidence``), not folded into the score. Scoring is a
measurement; deciding how much a broad-exposure caveat should count against a
hypothesis is judgement, which this deterministic module deliberately leaves to
whatever investigates with it (see the analysis-core plan's code/judgement split).
"Degraded" here is a small, self-contained, documented heuristic (a
``DEFAULT_DEGRADED_MULTIPLIER``-fold worsening vs a same-time-of-day baseline one week
earlier) -- it does not reuse baseline.py's median/MAD machinery, because correlate.py
has no dependency on the seasonality stack (see the analysis-core plan's task table:
correlate.py depends only on slices.py + metrics.py) and a second, heavier statistical
engine for a secondary transparency feature would be the wrong trade here.

REJECTED CANDIDATES are a first-class output, not debug logging: every change_log row
considered and ruled out is returned with a human-readable reason, so "we looked at
these and ruled them out" is something a caller (or a generated brief) can show.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from continuity.analysis.metrics import Metric, get_metric
from continuity.analysis.slices import (
    ALLOWED_DIMENSIONS,
    RAW_EVENTS_TABLE,
    ROLLUP_TABLE,
    TITLE_ID_DIMENSION,
    InvalidSliceError,
    Slice,
)
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_LOOKBACK = timedelta(hours=6)
DEFAULT_TOLERANCE = timedelta(0)
DEFAULT_METRIC = "rebuffer"

# Weight for a full dimensional match vs. a change on a dimension the blast radius says
# nothing about. Never zero -- an unrelated-looking dimension is weak evidence, not no
# evidence -- but never able to outrank a genuine match at comparable recency.
OVERLAP_WEIGHT = 1.0
NO_OVERLAP_WEIGHT = 0.3

# Disconfirming-evidence heuristic: a sibling value counts as "also degraded" when its
# in-window metric moved by at least this multiplicative factor from its own baseline,
# in the metric's bad direction. A documented assumption -- "clearly worse", not "any
# wobble" -- not a fitted threshold.
DEFAULT_DEGRADED_MULTIPLIER = 1.5
# Same-time-of-day one week earlier, matching split.py's own self-contained (non
# baseline.py) comparison-window convention.
_SIBLING_BASELINE_OFFSET = timedelta(days=7)

# The window candidate changes are fetched over extends this far PAST onset+tolerance so
# that changes shortly after onset are genuinely fetched and then explicitly REJECTED by
# classify_change, rather than silently absent because the SQL WHERE clause already
# excluded them. Fetching only up to onset+tolerance would make the "changes after onset
# are excluded" guarantee untested against real data -- this is deliberately wider so the
# rejection path is exercised, not just assumed. Capped by lookback so the fetch window
# never grows unboundedly for a large tolerance.
_POST_ONSET_FETCH_MARGIN_FACTOR = 1.0


class InvalidCorrelationWindowError(ValueError):
    """Raised when the anomaly window or lookback/tolerance parameters are invalid."""


@dataclass(frozen=True)
class ChangeRow:
    """One change_log row, already typed."""

    change_id: int
    changed_at: datetime
    change_type: str
    component: str
    description: str
    dimension_key: str
    dimension_value: str


@dataclass(frozen=True)
class SiblingMeasurement:
    """One sibling-dimension value's in-window metric vs. its own baseline."""

    value: str
    metric_value: float | None
    baseline_value: float | None


@dataclass(frozen=True)
class SiblingCheck:
    """A ``SiblingMeasurement`` with the degraded verdict attached.

    ``degraded`` is ``None`` (never a fabricated ``False``) when either value is
    missing -- e.g. a device_type that carried no traffic in the baseline period.
    """

    value: str
    metric_value: float | None
    baseline_value: float | None
    degraded: bool | None


@dataclass(frozen=True)
class DisconfirmingEvidence:
    """Whether the same change also touched slices that did NOT degrade.

    ``sibling_dimension`` is ``None`` when the blast radius has no dimension besides
    the change's own to check against -- there is nothing to disconfirm with, and that
    is recorded rather than silently omitted.
    """

    dimension_key: str
    dimension_value: str
    sibling_dimension: str | None
    siblings: tuple[SiblingCheck, ...]
    note: str

    @property
    def siblings_checked(self) -> int:
        return len(self.siblings)

    @property
    def siblings_degraded(self) -> int:
        return sum(1 for s in self.siblings if s.degraded is True)

    @property
    def siblings_not_degraded(self) -> int:
        return sum(1 for s in self.siblings if s.degraded is False)


@dataclass(frozen=True)
class RankedChange:
    """One accepted candidate cause, with everything needed to audit the ranking."""

    change_id: int
    changed_at: datetime
    change_type: str
    component: str
    description: str
    dimension_key: str
    dimension_value: str
    score: float
    temporal_delta: timedelta
    dimensional_overlap: bool
    disconfirming_evidence: DisconfirmingEvidence
    sql: str


@dataclass(frozen=True)
class RejectedChange:
    """A change_log row that was considered and ruled out, with a human-readable why."""

    change_id: int
    changed_at: datetime
    change_type: str
    component: str
    description: str
    dimension_key: str
    dimension_value: str
    reason: str
    sql: str


@dataclass(frozen=True)
class CorrelationResult:
    """Every change considered for one blast radius / anomaly window, ranked and
    rejected, plus the query that found them."""

    blast_radius: Slice
    onset: datetime
    end: datetime
    lookback: timedelta
    tolerance: timedelta
    candidates: tuple[RankedChange, ...]
    rejected: tuple[RejectedChange, ...]
    sql: str


# ---------------------------------------------------------------------------
# Pure scoring and classification -- no I/O, fully unit-testable.
# ---------------------------------------------------------------------------


def _temporal_score(delta: timedelta, lookback: timedelta) -> float:
    """1.0 at onset, decaying linearly to 0.0 at ``onset - lookback``.

    ``delta`` is ``onset - changed_at``: positive when the change precedes onset.
    A non-positive delta (a change at or after onset, only reachable within
    ``tolerance``) is treated as maximally proximate -- tolerance exists to absorb
    detection-granularity noise around the boundary, not to be penalised.
    """
    seconds = max(0.0, delta.total_seconds())
    lookback_seconds = lookback.total_seconds()
    if lookback_seconds <= 0:
        return 1.0 if seconds == 0 else 0.0
    return max(0.0, 1.0 - seconds / lookback_seconds)


def _blast_radius_value(blast_radius: Slice, dimension_key: str) -> str | None:
    for key, value in blast_radius.predicates:
        if key == dimension_key:
            return value
    return None


def _dimensional_overlap(row: ChangeRow, blast_radius: Slice) -> bool | None:
    """True: matches a blast-radius predicate exactly. False: contradicts one (same
    dimension, different value) -- the change targeted a population disjoint from the
    blast radius. None: the blast radius says nothing about this dimension at all."""
    blast_value = _blast_radius_value(blast_radius, row.dimension_key)
    if blast_value is None:
        return None
    return blast_value == row.dimension_value


def classify_change(
    row: ChangeRow,
    *,
    blast_radius: Slice,
    onset: datetime,
    lookback: timedelta = DEFAULT_LOOKBACK,
    tolerance: timedelta = DEFAULT_TOLERANCE,
    sql: str = "",
) -> RankedChange | RejectedChange:
    """Classify one change_log row as an accepted, scored candidate or a rejection.

    Temporal filter first (see the module docstring: this is the single most common way
    causal-attribution code goes wrong), then dimensional filter. A candidate's
    ``disconfirming_evidence`` starts as a placeholder recording "not yet checked" --
    ``correlate_changes`` fills it in once sibling data has been queried, since that
    step needs the gateway and this function must not.
    """
    delta = onset - row.changed_at

    if row.changed_at > onset + tolerance:
        return RejectedChange(
            change_id=row.change_id,
            changed_at=row.changed_at,
            change_type=row.change_type,
            component=row.component,
            description=row.description,
            dimension_key=row.dimension_key,
            dimension_value=row.dimension_value,
            reason=(
                f"too late: changed at {row.changed_at.isoformat()}, after the anomaly "
                f"onset {onset.isoformat()} (+ tolerance {tolerance}); a change after "
                "onset cannot have caused it"
            ),
            sql=sql,
        )
    if row.changed_at < onset - lookback:
        return RejectedChange(
            change_id=row.change_id,
            changed_at=row.changed_at,
            change_type=row.change_type,
            component=row.component,
            description=row.description,
            dimension_key=row.dimension_key,
            dimension_value=row.dimension_value,
            reason=(
                f"outside window: changed at {row.changed_at.isoformat()}, before the "
                f"{lookback} lookback horizon ({(onset - lookback).isoformat()})"
            ),
            sql=sql,
        )

    overlap = _dimensional_overlap(row, blast_radius)
    if overlap is False:
        blast_value = _blast_radius_value(blast_radius, row.dimension_key)
        return RejectedChange(
            change_id=row.change_id,
            changed_at=row.changed_at,
            change_type=row.change_type,
            component=row.component,
            description=row.description,
            dimension_key=row.dimension_key,
            dimension_value=row.dimension_value,
            reason=(
                f"no dimensional overlap: change touched {row.dimension_key}="
                f"{row.dimension_value!r}, but the blast radius pins {row.dimension_key}="
                f"{blast_value!r} -- disjoint populations"
            ),
            sql=sql,
        )

    weight = OVERLAP_WEIGHT if overlap else NO_OVERLAP_WEIGHT
    score = _temporal_score(delta, lookback) * weight
    placeholder_evidence = DisconfirmingEvidence(
        dimension_key=row.dimension_key,
        dimension_value=row.dimension_value,
        sibling_dimension=None,
        siblings=(),
        note="not yet checked",
    )
    return RankedChange(
        change_id=row.change_id,
        changed_at=row.changed_at,
        change_type=row.change_type,
        component=row.component,
        description=row.description,
        dimension_key=row.dimension_key,
        dimension_value=row.dimension_value,
        score=score,
        temporal_delta=delta,
        dimensional_overlap=bool(overlap),
        disconfirming_evidence=placeholder_evidence,
        sql=sql,
    )


def rank_candidates(
    rows: Sequence[ChangeRow],
    *,
    blast_radius: Slice,
    onset: datetime,
    lookback: timedelta = DEFAULT_LOOKBACK,
    tolerance: timedelta = DEFAULT_TOLERANCE,
    sql: str = "",
) -> tuple[tuple[RankedChange, ...], tuple[RejectedChange, ...]]:
    """Classify and rank every row. No changes in ``rows`` is not an error -- both
    outputs are simply empty.

    Ordering is fully deterministic even when several changes share an instant: by
    score descending, then most-recent-first, then ``change_id`` ascending as a final,
    unconditional tiebreak.
    """
    classified = [
        classify_change(
            row,
            blast_radius=blast_radius,
            onset=onset,
            lookback=lookback,
            tolerance=tolerance,
            sql=sql,
        )
        for row in rows
    ]
    candidates = [c for c in classified if isinstance(c, RankedChange)]
    rejected = [c for c in classified if isinstance(c, RejectedChange)]
    candidates.sort(key=lambda c: (-c.score, -c.changed_at.timestamp(), c.change_id))
    return tuple(candidates), tuple(rejected)


def _is_degraded(
    actual: float | None,
    baseline: float | None,
    *,
    higher_is_worse: bool,
    multiplier: float,
) -> bool | None:
    if actual is None or baseline is None:
        return None
    if baseline == 0:
        # No usable baseline level to multiply -- only a move away from exactly zero,
        # in the bad direction, counts; this never fabricates a degraded=True for a
        # value that is itself at or below zero.
        return actual > 0 if higher_is_worse else False
    if higher_is_worse:
        return actual >= baseline * multiplier
    return actual <= baseline / multiplier


def compute_disconfirming_evidence(
    dimension_key: str,
    dimension_value: str,
    *,
    sibling_dimension: str | None,
    sibling_measurements: Sequence[SiblingMeasurement],
    higher_is_worse: bool,
    degraded_multiplier: float = DEFAULT_DEGRADED_MULTIPLIER,
) -> DisconfirmingEvidence:
    """Pure: turn sibling measurements into a disconfirming-evidence verdict.

    Never raises on missing data -- a sibling with no baseline or no in-window traffic
    gets ``degraded=None``, not a fabricated verdict.
    """
    if sibling_dimension is None:
        return DisconfirmingEvidence(
            dimension_key=dimension_key,
            dimension_value=dimension_value,
            sibling_dimension=None,
            siblings=(),
            note="blast radius has no dimension besides the change's own to check against",
        )

    checks = tuple(
        sorted(
            (
                SiblingCheck(
                    value=m.value,
                    metric_value=m.metric_value,
                    baseline_value=m.baseline_value,
                    degraded=_is_degraded(
                        m.metric_value,
                        m.baseline_value,
                        higher_is_worse=higher_is_worse,
                        multiplier=degraded_multiplier,
                    ),
                )
                for m in sibling_measurements
            ),
            key=lambda c: c.value,
        )
    )

    if not checks:
        # The defining "went only to Roku" case from the module docstring: this change
        # touched no OTHER value of `sibling_dimension` at all, so there is nothing that
        # could disconfirm it -- the narrowest, strongest form of evidence.
        note = (
            f"this change touched no other {sibling_dimension} value in the window -- "
            "narrow exposure, strong specific evidence for this blast radius"
        )
    else:
        degraded_count = sum(1 for c in checks if c.degraded is True)
        not_degraded_count = sum(1 for c in checks if c.degraded is False)
        if not_degraded_count > 0:
            # The defining "went to every device type but only Roku degraded" case: at
            # least one other value the change also touched did NOT degrade, which is
            # exactly the disconfirming evidence this function exists to surface.
            note = (
                f"{not_degraded_count} of {len(checks)} other {sibling_dimension} "
                "value(s) touched by this change did NOT degrade -- weaker, "
                "broader-exposure evidence for this blast radius"
            )
        elif degraded_count > 0:
            note = (
                f"every other {sibling_dimension} value touched by this change also "
                f"degraded ({degraded_count} checked) -- broad exposure, not uniquely "
                "tied to this blast radius"
            )
        else:
            note = (
                f"degraded status could not be determined for the {len(checks)} other "
                f"{sibling_dimension} value(s) touched by this change"
            )

    return DisconfirmingEvidence(
        dimension_key=dimension_key,
        dimension_value=dimension_value,
        sibling_dimension=sibling_dimension,
        siblings=checks,
        note=note,
    )


def _sibling_dimension(blast_radius: Slice, dimension_key: str) -> str | None:
    """The first blast-radius dimension (hierarchy order) other than the change's own,
    or None if the blast radius pins nothing else."""
    for dimension in blast_radius.dimensions:
        if dimension != dimension_key:
            return dimension
    return None


# ---------------------------------------------------------------------------
# SQL construction and gateway integration.
# ---------------------------------------------------------------------------


def _fmt(dt: datetime) -> str:
    return dt.strftime(_DATETIME_FORMAT)


def build_candidates_sql(fetch_start: datetime, fetch_end: datetime) -> str:
    """The single query that fetches every change_log row in the fetch window. One
    query for the whole window, never one query per change."""
    return (
        "SELECT change_id, changed_at, change_type, component, description, "
        "dimension_key, dimension_value FROM change_log "
        f"WHERE changed_at >= '{_fmt(fetch_start)}' AND changed_at <= '{_fmt(fetch_end)}' "
        "ORDER BY changed_at"
    )


def _build_sibling_sql(
    change_slice: Slice, metric: Metric, sibling_dimension: str, start: datetime, end: datetime
) -> str:
    raw_events = change_slice.requires_raw_events or sibling_dimension == TITLE_ID_DIMENSION
    table = RAW_EVENTS_TABLE if raw_events else ROLLUP_TABLE
    time_col = "event_time" if raw_events else "bucket"
    expr = metric.sql_for(raw_events=raw_events)
    return (
        f"SELECT {sibling_dimension} AS value, {expr} AS metric_value FROM {table} "
        f"WHERE {change_slice.where_sql()} "
        f"AND {time_col} >= '{_fmt(start)}' AND {time_col} < '{_fmt(end)}' "
        f"GROUP BY {sibling_dimension}"
    )


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _parse_datetime(value: Any) -> datetime:
    return datetime.strptime(str(value), _DATETIME_FORMAT)


def _row_to_change(row: dict[str, Any]) -> ChangeRow:
    return ChangeRow(
        change_id=int(row["change_id"]),
        changed_at=_parse_datetime(row["changed_at"]),
        change_type=str(row["change_type"]),
        component=str(row["component"]),
        description=str(row["description"]),
        dimension_key=str(row["dimension_key"]),
        dimension_value=str(row["dimension_value"]),
    )


async def _attach_disconfirming_evidence(
    gateway: ClickHouseMCPGateway,
    candidate: RankedChange,
    blast_radius: Slice,
    metric: Metric,
    window_start: datetime,
    window_end: datetime,
) -> RankedChange:
    sibling_dimension = _sibling_dimension(blast_radius, candidate.dimension_key)
    if sibling_dimension is None:
        evidence = compute_disconfirming_evidence(
            candidate.dimension_key,
            candidate.dimension_value,
            sibling_dimension=None,
            sibling_measurements=(),
            higher_is_worse=metric.higher_is_worse,
        )
        return replace(candidate, disconfirming_evidence=evidence)

    if candidate.dimension_key not in ALLOWED_DIMENSIONS:
        evidence = DisconfirmingEvidence(
            dimension_key=candidate.dimension_key,
            dimension_value=candidate.dimension_value,
            sibling_dimension=sibling_dimension,
            siblings=(),
            note=(
                f"dimension_key {candidate.dimension_key!r} is not a recognised "
                "dimension; disconfirming evidence not computed"
            ),
        )
        return replace(candidate, disconfirming_evidence=evidence)

    try:
        change_slice = Slice().refine(candidate.dimension_key, candidate.dimension_value)
    except InvalidSliceError as exc:
        evidence = DisconfirmingEvidence(
            dimension_key=candidate.dimension_key,
            dimension_value=candidate.dimension_value,
            sibling_dimension=sibling_dimension,
            siblings=(),
            note=f"could not evaluate this change's dimension: {exc}",
        )
        return replace(candidate, disconfirming_evidence=evidence)

    baseline_start = window_start - _SIBLING_BASELINE_OFFSET
    baseline_end = window_end - _SIBLING_BASELINE_OFFSET
    window_sql = _build_sibling_sql(
        change_slice, metric, sibling_dimension, window_start, window_end
    )
    baseline_sql = _build_sibling_sql(
        change_slice, metric, sibling_dimension, baseline_start, baseline_end
    )
    window_result = await gateway.query(window_sql)
    baseline_result = await gateway.query(baseline_sql)

    baseline_by_value = {
        str(r["value"]): _to_float(r.get("metric_value")) for r in baseline_result.rows
    }
    # The blast radius's own pinned value for `sibling_dimension` (if any) is the
    # incident itself, not a disconfirming candidate -- it trivially "degraded" and
    # including it would always show at least one degraded sibling, which would wrongly
    # dilute a change that in fact touched nothing else at all.
    own_value = _blast_radius_value(blast_radius, sibling_dimension)
    measurements = [
        SiblingMeasurement(
            value=str(r["value"]),
            metric_value=_to_float(r.get("metric_value")),
            baseline_value=baseline_by_value.get(str(r["value"])),
        )
        for r in window_result.rows
        if str(r["value"]) != own_value
    ]

    evidence = compute_disconfirming_evidence(
        candidate.dimension_key,
        candidate.dimension_value,
        sibling_dimension=sibling_dimension,
        sibling_measurements=measurements,
        higher_is_worse=metric.higher_is_worse,
    )
    return replace(candidate, disconfirming_evidence=evidence)


async def correlate_changes(
    gateway: ClickHouseMCPGateway,
    *,
    blast_radius: Slice,
    anomaly_window: tuple[datetime, datetime],
    lookback: timedelta = DEFAULT_LOOKBACK,
    tolerance: timedelta = DEFAULT_TOLERANCE,
    metric_name: str = DEFAULT_METRIC,
) -> CorrelationResult:
    """Find and rank change_log candidates for one blast radius's anomaly window.

    Issues exactly one query to fetch candidate changes, regardless of how many rows
    come back. The fetch window intentionally extends past ``onset + tolerance`` (see
    ``_POST_ONSET_FETCH_MARGIN_FACTOR``) so that changes shortly after onset are
    genuinely fetched and then explicitly rejected as "too late" by ``classify_change``,
    rather than silently absent because the SQL already excluded them.

    Disconfirming evidence for each accepted candidate costs two further queries
    (window + baseline) -- not batched across candidates, because change_log is tiny by
    design (three rows in the real 63.85M-event dataset) and batching would add
    UNION ALL complexity for no measurable benefit here.
    """
    onset, end = anomaly_window
    if not onset < end:
        raise InvalidCorrelationWindowError(
            f"anomaly_window start must be before end, got {anomaly_window!r}"
        )
    if lookback < timedelta(0):
        raise InvalidCorrelationWindowError(f"lookback must be >= 0, got {lookback!r}")
    if tolerance < timedelta(0):
        raise InvalidCorrelationWindowError(f"tolerance must be >= 0, got {tolerance!r}")

    fetch_start = onset - lookback
    fetch_end = onset + tolerance + lookback * _POST_ONSET_FETCH_MARGIN_FACTOR
    sql = build_candidates_sql(fetch_start, fetch_end)
    result = await gateway.query(sql)
    rows = [_row_to_change(row) for row in result.rows]

    candidates, rejected = rank_candidates(
        rows,
        blast_radius=blast_radius,
        onset=onset,
        lookback=lookback,
        tolerance=tolerance,
        sql=sql,
    )

    metric = get_metric(metric_name)
    enriched = []
    for candidate in candidates:
        enriched.append(
            await _attach_disconfirming_evidence(
                gateway, candidate, blast_radius, metric, onset, end
            )
        )
    enriched = tuple(enriched)

    return CorrelationResult(
        blast_radius=blast_radius,
        onset=onset,
        end=end,
        lookback=lookback,
        tolerance=tolerance,
        candidates=enriched,
        rejected=rejected,
        sql=sql,
    )
