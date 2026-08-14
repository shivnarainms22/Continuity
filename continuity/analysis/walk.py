"""Deterministic greedy drill-down: the control arm.

Starting from the whole population over a given anomaly window (typically the output
of `detect.detect()` -- this module takes the window as an input, it does not discover
one), repeatedly:

1. Split every remaining (not-yet-fixed) dimension in one batched call
   (`split.split_dimensions_median_baseline`), against a ROBUST baseline: the median of
   several trailing comparison windows rather than one -- see that function's docstring
   for why a single comparison window is fragile.
2. Among the dimensions where splitting carries any information, pick the dimension
   whose best value explains the LARGEST share of the current slice's deviation.
3. Descend into that value (refine the slice) and repeat.
4. Stop -- and RECORD why -- the moment descending stops being informative.

No LLM calls, and none will ever be added here: every number is produced by `split.py`
and `baseline.py`'s SQL-backed maths. This is (a) proof the primitives compose into a
working investigation with no model involved, (b) a fallback if the agent misbehaves
during a live demo, and (c) the CONTROL ARM for the evaluation -- sub-project 3's
Gemini-driven investigator is scored against this on the same primitives and the same
data, so the walker has to be genuinely good, not a deliberately weak strawman.

Pure decision logic lives in `choose_next_step` -- no SQL, no I/O, fully testable
without Docker, exactly like `split.py`'s `rank_contributions` / `is_informative`.
`walk()` is the thin async loop that feeds it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from continuity.analysis.baseline import DEFAULT_LOOKBACK_WEEKS
from continuity.analysis.metrics import get_metric
from continuity.analysis.slices import Slice
from continuity.analysis.split import SplitResult, split_dimensions_median_baseline
from continuity.data.topology import DIMENSION_HIERARCHY
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway, ExecutedQuery

# The full coarse-to-fine dimension universe the walker will consider unless the
# caller narrows it. title_id is deliberately excluded -- it sits outside the
# hierarchy (see slices.py) and forces the raw-events table; a caller investigating a
# per-title fault should pass it explicitly via `dimensions`.
DEFAULT_DIMENSIONS: tuple[str, ...] = DIMENSION_HIERARCHY

# The PRIMARY stopping criterion. `lift = share_of_deviation / weight_share` (see
# split.py's module docstring): a value whose share of the deviation merely matches
# its share of the population has lift == 1 and explains nothing beyond being big --
# splitting a UNIFORMLY degraded slice on any dimension reproduces exactly this, since
# every value then carries the same signed delta and share_of_deviation collapses to
# weight_share by construction. A raw share threshold cannot tell that case apart from
# a genuine localized fault (both can show a high absolute share, if the affected
# population also happens to be a large one) -- lift can, because it divides the
# population-size effect out.
#
# 1.5 is chosen by REASONING about the ratio itself, not by sweeping this dataset for
# a number that happens to separate it (that is precisely the failure mode this
# threshold replaces): it requires the deviation to be at least 50% more concentrated
# in the candidate value than its population size alone would produce. That margin is
# wide enough to absorb ordinary sampling noise and a partially-diluted true signal
# (e.g. a fault that affects most, but not all, of a value's traffic, or a value that
# is a strict superset of the true affected population) without crossing 1.5 by
# accident, while a genuine localized fault -- where the deviation concentrates in a
# minority of the traffic -- clears it by several multiples, not by a hair. See
# tests/analysis/test_walk.py::test_choose_next_step_stops_on_uniform_degradation_not_
# the_biggest_segment for the degenerate case (lift == 1 exactly) this threshold
# exists to reject.
DEFAULT_MIN_LIFT = 1.5

# An OPTIONAL secondary guard, off by default (`None`): if a caller supplies it, the
# best candidate's raw share_of_deviation must also clear this bar. It must never be
# the primary criterion -- see the coordinator's own rejection of a tuned absolute
# threshold, which is exactly what a load-bearing default here would reintroduce.
DEFAULT_MIN_SHARE: float | None = None

# Cannot usefully exceed the number of candidate dimensions -- once every dimension is
# fixed there is nothing left to split on (DIMENSIONS_EXHAUSTED fires first anyway).
DEFAULT_MAX_DEPTH = len(DIMENSION_HIERARCHY)

# A candidate value must carry at least this fraction of the ROOT slice's total weight
# (same metric, same units, so the fraction is unit-agnostic within one walk) to be
# descended into. Guards against chasing a slice a handful of sessions wide, whose
# ratio is dominated by noise rather than signal -- exactly the failure mode
# mean/sigma baselines have on thin slices (baseline.py's INSUFFICIENT_DATA guard),
# just applied to slice SIZE rather than to the baseline's own sample size.
DEFAULT_MIN_WEIGHT_FRACTION = 0.01


class StopReason(Enum):
    """Why the walk stopped where it did. Recorded on every `WalkResult` -- "why did it
    stop here" is a product feature, not an implementation detail to be inferred."""

    LOW_LIFT = "low_lift"
    """The best-explaining value's LIFT (share_of_deviation / weight_share) fell below
    `min_lift`: it explains no more of the deviation than its population size alone
    would predict -- the "biggest population segment" trap, not a real signal."""

    LOW_SHARE = "low_share"
    """The optional secondary `min_share` guard fired: the best-explaining value's raw
    share of the slice's deviation fell below it. Only reachable when a caller
    supplies `min_share`, or when no candidate has a positive share at all (nothing to
    rank by lift in the first place)."""

    SINGLE_VALUE = "single_value"
    """Every remaining dimension had at most one usable value present -- no dimension
    could gain any information by comparison."""

    MAX_DEPTH = "max_depth"
    """The walk reached `max_depth` refinements."""

    TOO_SMALL = "too_small"
    """The best-explaining value's weight fell below `min_weight_fraction` of the root
    slice's weight -- too small a slice to trust its ratio."""

    DIMENSIONS_EXHAUSTED = "dimensions_exhausted"
    """Every candidate dimension is already fixed in the current slice."""


@dataclass(frozen=True)
class RefinementStep:
    """One step of the drill-down: which dimension, which value, how much of the
    slice's deviation it explained, its LIFT over its own population share, and the
    SQL behind that number."""

    dimension: str
    value: str
    share_of_deviation: float
    lift: float
    contribution: float
    weight: float
    sql: str
    baseline_sql: str


@dataclass(frozen=True)
class WalkResult:
    """The full, ordered drill-down, with everything needed to audit it."""

    metric: str
    window: tuple[datetime, datetime]
    baseline_windows: tuple[tuple[datetime, datetime], ...]
    path: tuple[RefinementStep, ...]
    final_slice: Slice
    stop_reason: StopReason
    stop_detail: str
    elapsed_ms: float
    query_log: tuple[ExecutedQuery, ...]


def week_over_week_baseline_windows(
    window: tuple[datetime, datetime], lookback_weeks: int
) -> tuple[tuple[datetime, datetime], ...]:
    """The trailing comparison windows the walker compares `window` against: the same
    weekday and time-of-day, `1..lookback_weeks` weeks earlier -- baseline.py's
    week-over-week convention (`select_week_over_week_window`), applied to a whole
    window rather than a single 5-minute bucket.
    """
    if lookback_weeks <= 0:
        raise ValueError(f"lookback_weeks must be > 0, got {lookback_weeks}")
    start, end = window
    if not start < end:
        raise ValueError(f"window start must be before end, got {window!r}")
    return tuple(
        (start - timedelta(weeks=i), end - timedelta(weeks=i)) for i in range(1, lookback_weeks + 1)
    )


def _population_weight(splits: dict[str, SplitResult]) -> float | None:
    """The current slice's total weight, estimated from whichever split has the
    largest total -- every dimension partitions the same population, so their totals
    should agree up to rows a GROUP BY happened to drop for one dimension but not
    another (e.g. a NULL value). `None` when no split carries any weighted value."""
    totals = [sum(c.weight for c in result.values) for result in splits.values() if result.values]
    return max(totals) if totals else None


def choose_next_step(
    splits: dict[str, SplitResult],
    *,
    min_lift: float,
    min_weight_fraction: float,
    root_weight: float | None,
    min_share: float | None = None,
) -> tuple[RefinementStep | None, StopReason | None, str | None]:
    """The pure decision at one level of the walk: which dimension/value to descend
    into next, or why to stop here instead. Exactly one of (`step`) or
    (`stop_reason`, `stop_detail`) is populated; `stop_detail` always carries the
    measured number that triggered the stop, not just its label.

    Only dimensions `split_dimensions_median_baseline` marked `informative` are
    considered (requirement: a dimension with a single usable value gains no
    information by comparison). Among those, the candidate is each dimension's
    top-ranked (by `split.py`'s weighted contribution) value, provided its
    `share_of_deviation` is a positive number -- `None` shares (zero net deviation, a
    single usable value) and non-positive shares (this value moved the *opposite* way
    from the slice's overall deviation) cannot explain the slice's problem and are not
    candidates. The dimension whose candidate has the LARGEST share wins: "the
    dimension whose best value explains the most of the current slice's deviation."

    The PRIMARY gate on that winner is its LIFT (`share_of_deviation / weight_share`),
    not its raw share -- see `DEFAULT_MIN_LIFT`'s module-level comment for why a raw
    share threshold cannot distinguish a genuine localized fault from a big value that
    merely inherited its share from its own population size. `min_share`, if supplied,
    is an OPTIONAL secondary guard checked only after the lift gate has already
    passed -- it is never the reason a step is accepted.
    """
    informative = {
        dimension: result
        for dimension, result in splits.items()
        if result.informative and result.values
    }
    if not informative:
        return (
            None,
            StopReason.SINGLE_VALUE,
            "no remaining dimension had more than one usable value",
        )

    candidates = [
        (dimension, result.values[0])
        for dimension, result in informative.items()
        if result.values[0].share_of_deviation is not None
        and result.values[0].share_of_deviation > 0
    ]
    if not candidates:
        return (
            None,
            StopReason.LOW_SHARE,
            "every remaining dimension's top value had an undefined or non-positive "
            "share of the slice's deviation",
        )

    best_dimension, best = max(candidates, key=lambda item: item[1].share_of_deviation)

    if best.lift is None or best.lift < min_lift:
        measured = "undefined" if best.lift is None else f"{best.lift:.2f}"
        return (
            None,
            StopReason.LOW_LIFT,
            f"{best_dimension}={best.value!r} has lift {measured} "
            f"(share_of_deviation={best.share_of_deviation:.3f}, "
            f"weight_share={(best.weight_share or 0.0):.3f}) < min_lift {min_lift:.2f} "
            "-- explains no more of the deviation than its population size alone would predict",
        )

    if min_share is not None and best.share_of_deviation < min_share:
        return (
            None,
            StopReason.LOW_SHARE,
            f"{best_dimension}={best.value!r} share_of_deviation "
            f"{best.share_of_deviation:.3f} < min_share {min_share:.2f}",
        )

    if (
        root_weight is not None
        and root_weight > 0
        and best.weight < root_weight * min_weight_fraction
    ):
        return (
            None,
            StopReason.TOO_SMALL,
            f"{best_dimension}={best.value!r} weight {best.weight:.1f} < "
            f"min_weight_fraction {min_weight_fraction:.4f} of root weight {root_weight:.1f}",
        )

    step = RefinementStep(
        dimension=best_dimension,
        value=best.value,
        share_of_deviation=best.share_of_deviation,
        lift=best.lift,
        contribution=best.contribution if best.contribution is not None else 0.0,
        weight=best.weight,
        sql=best.sql,
        baseline_sql=best.baseline_sql,
    )
    return step, None, None


async def walk(
    gateway: ClickHouseMCPGateway,
    *,
    metric_name: str,
    window: tuple[datetime, datetime],
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_lift: float = DEFAULT_MIN_LIFT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_weight_fraction: float = DEFAULT_MIN_WEIGHT_FRACTION,
    min_share: float | None = DEFAULT_MIN_SHARE,
) -> WalkResult:
    """Drill down from the whole population into `window`, one dimension at a time.

    Never re-splits a dimension already fixed in the current slice (`remaining` below
    excludes it at every level). Every level batches all remaining dimensions into one
    query per comparison window (per Task 1's benchmark: 43ms batched vs 183ms
    sequential) via `split_dimensions_median_baseline`, which itself compares against
    the MEDIAN of `lookback_weeks` trailing same-weekday/time-of-day windows rather
    than a single one -- see that function's docstring.

    This is the only function in this module that performs I/O. `choose_next_step` is
    the pure per-level decision it delegates to.
    """
    if not dimensions:
        raise ValueError("dimensions must be non-empty")
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")

    metric = get_metric(metric_name)
    baseline_windows = week_over_week_baseline_windows(window, lookback_weeks)

    start_log_index = len(gateway.query_log)
    started = time.perf_counter()

    current_slice = Slice()
    path: list[RefinementStep] = []
    root_weight: float | None = None
    stop_reason = StopReason.MAX_DEPTH
    stop_detail = f"reached max_depth={max_depth} before any dimension could be split"

    while True:
        if len(path) >= max_depth:
            stop_reason = StopReason.MAX_DEPTH
            stop_detail = f"reached max_depth={max_depth}"
            break

        remaining = [d for d in dimensions if d not in current_slice.dimensions]
        if not remaining:
            stop_reason = StopReason.DIMENSIONS_EXHAUSTED
            stop_detail = f"every candidate dimension {tuple(dimensions)} is already fixed"
            break

        splits = await split_dimensions_median_baseline(
            gateway,
            slice_=current_slice,
            metric=metric,
            dimensions=remaining,
            window=window,
            baseline_windows=baseline_windows,
        )

        if root_weight is None:
            root_weight = _population_weight(splits)

        step, reason, detail = choose_next_step(
            splits,
            min_lift=min_lift,
            min_share=min_share,
            min_weight_fraction=min_weight_fraction,
            root_weight=root_weight,
        )
        if step is None:
            assert reason is not None and detail is not None
            stop_reason = reason
            stop_detail = detail
            break

        path.append(step)
        current_slice = current_slice.refine(step.dimension, step.value)

    elapsed_ms = (time.perf_counter() - started) * 1000
    query_log = tuple(gateway.query_log[start_log_index:])
    return WalkResult(
        metric=metric_name,
        window=window,
        baseline_windows=baseline_windows,
        path=tuple(path),
        final_slice=current_slice,
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        elapsed_ms=elapsed_ms,
        query_log=query_log,
    )
