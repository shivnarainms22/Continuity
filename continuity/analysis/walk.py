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

# A value must explain at least this share of the slice's deviation to be worth
# descending into. Measured live against the real dataset (see
# tests/integration/test_walk_real.py): the true fault dimensions score 93-107%
# (device_type=roku 107%, app_version=8.2.0 99%, cdn=cdn_northwind 94%, pop=nw-atl-2
# 96%), while the first split AFTER the true fault is already isolated -- which only
# reflects where that (now uniformly affected) traffic happens to live, e.g.
# country=US, not an additional cause -- drops to 42-59%. That drop is not
# coincidental: once a dimension's degradation is uniform across its values, each
# value's contribution reduces to weight_share * (the same delta for everyone), so its
# share_of_deviation converges to its plain weight_share -- "explains a lot because
# it's a big population," not "because it's the cause." 0.6 sits cleanly between the
# two measured clusters on this dataset.
DEFAULT_MIN_SHARE = 0.6

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

    LOW_SHARE = "low_share"
    """The best-explaining value's share of the slice's deviation fell below
    `min_share`: no single value explains enough to justify descending."""

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
    slice's deviation it explained, and the SQL behind that number."""

    dimension: str
    value: str
    share_of_deviation: float
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
    min_share: float,
    min_weight_fraction: float,
    root_weight: float | None,
) -> tuple[RefinementStep | None, StopReason | None]:
    """The pure decision at one level of the walk: which dimension/value to descend
    into next, or why to stop here instead. Exactly one of the two return values is
    non-`None`.

    Only dimensions `split_dimensions_median_baseline` marked `informative` are
    considered (requirement: a dimension with a single usable value gains no
    information by comparison). Among those, the candidate is each dimension's
    top-ranked (by `split.py`'s weighted contribution) value, provided its
    `share_of_deviation` is a positive number -- `None` shares (zero net deviation, a
    single usable value) and non-positive shares (this value moved the *opposite* way
    from the slice's overall deviation) cannot explain the slice's problem and are not
    candidates. The dimension whose candidate has the LARGEST share wins: "the
    dimension whose best value explains the most of the current slice's deviation."
    """
    informative = {
        dimension: result
        for dimension, result in splits.items()
        if result.informative and result.values
    }
    if not informative:
        return None, StopReason.SINGLE_VALUE

    candidates = [
        (dimension, result.values[0])
        for dimension, result in informative.items()
        if result.values[0].share_of_deviation is not None
        and result.values[0].share_of_deviation > 0
    ]
    if not candidates:
        return None, StopReason.LOW_SHARE

    best_dimension, best = max(candidates, key=lambda item: item[1].share_of_deviation)

    if best.share_of_deviation < min_share:
        return None, StopReason.LOW_SHARE

    if (
        root_weight is not None
        and root_weight > 0
        and best.weight < root_weight * min_weight_fraction
    ):
        return None, StopReason.TOO_SMALL

    step = RefinementStep(
        dimension=best_dimension,
        value=best.value,
        share_of_deviation=best.share_of_deviation,
        contribution=best.contribution if best.contribution is not None else 0.0,
        weight=best.weight,
        sql=best.sql,
        baseline_sql=best.baseline_sql,
    )
    return step, None


async def walk(
    gateway: ClickHouseMCPGateway,
    *,
    metric_name: str,
    window: tuple[datetime, datetime],
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    min_share: float = DEFAULT_MIN_SHARE,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_weight_fraction: float = DEFAULT_MIN_WEIGHT_FRACTION,
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

    while True:
        if len(path) >= max_depth:
            stop_reason = StopReason.MAX_DEPTH
            break

        remaining = [d for d in dimensions if d not in current_slice.dimensions]
        if not remaining:
            stop_reason = StopReason.DIMENSIONS_EXHAUSTED
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

        step, reason = choose_next_step(
            splits,
            min_share=min_share,
            min_weight_fraction=min_weight_fraction,
            root_weight=root_weight,
        )
        if step is None:
            assert reason is not None
            stop_reason = reason
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
        elapsed_ms=elapsed_ms,
        query_log=query_log,
    )
