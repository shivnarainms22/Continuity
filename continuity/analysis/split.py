"""Dimensional decomposition: how much of a slice's deviation each value explains.

Pure maths lives in ``rank_contributions`` / ``is_informative`` -- no SQL, no I/O, fully
testable without Docker. ``split_dimension`` / ``split_dimensions`` are the thin async
wrappers that issue the SQL through the MCP gateway and feed the pure maths.

The core identity, from ``docs/superpowers/plans/2026-08-08-continuity-02-analysis-core.md``:

    rebuffer_ratio = sum(rebuffer_ms) / sum(watched_ms)

is a RATIO, and a parent's ratio deviation is NOT the sum of its children's ratio
deviations -- children carry different weights. For dimension values *v* with weight
share *w_v* and metric value *r_v*:

    parent_metric      = sum over v of (w_v * r_v)
    contribution of v  = w_v * (r_v - r_v_baseline)          [higher_is_worse metrics]
                       = w_v * (r_v_baseline - r_v)          [lower_is_worse metrics]
    share of deviation = contribution_v / sum of all contributions

Ranking naively by the raw deviation (r_v - r_v_baseline) promotes tiny slices with wild
ratios and buries the real cause -- see
``test_naive_ranking_by_raw_deviation_would_wrongly_promote_the_tiny_slice`` in
tests/analysis/test_split.py.

``lift = share_of_deviation / weight_share`` is carried on every ``Contribution`` for
the same reason ``share_of_deviation`` alone is not sufficient to decide whether a
value is worth pursuing further (see ``continuity/analysis/walk.py``'s stopping rule):
when a dimension's degradation is UNIFORM across its values, ``share_of_deviation``
mathematically collapses to that value's plain ``weight_share`` (every value's signed
delta is the same, so ``contribution_v = weight_share_v * delta`` and
``total_contribution = delta``, giving ``share_v = weight_share_v`` exactly). A value
that merely explains its own population share of the deviation -- lift ~= 1 -- is
"big," not "the cause." Lift is undefined (``None``) whenever ``share_of_deviation``
or ``weight_share`` is undefined, or ``weight_share`` is 0.

Weighting is explicit per metric (requirement 2 of Task 5):

* Ratio metrics (rebuffer, errors) are weighted by their own denominator -- watched_ms
  for rebuffer, starts for errors -- so that ``sum(w_v * r_v)`` reconstructs the parent's
  ratio exactly.
* Non-ratio metrics (startup p95, average bitrate) have no natural denominator, so they
  are weighted by distinct session count. Weighting them by watch time or start count
  would let a handful of unusually long sessions or frequent starters dominate; session
  count bounds each session's influence to one unit of weight.

Two queries per split -- one over ``window``, one over ``baseline_window`` -- regardless
of how many dimensions are requested, AS LONG AS every dimension shares one required
table: ``split_dimensions`` batches all of them into a single ``UNION ALL`` per window
(see ``scripts/benchmark_queries.py``, which measured this at ~43ms for 8 dimensions vs
~183ms issued one at a time). A dimension that forces ``playback_events`` instead of
``qoe_rollup_5m`` (currently only ``title_id`` -- see ``slices.dimension_required_table``)
is issued as its OWN query per window rather than unioned with the rollup-backed arms:
mixing a raw-events scan into the same statement as the cheap rollup arms would force
every arm to share that one query's memory budget for its whole lifetime, which is
exactly the defect this module used to have. ``split_dimensions``/
``split_dimensions_median_baseline`` therefore issue one query pair (or query-plus-
medians) PER REQUIRED TABLE among the requested dimensions, not per dimension -- the
batching win is preserved within each table, just no longer smeared across tables.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any

from continuity.analysis.metrics import Metric
from continuity.analysis.slices import (
    ALLOWED_DIMENSIONS,
    RAW_EVENTS_TABLE,
    TITLE_ID_DIMENSION,
    InvalidSliceError,
    Slice,
    dimension_required_table,
)
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# metric name -> (rollup weight expression, raw-events weight expression). See module
# docstring for the rationale behind each choice.
_WEIGHT_SQL: dict[str, tuple[str, str]] = {
    "rebuffer": ("sum(watched_ms)", "sum(watched_ms)"),
    "errors": ("sum(starts)", "sum(toUInt64(event_type = 'start'))"),
    "startup": ("uniqMerge(sessions)", "uniq(session_id)"),
    "bitrate": ("uniqMerge(sessions)", "uniq(session_id)"),
}


@dataclass(frozen=True)
class ValueMeasurement:
    """One dimension value's raw measurement, before any contribution maths.

    ``weight`` is the volume basis for this metric (see ``_WEIGHT_SQL``).
    ``baseline_value`` is ``None`` when the value has no observations at all in the
    baseline period -- e.g. an app version that only started shipping during the
    incident window. That must not crash and must not be silently dropped; it is
    carried through to the corresponding ``Contribution`` with an explanatory note.
    """

    value: str
    metric_value: float | None
    baseline_value: float | None
    weight: float


@dataclass(frozen=True)
class Contribution:
    """A ranked dimension value, carrying everything needed to audit the ranking."""

    dimension: str
    value: str
    metric_value: float | None
    baseline_value: float | None
    weight: float
    weight_share: float | None
    contribution: float | None
    share_of_deviation: float | None
    lift: float | None
    note: str | None
    sql: str
    baseline_sql: str


@dataclass(frozen=True)
class SplitResult:
    """The ranked decomposition of one dimension, plus provenance."""

    dimension: str
    values: tuple[Contribution, ...]
    informative: bool
    sql: str
    baseline_sql: str


def _is_usable(m: ValueMeasurement) -> bool:
    return m.metric_value is not None and not math.isnan(m.metric_value) and m.weight > 0


def is_informative(measurements: Sequence[ValueMeasurement]) -> bool:
    """False when splitting on this dimension carries no information.

    True information requires at least two distinct, usable values -- a dimension
    pinned to a single value (or where every other value has zero weight or no data)
    cannot explain anything by comparison.
    """
    usable_values = {m.value for m in measurements if _is_usable(m)}
    return len(usable_values) > 1


def rank_contributions(
    measurements: Sequence[ValueMeasurement],
    *,
    dimension: str,
    higher_is_worse: bool,
    sql: str = "",
    baseline_sql: str = "",
) -> tuple[Contribution, ...]:
    """The core maths: weighted contribution-to-deviation, ranked, edge-case safe.

    Never raises on missing data. Every measurement in ``measurements`` produces
    exactly one ``Contribution`` -- values that cannot be scored (no metric value, no
    baseline, no weight) come back with ``contribution=None`` and an explanatory
    ``note`` rather than being dropped or defaulted to a misleading number.
    """
    usable = [m for m in measurements if _is_usable(m)]
    total_weight = sum(m.weight for m in usable)
    single_value = len({m.value for m in usable}) <= 1

    staged: list[tuple[ValueMeasurement, float | None, float | None, str | None]] = []
    for m in measurements:
        if not _is_usable(m):
            note = (
                "no metric value recorded for this value in the window"
                if m.metric_value is None or math.isnan(m.metric_value)
                else "no weight (zero volume) for this value in the window"
            )
            staged.append((m, None, None, note))
            continue

        weight_share = m.weight / total_weight if total_weight > 0 else None
        if m.baseline_value is None:
            staged.append(
                (m, weight_share, None, "absent from the baseline period -- new to this window")
            )
            continue

        signed_delta = (
            (m.metric_value - m.baseline_value)
            if higher_is_worse
            else (m.baseline_value - m.metric_value)
        )
        contribution = (weight_share or 0.0) * signed_delta
        staged.append((m, weight_share, contribution, None))

    total_contribution = sum(c for _, _, c, _ in staged if c is not None)

    results: list[Contribution] = []
    for m, weight_share, contribution, note in staged:
        share: float | None = None
        final_note = note
        if contribution is not None:
            if single_value:
                final_note = "only one usable value for this dimension -- no information gained"
            elif total_contribution == 0:
                final_note = "zero net deviation across values -- share of deviation is undefined"
            else:
                share = contribution / total_contribution
        lift = (
            share / weight_share
            if share is not None and weight_share is not None and weight_share > 0
            else None
        )
        results.append(
            Contribution(
                dimension=dimension,
                value=m.value,
                metric_value=m.metric_value,
                baseline_value=m.baseline_value,
                weight=m.weight,
                weight_share=weight_share,
                contribution=contribution,
                share_of_deviation=share,
                lift=lift,
                note=final_note,
                sql=sql,
                baseline_sql=baseline_sql,
            )
        )

    def sort_key(c: Contribution) -> tuple[bool, float]:
        return (c.contribution is None, -(c.contribution or 0.0))

    return tuple(sorted(results, key=sort_key))


# ---------------------------------------------------------------------------
# SQL construction and gateway integration.
# ---------------------------------------------------------------------------


def _validate_dimension(dimension: str) -> None:
    if dimension not in ALLOWED_DIMENSIONS:
        raise InvalidSliceError(
            f"Unknown dimension {dimension!r}. Known: {sorted(ALLOWED_DIMENSIONS)}"
        )


def _validate_window(window: tuple[datetime, datetime]) -> None:
    start, end = window
    if not start < end:
        raise ValueError(f"window start must be before end, got {window!r}")


def _fmt(dt: datetime) -> str:
    return dt.strftime(_DATETIME_FORMAT)


def _weight_sql(metric: Metric, *, raw_events: bool) -> str:
    try:
        rollup_expr, raw_expr = _WEIGHT_SQL[metric.name]
    except KeyError:
        raise ValueError(
            f"No weighting rule defined for metric {metric.name!r}. Known: {sorted(_WEIGHT_SQL)}"
        ) from None
    return raw_expr if raw_events else rollup_expr


def _split_arm_sql(
    slice_: Slice,
    metric: Metric,
    dimension: str,
    window: tuple[datetime, datetime],
    *,
    tag: bool,
) -> str:
    """One SELECT ... GROUP BY arm. ``tag`` prefixes a literal dimension-name column
    so several arms can be told apart after a UNION ALL.

    ``title_id`` is numeric (``playback_events.title_id``); every other dimension's
    ``value`` column is a string. Batching them into one ``UNION ALL`` requires every
    arm's ``value`` column to share a common type, so the ``title_id`` arm projects
    ``toString(title_id)`` -- the ``GROUP BY`` stays on the raw numeric column, only the
    projected value changes. Without this cast, ClickHouse rejects the batched query
    with a ``NO_COMMON_TYPE`` error the moment ``title_id`` is mixed with any other
    dimension.
    """
    start, end = window
    table = dimension_required_table(slice_, dimension)
    raw_events = table == RAW_EVENTS_TABLE
    time_col = "event_time" if raw_events else "bucket"
    metric_expr = metric.sql_for(raw_events=raw_events)
    weight_expr = _weight_sql(metric, raw_events=raw_events)
    dim_tag = f"'{dimension}' AS dim, " if tag else ""
    value_expr = f"toString({dimension})" if dimension == TITLE_ID_DIMENSION else dimension
    return (
        f"SELECT {dim_tag}{value_expr} AS value, {metric_expr} AS metric_value, "
        f"{weight_expr} AS weight FROM {table} "
        f"WHERE {slice_.where_sql()} "
        f"AND {time_col} >= '{_fmt(start)}' AND {time_col} < '{_fmt(end)}' "
        f"GROUP BY {dimension}"
    )


def _build_split_sql(
    slice_: Slice, metric: Metric, dimension: str, window: tuple[datetime, datetime]
) -> str:
    return _split_arm_sql(slice_, metric, dimension, window, tag=False)


def _build_batched_split_sql(
    slice_: Slice, metric: Metric, dimensions: Sequence[str], window: tuple[datetime, datetime]
) -> str:
    return " UNION ALL ".join(
        _split_arm_sql(slice_, metric, dimension, window, tag=True) for dimension in dimensions
    )


def _group_dimensions_by_table(slice_: Slice, dimensions: Sequence[str]) -> list[list[str]]:
    """Group `dimensions` by the table a GROUP BY on each requires (see
    `slices.dimension_required_table`), preserving relative order both across and
    within groups. A rollup-backed dimension and a raw-events-backed one (currently
    only title_id) must never end up batched into the same UNION ALL -- see the module
    docstring's benchmark and the `Slice`/query-cost defect this fixes: unioning a
    handful of cheap rollup arms with one arm that scans `playback_events` over the
    full window forces every arm to share that one query's memory budget for its
    whole lifetime, defeating the reason the rollup exists. This function does not
    name `title_id` itself -- a future raw-events-only dimension is grouped correctly
    with no change here.
    """
    order: list[str] = []
    groups: dict[str, list[str]] = {}
    for dimension in dimensions:
        table = dimension_required_table(slice_, dimension)
        if table not in groups:
            groups[table] = []
            order.append(table)
        groups[table].append(dimension)
    return [groups[table] for table in order]


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _rows_to_measurements(
    window_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]
) -> list[ValueMeasurement]:
    baseline_by_value = {str(row["value"]): row for row in baseline_rows}
    measurements: list[ValueMeasurement] = []
    for row in window_rows:
        value = str(row["value"])
        baseline_row = baseline_by_value.get(value)
        baseline_value = (
            _to_float_or_none(baseline_row["metric_value"]) if baseline_row is not None else None
        )
        measurements.append(
            ValueMeasurement(
                value=value,
                metric_value=_to_float_or_none(row.get("metric_value")),
                baseline_value=baseline_value,
                weight=_to_float_or_none(row.get("weight")) or 0.0,
            )
        )
    return measurements


def _group_by_dim(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dim"])].append(row)
    return grouped


async def split_dimension(
    gateway: ClickHouseMCPGateway,
    *,
    slice_: Slice,
    metric: Metric,
    dimension: str,
    window: tuple[datetime, datetime],
    baseline_window: tuple[datetime, datetime],
) -> SplitResult:
    """Split ``slice_`` on ONE ``dimension``, ranking values by contribution.

    Issues exactly two queries -- one GROUP BY over ``window``, one over
    ``baseline_window`` -- regardless of how many distinct values the dimension has.
    """
    _validate_dimension(dimension)
    _validate_window(window)
    _validate_window(baseline_window)

    sql = _build_split_sql(slice_, metric, dimension, window)
    baseline_sql = _build_split_sql(slice_, metric, dimension, baseline_window)
    window_result = await gateway.query(sql)
    baseline_result = await gateway.query(baseline_sql)

    measurements = _rows_to_measurements(window_result.rows, baseline_result.rows)
    values = rank_contributions(
        measurements,
        dimension=dimension,
        higher_is_worse=metric.higher_is_worse,
        sql=sql,
        baseline_sql=baseline_sql,
    )
    return SplitResult(
        dimension=dimension,
        values=values,
        informative=is_informative(measurements),
        sql=sql,
        baseline_sql=baseline_sql,
    )


async def split_dimensions(
    gateway: ClickHouseMCPGateway,
    *,
    slice_: Slice,
    metric: Metric,
    dimensions: Sequence[str],
    window: tuple[datetime, datetime],
    baseline_window: tuple[datetime, datetime],
) -> dict[str, SplitResult]:
    """Split ``slice_`` on MANY dimensions, batched into one UNION ALL query per
    window PER REQUIRED TABLE -- two queries total when every dimension shares a
    table (the common case), never one query per dimension. A dimension that forces
    ``playback_events`` (currently only ``title_id``) is issued as its own query
    rather than unioned with the ``qoe_rollup_5m``-backed arms -- see
    ``_group_dimensions_by_table``.
    """
    if not dimensions:
        raise ValueError("dimensions must be non-empty")
    for dimension in dimensions:
        _validate_dimension(dimension)
    _validate_window(window)
    _validate_window(baseline_window)

    groups = _group_dimensions_by_table(slice_, dimensions)

    window_by_dim: dict[str, list[dict[str, Any]]] = {}
    baseline_by_dim: dict[str, list[dict[str, Any]]] = {}
    sql_by_dim: dict[str, str] = {}
    baseline_sql_by_dim: dict[str, str] = {}
    for group in groups:
        sql = _build_batched_split_sql(slice_, metric, group, window)
        baseline_sql = _build_batched_split_sql(slice_, metric, group, baseline_window)
        window_result = await gateway.query(sql)
        baseline_result = await gateway.query(baseline_sql)

        group_window_by_dim = _group_by_dim(window_result.rows)
        group_baseline_by_dim = _group_by_dim(baseline_result.rows)
        for dimension in group:
            window_by_dim[dimension] = group_window_by_dim.get(dimension, [])
            baseline_by_dim[dimension] = group_baseline_by_dim.get(dimension, [])
            sql_by_dim[dimension] = sql
            baseline_sql_by_dim[dimension] = baseline_sql

    results: dict[str, SplitResult] = {}
    for dimension in dimensions:
        measurements = _rows_to_measurements(
            window_by_dim[dimension], baseline_by_dim[dimension]
        )
        values = rank_contributions(
            measurements,
            dimension=dimension,
            higher_is_worse=metric.higher_is_worse,
            sql=sql_by_dim[dimension],
            baseline_sql=baseline_sql_by_dim[dimension],
        )
        results[dimension] = SplitResult(
            dimension=dimension,
            values=values,
            informative=is_informative(measurements),
            sql=sql_by_dim[dimension],
            baseline_sql=baseline_sql_by_dim[dimension],
        )
    return results


# ---------------------------------------------------------------------------
# Robust baseline: median of SEVERAL comparison windows, not one.
#
# The functions above compare against a single `baseline_window` (same hours, N days
# earlier). That is fragile for the same reason mean/sigma was rejected in baseline.py:
# if that one window happens to contain an incident, every contribution computed from
# it is wrong. With three planted incidents in 56 days that is a live possibility, not
# a hypothetical (see docs/superpowers/plans/2026-08-08-continuity-02-analysis-core.md,
# "Carried forward into Task 8"). `walk.py` (the greedy drill-down) uses this variant
# exclusively; the single-window functions above are unchanged and keep their own tests.
# ---------------------------------------------------------------------------


def _median_baseline_value(
    value: str, baseline_by_value_per_window: Sequence[dict[str, dict[str, Any]]]
) -> float | None:
    """Median `metric_value` for `value` across several baseline windows.

    Windows where `value` has no data are skipped, not treated as zero -- `None` only
    when EVERY window lacks it, matching `_rows_to_measurements`'s "absent from the
    baseline period" contract.
    """
    values = [
        v
        for by_value in baseline_by_value_per_window
        if (row := by_value.get(value)) is not None
        and (v := _to_float_or_none(row.get("metric_value"))) is not None
    ]
    return median(values) if values else None


def _rows_to_measurements_median_baseline(
    window_rows: list[dict[str, Any]], baseline_rows_per_window: Sequence[list[dict[str, Any]]]
) -> list[ValueMeasurement]:
    """Like `_rows_to_measurements`, but `baseline_value` is the MEDIAN across several
    comparison windows rather than one -- see the section docstring above."""
    baseline_by_value_per_window = [
        {str(row["value"]): row for row in rows} for rows in baseline_rows_per_window
    ]
    measurements: list[ValueMeasurement] = []
    for row in window_rows:
        value = str(row["value"])
        measurements.append(
            ValueMeasurement(
                value=value,
                metric_value=_to_float_or_none(row.get("metric_value")),
                baseline_value=_median_baseline_value(value, baseline_by_value_per_window),
                weight=_to_float_or_none(row.get("weight")) or 0.0,
            )
        )
    return measurements


async def split_dimensions_median_baseline(
    gateway: ClickHouseMCPGateway,
    *,
    slice_: Slice,
    metric: Metric,
    dimensions: Sequence[str],
    window: tuple[datetime, datetime],
    baseline_windows: Sequence[tuple[datetime, datetime]],
) -> dict[str, SplitResult]:
    """Split `slice_` on MANY dimensions against a ROBUST baseline: the MEDIAN of
    several comparison windows (typically the same weekday/time-of-day over the
    preceding `lookback_weeks` weeks, matching baseline.py's week-over-week convention)
    rather than one -- see the module-level section docstring above for why.

    Issues `1 + len(baseline_windows)` queries PER REQUIRED TABLE, each still batched
    across every dimension that shares that table in one UNION ALL (per Task 1's
    benchmark: batching all dimensions into one query measured at 43ms versus 183ms
    issued one at a time) -- never one query per dimension. A dimension that forces
    `playback_events` (currently only `title_id`) is issued in its own set of queries
    rather than unioned with the `qoe_rollup_5m`-backed arms -- see
    `_group_dimensions_by_table`; putting an arm that scans the full raw-events window
    in the same statement as the cheap rollup arms would make every arm share that one
    query's memory budget for its whole lifetime.
    """
    if not dimensions:
        raise ValueError("dimensions must be non-empty")
    for dimension in dimensions:
        _validate_dimension(dimension)
    _validate_window(window)
    if not baseline_windows:
        raise ValueError("baseline_windows must be non-empty")
    for baseline_window in baseline_windows:
        _validate_window(baseline_window)

    groups = _group_dimensions_by_table(slice_, dimensions)

    window_by_dim: dict[str, list[dict[str, Any]]] = {}
    baseline_by_dim_per_window: dict[str, list[list[dict[str, Any]]]] = {}
    sql_by_dim: dict[str, str] = {}
    baseline_sqls_by_dim: dict[str, list[str]] = {}
    for group in groups:
        sql = _build_batched_split_sql(slice_, metric, group, window)
        baseline_sqls = [
            _build_batched_split_sql(slice_, metric, group, bw) for bw in baseline_windows
        ]
        window_result = await gateway.query(sql)
        baseline_results = [await gateway.query(baseline_sql) for baseline_sql in baseline_sqls]

        group_window_by_dim = _group_by_dim(window_result.rows)
        group_baseline_by_dim_per_window = [_group_by_dim(r.rows) for r in baseline_results]
        for dimension in group:
            window_by_dim[dimension] = group_window_by_dim.get(dimension, [])
            baseline_by_dim_per_window[dimension] = [
                by_dim.get(dimension, []) for by_dim in group_baseline_by_dim_per_window
            ]
            sql_by_dim[dimension] = sql
            baseline_sqls_by_dim[dimension] = baseline_sqls

    results: dict[str, SplitResult] = {}
    for dimension in dimensions:
        measurements = _rows_to_measurements_median_baseline(
            window_by_dim[dimension], baseline_by_dim_per_window[dimension]
        )
        # Provenance carries every baseline query that fed the median, not just one.
        baseline_sql = "\n-- UNION (median across windows) --\n".join(
            baseline_sqls_by_dim[dimension]
        )
        values = rank_contributions(
            measurements,
            dimension=dimension,
            higher_is_worse=metric.higher_is_worse,
            sql=sql_by_dim[dimension],
            baseline_sql=baseline_sql,
        )
        results[dimension] = SplitResult(
            dimension=dimension,
            values=values,
            informative=is_informative(measurements),
            sql=sql_by_dim[dimension],
            baseline_sql=baseline_sql,
        )
    return results
