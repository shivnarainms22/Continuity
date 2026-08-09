"""The tool layer: analysis primitives exposed as ADK ``FunctionTool``s.

This is the entire boundary between judgement (the model's job) and measurement
(SQL's job). Gemini decides WHAT to investigate by calling these tools; the tools
decide WHAT IS TRUE by running SQL through the MCP gateway and returning grounded
numbers. A number must never originate in the model -- every tool result below
carries the SQL that produced it, so any figure that later shows up in a brief
without a matching logged query is mechanically detectable as a fabrication.

Five tools, one per analysis primitive from ``continuity/analysis``:

* ``detect_anomalies``  -- wraps ``detect.detect``
* ``measure_slice``     -- wraps ``baseline.compute_baseline`` over a whole-window
  aggregate (see its docstring for why this is coarser than bucket-level detection)
* ``split_on_dimension`` -- wraps ``split.split_dimensions_median_baseline``
* ``find_changes``      -- wraps ``correlate.correlate_changes``
* ``quantify_impact``   -- wraps ``impact.compute_impact``

This module makes **no LLM calls** and imports nothing outside the analysis core,
the gateway and ``google.adk.tools`` (for the ``FunctionTool`` wrapper type itself,
which is construction-only -- it never talks to a model). It is pure wrapping:
input validation, a thin SQL query for ``measure_slice``'s aggregate built from
already-validated ``Slice``/``Metric`` objects, and dict-shaped, JSON-serialisable
results.

Every tool method lives on ``AnalysisTools``, bound to one live gateway. The model
never sees ``gateway`` as an argument -- it is a bound instance attribute, not a
parameter of the wrapped function, so ADK's schema derivation (which reads the
function's signature) never exposes it and never asks the model to supply one.
``build_function_tools(gateway)`` is the one-line entry point sub-project 3's
agent uses to get the five ``FunctionTool``s for an ``LlmAgent``.

ERROR SHAPE. Every tool method returns a plain dict. On success, the dict carries
the measurement plus its SQL. On failure, it carries ``{"error": <message>,
"error_type": <kind>}`` instead of raising -- a Gemini tool call cannot read a
Python traceback, so a bad dimension name or an empty window must come back as
something the model can act on, never an opaque exception. ``error_type`` is one
of:

* ``"invalid_input"``          -- the model's own arguments were bad (unknown
  dimension, malformed slice JSON, non-ISO datetime, end before start, negative
  severity ratio). The message names the valid alternative where there is one
  (e.g. every allowed dimension) so the model can immediately retry correctly.
* ``"no_data"``                -- the arguments were valid but there is nothing to
  measure (e.g. a slice with zero traffic in the window). This is a real,
  actionable finding, not a bug -- but it must never be silently returned as a
  zero or a fabricated value.
* ``"infrastructure_failure"`` -- the ClickHouse query itself failed
  (``QueryError`` from the gateway: connection lost, malformed SQL, server
  error). This is NEVER collapsed into ``"no_data"``: a broken pipe and a slice
  with no traffic look identical to a caller that only checks "did I get
  results", and conflating them is exactly the failure mode this project has
  fought all day. A truly unexpected exception (a bug in this module, not an
  expected input or infra failure) is deliberately NOT caught here -- it
  propagates, so it shows up in logs and tests instead of being silently
  swallowed as a third kind of "error" the model would just shrug off.

COMPACTNESS. Every tool that can return an unbounded number of rows (dimension
values, anomaly windows, change candidates, rejections) is truncated to the top
``top_n`` (default ``DEFAULT_TOP_N`` = 8) by the metric that matters for that
tool, with an accompanying ``*_omitted`` count -- the model always knows it is
looking at a bounded, ranked view, never a silent partial dump.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from google.adk.tools import FunctionTool

from continuity.analysis.baseline import DEFAULT_LOOKBACK_WEEKS, Baseline, compute_baseline
from continuity.analysis.correlate import (
    InvalidCorrelationWindowError,
    RankedChange,
    RejectedChange,
    correlate_changes,
)
from continuity.analysis.detect import AnomalyWindow, detect
from continuity.analysis.impact import Methodology, compute_impact
from continuity.analysis.metrics import Metric, get_metric
from continuity.analysis.slices import InvalidSliceError, Slice
from continuity.analysis.split import Contribution, split_dimensions_median_baseline
from continuity.analysis.walk import week_over_week_baseline_windows
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway, QueryError

DEFAULT_TOP_N = 8

# find_changes takes only a slice and an onset (no explicit end), so disconfirming
# evidence for each candidate is measured over this fixed window following onset --
# long enough to see a real effect on siblings, short enough to stay inside the
# incident rather than blurring into whatever comes after it.
_FIND_CHANGES_EVIDENCE_WINDOW = timedelta(hours=1)

_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _error(message: str, error_type: str) -> dict[str, Any]:
    return {"error": message, "error_type": error_type}


def _truncate[T](items: Sequence[T], *, top_n: int) -> tuple[list[T], int]:
    """The first `top_n` items (caller must have already sorted for relevance),
    plus how many were left out -- the compactness contract every tool honours."""
    if top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")
    kept = list(items[:top_n])
    omitted = max(0, len(items) - top_n)
    return kept, omitted


def _parse_slice(slice_json: Mapping[str, Any] | str | None) -> Slice:
    """Parse a tool-supplied slice into a validated `Slice`.

    Accepts a plain JSON object (dict) of dimension -> value, a JSON-encoded
    string of the same, or `None`/empty for the whole population. Raises
    `InvalidSliceError` -- caught by every tool method and turned into a
    structured `"invalid_input"` result, never a bare traceback -- on malformed
    JSON, a non-object payload, or a dimension name that is not one of
    `continuity.analysis.slices.ALLOWED_DIMENSIONS` (the message names every
    valid dimension, since a model WILL guess a plausible-sounding wrong one).
    """
    if slice_json is None:
        return Slice()
    if isinstance(slice_json, str):
        if slice_json.strip() == "":
            return Slice()
        try:
            payload = json.loads(slice_json)
        except json.JSONDecodeError as exc:
            raise InvalidSliceError(f"slice is not valid JSON: {exc}") from exc
    elif isinstance(slice_json, Mapping):
        payload = slice_json
    else:
        raise InvalidSliceError(
            "slice must be a JSON object or a JSON-encoded string of dimension -> "
            f"value pairs, got {type(slice_json).__name__}"
        )
    if not isinstance(payload, Mapping):
        raise InvalidSliceError(f"slice must decode to a JSON object, got {type(payload).__name__}")

    slice_ = Slice()
    for dimension, value in payload.items():
        slice_ = slice_.refine(dimension, value)
    return slice_


def _slice_repr(slice_: Slice) -> dict[str, str]:
    return dict(slice_.predicates)


def _parse_datetime(value: str, *, field_name: str) -> datetime:
    """Parse an ISO-8601 datetime string. Naive throughout the rest of the
    codebase (ClickHouse timestamps carry no timezone), so a timezone-aware
    string (as `data/ground_truth.json` uses) has its offset dropped rather
    than rejected -- the caller almost always means "this instant in UTC"."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty ISO-8601 datetime string, got {value!r}"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name}={value!r} is not a valid ISO-8601 datetime") from exc
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def _parse_window(
    start: str, end: str, *, start_name: str = "start", end_name: str = "end"
) -> tuple[datetime, datetime]:
    start_dt = _parse_datetime(start, field_name=start_name)
    end_dt = _parse_datetime(end, field_name=end_name)
    if not start_dt < end_dt:
        raise ValueError(f"{start_name} ({start}) must be before {end_name} ({end})")
    return start_dt, end_dt


def _fmt(dt: datetime) -> str:
    return dt.strftime(_DATETIME_FORMAT)


def _aggregate_sql(slice_: Slice, metric: Metric, window: tuple[datetime, datetime]) -> str:
    """The single-row aggregate of `metric` over `slice_` and `window` -- the same
    table/column choice `split.py` and `detect.py` use (raw events only when the
    slice forces it), just without their `GROUP BY`."""
    start, end = window
    raw_events = slice_.requires_raw_events
    table = slice_.required_table
    time_col = "event_time" if raw_events else "bucket"
    expr = metric.sql_for(raw_events=raw_events)
    return (
        f"SELECT {expr} AS value FROM {table} WHERE {slice_.where_sql()} "
        f"AND {time_col} >= '{_fmt(start)}' AND {time_col} < '{_fmt(end)}'"
    )


def _window_dict(window: AnomalyWindow) -> dict[str, Any]:
    return {
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "span_minutes": (window.end - window.start).total_seconds() / 60,
        "peak_z": window.peak_z,
        "peak_value": window.peak_value,
        "expected_at_peak": window.expected_at_peak,
        "bucket_count": window.bucket_count,
    }


def _contribution_dict(contribution: Contribution) -> dict[str, Any]:
    return {
        "value": contribution.value,
        "metric_value": contribution.metric_value,
        "baseline_value": contribution.baseline_value,
        "weight": contribution.weight,
        "weight_share": contribution.weight_share,
        "contribution": contribution.contribution,
        "share_of_deviation": contribution.share_of_deviation,
        "lift": contribution.lift,
        "note": contribution.note,
    }


def _candidate_dict(candidate: RankedChange) -> dict[str, Any]:
    evidence = candidate.disconfirming_evidence
    return {
        "change_id": candidate.change_id,
        "changed_at": candidate.changed_at.isoformat(),
        "change_type": candidate.change_type,
        "component": candidate.component,
        "description": candidate.description,
        "dimension_key": candidate.dimension_key,
        "dimension_value": candidate.dimension_value,
        "score": candidate.score,
        "temporal_delta_seconds": candidate.temporal_delta.total_seconds(),
        "dimensional_overlap": candidate.dimensional_overlap,
        "disconfirming_evidence": {
            "sibling_dimension": evidence.sibling_dimension,
            "siblings_checked": evidence.siblings_checked,
            "siblings_degraded": evidence.siblings_degraded,
            "siblings_not_degraded": evidence.siblings_not_degraded,
            "note": evidence.note,
        },
    }


def _rejected_dict(rejected: RejectedChange) -> dict[str, Any]:
    return {
        "change_id": rejected.change_id,
        "changed_at": rejected.changed_at.isoformat(),
        "change_type": rejected.change_type,
        "component": rejected.component,
        "description": rejected.description,
        "dimension_key": rejected.dimension_key,
        "dimension_value": rejected.dimension_value,
        "reason": rejected.reason,
    }


def _methodology_dict(methodology: Methodology) -> dict[str, Any]:
    return {
        "base_monthly_churn": str(methodology.base_monthly_churn),
        "base_churn_variation": str(methodology.base_churn_variation),
        "tenure_multiplier_at_signup": str(methodology.tenure_multiplier_at_signup),
        "tenure_multiplier_floor": str(methodology.tenure_multiplier_floor),
        "tenure_half_life_days": str(methodology.tenure_half_life_days),
        "severity_multiplier_max": str(methodology.severity_multiplier_max),
        "severity_sessions_half_saturation": str(methodology.severity_sessions_half_saturation),
        "severity_qoe_half_saturation": str(methodology.severity_qoe_half_saturation),
        "churn_risk_ceiling": str(methodology.churn_risk_ceiling),
        "qoe_delta_ratio": str(methodology.qoe_delta_ratio),
        "affected_subscriber_count": methodology.affected_subscriber_count,
        "notes": methodology.notes,
    }


class AnalysisTools:
    """The five analysis primitives, bound to one live `ClickHouseMCPGateway`.

    Construct once per investigation (or per agent session) and either call the
    methods directly, or hand `function_tools()` to an ADK `LlmAgent`. `gateway`
    is an instance attribute, never a parameter of the wrapped methods -- the
    model can never be asked to supply one, and never sees it in a tool schema.
    """

    def __init__(
        self, gateway: ClickHouseMCPGateway, *, default_top_n: int = DEFAULT_TOP_N
    ) -> None:
        self._gateway = gateway
        self._default_top_n = default_top_n

    async def _scalar_metric(self, sql: str) -> float | None:
        """Run `sql` (built by `_aggregate_sql`, always exactly one row, one
        column) and return its value, or `None` if the aggregate had nothing to
        aggregate (e.g. `nullIf` on a zero denominator)."""
        result = await self._gateway.query(sql)
        if not result.rows:
            return None
        value = result.rows[0].get("value")
        return None if value is None else float(value)

    # ------------------------------------------------------------------
    # detect_anomalies
    # ------------------------------------------------------------------

    async def detect_anomalies(
        self,
        slice_json: dict[str, str] | str,
        metric: str,
        start: str,
        end: str,
    ) -> dict[str, Any]:
        """Scan `slice_json` for sustained anomalous periods in `metric` over [start, end).

        Splits [start, end) into 5-minute buckets and compares each one against
        its own weekday/time-of-day baseline from the preceding weeks (robust
        median/MAD, not mean/sigma, so one historical incident does not blind the
        detector to the next). A lone anomalous bucket is noise; only a run of
        several consecutive anomalous buckets is reported as an anomaly window.
        Use this FIRST, before drilling down, to find out WHEN something went
        wrong and how bad the worst moment was.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension ->
                value, e.g. `{"device_type": "roku"}`, or `{}`/`""` for the whole
                population. Unknown dimension names come back as an
                `"invalid_input"` error naming every valid dimension.
            metric: One of the known metric names (`"rebuffer"`, `"startup"`,
                `"bitrate"`, `"errors"`). An unknown name comes back as an
                `"invalid_input"` error naming every known metric.
            start: ISO-8601 datetime, inclusive start of the scan window.
            end: ISO-8601 datetime, exclusive end of the scan window. Must be
                after `start`.

        Returns:
            On success, a dict with:
            - `windows`: up to `DEFAULT_TOP_N` anomaly windows, ranked by the
              size of their `peak_z` (largest deviation first), each carrying
              `start`, `end`, `span_minutes`, `peak_z` (robust standard
              deviations from baseline at the worst bucket), `peak_value`,
              `expected_at_peak` (what the baseline predicted there), and
              `bucket_count`.
            - `windows_omitted`: how many further anomaly windows were left out
              of `windows` by the top-N cutoff -- 0 means nothing was omitted.
            - `total_buckets`, `anomalous_buckets`, `unknown_buckets`,
              `unknown_fraction`: bucket accounting for the WHOLE scan range, not
              just the reported windows -- a high `unknown_fraction` means most
              of the range could not be measured at all (too little history),
              so an empty `windows` list there means "unmeasurable", not "quiet".
              A low `unknown_fraction` with an empty `windows` list genuinely
              means quiet.
            - `sql`: the query that fetched the series this was computed from.

            On failure, `{"error": ..., "error_type": ...}` -- see the module
            docstring for `error_type` values.

            What this tool does NOT tell you: WHY a window is anomalous, or
            which sub-population within `slice_json` is responsible -- call
            `split_on_dimension` on the window it reports to find that out.
        """
        try:
            slice_ = _parse_slice(slice_json)
            window = _parse_window(start, end)
        except (InvalidSliceError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        try:
            result = await detect(self._gateway, slice_, metric, window[0], window[1])
        except KeyError as exc:
            return _error(str(exc), "invalid_input")
        except ValueError as exc:
            return _error(str(exc), "invalid_input")
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        ranked_windows = sorted(result.windows, key=lambda w: -abs(w.peak_z))
        kept, omitted = _truncate(ranked_windows, top_n=self._default_top_n)
        return {
            "slice": _slice_repr(slice_),
            "metric": metric,
            "start": window[0].isoformat(),
            "end": window[1].isoformat(),
            "windows": [_window_dict(w) for w in kept],
            "windows_omitted": omitted,
            "total_buckets": result.total_buckets,
            "anomalous_buckets": result.anomalous_buckets,
            "unknown_buckets": result.unknown_buckets,
            "unknown_fraction": result.unknown_fraction,
            "sql": result.sql,
        }

    # ------------------------------------------------------------------
    # measure_slice
    # ------------------------------------------------------------------

    async def measure_slice(
        self,
        slice_json: dict[str, str] | str,
        metric: str,
        window_start: str,
        window_end: str,
    ) -> dict[str, Any]:
        """Measure `metric` for `slice_json`, aggregated over one whole window, against
        its own weekday/time-of-day baseline.

        This is coarser than `detect_anomalies`: it does not look bucket-by-bucket,
        it collapses the whole `[window_start, window_end)` span into ONE number
        (the same ratio-of-sums `metrics.py` defines) and compares that single
        number against the same aggregate computed over the same span on each of
        the preceding `DEFAULT_LOOKBACK_WEEKS` weeks (same weekday, same
        time-of-day). Use this to get one clean, comparable number for a
        candidate slice and window -- e.g. after `split_on_dimension` suggests a
        value, confirm how anomalous that value's own window really is.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension ->
                value, or `{}`/`""` for the whole population.
            metric: One of the known metric names (`"rebuffer"`, `"startup"`,
                `"bitrate"`, `"errors"`).
            window_start: ISO-8601 datetime, inclusive start of the window to
                measure.
            window_end: ISO-8601 datetime, exclusive end of the window to
                measure. Must be after `window_start`.

        Returns:
            On success, a dict with:
            - `value`: the metric's actual aggregate value over the window.
            - `baseline`: the median of the same aggregate over the comparison
              weeks -- what "normal" looks like for this slice at this
              time-of-day.
            - `spread`: the robust (MAD-based) spread of the comparison values.
            - `z`: `(value - baseline) / spread`, robust standard deviations.
              Positive is not automatically "bad" -- check the metric's
              direction (`metrics.py`'s `higher_is_worse`) before concluding
              anything from the sign.
            - `sample_size`: how many of the comparison weeks had usable data
              (out of `DEFAULT_LOOKBACK_WEEKS`). `status` is `"insufficient_data"`
              when this is too small to trust `z`, in which case `baseline`,
              `spread` and `z` are all `None` -- never a fabricated number.
            - `sql`, `baseline_sql`: the queries behind `value` and the
              comparison-week values respectively.

            If the slice/window combination has literally no data to aggregate
            (e.g. a thin slice with zero traffic in that window), this returns
            `{"error": "no data for ...", "error_type": "no_data"}` rather than
            a fabricated `0`. A genuine ClickHouse failure comes back as
            `"infrastructure_failure"` instead -- never silently folded into
            `"no_data"`.

            What this tool does NOT tell you: WHICH sub-population within
            `slice_json` drove the number -- call `split_on_dimension` for that.
        """
        try:
            slice_ = _parse_slice(slice_json)
            metric_obj = get_metric(metric)
            window = _parse_window(
                window_start, window_end, start_name="window_start", end_name="window_end"
            )
        except (InvalidSliceError, KeyError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        sql = _aggregate_sql(slice_, metric_obj, window)
        try:
            actual = await self._scalar_metric(sql)
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        if actual is None:
            # Short-circuit before spending 4 more queries on comparison weeks --
            # there is nothing to compare a missing actual against anyway.
            return _error(
                f"no data for this slice/metric in the window {window_start}..{window_end}",
                "no_data",
            )

        baseline_windows = week_over_week_baseline_windows(window, DEFAULT_LOOKBACK_WEEKS)
        baseline_sqls = [_aggregate_sql(slice_, metric_obj, bw) for bw in baseline_windows]
        try:
            comparison_values = [await self._scalar_metric(bsql) for bsql in baseline_sqls]
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        baseline: Baseline = compute_baseline(actual, comparison_values)
        return {
            "slice": _slice_repr(slice_),
            "metric": metric,
            "window_start": window[0].isoformat(),
            "window_end": window[1].isoformat(),
            "value": actual,
            "baseline": baseline.expected,
            "spread": baseline.spread,
            "z": baseline.z,
            "sample_size": baseline.sample_size,
            "status": baseline.status.value,
            "sql": sql,
            "baseline_sql": "\n-- comparison window --\n".join(baseline_sqls),
        }

    # ------------------------------------------------------------------
    # split_on_dimension
    # ------------------------------------------------------------------

    async def split_on_dimension(
        self,
        slice_json: dict[str, str] | str,
        metric: str,
        dimension: str,
        window_start: str,
        window_end: str,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """Break `slice_json` down by `dimension` and rank each value by how much of the
        slice's deviation in `metric` it explains -- this is THE tool for deciding what to
        descend into next.

        For every value of `dimension` (e.g. every `device_type`), this computes
        that value's own share of the current population, its own deviation from
        its own baseline, and from those two numbers a `lift`:

            lift = share_of_deviation / weight_share

        `share_of_deviation` is how much of the SLICE's total deviation this one
        value accounts for. `weight_share` is how much of the SLICE's total
        traffic this one value accounts for. LIFT IS THE SIGNAL TO ACT ON, not
        `share_of_deviation` alone: a value can carry a huge share of the
        deviation simply because it is a huge share of the population, while
        being no more broken than any other value. Lift divides that population-
        size effect out. Concretely: lift 1.0 means this value explains exactly
        its own share of the population and is therefore just big, not causal;
        lift above ~1.5 means the deviation is genuinely concentrated here --
        this value is meaningfully worse than its size alone would predict, and
        is worth descending into. Lift below 1.0, or `None` (undefined -- no
        weight, no baseline, or a single usable value), means this value is NOT
        where the problem lives, however large its raw share or contribution
        looks; do not chase it.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension ->
                value already fixed (e.g. `{"device_type": "roku"}` to split the
                roku population further), or `{}`/`""` to split the whole
                population. `dimension` must not already be a key here.
            metric: One of the known metric names (`"rebuffer"`, `"startup"`,
                `"bitrate"`, `"errors"`).
            dimension: The dimension to split on (e.g. `"device_type"`,
                `"app_version"`, `"cdn"`, `"pop"`, `"isp"`, `"title_id"`). An
                unknown name comes back as `"invalid_input"`, naming every valid
                dimension -- guess again with one of those, do not retry the same
                name.
            window_start: ISO-8601 datetime, inclusive start of the window being
                investigated.
            window_end: ISO-8601 datetime, exclusive end of the window being
                investigated. Must be after `window_start`.
            top_n: How many values to return, ranked by contribution (largest
                first). Default `DEFAULT_TOP_N`. Lower this for a dimension you
                already expect to have many values (e.g. `title_id`).

        Returns:
            On success, a dict with:
            - `informative`: `False` when this dimension has at most one usable
              value here and therefore cannot explain anything by comparison --
              splitting further on it is pointless.
            - `values`: up to `top_n` values, ranked by contribution, each with
              `value`, `metric_value`, `baseline_value`, `weight`,
              `weight_share`, `contribution`, `share_of_deviation`, `lift`, and
              `note` (a human-readable explanation whenever a field is `None`,
              e.g. "absent from the baseline period -- new to this window").
            - `values_omitted`: how many further values were left out by the
              `top_n` cutoff. 0 means nothing was omitted -- you are seeing
              every value this dimension has.
            - `sql`, `baseline_sql`: the queries behind the window and baseline
              measurements respectively.

            If the dimension has no rows at all for this slice/window, this
            returns `{"error": ..., "error_type": "no_data"}` instead of an
            empty, misleadingly "successful" split.

            What this tool does NOT tell you: whether the SAME dimension's split
            would look the same in a DIFFERENT, already-fixed sub-slice -- lift
            is only ever measured within the slice you passed in. It also does
            not tell you WHAT CHANGED to cause the deviation -- call
            `find_changes` for that once you have a slice you trust.
        """
        try:
            slice_ = _parse_slice(slice_json)
            metric_obj = get_metric(metric)
            window = _parse_window(
                window_start, window_end, start_name="window_start", end_name="window_end"
            )
        except (InvalidSliceError, KeyError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        baseline_windows = week_over_week_baseline_windows(window, DEFAULT_LOOKBACK_WEEKS)
        try:
            results = await split_dimensions_median_baseline(
                self._gateway,
                slice_=slice_,
                metric=metric_obj,
                dimensions=[dimension],
                window=window,
                baseline_windows=baseline_windows,
            )
        except InvalidSliceError as exc:
            return _error(str(exc), "invalid_input")
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        result = results[dimension]
        if not result.values:
            return _error(
                f"no data for dimension {dimension!r} in the window "
                f"{window_start}..{window_end}",
                "no_data",
            )

        values, omitted = _truncate(result.values, top_n=top_n)
        return {
            "slice": _slice_repr(slice_),
            "metric": metric,
            "dimension": dimension,
            "window_start": window[0].isoformat(),
            "window_end": window[1].isoformat(),
            "informative": result.informative,
            "values": [_contribution_dict(c) for c in values],
            "values_omitted": omitted,
            "sql": result.sql,
            "baseline_sql": result.baseline_sql,
        }

    # ------------------------------------------------------------------
    # find_changes
    # ------------------------------------------------------------------

    async def find_changes(self, slice_json: dict[str, str] | str, onset: str) -> dict[str, Any]:
        """Rank `change_log` entries as plausible causes of an anomaly that started at `onset`
        in `slice_json`, WITH the disconfirming evidence for each and the candidates ruled out.

        A change strictly after `onset` is never a cause and is REJECTED outright,
        never merely down-ranked -- causality only runs forward in time. A change
        that touches a dimension `slice_json` pins to a DIFFERENT value is also
        rejected: it could not have caused THIS slice's problem. Everything else
        is scored by how recent it is and how well its own dimension overlaps
        `slice_json`'s predicates.

        Every accepted candidate also carries disconfirming evidence: whether the
        same change touched OTHER values of some dimension `slice_json` also
        predicates on, and whether those other values ALSO degraded in the hour
        after `onset`. A change that shipped everywhere but only this slice
        degraded is much stronger evidence than one that shipped everywhere and
        everything degraded -- READ `disconfirming_evidence.note` before trusting
        a candidate; do not just take the top score.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension ->
                value describing the blast radius to find causes for, or
                `{}`/`""` for the whole population.
            onset: ISO-8601 datetime -- when the anomaly began. Disconfirming
                evidence is checked over the hour following this instant.

        Returns:
            On success, a dict with:
            - `candidates`: up to `DEFAULT_TOP_N` accepted candidates, ranked by
              score (highest first), each with `change_id`, `changed_at`,
              `change_type`, `component`, `description`, `dimension_key`,
              `dimension_value`, `score`, `temporal_delta_seconds` (how long
              before `onset`), `dimensional_overlap`, and
              `disconfirming_evidence` (`sibling_dimension`, `siblings_checked`,
              `siblings_degraded`, `siblings_not_degraded`, `note`).
            - `candidates_omitted`: how many further accepted candidates were
              left out by the top-N cutoff.
            - `rejected`: up to `DEFAULT_TOP_N` changes that were considered and
              ruled out, each with the same identifying fields plus `reason` --
              show these in a brief as "we looked at these and ruled them out",
              not just the ones that scored well.
            - `rejected_omitted`: how many further rejections were left out.
            - `sql`: the query that fetched the candidate change_log rows.

            An empty `candidates` list with an empty `rejected` list means there
            were no changes at all near `onset` -- a real finding (nothing in
            `change_log` explains this), not an error.

            What this tool does NOT tell you: whether a candidate ACTUALLY caused
            the anomaly -- it ranks plausibility from timing and dimensional
            overlap only. Weighing the disconfirming evidence and deciding
            confidence is your job, not this tool's.
        """
        try:
            blast_radius = _parse_slice(slice_json)
            onset_dt = _parse_datetime(onset, field_name="onset")
        except (InvalidSliceError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        anomaly_window = (onset_dt, onset_dt + _FIND_CHANGES_EVIDENCE_WINDOW)
        try:
            result = await correlate_changes(
                self._gateway, blast_radius=blast_radius, anomaly_window=anomaly_window
            )
        except InvalidCorrelationWindowError as exc:
            return _error(str(exc), "invalid_input")
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        candidates, candidates_omitted = _truncate(result.candidates, top_n=self._default_top_n)
        rejected, rejected_omitted = _truncate(result.rejected, top_n=self._default_top_n)
        return {
            "slice": _slice_repr(blast_radius),
            "onset": onset_dt.isoformat(),
            "candidates": [_candidate_dict(c) for c in candidates],
            "candidates_omitted": candidates_omitted,
            "rejected": [_rejected_dict(r) for r in rejected],
            "rejected_omitted": rejected_omitted,
            "sql": result.sql,
        }

    # ------------------------------------------------------------------
    # quantify_impact
    # ------------------------------------------------------------------

    async def quantify_impact(
        self,
        slice_json: dict[str, str] | str,
        window_start: str,
        window_end: str,
        severity_ratio: float,
    ) -> dict[str, Any]:
        """Turn `slice_json`'s affected subscribers over `[window_start, window_end)` into an
        ARR-at-risk band, using a transparent, documented churn heuristic (not a trained model).

        `severity_ratio` -- how much worse the driving metric got vs its own
        baseline, e.g. `(actual - baseline) / baseline` from `measure_slice` or
        `split_on_dimension` -- feeds a saturating severity multiplier; it does
        not change WHICH subscribers count as affected (that is purely
        `slice_json` and the window), only how severely each affected
        subscriber's churn risk is scored.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension ->
                value describing who was affected, or `{}`/`""` for the whole
                population.
            window_start: ISO-8601 datetime, inclusive start of the impact
                window.
            window_end: ISO-8601 datetime, exclusive end of the impact window.
                Must be after `window_start`.
            severity_ratio: Non-negative. How much worse the metric got vs
                baseline; `0.0` if you have no better estimate.

        Returns:
            On success, a dict with:
            - `affected_subscribers`: count of distinct subscribers with at
              least one session in `slice_json` during the window.
            - `arr_at_risk_low`/`_expected`/`_high`: a BAND, not a point
              estimate, as decimal strings (exact, never a rounded float) --
              always report the band, never just `_expected`.
            - `methodology`: every coefficient the heuristic used (all stated,
              documented ASSUMPTIONS -- there is no churn-event ground truth in
              this dataset to calibrate against) plus free-text `notes`. Surface
              this alongside the number, not just the number -- that is the
              whole point of it being data rather than a docstring.
            - `sql`: the query that found the affected subscribers.

            `affected_subscribers = 0` is a real, legitimate finding (this slice
            genuinely affected nobody), not an error.

            What this tool does NOT tell you: whether `severity_ratio` you
            supplied is itself correct -- that number must come from
            `measure_slice` or `split_on_dimension`, never invented.
        """
        try:
            slice_ = _parse_slice(slice_json)
            window = _parse_window(
                window_start, window_end, start_name="window_start", end_name="window_end"
            )
            if severity_ratio < 0:
                raise ValueError(f"severity_ratio must be >= 0, got {severity_ratio}")
        except (InvalidSliceError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        try:
            result = await compute_impact(
                self._gateway, slice_=slice_, window=window, qoe_delta_ratio=severity_ratio
            )
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        return {
            "slice": _slice_repr(slice_),
            "window_start": window[0].isoformat(),
            "window_end": window[1].isoformat(),
            "severity_ratio": severity_ratio,
            "affected_subscribers": result.affected_subscribers,
            "arr_at_risk_low": str(result.arr_at_risk_low),
            "arr_at_risk_expected": str(result.arr_at_risk_expected),
            "arr_at_risk_high": str(result.arr_at_risk_high),
            "methodology": _methodology_dict(result.methodology),
            "sql": result.sql,
        }

    # ------------------------------------------------------------------
    # ADK wiring
    # ------------------------------------------------------------------

    def function_tools(self) -> list[FunctionTool]:
        """The five bound methods above, wrapped as ADK `FunctionTool`s ready to
        hand to an `LlmAgent`. Construction only -- this never calls a model."""
        return [
            FunctionTool(self.detect_anomalies),
            FunctionTool(self.measure_slice),
            FunctionTool(self.split_on_dimension),
            FunctionTool(self.find_changes),
            FunctionTool(self.quantify_impact),
        ]


def build_function_tools(
    gateway: ClickHouseMCPGateway, *, default_top_n: int = DEFAULT_TOP_N
) -> list[FunctionTool]:
    """Bind `gateway` and return the five analysis-primitive tools as ADK
    `FunctionTool`s -- the one-line entry point sub-project 3's agent uses."""
    return AnalysisTools(gateway, default_top_n=default_top_n).function_tools()
