"""The tool layer: analysis primitives exposed as ADK ``FunctionTool``s.

This is the entire boundary between judgement (the model's job) and measurement
(SQL's job). Gemini decides WHAT to investigate by calling these tools; the tools
decide WHAT IS TRUE by running SQL through the MCP gateway and returning grounded
numbers. A number must never originate in the model -- every tool result below
carries the SQL that produced it, so any figure that later shows up in a brief
without a matching logged query is mechanically detectable as a fabrication.

Seven tools, composed from the analysis primitives in ``continuity/analysis``:

* ``detect_anomalies``     -- wraps ``detect.detect``
* ``measure_slice``        -- wraps ``baseline.compute_baseline`` over a whole-window
  aggregate (see its docstring for why this is coarser than bucket-level detection)
* ``split_on_dimension``   -- wraps ``split.split_dimensions_median_baseline`` for ONE
  dimension
* ``split_all_dimensions`` -- wraps the same batched primitive for EVERY candidate
  dimension not already pinned in the slice, in one call -- the tool that replaces
  what used to be many separate ``split_on_dimension`` calls
* ``refine_incident_span`` -- composes ``detect.detect`` and
  ``continuity.analysis.cli._typical_and_peak_deviation`` (the same functions
  ``cli.refine_incident`` composes) to find a localized slice's TRUE onset/end and its
  TYPICAL/PEAK severity, instead of the diluted population-level span that first
  surfaced it
* ``find_changes``         -- wraps ``correlate.correlate_changes``
* ``quantify_impact``      -- wraps ``impact.compute_impact``, computing its own
  severity input (the TYPICAL deviation ratio, via ``detect`` +
  ``_typical_and_peak_deviation``) rather than accepting one as a parameter -- see that
  method's docstring for why severity must never be something the model supplies.

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
agent uses to get the seven ``FunctionTool``s for an ``LlmAgent``.

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
from continuity.analysis.cli import DEFAULT_REFINE_PADDING, _typical_and_peak_deviation
from continuity.analysis.correlate import (
    InvalidCorrelationWindowError,
    RankedChange,
    RejectedChange,
    correlate_changes,
)
from continuity.analysis.detect import AnomalyWindow, detect
from continuity.analysis.impact import Methodology, compute_impact
from continuity.analysis.metrics import Metric, get_metric
from continuity.analysis.slices import ALLOWED_DIMENSIONS, InvalidSliceError, Slice
from continuity.analysis.split import Contribution, split_dimensions_median_baseline
from continuity.analysis.walk import DEFAULT_MIN_LIFT, week_over_week_baseline_windows
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


def _meets_lift_gate(lift: float | None) -> bool:
    """`True` when `lift` clears `walk.DEFAULT_MIN_LIFT` -- the SAME gate the
    deterministic walker (`walk.py::choose_next_step`) uses to decide whether a
    candidate is worth descending into at all. Reused here, never reinvented, so
    this tool layer and the walker it is scored against can never disagree on what
    counts as "genuinely concentrated" versus "just big"."""
    return lift is not None and lift >= DEFAULT_MIN_LIFT


def _share_rank_key(lift: float | None, share: float | None) -> tuple[int, float, float]:
    """The walker's own two-part ranking, exactly as `choose_next_step` applies it:
    `lift` GATES (candidates that clear it sort before every candidate that does
    not), `share_of_deviation` RANKS -- descending -- among the ones that clear the
    gate. A candidate that fails the gate is never dropped, only sorted after every
    qualifying one; among the failing candidates themselves, the ordering is by
    `lift` descending (a `None` lift sorts last of all) so "how close it came" is
    still visible rather than arbitrary.
    """
    if _meets_lift_gate(lift):
        return (0, -(share if share is not None else 0.0), 0.0)
    return (1, 0.0 if lift is not None else 1.0, -(lift if lift is not None else 0.0))


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
        "meets_lift_gate": _meets_lift_gate(contribution.lift),
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
    """The seven analysis primitives, bound to one live `ClickHouseMCPGateway`.

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

        `lift` and `share_of_deviation` answer DIFFERENT questions, and ranking
        needs BOTH, in this order -- exactly the criterion the deterministic
        walker applies (see `continuity.analysis.walk.choose_next_step`):

        1. `lift` is a GATE, not a ranking. It answers "is this value genuinely
           worse than its size predicts, or is it just big?" Lift ~1.0 means the
           value explains exactly its own population share and is therefore just
           big, not causal. Below `continuity.analysis.walk.DEFAULT_MIN_LIFT`
           (~1.5 -- the SAME threshold the walker gates on, not a second one), or
           `None` (undefined -- no weight, no baseline, or a single usable
           value): do not descend into this value, no matter how large its raw
           share or contribution looks.
        2. `share_of_deviation` is the RANKING, among values that clear the lift
           gate. It answers "how much of the problem does this value account
           for?" Prefer the value with the HIGHEST share.

        THE TRAP, because it is subtle and worth stating explicitly: lift is
        SCALE-INVARIANT, so a value that is a proportional SUBSET of the true
        affected population -- e.g. one `os_version` out of several that make up
        a genuinely-affected `device_type` -- can carry the SAME lift as the
        broader, true value while its `share_of_deviation` is only a FRACTION of
        the broader value's (the subset is thinner, so its `weight_share` shrinks
        by the same factor lift would otherwise be inflated by -- they cancel).
        Ranking on lift alone picks the narrower subset and understates the
        incident. `values` below is therefore ranked by `share_of_deviation`
        DESCENDING among values whose lift clears the gate; values that do not
        clear it are still returned -- never hidden -- but sorted after every
        qualifying value, each carrying `meets_lift_gate: false`.

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
            top_n: How many values to return, ranked by `share_of_deviation`
                among the values whose lift clears the gate (see above), with
                non-qualifying values sorted after. Default `DEFAULT_TOP_N`.
                Lower this for a dimension you already expect to have many
                values (e.g. `title_id`).

        Returns:
            On success, a dict with:
            - `informative`: `False` when this dimension has at most one usable
              value here and therefore cannot explain anything by comparison --
              splitting further on it is pointless.
            - `values`: up to `top_n` values, ranked by `share_of_deviation`
              descending among values whose lift clears
              `walk.DEFAULT_MIN_LIFT`, then by `lift` descending among the rest
              (see the ranking above), each with `value`, `metric_value`,
              `baseline_value`, `weight`, `weight_share`, `contribution`,
              `share_of_deviation`, `lift`, `meets_lift_gate` (whether this
              value's lift cleared the gate -- `False` means it is not worth
              descending into, however high its share looks), and `note` (a
              human-readable explanation whenever a field is `None`, e.g.
              "absent from the baseline period -- new to this window").
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

        ranked_values = sorted(
            result.values, key=lambda c: _share_rank_key(c.lift, c.share_of_deviation)
        )
        values, omitted = _truncate(ranked_values, top_n=top_n)
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
    # split_all_dimensions
    # ------------------------------------------------------------------

    async def split_all_dimensions(
        self,
        slice_json: dict[str, str] | str,
        metric: str,
        window_start: str,
        window_end: str,
    ) -> dict[str, Any]:
        """Break `slice_json` down by EVERY candidate dimension not already pinned in it, all
        in ONE call -- CALL THIS FIRST when deciding what to descend into next, before
        reaching for `split_on_dimension` one dimension at a time.

        This is the same batched, single-UNION-ALL-per-window primitive
        `split_on_dimension` uses, just run once for every remaining dimension instead of
        once per dimension -- measured at ~43ms for all 8 dimensions versus ~183ms
        issuing them one at a time. Splitting device_type, app_version, cdn, isp,
        country, region, os_version and pop as 8 separate tool calls costs 8 turns of
        model reasoning and 8x the tokens for information this ONE call already
        contains. `title_id` IS included as a candidate here too -- it forces the
        raw-events table and its value column is numeric, but its arm of the batched
        query casts it (`toString(title_id)`) so it unions cleanly with every other
        dimension's string-valued arm. There are ~500 titles, so only its single
        TOP-ranked value (by contribution) is ever surfaced here, exactly like every
        other dimension -- the ~500 candidate values are never dumped into your
        context. Only fall back to `split_on_dimension` for a closer look at one
        dimension you have already identified as promising -- a larger `top_n` (e.g.
        to see more than one title beyond the top-ranked one), or to re-check it after
        refining the slice further.

        For each candidate dimension, only its TOP-ranked value (by contribution) is
        returned, and dimensions are ranked across each other exactly as
        `split_on_dimension` ranks values within one: `lift` GATES, `share_of_deviation`
        RANKS. `lift` answers "is this value genuinely worse than its size predicts, or
        is it just big?" -- below `continuity.analysis.walk.DEFAULT_MIN_LIFT` (~1.5, the
        SAME threshold the deterministic walker gates on), or `None`, do not descend
        into it regardless of anything else. `share_of_deviation` answers "how much of
        the problem does this value account for?" -- among dimensions whose top value
        clears the lift gate, prefer the one with the HIGHEST share.

        THE TRAP: lift is SCALE-INVARIANT, so a dimension whose top value is a
        proportional SUBSET of the true affected population (e.g. `os_version` sliced
        out of an already-affected `device_type`) can carry the SAME lift as the
        broader dimension's top value while explaining only a FRACTION of its
        share_of_deviation -- the subset is thinner, so its own weight_share shrinks by
        the same factor that would otherwise inflate its lift, and the two cancel.
        Ranking by lift alone picks the narrower subset and understates the incident.
        `dimensions` below is therefore ranked by `share_of_deviation` DESCENDING among
        dimensions whose top value clears the lift gate; dimensions that do not clear it
        are still returned -- never hidden -- but sorted after every qualifying one,
        each carrying `meets_lift_gate: false` to mark it as not worth descending into.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension -> value
                already fixed, or `{}`/`""` to split the whole population. Every
                dimension already a key here is automatically excluded from the result
                -- you cannot split a dimension you have already pinned.
            metric: One of the known metric names (`"rebuffer"`, `"startup"`, `"bitrate"`,
                `"errors"`). An unknown name comes back as `"invalid_input"`.
            window_start: ISO-8601 datetime, inclusive start of the window being
                investigated.
            window_end: ISO-8601 datetime, exclusive end of the window being
                investigated. Must be after `window_start`.

        Returns:
            On success, a dict with:
            - `dimensions`: one entry per candidate dimension (every dimension in
              `slices.ALLOWED_DIMENSIONS` -- the 8-dimension hierarchy plus `title_id`
              -- not already pinned in `slice_json`), sorted by `share_of_deviation`
              descending among dimensions whose top value's lift clears
              `walk.DEFAULT_MIN_LIFT`, then by `lift` descending among the rest (see the
              ranking above). Each entry carries `dimension`, `informative` (see
              `split_on_dimension`), `top_value`, `weight_share`, `share_of_deviation`,
              `lift`, `meets_lift_gate` (`False` means this dimension's top value is
              not worth descending into, however high its share looks), and `note`
              (why a field is `None`, or `"no data for this dimension in the window"`
              when the dimension had no rows at all here).
            - `sql`, `baseline_sql`: the one batched query (per window) behind every
              dimension's result -- all dimensions share the same two queries, so this
              is not repeated per dimension.

            If `slice_json` already pins every candidate dimension (including
            `title_id`), `dimensions` is an empty list with a `note` explaining there
            is nothing left to split -- a real finding, not an error.

            On failure, `{"error": ..., "error_type": ...}` -- see the module docstring.

            What this tool does NOT tell you: a value beyond each dimension's own top
            one -- call `split_on_dimension` with a larger `top_n` for that. It also
            does not tell you the TRUE onset/end of the incident on the slice you pick
            -- call `refine_incident_span` for that once localized.
        """
        try:
            slice_ = _parse_slice(slice_json)
            metric_obj = get_metric(metric)
            window = _parse_window(
                window_start, window_end, start_name="window_start", end_name="window_end"
            )
        except (InvalidSliceError, KeyError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        pinned = {*slice_.dimensions}
        candidate_dimensions = sorted(d for d in ALLOWED_DIMENSIONS if d not in pinned)
        if not candidate_dimensions:
            return {
                "slice": _slice_repr(slice_),
                "metric": metric,
                "window_start": window[0].isoformat(),
                "window_end": window[1].isoformat(),
                "dimensions": [],
                "note": "every dimension is already pinned in this slice -- nothing left to split",
            }

        baseline_windows = week_over_week_baseline_windows(window, DEFAULT_LOOKBACK_WEEKS)
        try:
            results = await split_dimensions_median_baseline(
                self._gateway,
                slice_=slice_,
                metric=metric_obj,
                dimensions=candidate_dimensions,
                window=window,
                baseline_windows=baseline_windows,
            )
        except InvalidSliceError as exc:
            return _error(str(exc), "invalid_input")
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        ranked: list[dict[str, Any]] = []
        for dimension in candidate_dimensions:
            result = results[dimension]
            top = result.values[0] if result.values else None
            top_lift = top.lift if top is not None else None
            ranked.append(
                {
                    "dimension": dimension,
                    "informative": result.informative,
                    "top_value": top.value if top is not None else None,
                    "weight_share": top.weight_share if top is not None else None,
                    "share_of_deviation": top.share_of_deviation if top is not None else None,
                    "lift": top_lift,
                    "meets_lift_gate": _meets_lift_gate(top_lift),
                    "note": (
                        top.note
                        if top is not None
                        else "no data for this dimension in the window"
                    ),
                }
            )

        ranked.sort(key=lambda entry: _share_rank_key(entry["lift"], entry["share_of_deviation"]))

        sample_result = next(iter(results.values()))
        return {
            "slice": _slice_repr(slice_),
            "metric": metric,
            "window_start": window[0].isoformat(),
            "window_end": window[1].isoformat(),
            "dimensions": ranked,
            "sql": sample_result.sql,
            "baseline_sql": sample_result.baseline_sql,
        }

    # ------------------------------------------------------------------
    # refine_incident_span
    # ------------------------------------------------------------------

    async def refine_incident_span(
        self,
        slice_json: dict[str, str] | str,
        metric: str,
        approx_start: str,
        approx_end: str,
    ) -> dict[str, Any]:
        """Re-detect `metric` directly on `slice_json` (never the whole population) to find
        this incident's TRUE onset/end and its TYPICAL severity -- call this AFTER you
        have localized a blast radius and BEFORE `quantify_impact`.

        `approx_start`/`approx_end` is typically the population-level window a
        population-level `detect_anomalies` call handed you. A fault scoped to a narrow
        slice is a diluted signal at population level -- it only breaches the detection
        threshold at its worst peaks, understating both the true span and the true
        severity. Re-running detection directly on the isolated slice sees a far
        stronger signal (measured on a real incident: z 23 on the isolated slice versus
        z 7 at population level) and recovers the shoulders before the first peak and
        after the last one that population-level detection misses entirely. This
        composes `continuity.analysis.detect.detect` and the same median-severity
        computation `continuity.analysis.cli.refine_incident` uses -- it does not
        reimplement either.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension -> value
                describing the blast radius to refine, e.g. `{"device_type": "roku",
                "app_version": "8.2.0"}`. Refining the whole population (`{}`/`""`) is
                allowed but rarely useful.
            metric: One of the known metric names (`"rebuffer"`, `"startup"`, `"bitrate"`,
                `"errors"`). An unknown name comes back as `"invalid_input"`.
            approx_start: ISO-8601 datetime, your current best guess at when the
                incident started (e.g. from `detect_anomalies` at population level).
            approx_end: ISO-8601 datetime, your current best guess at when it ended.
                Must be after `approx_start`.

        Returns:
            On success, a dict with:
            - `input_start`/`input_end`: the population-level span you started from,
              echoed back for reference.
            - `refined`: `True` if re-detecting directly on the isolated slice found a
              signal, `False` if it found nothing (a thin slice, or genuinely no
              incident here at this grain).
            - `start`/`end`: the TRUE onset/end on this slice when `refined` is `True`;
              otherwise EXACTLY `input_start`/`input_end`, unchanged -- refinement never
              silently returns something worse than the span you already had.
            - `buckets_breached`: how many 5-minute buckets crossed the anomaly
              threshold on this slice while re-detecting (0 when `refined` is `False`).
            - `typical_severity_ratio`: the MEDIAN `(actual - expected) / expected`
              across every anomalous bucket in the refined span -- what subscribers
              TYPICALLY experienced. `None` when `refined` is `False`.
            - `peak_severity_ratio`: the single worst bucket's own ratio, carried
              alongside the typical figure for transparency only -- `quantify_impact`
              computes severity itself from the typical figure, never from a number you
              pass it. `None` when `refined` is `False`.
            - `sql`: the query that scanned the isolated slice for anomaly windows.
            - `severity_sql`: the query behind `typical_severity_ratio` /
              `peak_severity_ratio` (present only when `refined` is `True`).
            - `note`: present only when `refined` is `False`, explaining why.

            On failure, `{"error": ..., "error_type": ...}` -- see the module docstring.

            What this tool does NOT do: compute business impact -- call
            `quantify_impact` with the refined `start`/`end` for that; it derives
            severity on its own, never from this tool's output.
        """
        try:
            slice_ = _parse_slice(slice_json)
            get_metric(metric)
            window = _parse_window(
                approx_start, approx_end, start_name="approx_start", end_name="approx_end"
            )
        except (InvalidSliceError, KeyError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        search_start = window[0] - DEFAULT_REFINE_PADDING
        search_end = window[1] + DEFAULT_REFINE_PADDING
        try:
            refine_detection = await detect(self._gateway, slice_, metric, search_start, search_end)
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        base = {
            "slice": _slice_repr(slice_),
            "metric": metric,
            "input_start": window[0].isoformat(),
            "input_end": window[1].isoformat(),
            "sql": refine_detection.sql,
        }

        if not refine_detection.windows:
            return {
                **base,
                "refined": False,
                "start": window[0].isoformat(),
                "end": window[1].isoformat(),
                "buckets_breached": 0,
                "typical_severity_ratio": None,
                "peak_severity_ratio": None,
                "note": (
                    "re-detecting directly on this slice found no anomaly window over "
                    f"{_fmt(search_start)}..{_fmt(search_end)} -- returning the input span "
                    "unchanged (thin slice, or genuinely no signal here at this grain), "
                    "never a worse guess than what you already had"
                ),
            }

        windows = refine_detection.windows
        try:
            typical_ratio, peak_ratio, severity_sql = await _typical_and_peak_deviation(
                self._gateway, slice_=slice_, metric_name=metric, windows=windows
            )
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        return {
            **base,
            "refined": True,
            "start": windows[0].start.isoformat(),
            "end": windows[-1].end.isoformat(),
            "buckets_breached": refine_detection.anomalous_buckets,
            "typical_severity_ratio": float(typical_ratio),
            "peak_severity_ratio": float(peak_ratio),
            "severity_sql": severity_sql,
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
        metric: str,
        window_start: str,
        window_end: str,
    ) -> dict[str, Any]:
        """Turn `slice_json`'s affected subscribers over `[window_start, window_end)` into an
        ARR-at-risk band, using a transparent, documented churn heuristic (not a trained model).

        THERE IS NO SEVERITY PARAMETER HERE -- this tool measures severity itself,
        directly from `slice_json` and the window, exactly as
        `continuity.analysis.cli.refine_incident` does: it re-detects anomalous buckets
        on this slice over this window and takes the MEDIAN `(actual - expected) /
        expected` across them (what subscribers TYPICALLY experienced), never the peak
        or a number you invent. `window_start`/`window_end` should be the slice's TRUE
        span -- call `refine_incident_span` first if you have not already, since a
        loosely-guessed window may contain no measurable anomaly here at all.

        Args:
            slice_json: A JSON object (or JSON-encoded string) of dimension ->
                value describing who was affected, or `{}`/`""` for the whole
                population.
            metric: One of the known metric names (`"rebuffer"`, `"startup"`,
                `"bitrate"`, `"errors"`) -- the metric that drove the incident, used
                to measure severity on this slice. An unknown name comes back as
                `"invalid_input"`.
            window_start: ISO-8601 datetime, inclusive start of the impact
                window.
            window_end: ISO-8601 datetime, exclusive end of the impact window.
                Must be after `window_start`.

        Returns:
            On success, a dict with:
            - `typical_severity_ratio`: the MEDIAN deviation ratio this tool measured
              and used as the churn heuristic's severity input.
            - `peak_severity_ratio`: the single worst bucket's own ratio, carried
              alongside for transparency only -- NOT what was used for the churn
              multiplier.
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
            - `sql`, `severity_sql`, `detect_sql`: every query behind these figures.

            `affected_subscribers = 0` is a real, legitimate finding (this slice
            genuinely affected nobody), not an error.

            If no anomalous bucket can be found on this slice in this window,
            severity cannot be measured here -- this returns `{"error": ...,
            "error_type": "no_data"}` naming `detect_anomalies`/`refine_incident_span`
            as the next step, rather than fabricating a severity of `0.0`.
        """
        try:
            slice_ = _parse_slice(slice_json)
            get_metric(metric)
            window = _parse_window(
                window_start, window_end, start_name="window_start", end_name="window_end"
            )
        except (InvalidSliceError, KeyError, ValueError) as exc:
            return _error(str(exc), "invalid_input")

        try:
            detection = await detect(self._gateway, slice_, metric, window[0], window[1])
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        if not detection.windows:
            return _error(
                f"no anomalous window found for this slice in {window_start}..{window_end} -- "
                "severity cannot be measured here. Call detect_anomalies or "
                "refine_incident_span first to find this slice's true anomalous span "
                "before quantifying impact.",
                "no_data",
            )

        try:
            typical_ratio, peak_ratio, severity_sql = await _typical_and_peak_deviation(
                self._gateway, slice_=slice_, metric_name=metric, windows=detection.windows
            )
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        try:
            result = await compute_impact(
                self._gateway, slice_=slice_, window=window, qoe_delta_ratio=typical_ratio
            )
        except QueryError as exc:
            return _error(str(exc), "infrastructure_failure")

        return {
            "slice": _slice_repr(slice_),
            "metric": metric,
            "window_start": window[0].isoformat(),
            "window_end": window[1].isoformat(),
            "typical_severity_ratio": float(typical_ratio),
            "peak_severity_ratio": float(peak_ratio),
            "affected_subscribers": result.affected_subscribers,
            "arr_at_risk_low": str(result.arr_at_risk_low),
            "arr_at_risk_expected": str(result.arr_at_risk_expected),
            "arr_at_risk_high": str(result.arr_at_risk_high),
            "methodology": _methodology_dict(result.methodology),
            "sql": result.sql,
            "severity_sql": severity_sql,
            "detect_sql": detection.sql,
        }

    # ------------------------------------------------------------------
    # ADK wiring
    # ------------------------------------------------------------------

    def function_tools(self) -> list[FunctionTool]:
        """The seven bound methods above, wrapped as ADK `FunctionTool`s ready to
        hand to an `LlmAgent`. Construction only -- this never calls a model."""
        return [
            FunctionTool(self.detect_anomalies),
            FunctionTool(self.measure_slice),
            FunctionTool(self.split_on_dimension),
            FunctionTool(self.split_all_dimensions),
            FunctionTool(self.refine_incident_span),
            FunctionTool(self.find_changes),
            FunctionTool(self.quantify_impact),
        ]


def build_function_tools(
    gateway: ClickHouseMCPGateway, *, default_top_n: int = DEFAULT_TOP_N
) -> list[FunctionTool]:
    """Bind `gateway` and return the seven analysis-primitive tools as ADK
    `FunctionTool`s -- the one-line entry point sub-project 3's agent uses."""
    return AnalysisTools(gateway, default_top_n=default_top_n).function_tools()
