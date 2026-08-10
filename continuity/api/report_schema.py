"""JSON-serializable view of an `InvestigationReport` for the SSE `done` event.

Pure reshaping of the same dataclasses `continuity/analysis/cli.py`'s `render_brief`
already walks -- no new analysis logic, no number computed here that the analysis
package did not already produce. This module hands the frontend structured FACTS
(dimension/value pairs, enums, numbers, SQL) rather than prose; turning those into
sentences ("roku devices and app 8.2.0", "stopped because...") is left to the UI,
the same separation `render_brief`'s own `_render_*` helpers would have used had
this API existed first.

The one piece of I/O this module owns (`fetch_incident_series`) reuses detect.py's
public building blocks (`fetch_window_start`, `build_series_sql`, `label_buckets`)
exactly as cli.py's `_typical_and_peak_deviation` already does -- `DetectionResult`
itself only exposes anomaly windows and aggregate bucket counts, never every
bucket's own baseline band, which the hero chart needs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from continuity.analysis.baseline import (
    DEFAULT_LOOKBACK_WEEKS,
    DEFAULT_TRAILING_DAYS,
    BaselineStatus,
    ComparisonMode,
)
from continuity.analysis.cli import (
    DEFAULT_REFINE_PADDING,
    IncidentInvestigation,
    InvestigationReport,
)
from continuity.analysis.correlate import DisconfirmingEvidence, RankedChange, RejectedChange
from continuity.analysis.detect import (
    DEFAULT_MODE,
    DEFAULT_THRESHOLD,
    build_series_sql,
    fetch_window_start,
    label_buckets,
)
from continuity.analysis.metrics import METRICS, get_metric
from continuity.analysis.slices import Slice
from continuity.analysis.walk import WalkResult
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

_BUCKET_FORMAT = "%Y-%m-%d %H:%M:%S"


def iso(dt: datetime) -> str:
    """ISO-8601 with no timezone suffix -- every datetime in this codebase is naive
    UTC (see ClickHouseConfig / the generator), and `Date.parse` in every evergreen
    browser reads this form unambiguously, unlike the space-separated form
    `str(datetime)` produces."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_bucket_datetime(value: str) -> datetime:
    """Matches detect.py's own bucket timestamp format exactly -- it is not
    exported, so independently duplicated here exactly as cli.py, correlate.py and
    impact.py already each do for their own timestamp parsing."""
    return datetime.strptime(value, _BUCKET_FORMAT)


def slice_predicates(slice_: Slice) -> list[dict[str, str]]:
    """This slice's predicates as an ordered list, coarse-to-fine hierarchy order --
    a JSON array preserves order; a JSON object's key order is not a contract."""
    values = dict(slice_.predicates)
    return [{"dimension": d, "value": values[d]} for d in slice_.dimensions]


def serialize_detection(report: InvestigationReport) -> dict:
    detection = report.detection
    return {
        "total_buckets": detection.total_buckets,
        "anomalous_buckets": detection.anomalous_buckets,
        "unknown_buckets": detection.unknown_buckets,
        "unknown_fraction": detection.unknown_fraction,
        "windows_found": len(detection.windows),
        "sql": detection.sql,
    }


def _serialize_walk_path(walk_result: WalkResult) -> list[dict]:
    return [
        {
            "dimension": step.dimension,
            "value": step.value,
            "share_of_deviation": step.share_of_deviation,
            "lift": step.lift,
            "weight": step.weight,
            "sql": step.sql,
            "baseline_sql": step.baseline_sql,
        }
        for step in walk_result.path
    ]


def serialize_what_happened(ir: IncidentInvestigation, metric_label: str) -> dict:
    refined = ir.incident
    population = refined.population_incident
    pop_start, pop_end = population.span
    pop_peak = population.peak_window
    peak = refined.peak_window

    refined_span = None
    refined_peak_z = None
    refined_anomalous_buckets = None
    refined_span_buckets = None
    if not refined.used_fallback:
        refined_start, refined_end = refined.span
        refined_span = {"start": iso(refined_start), "end": iso(refined_end)}
        refined_peak_z = peak.peak_z
        refined_anomalous_buckets = refined.anomalous_bucket_count
        refined_span_buckets = refined.span_bucket_count

    return {
        "metric_label": metric_label,
        "population_span": {"start": iso(pop_start), "end": iso(pop_end)},
        "population_anomalous_buckets": population.anomalous_bucket_count,
        "population_span_buckets": population.span_bucket_count,
        "population_burst_count": len(population.windows),
        "population_peak_z": pop_peak.peak_z,
        "used_fallback": refined.used_fallback,
        "fallback_reason": refined.fallback_reason,
        "refined_span": refined_span,
        "refined_peak_z": refined_peak_z,
        "refined_anomalous_buckets": refined_anomalous_buckets,
        "refined_span_buckets": refined_span_buckets,
        "peak_value": peak.peak_value,
        "expected_at_peak": peak.expected_at_peak,
        "typical_multiple": float(refined.typical_deviation_ratio) + 1.0,
        "peak_multiple": float(refined.peak_deviation_ratio) + 1.0,
        "severity_sql": refined.severity_sql,
    }


def serialize_who_affected(ir: IncidentInvestigation) -> dict:
    population = ir.incident.population_incident
    walk_result = population.representative_walk
    peak = population.peak_window
    return {
        "predicates": slice_predicates(walk_result.final_slice),
        "drill_down": _serialize_walk_path(walk_result),
        "stop_reason": walk_result.stop_reason.value,
        "stop_detail": walk_result.stop_detail,
        "peak_window": {"start": iso(peak.start), "end": iso(peak.end)},
    }


def _serialize_disconfirming_evidence(evidence: DisconfirmingEvidence) -> dict:
    return {
        "sibling_dimension": evidence.sibling_dimension,
        "note": evidence.note,
        "siblings_checked": evidence.siblings_checked,
        "siblings_degraded": evidence.siblings_degraded,
        "siblings_not_degraded": evidence.siblings_not_degraded,
    }


def _serialize_candidate(candidate: RankedChange) -> dict:
    return {
        "change_id": candidate.change_id,
        "changed_at": iso(candidate.changed_at),
        "change_type": candidate.change_type,
        "component": candidate.component,
        "description": candidate.description,
        "dimension_key": candidate.dimension_key,
        "dimension_value": candidate.dimension_value,
        "score": candidate.score,
        "temporal_delta_hours": candidate.temporal_delta.total_seconds() / 3600,
        "dimensional_overlap": candidate.dimensional_overlap,
        "disconfirming_evidence": _serialize_disconfirming_evidence(
            candidate.disconfirming_evidence
        ),
        "sql": candidate.sql,
    }


def _serialize_rejected(rejected: RejectedChange) -> dict:
    return {
        "change_id": rejected.change_id,
        "changed_at": iso(rejected.changed_at),
        "description": rejected.description,
        "reason": rejected.reason,
    }


def serialize_probable_cause(ir: IncidentInvestigation) -> dict:
    correlation = ir.correlation
    return {
        "top": _serialize_candidate(correlation.candidates[0]) if correlation.candidates else None,
        "others": [_serialize_candidate(c) for c in correlation.candidates[1:]],
        "rejected": [_serialize_rejected(r) for r in correlation.rejected],
        "sql": correlation.sql,
    }


def serialize_impact(ir: IncidentInvestigation) -> dict:
    impact = ir.impact
    m = impact.methodology
    return {
        "affected_subscribers": impact.affected_subscribers,
        "arr_at_risk_low": float(impact.arr_at_risk_low),
        "arr_at_risk_expected": float(impact.arr_at_risk_expected),
        "arr_at_risk_high": float(impact.arr_at_risk_high),
        "methodology": {
            "base_monthly_churn": float(m.base_monthly_churn),
            "base_churn_variation": float(m.base_churn_variation),
            "qoe_delta_ratio": float(m.qoe_delta_ratio),
            "peak_deviation_ratio": float(ir.incident.peak_deviation_ratio),
            "notes": m.notes,
        },
        "sql": impact.sql,
    }


def serialize_recommended_action(ir: IncidentInvestigation) -> dict:
    if ir.correlation.candidates:
        top = ir.correlation.candidates[0]
        return {
            "has_candidate": True,
            "change_id": top.change_id,
            "component": top.component,
            "description": top.description,
        }
    return {"has_candidate": False, "change_id": None, "component": None, "description": None}


def serialize_performance(report: InvestigationReport) -> list[dict]:
    return [
        {
            "stage": stage.name,
            "elapsed_ms": round(stage.elapsed_ms, 1),
            "queries": len(stage.queries),
        }
        for stage in report.stage_timings
    ]


async def fetch_incident_series(
    gateway: ClickHouseMCPGateway,
    *,
    slice_: Slice,
    metric_name: str,
    span: tuple[datetime, datetime],
    padding: timedelta = DEFAULT_REFINE_PADDING,
) -> dict:
    """One extra query per incident: the per-bucket series (value, baseline band,
    status) the hero chart renders. `detect()` never returns per-bucket detail --
    only anomaly windows and aggregate counts -- so this composes the same public
    primitives `detect()` and cli.py's `_typical_and_peak_deviation` already use,
    over `span` padded on both sides for chart context.
    """
    metric = get_metric(metric_name)
    chart_start = span[0] - padding
    chart_end = span[1] + padding
    days_of_history = (
        DEFAULT_TRAILING_DAYS
        if DEFAULT_MODE is ComparisonMode.TRAILING_DAYS
        else DEFAULT_LOOKBACK_WEEKS * 7
    )
    fetch_start = fetch_window_start(chart_start, days_of_history)
    sql = build_series_sql(slice_, metric, fetch_start, chart_end)
    result = await gateway.query(sql)
    observations = [(_parse_bucket_datetime(row["bucket"]), row["value"]) for row in result.rows]
    labels = label_buckets(observations, start=chart_start, end=chart_end, metric=metric)

    points = []
    for label in labels:
        baseline = label.baseline
        if baseline.status is BaselineStatus.OK and baseline.expected is not None:
            spread = baseline.spread or 0.0
            expected = baseline.expected
            lower = expected - DEFAULT_THRESHOLD * spread
            upper = expected + DEFAULT_THRESHOLD * spread
        else:
            expected = lower = upper = None
        points.append(
            {
                "bucket": iso(label.bucket),
                "value": label.value,
                "expected": expected,
                "lower": lower,
                "upper": upper,
                "status": label.status.value,
            }
        )
    return {"points": points, "sql": sql, "metric": metric_name}


def serialize_incident(ir: IncidentInvestigation, *, metric_label: str, series: dict) -> dict:
    anomaly_windows = [{"start": iso(w.start), "end": iso(w.end)} for w in ir.incident.windows]
    return {
        "what_happened": serialize_what_happened(ir, metric_label),
        "who_affected": serialize_who_affected(ir),
        "probable_cause": serialize_probable_cause(ir),
        "impact": serialize_impact(ir),
        "recommended_action": serialize_recommended_action(ir),
        "series": {**series, "anomaly_windows": anomaly_windows},
    }


def serialize_report(report: InvestigationReport, incident_series: list[dict]) -> dict:
    """Everything the brief view and hero chart need, as JSON.

    `incident_series` must be the same length and order as `report.incidents` --
    fetching it needs gateway I/O this module's own pure functions cannot perform,
    so the caller (the SSE endpoint) fetches it and hands it in.
    """
    if len(incident_series) != len(report.incidents):
        raise ValueError(
            f"incident_series has {len(incident_series)} entries, "
            f"report.incidents has {len(report.incidents)}"
        )
    metric_label = METRICS[report.metric_name].label
    return {
        "metric": report.metric_name,
        "metric_label": metric_label,
        "description": report.description,
        "window": {"start": iso(report.window[0]), "end": iso(report.window[1])},
        "detection": serialize_detection(report),
        "incidents": [
            serialize_incident(ir, metric_label=metric_label, series=series)
            for ir, series in zip(report.incidents, incident_series, strict=True)
        ],
        "performance": serialize_performance(report),
        "total_elapsed_ms": round(report.total_elapsed_ms, 1),
    }
