"""`python -m continuity.analysis.cli investigate` -- a full investigation, no LLM.

Composes every primitive built in this sub-project into the deterministic twin of what
sub-project 3's Gemini-driven investigator will produce:

    detect() on the whole population -> for each anomaly window, walk() to localise the
    blast radius -> correlate_changes() against change_log -> compute_impact() for
    ARR at risk -> a plain-text incident brief.

This module contains no analysis logic of its own -- every number comes from detect.py,
walk.py, correlate.py or impact.py. It only orchestrates the calls, times each stage, and
renders the result. See CLAUDE.md hard constraint 4: the deterministic analysis engine
(this included) contains zero LLM calls.

Every claim in the printed brief is backed by a query the gateway actually ran --
`--show-sql` prints them. This is not a debugging aid; it is the anti-hallucination
property the whole project rests on (`ClickHouseMCPGateway.query_log`).

Ground truth (`data/ground_truth.json`) is used ONLY to look up a known incident's time
window and its primary metric when `--incident` is passed -- never its predicate. The
blast radius is always rediscovered from scratch by `walk()`, exactly as it would be with
an explicit `--start`/`--end` window; `--incident` is a convenience for demos, not a
shortcut that leaks the answer into the analysis.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import typer
from dotenv import load_dotenv

from continuity.analysis.correlate import CorrelationResult, correlate_changes
from continuity.analysis.detect import AnomalyWindow, DetectionResult, detect
from continuity.analysis.impact import ImpactResult
from continuity.analysis.impact import compute_impact as _compute_impact
from continuity.analysis.metrics import METRICS, get_metric
from continuity.analysis.slices import Slice
from continuity.analysis.walk import StopReason, WalkResult
from continuity.analysis.walk import walk as _walk
from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway, ExecutedQuery, QueryError

DEFAULT_METRIC = "rebuffer"
DEFAULT_GROUND_TRUTH_PATH = Path("data/ground_truth.json")

# How far to pad a known incident's ground-truth window before handing it to detect().
# Matches the margin used throughout tests/integration/test_detect_real.py and
# test_walk_real.py: wide enough that the anomaly's true onset/offset are not clipped
# at the search boundary, without pulling in an adjacent incident (the closest two
# planted incidents are >2 days apart).
INCIDENT_SEARCH_PADDING = timedelta(hours=6)

_SEPARATOR = "=" * 78
_RULE = "-" * 78

_DIMENSION_PHRASES: dict[str, str] = {
    "device_type": "{value} devices",
    "app_version": "app {value}",
    "os_version": "OS {value}",
    "cdn": "CDN {value}",
    "pop": "PoP {value}",
    "isp": "ISP {value}",
    "country": "country {value}",
    "region": "region {value}",
    "title_id": "title {value}",
}

_STOP_REASON_TEXT: dict[StopReason, str] = {
    StopReason.LOW_LIFT: (
        "the best-explaining value only explained its own share of the population -- "
        "big, not causal"
    ),
    StopReason.LOW_SHARE: (
        "no remaining dimension explained enough of the deviation to justify "
        "descending further"
    ),
    StopReason.SINGLE_VALUE: (
        "every remaining dimension had only one usable value -- nothing left to compare"
    ),
    StopReason.MAX_DEPTH: "reached the maximum drill-down depth",
    StopReason.TOO_SMALL: (
        "the next candidate slice was too small a share of the population to trust its ratio"
    ),
    StopReason.DIMENSIONS_EXHAUSTED: "every candidate dimension is already fixed in this slice",
}

app = typer.Typer(add_completion=False)


@app.callback()
def _callback() -> None:
    """Deterministic incident investigation (detect -> walk -> correlate -> quantify).

    A no-op callback -- its only job is to force Typer into subcommand-group mode so
    `investigate` must be named explicitly, matching every documented invocation of
    this CLI (`python -m continuity.analysis.cli investigate ...`). With a single
    command and no callback, Typer collapses to a bare single-command CLI instead.
    """


class InvestigationInputError(ValueError):
    """Bad CLI input: unknown metric/incident, missing or malformed window, etc.

    Always caught at the CLI boundary and reported with exit code 1 -- never allowed to
    surface as a raw traceback.
    """


# ---------------------------------------------------------------------------
# Pipeline orchestration -- no analysis logic, only composition and timing.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageTiming:
    """Wall-clock time spent in one pipeline stage, plus every query it ran."""

    name: str
    elapsed_ms: float
    queries: tuple[ExecutedQuery, ...]


@dataclass(frozen=True)
class WindowInvestigation:
    """The full drill-down for one anomaly window: blast radius, cause, impact."""

    anomaly: AnomalyWindow
    walk: WalkResult
    correlation: CorrelationResult
    impact: ImpactResult
    qoe_delta_ratio: Decimal


@dataclass(frozen=True)
class InvestigationReport:
    """Everything needed to render a brief, or to prove there was nothing to report."""

    metric_name: str
    window: tuple[datetime, datetime]
    description: str
    detection: DetectionResult
    windows: tuple[WindowInvestigation, ...]
    stage_timings: tuple[StageTiming, ...]
    total_elapsed_ms: float


def _qoe_delta_ratio(anomaly: AnomalyWindow) -> Decimal:
    """(actual - expected) / expected at the anomaly's peak bucket, as impact.py expects.

    When the expected (baseline) level is zero or unmeasurable, there is no ratio to
    form -- rather than fabricate a small one, this falls back to 1.0 ("clearly
    doubled"), a documented, conservative stand-in. The anomaly's own peak_z already
    established that this is a genuine, statistically significant deviation regardless.
    """
    expected = anomaly.expected_at_peak
    if expected is None or expected == 0:
        return Decimal("1.0")
    ratio = abs(anomaly.peak_value - expected) / abs(expected)
    return Decimal(str(ratio))


async def investigate_pipeline(
    gateway: ClickHouseMCPGateway,
    *,
    metric_name: str,
    window: tuple[datetime, datetime],
    description: str,
) -> InvestigationReport:
    """Run detect -> {walk -> correlate -> quantify}* and time every stage.

    Raises on the first failing query (QueryError) or bad input -- never builds a
    partial report. A caller only ever sees a report once every stage that ran
    succeeded completely.
    """
    total_started = time.perf_counter()

    detect_start_idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    detection = await detect(gateway, Slice(), metric_name, window[0], window[1])
    detect_timing = StageTiming(
        "detect",
        (time.perf_counter() - stage_started) * 1000,
        tuple(gateway.query_log[detect_start_idx:]),
    )

    walk_queries: list[ExecutedQuery] = []
    correlate_queries: list[ExecutedQuery] = []
    quantify_queries: list[ExecutedQuery] = []
    walk_elapsed_ms = 0.0
    correlate_elapsed_ms = 0.0
    quantify_elapsed_ms = 0.0
    window_results: list[WindowInvestigation] = []

    for anomaly in detection.windows:
        stage_started = time.perf_counter()
        walk_result = await _walk(
            gateway, metric_name=metric_name, window=(anomaly.start, anomaly.end)
        )
        walk_elapsed_ms += (time.perf_counter() - stage_started) * 1000
        walk_queries.extend(walk_result.query_log)

        idx = len(gateway.query_log)
        stage_started = time.perf_counter()
        correlation = await correlate_changes(
            gateway,
            blast_radius=walk_result.final_slice,
            anomaly_window=(anomaly.start, anomaly.end),
            metric_name=metric_name,
        )
        correlate_elapsed_ms += (time.perf_counter() - stage_started) * 1000
        correlate_queries.extend(gateway.query_log[idx:])

        qoe_delta_ratio = _qoe_delta_ratio(anomaly)
        idx = len(gateway.query_log)
        stage_started = time.perf_counter()
        impact = await _compute_impact(
            gateway,
            slice_=walk_result.final_slice,
            window=(anomaly.start, anomaly.end),
            qoe_delta_ratio=qoe_delta_ratio,
        )
        quantify_elapsed_ms += (time.perf_counter() - stage_started) * 1000
        quantify_queries.extend(gateway.query_log[idx:])

        window_results.append(
            WindowInvestigation(
                anomaly=anomaly,
                walk=walk_result,
                correlation=correlation,
                impact=impact,
                qoe_delta_ratio=qoe_delta_ratio,
            )
        )

    total_elapsed_ms = (time.perf_counter() - total_started) * 1000
    stage_timings = (
        detect_timing,
        StageTiming("walk", walk_elapsed_ms, tuple(walk_queries)),
        StageTiming("correlate", correlate_elapsed_ms, tuple(correlate_queries)),
        StageTiming("quantify", quantify_elapsed_ms, tuple(quantify_queries)),
    )
    return InvestigationReport(
        metric_name=metric_name,
        window=window,
        description=description,
        detection=detection,
        windows=tuple(window_results),
        stage_timings=stage_timings,
        total_elapsed_ms=total_elapsed_ms,
    )


# ---------------------------------------------------------------------------
# CLI input resolution -- pure, no I/O besides reading ground_truth.json.
# ---------------------------------------------------------------------------


def _parse_datetime_arg(name: str, value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvestigationInputError(
            f"{name}={value!r} is not a valid datetime. Use e.g. '2026-02-12 18:00:00'."
        ) from exc
    return parsed.replace(tzinfo=None)


def _load_ground_truth(path: Path) -> list[dict]:
    if not path.exists():
        raise InvestigationInputError(
            f"Ground truth file not found: {path}. Pass --ground-truth-path, or use "
            "--start/--end instead of --incident."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvestigationInputError(f"Ground truth file {path} is not valid JSON: {exc}") from exc
    incidents = payload.get("incidents")
    if not isinstance(incidents, list):
        raise InvestigationInputError(f"Ground truth file {path} has no 'incidents' list.")
    return incidents


def _find_incident(incidents: list[dict], incident_id: str) -> dict:
    for row in incidents:
        if row.get("incident_id") == incident_id:
            return row
    known = sorted(str(row.get("incident_id")) for row in incidents)
    raise InvestigationInputError(f"Unknown incident {incident_id!r}. Known incident ids: {known}")


def _incident_window(incident: dict) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(incident["start"]).replace(tzinfo=None)
    end = datetime.fromisoformat(incident["end"]).replace(tzinfo=None)
    return start, end


def _incident_default_metric(incident: dict) -> str:
    effects = incident.get("effects") or []
    if effects:
        return str(effects[0]["metric"])
    return DEFAULT_METRIC


def resolve_investigation(
    *,
    metric: str | None,
    start: str | None,
    end: str | None,
    incident: str | None,
    ground_truth_path: Path,
) -> tuple[str, tuple[datetime, datetime], str]:
    """Turn CLI arguments into (metric_name, detect_window, human description).

    Exactly one of `--incident` or `--start`+`--end` must be given. `--incident`
    widens the ground-truth window by `INCIDENT_SEARCH_PADDING` before handing it to
    detect() -- an explicit `--start`/`--end` window is used exactly as given.
    """
    has_explicit_window = start is not None or end is not None
    if incident is not None and has_explicit_window:
        raise InvestigationInputError("Pass either --incident or --start/--end, not both.")

    if incident is not None:
        incidents = _load_ground_truth(ground_truth_path)
        row = _find_incident(incidents, incident)
        true_start, true_end = _incident_window(row)
        window = (true_start - INCIDENT_SEARCH_PADDING, true_end + INCIDENT_SEARCH_PADDING)
        metric_name = metric or _incident_default_metric(row)
        description = f"incident {incident} (planted window {true_start} to {true_end})"
        return metric_name, window, description

    if start is None or end is None:
        raise InvestigationInputError("Provide either --incident <id>, or both --start and --end.")

    window_start = _parse_datetime_arg("--start", start)
    window_end = _parse_datetime_arg("--end", end)
    if not window_start < window_end:
        raise InvestigationInputError(
            f"--start ({window_start}) must be before --end ({window_end})."
        )
    metric_name = metric or DEFAULT_METRIC
    return metric_name, (window_start, window_end), f"{window_start} to {window_end}"


# ---------------------------------------------------------------------------
# Rendering -- plain text, no colour codes, no box-drawing characters.
# ---------------------------------------------------------------------------


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_usd(value: Decimal) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "n/a"


def _humanize_slice(slice_: Slice) -> str:
    if not slice_.predicates:
        return "the whole population -- no dimension isolated a narrower blast radius"
    predicates = dict(slice_.predicates)
    phrases = [
        _DIMENSION_PHRASES.get(dim, f"{dim}={predicates[dim]}").format(value=predicates[dim])
        for dim in slice_.dimensions
    ]
    if len(phrases) <= 2:
        return " and ".join(phrases)
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

def _render_sql(sql: str) -> list[str]:
    """The raw SQL text for one query, indented for a brief's SQL sub-section.

    Deliberately carries no duration/row-count placeholder here -- those numbers are
    only known from the gateway's own `ExecutedQuery` records, which the PERFORMANCE
    section at the end of the brief lists exactly once, accurately, per query actually
    run. Fabricating a "0.0ms" here would be exactly the kind of invented number this
    project's anti-hallucination property exists to rule out.
    """
    return [f"    {sql}", ""]


def _render_detect_section(report: InvestigationReport, show_sql: bool) -> list[str]:
    detection = report.detection
    lines = [
        _SEPARATOR,
        f"INCIDENT BRIEF -- {METRICS[report.metric_name].label} -- {report.description}",
        _SEPARATOR,
        "",
        "WHAT WAS CHECKED",
        f"  Metric: {METRICS[report.metric_name].label} ({report.metric_name})",
        "  Population: whole population (detect stage scans before any drill-down)",
        f"  Window: {_fmt_dt(report.window[0])} to {_fmt_dt(report.window[1])}",
        f"  Buckets measured: {detection.total_buckets} "
        f"({detection.anomalous_buckets} anomalous, {detection.unknown_buckets} unknown, "
        f"{detection.unknown_fraction:.0%} unmeasurable)",
        "",
    ]
    if show_sql:
        lines.append("  SQL:")
        lines.extend(_render_sql(detection.sql))
    return lines


def _render_no_anomalies(report: InvestigationReport) -> list[str]:
    return [
        "RESULT: NO ANOMALIES DETECTED.",
        "",
        "This is a normal, healthy outcome, not a failure: the metric stayed within its",
        "seasonality-aware expected range for the whole window checked above.",
        "",
    ]


def _render_what_happened(w: WindowInvestigation, metric_label: str) -> list[str]:
    a = w.anomaly
    return [
        "WHAT HAPPENED",
        f"  {metric_label} moved to {a.peak_value:.6g} at its worst (expected "
        f"{a.expected_at_peak:.6g}), a robust z-score of {a.peak_z:.1f} sigma.",
        f"  Window: {_fmt_dt(a.start)} to {_fmt_dt(a.end)} "
        f"({a.bucket_count} anomalous 5-minute buckets).",
        "",
    ]


def _render_who_affected(w: WindowInvestigation, show_sql: bool) -> list[str]:
    slice_ = w.walk.final_slice
    lines = [
        "WHO WAS AFFECTED (blast radius)",
        f"  {_humanize_slice(slice_)}",
        "",
        "  Drill-down path:",
    ]
    if not w.walk.path:
        lines.append("    (no refinement -- the anomaly could not be localised below the")
        lines.append("     whole population)")
    for i, step in enumerate(w.walk.path, start=1):
        lines.append(
            f"    {i}. {step.dimension} = {step.value}  "
            f"({_fmt_pct(step.share_of_deviation)} of the deviation, weight {step.weight:.0f})"
        )
    lines.append(
        f"  Stopped because: {_STOP_REASON_TEXT.get(w.walk.stop_reason, w.walk.stop_reason.value)}"
    )
    lines.append("")
    if show_sql:
        lines.append("  SQL (one query per drill-down level, batched across dimensions):")
        for step in w.walk.path:
            lines.extend(_render_sql(step.sql))
            lines.extend(_render_sql(step.baseline_sql))
    return lines


def _render_probable_cause(w: WindowInvestigation, show_sql: bool) -> list[str]:
    correlation = w.correlation
    lines = ["PROBABLE CAUSE"]
    if not correlation.candidates:
        lines.append("  No change_log entry correlates with this anomaly (temporally and")
        lines.append("  dimensionally). No probable cause identified from recorded changes.")
    else:
        top = correlation.candidates[0]
        lines.extend(
            [
                f"  [change #{top.change_id}] {top.change_type} / {top.component}: "
                f"{top.description}",
                f"  Changed at: {_fmt_dt(top.changed_at)} "
                f"({top.temporal_delta} before the anomaly's onset)",
                f"  Confidence score: {top.score:.2f} "
                "(temporal proximity x dimensional overlap; 1.0 is the maximum)",
                f"  Dimensional match: {top.dimension_key} = {top.dimension_value} -- "
                f"{'matches' if top.dimensional_overlap else 'does not directly match'} "
                "a blast-radius predicate",
                "  Disconfirming evidence checked:",
                f"    {top.disconfirming_evidence.note}",
            ]
        )
        others = list(correlation.candidates[1:])
        if others or correlation.rejected:
            lines.append("")
            lines.append("  Other changes considered and ruled out or ranked lower:")
            for c in others:
                lines.append(f"    - [change #{c.change_id}] {c.description} (score {c.score:.2f})")
            for r in correlation.rejected:
                lines.append(f"    - [change #{r.change_id}] {r.description} -- {r.reason}")
    lines.append("")
    if show_sql:
        lines.append("  SQL:")
        lines.extend(_render_sql(correlation.sql))
    return lines


def _render_revenue_impact(w: WindowInvestigation, show_sql: bool) -> list[str]:
    impact = w.impact
    m = impact.methodology
    lines = [
        "SUBSCRIBERS AFFECTED AND ARR AT RISK",
        f"  Affected subscribers: {impact.affected_subscribers:,}",
        f"  ARR at risk: {_fmt_usd(impact.arr_at_risk_low)} - {_fmt_usd(impact.arr_at_risk_high)} "
        f"(expected {_fmt_usd(impact.arr_at_risk_expected)})",
        "  Methodology: churn_risk = base_monthly_churn x tenure_multiplier x "
        "severity_multiplier, capped at 1.0;",
        "    arr_at_risk = sum(churn_risk x monthly_arpu x 12) over affected subscribers.",
        f"    base_monthly_churn={m.base_monthly_churn} +/-{m.base_churn_variation:.0%} "
        f"(assumption, not measured); qoe_delta_ratio={m.qoe_delta_ratio:.3f}.",
        "    Every coefficient is a documented assumption -- see continuity/analysis/impact.py.",
        "",
    ]
    if show_sql:
        lines.append("  SQL:")
        lines.extend(_render_sql(impact.sql))
    return lines


def _recommended_action(w: WindowInvestigation) -> list[str]:
    slice_desc = _humanize_slice(w.walk.final_slice)
    if w.correlation.candidates:
        top = w.correlation.candidates[0]
        action = (
            f"Roll back or hotfix change #{top.change_id} ({top.component}): "
            f"{top.description}. It is the top-ranked probable cause for the impact on "
            f"{slice_desc}."
        )
    else:
        action = (
            f"No change_log entry explains this anomaly. Escalate to the on-call team for "
            f"manual investigation of {slice_desc}."
        )
    return [
        "RECOMMENDED ACTION  [PROPOSAL -- REQUIRES HUMAN APPROVAL]",
        f"  {action}",
        "",
    ]


def _render_performance(report: InvestigationReport, show_sql: bool) -> list[str]:
    lines = ["PERFORMANCE (measured this run)"]
    for stage in report.stage_timings:
        n_queries = len(stage.queries)
        lines.append(f"  {stage.name:<10s} {stage.elapsed_ms:8.1f}ms  ({n_queries} queries)")
    lines.append(f"  {'total':<10s} {report.total_elapsed_ms:8.1f}ms")
    lines.append("")
    if show_sql:
        lines.append("  Every query executed, in order:")
        for stage in report.stage_timings:
            for q in stage.queries:
                lines.append(f"    [{stage.name}] ({q.duration_ms:.1f}ms, {q.row_count} rows)")
                lines.append(f"    {q.sql}")
        lines.append("")
    return lines


def render_brief(report: InvestigationReport, *, show_sql: bool) -> str:
    """The full incident brief, as plain text. Never called with a partial report --
    `investigate_pipeline` only returns once every stage it ran has succeeded."""
    lines = _render_detect_section(report, show_sql)

    if not report.windows:
        lines.extend(_render_no_anomalies(report))
        lines.extend(_render_performance(report, show_sql))
        return "\n".join(lines)

    metric_label = METRICS[report.metric_name].label
    for i, w in enumerate(report.windows, start=1):
        if len(report.windows) > 1:
            lines.append(f"{_RULE}\nANOMALY WINDOW {i} of {len(report.windows)}\n{_RULE}")
            lines.append("")
        lines.extend(_render_what_happened(w, metric_label))
        lines.extend(_render_who_affected(w, show_sql))
        lines.extend(_render_probable_cause(w, show_sql))
        lines.extend(_render_revenue_impact(w, show_sql))
        lines.extend(_recommended_action(w))

    lines.extend(_render_performance(report, show_sql))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


async def _run(
    config: ClickHouseConfig, metric_name: str, window: tuple[datetime, datetime], description: str
) -> InvestigationReport:
    async with ClickHouseMCPGateway(config) as gateway:
        return await investigate_pipeline(
            gateway, metric_name=metric_name, window=window, description=description
        )


@app.command()
def investigate(
    metric: str | None = typer.Option(
        None, "--metric", help=f"One of: {', '.join(sorted(METRICS))}. Defaults per --incident."
    ),
    start: str | None = typer.Option(
        None, "--start", help="Window start, e.g. '2026-02-12 12:00:00'."
    ),
    end: str | None = typer.Option(
        None, "--end", help="Window end, e.g. '2026-02-13 06:00:00'."
    ),
    incident: str | None = typer.Option(
        None, "--incident", help="Incident id from ground truth, e.g. INC-APP-ROKU-820."
    ),
    ground_truth_path: Path = typer.Option(  # noqa: B008 -- read-only typer.Option singleton.
        DEFAULT_GROUND_TRUTH_PATH,
        "--ground-truth-path",
        help="Path to ground_truth.json. Only used with --incident.",
    ),
    show_sql: bool = typer.Option(
        False, "--show-sql", help="Print the SQL behind every claim in the brief."
    ),
) -> None:
    """Run a full investigation (detect -> walk -> correlate -> quantify) with no LLM
    involved, and print a plain-text incident brief. Every number is traceable to a
    query the gateway actually ran; pass --show-sql to see them."""
    load_dotenv(override=False)
    try:
        metric_name, window, description = resolve_investigation(
            metric=metric,
            start=start,
            end=end,
            incident=incident,
            ground_truth_path=ground_truth_path,
        )
        get_metric(metric_name)
        config = ClickHouseConfig.from_env()
    except (InvestigationInputError, KeyError, ValueError) as exc:
        typer.echo(f"INVESTIGATION FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        report = asyncio.run(_run(config, metric_name, window, description))
    except QueryError as exc:
        typer.echo(f"QUERY FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(render_brief(report, show_sql=show_sql))


if __name__ == "__main__":
    app()
