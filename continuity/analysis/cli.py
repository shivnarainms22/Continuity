"""`python -m continuity.analysis.cli investigate` -- a full investigation, no LLM.

Composes every primitive built in this sub-project into the deterministic twin of what
sub-project 3's Gemini-driven investigator will produce:

    detect() on the whole population -> for each anomaly window, walk() to localise the
    blast radius -> MERGE windows that resolve to the same blast radius and are close in
    time into one incident -> correlate_changes() and compute_impact() ONCE per merged
    incident -> a plain-text incident brief.

This module contains no analysis logic of its own -- every number comes from detect.py,
walk.py, correlate.py or impact.py. It only orchestrates the calls, times each stage, and
renders the result. See CLAUDE.md hard constraint 4: the deterministic analysis engine
(this included) contains zero LLM calls.

WHY MERGING MATTERS: detect() runs on the WHOLE POPULATION, where a fault scoped to a
narrow slice (e.g. roku + app 8.2.0, ~8% of sessions) is a diluted signal that only
breaches the population-level threshold at its worst peaks. One continuous incident
therefore surfaces as several short, separate anomaly windows with a quiet gap between
them. Reporting each window as its own incident (a) drastically understates the true
subscriber/revenue impact -- correlating and quantifying over a 25-minute fragment
instead of the true multi-hour incident -- and (b) is exactly the alert-fatigue behaviour
this project exists to argue against. `merge_windows_into_incidents` groups windows that
walk() resolved to the SAME blast radius and are within `merge_gap` of each other, and
correlation/impact run once over the merged span, not once per raw window.

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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import typer
from dotenv import load_dotenv

from continuity.analysis.correlate import CorrelationResult, correlate_changes
from continuity.analysis.detect import BUCKET_WIDTH, AnomalyWindow, DetectionResult, detect
from continuity.analysis.impact import ImpactResult
from continuity.analysis.impact import compute_impact as _compute_impact
from continuity.analysis.metrics import METRICS, get_metric
from continuity.analysis.slices import Slice
from continuity.analysis.walk import RefinementStep, StopReason, WalkResult
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

# Two anomaly windows that resolve to the SAME blast radius (walk()'s final_slice) are
# treated as one incident when the gap between them is at most this long. 2 hours is
# generous enough to bridge the quiet stretches a diluted, population-level detector
# leaves between the peaks of one continuous, narrowly-scoped fault (see the module
# docstring) without bridging across genuinely separate incidents days apart.
DEFAULT_MERGE_GAP = timedelta(hours=2)

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
class MergedIncident:
    """Several anomaly windows that walk() resolved to the SAME blast radius, within
    `merge_gap` of each other, merged into one incident.

    `windows` is kept in full (not collapsed away) so the brief can show exactly which
    5-minute buckets breached threshold -- never claiming the whole span was anomalous
    when only a fraction of it measurably was. `representative_walk` is the walk() run
    over the constituent window with the single worst (largest |peak_z|) deviation --
    used for the displayed drill-down path since it is the clearest signal, though by
    construction of the merge condition every constituent window resolved to the same
    final slice.
    """

    windows: tuple[AnomalyWindow, ...]
    walks: tuple[WalkResult, ...]

    @property
    def span(self) -> tuple[datetime, datetime]:
        return self.windows[0].start, self.windows[-1].end

    @property
    def final_slice(self) -> Slice:
        return self.representative_walk.final_slice

    @property
    def peak_index(self) -> int:
        return max(range(len(self.windows)), key=lambda i: abs(self.windows[i].peak_z))

    @property
    def peak_window(self) -> AnomalyWindow:
        return self.windows[self.peak_index]

    @property
    def representative_walk(self) -> WalkResult:
        return self.walks[self.peak_index]

    @property
    def anomalous_bucket_count(self) -> int:
        return sum(w.bucket_count for w in self.windows)

    @property
    def span_bucket_count(self) -> int:
        start, end = self.span
        return max(1, int((end - start) / BUCKET_WIDTH))


@dataclass(frozen=True)
class IncidentInvestigation:
    """One merged incident's cause and business impact."""

    incident: MergedIncident
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
    incidents: tuple[IncidentInvestigation, ...]
    stage_timings: tuple[StageTiming, ...]
    total_elapsed_ms: float


def _qoe_delta_ratio(anomaly: AnomalyWindow) -> Decimal:
    """(actual - expected) / expected at an anomaly window's peak bucket, as impact.py
    expects.

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


def merge_windows_into_incidents(
    entries: Sequence[tuple[AnomalyWindow, WalkResult]],
    *,
    merge_gap: timedelta = DEFAULT_MERGE_GAP,
) -> list[MergedIncident]:
    """Group (anomaly window, its walk result) pairs into incidents.

    Pure and I/O-free -- `entries` must already be in chronological order (exactly how
    `investigate_pipeline` builds them, walking `detect()`'s own windows in order). A
    new window extends the current group when its walk resolved to the SAME final
    slice as the group's most recent window AND it starts within `merge_gap` of that
    window's end; otherwise it starts a new group. Because the check is against the
    immediately preceding window, every window within one resulting group shares an
    identical final slice by transitivity -- there is exactly one blast radius per
    `MergedIncident`.
    """
    if not entries:
        return []
    groups: list[list[tuple[AnomalyWindow, WalkResult]]] = [[entries[0]]]
    for anomaly, walk_result in entries[1:]:
        last_anomaly, last_walk = groups[-1][-1]
        same_slice = walk_result.final_slice == last_walk.final_slice
        gap = anomaly.start - last_anomaly.end
        if same_slice and gap <= merge_gap:
            groups[-1].append((anomaly, walk_result))
        else:
            groups.append([(anomaly, walk_result)])
    return [
        MergedIncident(
            windows=tuple(a for a, _ in group), walks=tuple(w for _, w in group)
        )
        for group in groups
    ]


async def investigate_pipeline(
    gateway: ClickHouseMCPGateway,
    *,
    metric_name: str,
    window: tuple[datetime, datetime],
    description: str,
    merge_gap: timedelta = DEFAULT_MERGE_GAP,
) -> InvestigationReport:
    """Run detect -> walk (per window) -> merge -> {correlate -> quantify}* (per
    merged incident), timing every stage.

    Raises on the first failing query (QueryError) or bad input -- never builds a
    partial report. A caller only ever sees a report once every stage that ran
    succeeded completely.

    A trivial warm-up query runs BEFORE any stage timer starts, so the one-time
    mcp-clickhouse subprocess-spawn-and-first-connection cost (seconds; see CLAUDE.md's
    own benchmark notes) is never misattributed to detect -- it is reported as its own
    "session_startup" stage instead. Sub-project 4's long-lived session pays this cost
    once for the process's whole life, never per investigation.
    """
    total_started = time.perf_counter()

    startup_idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    await gateway.query("SELECT 1")
    startup_timing = StageTiming(
        "session_startup",
        (time.perf_counter() - stage_started) * 1000,
        tuple(gateway.query_log[startup_idx:]),
    )

    detect_start_idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    detection = await detect(gateway, Slice(), metric_name, window[0], window[1])
    detect_timing = StageTiming(
        "detect",
        (time.perf_counter() - stage_started) * 1000,
        tuple(gateway.query_log[detect_start_idx:]),
    )

    walk_queries: list[ExecutedQuery] = []
    walk_elapsed_ms = 0.0
    entries: list[tuple[AnomalyWindow, WalkResult]] = []
    for anomaly in detection.windows:
        stage_started = time.perf_counter()
        walk_result = await _walk(
            gateway, metric_name=metric_name, window=(anomaly.start, anomaly.end)
        )
        walk_elapsed_ms += (time.perf_counter() - stage_started) * 1000
        walk_queries.extend(walk_result.query_log)
        entries.append((anomaly, walk_result))

    incidents = merge_windows_into_incidents(entries, merge_gap=merge_gap)

    correlate_queries: list[ExecutedQuery] = []
    quantify_queries: list[ExecutedQuery] = []
    correlate_elapsed_ms = 0.0
    quantify_elapsed_ms = 0.0
    incident_results: list[IncidentInvestigation] = []

    for incident in incidents:
        idx = len(gateway.query_log)
        stage_started = time.perf_counter()
        correlation = await correlate_changes(
            gateway,
            blast_radius=incident.final_slice,
            anomaly_window=incident.span,
            metric_name=metric_name,
        )
        correlate_elapsed_ms += (time.perf_counter() - stage_started) * 1000
        correlate_queries.extend(gateway.query_log[idx:])

        qoe_delta_ratio = _qoe_delta_ratio(incident.peak_window)
        idx = len(gateway.query_log)
        stage_started = time.perf_counter()
        impact = await _compute_impact(
            gateway,
            slice_=incident.final_slice,
            window=incident.span,
            qoe_delta_ratio=qoe_delta_ratio,
        )
        quantify_elapsed_ms += (time.perf_counter() - stage_started) * 1000
        quantify_queries.extend(gateway.query_log[idx:])

        incident_results.append(
            IncidentInvestigation(
                incident=incident,
                correlation=correlation,
                impact=impact,
                qoe_delta_ratio=qoe_delta_ratio,
            )
        )

    total_elapsed_ms = (time.perf_counter() - total_started) * 1000
    stage_timings = (
        startup_timing,
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
        incidents=tuple(incident_results),
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


def _fmt_share(share: float | None) -> str:
    """Share of deviation, capped at 100% for display with the raw value kept visible
    when it exceeds that -- a share above 100% is mathematically legitimate (some
    values improved while this one worsened, so its contribution alone exceeds the
    slice's net deviation), not an error, and a reader must not mistake it for one."""
    if share is None:
        return "n/a"
    capped = min(share, 1.0)
    text = f"{capped:.0%} of the deviation"
    if share > 1.0:
        text += (
            f" (raw {share:.0%} -- contributions can exceed 100% when other values "
            "moved in the opposite direction)"
        )
    return text


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


def _render_step_line(index: int, step: RefinementStep) -> str:
    """One drill-down step, with its share of deviation (capped for display, see
    `_fmt_share`) and its LIFT -- how many times more of the problem this value
    explains than its own share of the population alone would predict. Lift is the
    number that is actually checkable by a reader; a raw share cannot distinguish "the
    cause" from "just a big population segment" (see walk.py's own module docstring)."""
    return (
        f"    {index}. {step.dimension} = {step.value}  ({_fmt_share(step.share_of_deviation)}, "
        f"lift {step.lift:.1f}x)\n"
        f"       -- {step.value} accounts for {step.lift:.1f}x more of this problem than its "
        "population share alone would explain"
    )


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


def _render_what_happened(ir: IncidentInvestigation, metric_label: str) -> list[str]:
    incident = ir.incident
    peak = incident.peak_window
    span_start, span_end = incident.span
    lines = [
        "WHAT HAPPENED",
        f"  {metric_label} breached its seasonality-aware threshold intermittently across "
        f"{_fmt_dt(span_start)} to {_fmt_dt(span_end)}: {incident.anomalous_bucket_count} of "
        f"{incident.span_bucket_count} five-minute buckets across this span, in "
        f"{len(incident.windows)} separate burst(s).",
        "  This is expected of a fault scoped to a narrow slice: measured against the WHOLE",
        "  population, its signal is diluted and only breaches threshold at its worst peaks;",
        "  the blast radius below is what makes it one continuous incident.",
        f"  Worst burst: {_fmt_dt(peak.start)} to {_fmt_dt(peak.end)} -- {metric_label} reached "
        f"{peak.peak_value:.6g} (expected {peak.expected_at_peak:.6g}), a robust z-score of "
        f"{peak.peak_z:.1f} sigma.",
        "",
    ]
    return lines


def _render_who_affected(ir: IncidentInvestigation, show_sql: bool) -> list[str]:
    incident = ir.incident
    walk_result = incident.representative_walk
    slice_ = walk_result.final_slice
    peak = incident.peak_window
    lines = [
        "WHO WAS AFFECTED (blast radius)",
        f"  {_humanize_slice(slice_)}",
        "",
        f"  Drill-down path (from the strongest burst, {_fmt_dt(peak.start)} to "
        f"{_fmt_dt(peak.end)}):",
    ]
    if not walk_result.path:
        lines.append("    (no refinement -- the anomaly could not be localised below the")
        lines.append("     whole population)")
    for i, step in enumerate(walk_result.path, start=1):
        lines.append(_render_step_line(i, step))
    lines.append(
        f"  Stopped because: "
        f"{_STOP_REASON_TEXT.get(walk_result.stop_reason, walk_result.stop_reason.value)} "
        f"({walk_result.stop_detail})"
    )
    lines.append("")
    if show_sql:
        lines.append("  SQL (one query per drill-down level, batched across dimensions):")
        for step in walk_result.path:
            lines.extend(_render_sql(step.sql))
            lines.extend(_render_sql(step.baseline_sql))
    return lines


def _render_probable_cause(ir: IncidentInvestigation, show_sql: bool) -> list[str]:
    correlation = ir.correlation
    lines = ["PROBABLE CAUSE"]
    if not correlation.candidates:
        lines.append("  No change_log entry correlates with this incident (temporally and")
        lines.append("  dimensionally). No probable cause identified from recorded changes.")
    else:
        top = correlation.candidates[0]
        lines.extend(
            [
                f"  [change #{top.change_id}] {top.change_type} / {top.component}: "
                f"{top.description}",
                f"  Changed at: {_fmt_dt(top.changed_at)} "
                f"({top.temporal_delta} before the incident's onset)",
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


def _render_revenue_impact(ir: IncidentInvestigation, show_sql: bool) -> list[str]:
    impact = ir.impact
    m = impact.methodology
    lines = [
        "SUBSCRIBERS AFFECTED AND ARR AT RISK",
        f"  Affected subscribers: {impact.affected_subscribers:,}",
        f"  ARR at risk: {_fmt_usd(impact.arr_at_risk_low)} - {_fmt_usd(impact.arr_at_risk_high)} "
        f"(expected {_fmt_usd(impact.arr_at_risk_expected)})",
        "  Methodology: churn_risk = base_monthly_churn x tenure_multiplier x "
        "severity_multiplier, capped at 1.0;",
        "    arr_at_risk = sum(churn_risk x monthly_arpu x 12) over affected subscribers,",
        "    measured over the full merged incident span, not a single 5-minute fragment.",
        f"    base_monthly_churn={m.base_monthly_churn} +/-{m.base_churn_variation:.0%} "
        f"(assumption, not measured); qoe_delta_ratio={m.qoe_delta_ratio:.3f} (from the "
        "incident's worst burst).",
        "    Every coefficient is a documented assumption -- see continuity/analysis/impact.py.",
        "",
    ]
    if show_sql:
        lines.append("  SQL:")
        lines.extend(_render_sql(impact.sql))
    return lines


def _recommended_action(ir: IncidentInvestigation) -> list[str]:
    slice_desc = _humanize_slice(ir.incident.final_slice)
    if ir.correlation.candidates:
        top = ir.correlation.candidates[0]
        action = (
            f"Roll back or hotfix change #{top.change_id} ({top.component}): "
            f"{top.description}. It is the top-ranked probable cause for the impact on "
            f"{slice_desc}."
        )
    else:
        action = (
            f"No change_log entry explains this incident. Escalate to the on-call team for "
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
        lines.append(f"  {stage.name:<16s} {stage.elapsed_ms:8.1f}ms  ({n_queries} queries)")
    lines.append(f"  {'total':<16s} {report.total_elapsed_ms:8.1f}ms")
    lines.append(
        "  (session_startup is the one-time mcp-clickhouse subprocess/connection cost, "
        "not analysis; a long-lived session pays it once, never per investigation.)"
    )
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

    if not report.incidents:
        lines.extend(_render_no_anomalies(report))
        lines.extend(_render_performance(report, show_sql))
        return "\n".join(lines)

    metric_label = METRICS[report.metric_name].label
    for i, ir in enumerate(report.incidents, start=1):
        if len(report.incidents) > 1:
            lines.append(f"{_RULE}\nINCIDENT {i} of {len(report.incidents)}\n{_RULE}")
            lines.append("")
        lines.extend(_render_what_happened(ir, metric_label))
        lines.extend(_render_who_affected(ir, show_sql))
        lines.extend(_render_probable_cause(ir, show_sql))
        lines.extend(_render_revenue_impact(ir, show_sql))
        lines.extend(_recommended_action(ir))

    lines.extend(_render_performance(report, show_sql))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


async def _run(
    config: ClickHouseConfig,
    metric_name: str,
    window: tuple[datetime, datetime],
    description: str,
    merge_gap: timedelta,
) -> InvestigationReport:
    async with ClickHouseMCPGateway(config) as gateway:
        return await investigate_pipeline(
            gateway,
            metric_name=metric_name,
            window=window,
            description=description,
            merge_gap=merge_gap,
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
    merge_gap_hours: float = typer.Option(
        DEFAULT_MERGE_GAP.total_seconds() / 3600,
        "--merge-gap-hours",
        help="Merge anomaly windows with the same blast radius into one incident if the "
        "gap between them is at most this many hours.",
    ),
    show_sql: bool = typer.Option(
        False, "--show-sql", help="Print the SQL behind every claim in the brief."
    ),
) -> None:
    """Run a full investigation (detect -> walk -> merge -> correlate -> quantify) with
    no LLM involved, and print a plain-text incident brief. Every number is traceable to
    a query the gateway actually ran; pass --show-sql to see them."""
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
        if merge_gap_hours < 0:
            raise InvestigationInputError(
                f"--merge-gap-hours must be >= 0, got {merge_gap_hours}"
            )
        merge_gap = timedelta(hours=merge_gap_hours)
        config = ClickHouseConfig.from_env()
    except (InvestigationInputError, KeyError, ValueError) as exc:
        typer.echo(f"INVESTIGATION FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        report = asyncio.run(_run(config, metric_name, window, description, merge_gap))
    except QueryError as exc:
        typer.echo(f"QUERY FAILED: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(render_brief(report, show_sql=show_sql))


if __name__ == "__main__":
    app()
