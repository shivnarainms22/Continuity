"""Head-to-head: the Gemini investigator vs the deterministic walker.

Runs BOTH arms over the SAME incidents (from ``data/ground_truth.json``), starting from
the SAME detected evidence, scored by the SAME code -- so "does the agent add value?"
becomes a number instead of an argument. See ``docs/superpowers/plans/
2026-08-09-continuity-03-agent-pipeline.md`` Task 6.

FAIRNESS, spelled out because it is the whole point of this script existing:

* Both arms see the SAME starting evidence. For every incident, ``detect()`` (no LLM,
  ``continuity.analysis.detect``) runs ONCE over the same padded search window with the
  same metric -- a pure function, so the walker's own internal ``detect()`` call inside
  ``investigate_pipeline`` and the agent's own ``detect()`` call in ``run_agent_arm``
  compute the identical windows. Neither arm is handed a hint the other lacks.
* Both are scored by the identical functions below (``blast_radius_score``,
  ``attribution_correct``, ``_impact_delta``), never by separate logic per arm.
* The walker used here is ``continuity.analysis.cli.investigate_pipeline`` -- the real,
  already-built control arm (detect -> walk per window -> merge -> refine -> correlate ->
  quantify), not a weakened stand-in. Its own documented limitation (``walk.py``'s
  ``DEFAULT_DIMENSIONS`` excludes ``title_id`` -- a per-title fault needs an explicit,
  caller-supplied dimension list) is left exactly as built: reporting where that costs it
  is more informative than hiding it behind a hand-tuned dimension list this script would
  be inventing only for this comparison.
* The agent is driven exactly as ``scripts/run_agent.py`` drives it: DETECT runs first
  (no model), then the agent is handed the FULL extent across every detected window (see
  that script's own comment on why a single peak window starved a real run of the
  evidence it needed) and the ``Workflow`` pipeline (INVESTIGATE -> CORRELATE -> QUANTIFY
  -> BRIEF) takes it from there.

COST. Every real (non ``--skip-agent``) run of this script spends real tokens: exactly
ONE agent investigation per incident named on the command line (default: every incident
in ``data/ground_truth.json`` -- 3 real incidents + 1 decoy = 4 calls). Use
``--skip-agent`` to iterate on the walker, the scoring functions, and the report/JSON
rendering below for free before spending any of that budget.

Guarding. Per CLAUDE.md's "errors are never swallowed" and this task's own rule that one
arm's failure on one incident must not abort the run: ``_guarded_arm`` catches any
exception -- a ClickHouse failure, a schema-invalid model response, an incomplete
pipeline -- and turns it into a loud, specific ``ArmResult(ok=False, error=...)`` rather
than raising. The exception text is always recorded and printed, never discarded.

Read-only throughout: every call below is a ``SELECT`` through the existing gateway/tool
layer. This script never writes to ClickHouse and never touches ``continuity/`` -- it
only composes what sub-projects 2 and 3 already built. Every window comes from
``data/ground_truth.json``; no date is ever hardcoded here (see CLAUDE.md).

Run:
    uv run python scripts/compare_arms.py --skip-agent        # free, walker-only
    uv run python scripts/compare_arms.py                     # spends real tokens
    uv run python scripts/compare_arms.py --incidents INC-APP-ROKU-820
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

from continuity.agent.agents import build_investigation_pipeline, extract_pipeline_result
from continuity.agent.tools import AnalysisTools, build_function_tools
from continuity.analysis.cli import (
    INCIDENT_SEARCH_PADDING,
    IncidentInvestigation,
    InvestigationReport,
    investigate_pipeline,
)
from continuity.analysis.detect import detect
from continuity.analysis.slices import Slice
from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

GROUND_TRUTH = Path("data/ground_truth.json")
RESULTS_JSON = Path("results/comparison.json")
DEFAULT_MODEL_ID = "gemini-3.6-flash"
APP_NAME = "continuity_compare"

_SEP = "=" * 78
_RULE = "-" * 78


# ---------------------------------------------------------------------------
# Shared result shape -- same fields for both arms, so the report and the JSON
# never special-case one arm's schema over the other's.
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """One arm's (walker or agent) investigation of one incident."""

    incident_id: str
    arm: str
    ok: bool
    error: str | None = None
    skipped: bool = False
    final_slice: dict[str, str] = field(default_factory=dict)
    window_start: str | None = None
    window_end: str | None = None
    stop_reason: str | None = None
    top_change_id: str | None = None
    confidence: str | None = None
    unresolved: bool | None = None
    affected_subscribers: int | None = None
    arr_at_risk_low: str | None = None
    arr_at_risk_expected: str | None = None
    arr_at_risk_high: str | None = None
    wall_ms: float = 0.0
    model_turns: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    queries: int = 0
    note: str = ""


@dataclass
class IncidentScore:
    """One ground-truth incident's evaluation: both arms' results plus every score
    computed against the same ground truth and the same deterministic reference."""

    incident_id: str
    kind: str
    is_decoy: bool
    true_predicate: dict[str, str]
    true_change_id: str | None
    reference_impact: dict[str, Any] | None
    walker: ArmResult
    agent: ArmResult
    walker_blast: dict[str, Any] | None = None
    agent_blast: dict[str, Any] | None = None
    walker_attribution: bool | None = None
    agent_attribution: bool | None = None
    walker_decoy_flagged: bool | None = None
    agent_decoy_flagged: bool | None = None


class Usage:
    """Accumulates token counts across every model turn in one agent investigation.

    Identical to ``scripts/run_agent.py``'s own ``Usage`` -- duplicated rather than
    imported, matching this project's existing convention of small, script-local
    helpers (see ``acceptance_sp2.py``'s own ``_parse``/``_fmt``/``_window``) rather
    than inventing a shared module inside ``scripts/`` for a ~15-line class used in
    exactly two places.
    """

    def __init__(self) -> None:
        self.prompt = 0
        self.output = 0
        self.turns = 0

    def add(self, event: Any) -> None:
        meta = getattr(event, "usage_metadata", None)
        if meta is None:
            return
        self.turns += 1
        self.prompt += getattr(meta, "prompt_token_count", 0) or 0
        self.output += getattr(meta, "candidates_token_count", 0) or 0


# ---------------------------------------------------------------------------
# Small parsing helpers -- ground truth is the only source of every date/predicate.
# ---------------------------------------------------------------------------


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)


def _metric_for(incident: dict[str, Any]) -> str:
    effects = incident.get("effects") or []
    return str(effects[0]["metric"]) if effects else "rebuffer"


def _fmt_slice(d: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items()) if d else "(whole population)"


# ---------------------------------------------------------------------------
# Walker arm -- continuity.analysis.cli.investigate_pipeline exactly as built.
# ---------------------------------------------------------------------------


def _overlap_seconds(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> float:
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0.0, (end - start).total_seconds())


def _pick_incident(
    report: InvestigationReport, true_window: tuple[datetime, datetime]
) -> tuple[IncidentInvestigation | None, float | None]:
    """The merged incident whose (refined) span overlaps `true_window` the most.

    A padded search window can produce more than one merged incident (noise, or a
    genuinely separate event nearby); this picks the one actually relevant to the
    ground-truth incident under test, exactly as a human reading the brief would.
    """
    if not report.incidents:
        return None, None
    scored = [(_overlap_seconds(ir.incident.span, true_window), ir) for ir in report.incidents]
    best_overlap, best_ir = max(scored, key=lambda item: item[0])
    return best_ir, best_overlap


async def run_walker_arm(
    gateway: ClickHouseMCPGateway,
    incident: dict[str, Any],
    metric: str,
    search_window: tuple[datetime, datetime],
    true_window: tuple[datetime, datetime],
) -> ArmResult:
    """The control arm: detect -> walk (per window) -> merge -> refine -> correlate ->
    quantify, composed exactly as sub-project 2 built it. No LLM call anywhere in this
    call. `search_window` is the SAME padded window handed to the agent's own DETECT
    step, so both arms scan the identical population-level slice/window before
    diverging into their own (very different) drill-down strategies.
    """
    t0 = time.perf_counter()
    report = await investigate_pipeline(
        gateway,
        metric_name=metric,
        window=search_window,
        description=f"walker arm: {incident['incident_id']}",
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    queries = sum(len(s.queries) for s in report.stage_timings)

    ir, overlap_s = _pick_incident(report, true_window)
    if ir is None:
        return ArmResult(
            incident_id=incident["incident_id"],
            arm="walker",
            ok=True,
            wall_ms=wall_ms,
            queries=queries,
            note="no incident detected/merged over the search window",
        )

    top = ir.correlation.candidates[0] if ir.correlation.candidates else None
    note = "deterministic walker -- $0 model cost, no LLM calls"
    if overlap_s == 0.0:
        note = "picked incident does not overlap the ground-truth window at all; " + note
    return ArmResult(
        incident_id=incident["incident_id"],
        arm="walker",
        ok=True,
        final_slice=dict(ir.incident.final_slice.predicates),
        window_start=ir.incident.span[0].isoformat(),
        window_end=ir.incident.span[1].isoformat(),
        stop_reason=ir.incident.population_incident.representative_walk.stop_reason.value,
        top_change_id=str(top.change_id) if top is not None else None,
        affected_subscribers=ir.impact.affected_subscribers,
        arr_at_risk_low=str(ir.impact.arr_at_risk_low),
        arr_at_risk_expected=str(ir.impact.arr_at_risk_expected),
        arr_at_risk_high=str(ir.impact.arr_at_risk_high),
        wall_ms=wall_ms,
        queries=queries,
        note=note,
    )


# ---------------------------------------------------------------------------
# Agent arm -- driven exactly as scripts/run_agent.py drives it.
# ---------------------------------------------------------------------------


async def run_agent_arm(
    gateway: ClickHouseMCPGateway,
    incident: dict[str, Any],
    metric: str,
    search_window: tuple[datetime, datetime],
    model: str,
) -> ArmResult:
    """The Gemini-driven investigator, given the SAME starting evidence as the walker:
    DETECT runs once (no model) over `search_window`, and the agent is handed the full
    detected extent across every window -- ``scripts/run_agent.py``'s own protocol,
    reproduced here so both arms are measured identically.
    """
    t0 = time.perf_counter()
    detection = await detect(gateway, Slice(), metric, search_window[0], search_window[1])
    if not detection.windows:
        wall_ms = (time.perf_counter() - t0) * 1000
        return ArmResult(
            incident_id=incident["incident_id"],
            arm="agent",
            ok=True,
            wall_ms=wall_ms,
            note="DETECT found no anomaly windows -- nothing handed to the agent",
        )
    worst = max(detection.windows, key=lambda w: abs(w.peak_z))
    span_start = min(w.start for w in detection.windows)
    span_end = max(w.end for w in detection.windows)

    tools = build_function_tools(gateway)
    pipeline, audit_log = build_investigation_pipeline(tools, model=model)
    runner = InMemoryRunner(agent=pipeline, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="cmp")
    prompt = (
        f"A population-level anomaly was detected on metric '{metric}'.\n"
        f"Anomaly window: {span_start.isoformat()} to {span_end.isoformat()}.\n"
        f"Within it, {len(detection.windows)} separate burst(s) breached threshold; "
        f"the worst peaked at robust z {worst.peak_z:.1f} "
        f"({worst.start.isoformat()} to {worst.end.isoformat()}).\n"
        "Investigate where this is concentrated and produce the full brief."
    )

    usage = Usage()
    async for event in runner.run_async(
        user_id="cmp",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        usage.add(event)
    wall_ms = (time.perf_counter() - t0) * 1000

    final_session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id="cmp", session_id=session.id
    )
    result = extract_pipeline_result(final_session.state)

    final_slice = {d.dimension: d.value for d in result.investigation.final_slice}
    top_id = result.correlation.top_candidate_change_id
    return ArmResult(
        incident_id=incident["incident_id"],
        arm="agent",
        ok=True,
        final_slice=final_slice,
        window_start=result.investigation.window_start,
        window_end=result.investigation.window_end,
        stop_reason=result.investigation.stop_reason,
        top_change_id=str(top_id) if top_id is not None else None,
        confidence=result.correlation.confidence,
        unresolved=result.brief.unresolved,
        affected_subscribers=result.quantify.affected_subscribers,
        arr_at_risk_low=result.quantify.arr_at_risk_low,
        arr_at_risk_expected=result.quantify.arr_at_risk_expected,
        arr_at_risk_high=result.quantify.arr_at_risk_high,
        wall_ms=wall_ms,
        model_turns=usage.turns,
        prompt_tokens=usage.prompt,
        output_tokens=usage.output,
        total_tokens=usage.prompt + usage.output,
        tool_calls=len(audit_log.entries),
        note=f"gemini model={model}",
    )


async def _guarded_arm(coro: Any, incident_id: str, arm: str) -> ArmResult:
    """Runs one arm's investigation of one incident. Any exception -- a ClickHouse
    failure, a schema-invalid model response, an incomplete pipeline -- is caught here
    and turned into a loud, specific `ArmResult` instead of aborting the whole
    comparison run; the exception text is always recorded on `.error` and printed by
    the caller, never discarded. This is the one place a broad `except Exception` is
    correct: it is the task's own requirement ("one failure must be reported as a loud
    specific FAIL, not abort the run"; "a schema-invalid result or error is a RESULT,
    not a crash"), and it always re-surfaces the failure rather than hiding it.
    """
    try:
        return await coro
    except Exception as exc:
        return ArmResult(
            incident_id=incident_id, arm=arm, ok=False, error=f"{type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------------------------
# Reference impact -- the deterministic engine's own output on the TRUE span/predicate.
# ---------------------------------------------------------------------------


async def _reference_impact(
    gateway: ClickHouseMCPGateway, incident: dict[str, Any], metric: str
) -> dict[str, Any]:
    """What the deterministic engine (``quantify_impact`` -- no LLM) produces when
    given the TRUE ground-truth predicate and TRUE span, unpadded. This is the best
    available reference for impact accuracy -- explicitly NOT an oracle: it is the
    same heuristic and the same self-measured severity every arm's own QUANTIFY step
    uses, just fed the answer key's span/predicate instead of a discovered one.
    """
    tools = AnalysisTools(gateway)
    start, end = _parse(incident["start"]), _parse(incident["end"])
    try:
        return await tools.quantify_impact(
            incident["predicate"], metric, start.isoformat(), end.isoformat()
        )
    except Exception as exc:  # defensive: quantify_impact itself already catches QueryError
        return {"error": f"{type(exc).__name__}: {exc}", "error_type": "harness_failure"}


# ---------------------------------------------------------------------------
# Scoring -- the SAME functions score both arms; nothing here special-cases one.
# ---------------------------------------------------------------------------


def _pairs(predicate: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted((str(k), str(v)) for k, v in predicate.items())


def blast_radius_score(found: dict[str, str], true: dict[str, str]) -> dict[str, Any]:
    """Exact-match plus partial credit (precision/recall over dimension/value pairs),
    so "found roku but missed 8.2.0" is distinguishable from "found something
    unrelated" -- both would fail `exact`, but only the first keeps any precision."""
    found_pairs = set(_pairs(found))
    true_pairs = set(_pairs(true))
    exact = found_pairs == true_pairs
    tp = len(found_pairs & true_pairs)
    precision = (tp / len(found_pairs)) if found_pairs else (1.0 if not true_pairs else 0.0)
    recall = (tp / len(true_pairs)) if true_pairs else (1.0 if not found_pairs else 0.0)
    return {
        "exact": exact,
        "precision": precision,
        "recall": recall,
        "found": sorted(found_pairs),
        "true": sorted(true_pairs),
    }


def attribution_correct(top_change_id: str | None, true_change_id: str) -> bool:
    return top_change_id is not None and str(top_change_id) == str(true_change_id)


def _impact_delta(value: str | int | None, reference: str | int | None) -> float | None:
    """Relative difference `(value - reference) / reference`, or `None` when either
    side is missing/zero -- never a divide-by-zero, never a fabricated 0%."""
    if value is None or reference is None:
        return None
    try:
        v, r = Decimal(str(value)), Decimal(str(reference))
    except InvalidOperation:
        return None
    if r == 0:
        return None
    return float((v - r) / r)


# ---------------------------------------------------------------------------
# Per-incident orchestration.
# ---------------------------------------------------------------------------


async def evaluate_incident(
    gateway: ClickHouseMCPGateway,
    incident: dict[str, Any],
    *,
    agent_model: str,
    run_agent: bool,
) -> IncidentScore:
    metric = _metric_for(incident)
    true_start, true_end = _parse(incident["start"]), _parse(incident["end"])
    true_window = (true_start, true_end)
    search_window = (true_start - INCIDENT_SEARCH_PADDING, true_end + INCIDENT_SEARCH_PADDING)

    walker = await _guarded_arm(
        run_walker_arm(gateway, incident, metric, search_window, true_window),
        incident["incident_id"],
        "walker",
    )
    if run_agent:
        agent = await _guarded_arm(
            run_agent_arm(gateway, incident, metric, search_window, agent_model),
            incident["incident_id"],
            "agent",
        )
    else:
        agent = ArmResult(
            incident_id=incident["incident_id"],
            arm="agent",
            ok=False,
            skipped=True,
            note="skipped (--skip-agent) -- not run, not scored, not an error",
        )

    is_decoy = bool(incident["is_decoy"])
    score = IncidentScore(
        incident_id=incident["incident_id"],
        kind=incident["kind"],
        is_decoy=is_decoy,
        true_predicate=dict(incident["predicate"]),
        true_change_id=None,
        reference_impact=None,
        walker=walker,
        agent=agent,
    )

    if is_decoy:
        decoy_title = incident["predicate"].get("title_id")
        if walker.ok:
            score.walker_decoy_flagged = walker.final_slice.get("title_id") == decoy_title
        if agent.ok:
            score.agent_decoy_flagged = agent.final_slice.get("title_id") == decoy_title
        return score

    score.true_change_id = str(incident["change"]["change_id"])
    if walker.ok:
        score.walker_blast = blast_radius_score(walker.final_slice, score.true_predicate)
        score.walker_attribution = attribution_correct(walker.top_change_id, score.true_change_id)
    if agent.ok:
        score.agent_blast = blast_radius_score(agent.final_slice, score.true_predicate)
        score.agent_attribution = attribution_correct(agent.top_change_id, score.true_change_id)
    score.reference_impact = await _reference_impact(gateway, incident, metric)
    return score


# ---------------------------------------------------------------------------
# Rendering -- plain text, matching acceptance_sp2.py's report style.
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def _render_arm(label: str, arm: ArmResult, score: IncidentScore) -> list[str]:
    lines = [f"  [{label:<6}]"]
    if arm.skipped:
        lines.append(f"      SKIPPED -- {arm.note}")
        return lines
    if not arm.ok:
        lines.append(f"      FAILED -- {arm.error}")
        return lines

    if score.is_decoy:
        flagged = score.walker_decoy_flagged if label == "walker" else score.agent_decoy_flagged
        if flagged is None:
            lines.append("      decoy check: skipped")
        else:
            verdict = "FLAGGED THE DECOY -- FAILURE" if flagged else "did not flag it -- correct"
            lines.append(
                f"      decoy check : {verdict} (final slice: {_fmt_slice(arm.final_slice)})"
            )
    else:
        blast = score.walker_blast if label == "walker" else score.agent_blast
        if blast is not None:
            match_text = "EXACT MATCH" if blast["exact"] else "NOT EXACT"
            lines.append(
                f"      blast radius : {_fmt_slice(arm.final_slice)} -- {match_text} "
                f"(precision={blast['precision']:.2f}, recall={blast['recall']:.2f})"
            )
        attr = score.walker_attribution if label == "walker" else score.agent_attribution
        if attr is not None:
            attr_text = "CORRECT" if attr else "WRONG"
            lines.append(
                f"      attribution  : top=#{arm.top_change_id} vs true=#{score.true_change_id} "
                f"-> {attr_text}"
            )
        if arm.unresolved:
            lines.append(
                "      UNRESOLVED   : localisation could not be corroborated by any "
                "plausible change -- the brief itself says so; treat the impact figure "
                "below as an unreliable estimate, not a confident finding"
            )
        ref = score.reference_impact
        if ref and "error" not in ref:
            subs_delta = _impact_delta(arm.affected_subscribers, ref.get("affected_subscribers"))
            arr_delta = _impact_delta(arm.arr_at_risk_expected, ref.get("arr_at_risk_expected"))
            lines.append(
                f"      impact       : subscribers={arm.affected_subscribers} "
                f"(ref {ref.get('affected_subscribers')}, {_fmt_pct(subs_delta)}); "
                f"ARR expected ${arm.arr_at_risk_expected} "
                f"(ref ${ref.get('arr_at_risk_expected')}, {_fmt_pct(arr_delta)})"
            )
        elif ref:
            lines.append(f"      impact       : reference unavailable -- {ref.get('error')}")
    if label == "walker":
        lines.append(
            f"      cost         : {arm.wall_ms:.0f}ms wall, $0 model cost (no LLM calls), "
            f"{arm.queries} SQL queries"
        )
    else:
        lines.append(
            f"      cost         : {arm.wall_ms:.0f}ms wall, {arm.model_turns} model turns, "
            f"{arm.total_tokens:,} tokens ({arm.prompt_tokens:,} prompt + "
            f"{arm.output_tokens:,} output), {arm.tool_calls} tool calls"
        )
    if arm.note:
        lines.append(f"      note         : {arm.note}")
    return lines


def _print_incident(score: IncidentScore) -> None:
    print()
    print(_RULE)
    tag = " (DECOY -- must produce no incident)" if score.is_decoy else ""
    print(f"{score.incident_id} [{score.kind}]{tag}")
    if not score.is_decoy:
        print(f"  true blast radius : {_fmt_slice(score.true_predicate)}")
        print(f"  true cause        : change #{score.true_change_id}")
        ref = score.reference_impact
        if ref and "error" not in ref:
            print(
                "  reference impact  : (deterministic engine, TRUE span -- a reference, "
                "not an oracle)"
            )
            print(
                f"                      subscribers={ref['affected_subscribers']}, "
                f"ARR expected ${ref['arr_at_risk_expected']}"
            )
        elif ref:
            print(f"  reference impact  : unavailable -- {ref.get('error')}")
    for label, arm in (("walker", score.walker), ("agent", score.agent)):
        for line in _render_arm(label, arm, score):
            print(line)


def _counts(scores: list[IncidentScore], label: str) -> dict[str, int]:
    real = [s for s in scores if not s.is_decoy]
    decoys = [s for s in scores if s.is_decoy]

    def arm_of(s: IncidentScore) -> ArmResult:
        return s.walker if label == "walker" else s.agent

    def blast_of(s: IncidentScore) -> dict[str, Any] | None:
        return s.walker_blast if label == "walker" else s.agent_blast

    def attr_of(s: IncidentScore) -> bool | None:
        return s.walker_attribution if label == "walker" else s.agent_attribution

    def flagged_of(s: IncidentScore) -> bool | None:
        return s.walker_decoy_flagged if label == "walker" else s.agent_decoy_flagged

    exact = sum(1 for s in real if blast_of(s) and blast_of(s)["exact"])
    attrib = sum(1 for s in real if attr_of(s))
    decoy_ok = sum(1 for s in decoys if flagged_of(s) is False)
    errors = sum(1 for s in scores if not arm_of(s).ok and not arm_of(s).skipped)
    skipped = sum(1 for s in scores if arm_of(s).skipped)
    return {
        "exact_matches": exact,
        "of_real_incidents": len(real),
        "skipped": skipped,
        "correct_attributions": attrib,
        "decoys_correctly_ignored": decoy_ok,
        "of_decoys": len(decoys),
        "errors": errors,
    }


def _print_summary(scores: list[IncidentScore]) -> None:
    print()
    print(_SEP)
    print("SUMMARY")
    print(_SEP)
    print(
        f"{'incident':<22}{'arm':<8}{'exact':<7}{'attrib':<8}"
        f"{'decoy_ok':<12}{'wall_ms':>10}{'tokens':>10}"
    )
    for s in scores:
        for label, arm in (("walker", s.walker), ("agent", s.agent)):
            if arm.skipped:
                exact, attrib, decoy_ok = "SKIP", "SKIP", "SKIP"
            elif not arm.ok:
                exact, attrib, decoy_ok = "ERR", "ERR", "ERR"
            elif s.is_decoy:
                flagged = s.walker_decoy_flagged if label == "walker" else s.agent_decoy_flagged
                exact, attrib = "-", "-"
                decoy_ok = "n/a" if flagged is None else ("N (FLAGGED)" if flagged else "Y")
            else:
                blast = s.walker_blast if label == "walker" else s.agent_blast
                attr = s.walker_attribution if label == "walker" else s.agent_attribution
                exact = "n/a" if blast is None else ("Y" if blast["exact"] else "N")
                attrib = "n/a" if attr is None else ("Y" if attr else "N")
                decoy_ok = "-"
            print(
                f"{s.incident_id:<22}{label:<8}{exact:<7}{attrib:<8}{decoy_ok:<12}"
                f"{arm.wall_ms:>10.0f}{arm.total_tokens:>10,}"
            )


def _print_verdict(scores: list[IncidentScore]) -> None:
    print()
    print(_SEP)
    print("VERDICT")
    print(_SEP)
    w = _counts(scores, "walker")
    a = _counts(scores, "agent")
    print(
        f"Walker: {w['exact_matches']}/{w['of_real_incidents']} exact blast-radius matches, "
        f"{w['correct_attributions']}/{w['of_real_incidents']} correct attributions, "
        f"{w['decoys_correctly_ignored']}/{w['of_decoys']} decoy(s) correctly ignored, "
        f"{w['errors']} error(s). Cost: $0, no model calls, by construction."
    )
    if a["skipped"] >= len(scores):
        print("Agent : skipped (--skip-agent) -- not run, no comparison to draw.")
        return
    agent_wall = sum(s.agent.wall_ms for s in scores if s.agent.ok)
    agent_tokens = sum(s.agent.total_tokens for s in scores if s.agent.ok)
    print(
        f"Agent : {a['exact_matches']}/{a['of_real_incidents']} exact blast-radius matches, "
        f"{a['correct_attributions']}/{a['of_real_incidents']} correct attributions, "
        f"{a['decoys_correctly_ignored']}/{a['of_decoys']} decoy(s) correctly ignored, "
        f"{a['errors']} error(s). Cost: {agent_tokens:,} tokens, {agent_wall / 1000:.1f}s "
        "wall total."
    )
    print()
    if w["exact_matches"] > a["exact_matches"]:
        print(
            "On blast-radius exactness, the WALKER BEATS THE AGENT this run "
            f"({w['exact_matches']} vs {a['exact_matches']} exact matches) -- the model did "
            "not add measurable localisation value here, and it spent real tokens/time to "
            "not add it."
        )
    elif a["exact_matches"] > w["exact_matches"]:
        print(
            "On blast-radius exactness, the AGENT BEATS THE WALKER this run "
            f"({a['exact_matches']} vs {w['exact_matches']} exact matches)."
        )
    else:
        print(
            "On blast-radius exactness, the WALKER MATCHES THE AGENT this run "
            f"({w['exact_matches']} vs {a['exact_matches']} exact matches) -- at zero cost for "
            "the walker versus a real token/time cost for the agent."
        )
    if w["correct_attributions"] > a["correct_attributions"]:
        print(
            "On attribution, the walker beats the agent "
            f"({w['correct_attributions']} vs {a['correct_attributions']})."
        )
    elif a["correct_attributions"] > w["correct_attributions"]:
        print(
            "On attribution, the agent beats the walker "
            f"({a['correct_attributions']} vs {w['correct_attributions']})."
        )
    else:
        print(
            "On attribution, the walker and agent tie "
            f"({w['correct_attributions']} vs {a['correct_attributions']})."
        )


def _print_where_each_better(scores: list[IncidentScore]) -> None:
    print()
    print(_SEP)
    print("WHERE EACH ARM IS BETTER")
    print(_SEP)
    walker_wins: list[str] = []
    agent_wins: list[str] = []
    for s in scores:
        if s.is_decoy or not s.walker.ok or not s.agent.ok or s.agent.skipped:
            # A head-to-head bullet requires BOTH arms to have actually produced a
            # result on this incident -- an agent that was skipped or errored is
            # neither "worse" nor "better" than the walker here, it simply did not run.
            continue
        wb, ab = s.walker_blast, s.agent_blast
        if wb and ab and wb["exact"] and not ab["exact"]:
            walker_wins.append(
                f"{s.incident_id}: walker found the exact blast radius, agent did not"
            )
        if wb and ab and ab["exact"] and not wb["exact"]:
            agent_wins.append(
                f"{s.incident_id}: agent found the exact blast radius, walker did not"
            )
        if s.walker_attribution and not s.agent_attribution:
            walker_wins.append(f"{s.incident_id}: walker attributed correctly, agent did not")
        if s.agent_attribution and not s.walker_attribution:
            agent_wins.append(f"{s.incident_id}: agent attributed correctly, walker did not")
    walker_wins.append("cost: $0 and zero model calls on every incident, by construction")
    walker_wins.append("determinism: byte-identical result on every re-run, nothing to retry")
    if any(s.agent.ok and s.agent.confidence for s in scores):
        agent_wins.append(
            "judgement: engages with disconfirming evidence and states a confidence level "
            "(low/medium/high) with reasoning -- the walker has no mechanism to do either"
        )
    print("Walker is better where:")
    for line in walker_wins:
        print(f"  - {line}")
    print("Agent is better where:")
    if not agent_wins:
        print("  - (none observed this run beyond the structural capability noted above)")
    for line in agent_wins:
        print(f"  - {line}")


def _print_total_cost(scores: list[IncidentScore], *, skipped_agent: bool) -> None:
    print()
    print(_SEP)
    print("TOTAL COST THIS RUN")
    print(_SEP)
    print("  walker: $0, 0 tokens -- no model calls, by construction (deterministic SQL only)")
    if skipped_agent:
        print("  agent : skipped (--skip-agent)")
        return
    n = sum(1 for s in scores if s.agent.ok)
    tokens = sum(s.agent.total_tokens for s in scores if s.agent.ok)
    turns = sum(s.agent.model_turns for s in scores if s.agent.ok)
    wall = sum(s.agent.wall_ms for s in scores if s.agent.ok)
    avg_s = wall / max(n, 1) / 1000
    print(
        f"  agent : {n} investigation(s) run, {tokens:,} total tokens, {turns} model turns, "
        f"{wall / 1000:.1f}s total wall clock ({avg_s:.1f}s average per investigation)"
    )


def _to_jsonable(scores: list[IncidentScore], *, model: str, skipped_agent: bool) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().isoformat(),
        "agent_model": model,
        "agent_skipped": skipped_agent,
        "incidents": [
            {
                "incident_id": s.incident_id,
                "kind": s.kind,
                "is_decoy": s.is_decoy,
                "true_predicate": s.true_predicate,
                "true_change_id": s.true_change_id,
                "reference_impact": s.reference_impact,
                "walker": asdict(s.walker),
                "agent": asdict(s.agent),
                "scores": {
                    "walker_blast_radius": s.walker_blast,
                    "agent_blast_radius": s.agent_blast,
                    "walker_attribution_correct": s.walker_attribution,
                    "agent_attribution_correct": s.agent_attribution,
                    "walker_decoy_flagged": s.walker_decoy_flagged,
                    "agent_decoy_flagged": s.agent_decoy_flagged,
                },
            }
            for s in scores
        ],
        "summary": {"walker": _counts(scores, "walker"), "agent": _counts(scores, "agent")},
    }


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Head-to-head: the Gemini investigator vs the deterministic walker, "
        "on the same incidents, over the same primitives, scored identically."
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="Agent model id (default: %(default)s)."
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Walker-only run: $0, no model calls. Use to debug the harness before "
        "spending tokens.",
    )
    parser.add_argument(
        "--incidents",
        default=None,
        help="Comma-separated incident ids to run (default: every incident in "
        "ground_truth.json).",
    )
    parser.add_argument("--out", default=str(RESULTS_JSON), help="Path for the JSON report.")
    args = parser.parse_args()

    load_dotenv(override=False)
    if not GROUND_TRUTH.exists():
        print(f"{GROUND_TRUTH} not found. Run the loader first.")
        return 2
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    all_incidents = truth["incidents"]
    if args.incidents:
        wanted = set(args.incidents.split(","))
        incidents = [i for i in all_incidents if i["incident_id"] in wanted]
        missing = wanted - {i["incident_id"] for i in incidents}
        if missing:
            print(f"Unknown incident id(s): {sorted(missing)}")
            return 2
    else:
        incidents = all_incidents

    print(_SEP)
    print("AGENT vs WALKER -- head-to-head comparison, same incidents, same primitives")
    print(_SEP)
    if args.skip_agent:
        print("\n*** --skip-agent: walker-only run. Agent rows are not run, not scored. ***")
    else:
        print(f"\nAgent model: {args.model}")

    scores: list[IncidentScore] = []
    async with ClickHouseMCPGateway(ClickHouseConfig.from_env()) as gateway:
        for incident in incidents:
            score = await evaluate_incident(
                gateway, incident, agent_model=args.model, run_agent=not args.skip_agent
            )
            scores.append(score)
            _print_incident(score)

    _print_summary(scores)
    _print_verdict(scores)
    _print_where_each_better(scores)
    _print_total_cost(scores, skipped_agent=args.skip_agent)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(scores, model=args.model, skipped_agent=args.skip_agent)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nMachine-readable comparison written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
