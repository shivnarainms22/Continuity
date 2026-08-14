"""The Gemini investigation, streamed to the browser one measurement at a time.

`investigate_stream.py` streams the DETERMINISTIC walker -- detect, walk, refine,
correlate, quantify -- which is the control arm. This module streams the AGENT: the
same `Workflow` pipeline `scripts/run_agent.py` drives, with one SSE frame per tool call
the model chooses to make, as it makes it.

WHY PER TOOL CALL, not per stage. A full investigation is ~40-56s, essentially all of it
model round-trips (profiled: 67.7s of 67.9s was model time, 0.2s was ClickHouse). Four
stage-level frames would leave the user staring at "investigating..." for ~28s during
INVESTIGATE alone. Streaming each measurement turns that wait into the product's actual
claim -- you watch it form a hypothesis, split the population, read the lift, decide
whether to descend, and stop -- and it puts the ClickHouse query behind every step on
screen while it runs.

The frames carry the AUDIT LOG's copy of each call, so the SQL shown is the full query
text. `continuity.agent.agents._without_query_text` trims the MODEL's copy for prompt
economy; that trimming is deliberately not applied here, because showing the query is
the whole point of the view.

Errors are never swallowed (CLAUDE.md): a failed query or an incomplete pipeline
surfaces as an `error` frame naming the cause, never as a stream that simply stops.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from datetime import datetime

from google.adk.runners import InMemoryRunner
from google.genai import types

from continuity.agent.agents import (
    DEFAULT_MODEL_ID,
    AuditLogEntry,
    Model,
    build_investigation_pipeline,
    extract_pipeline_result,
    verify_brief_citations,
)
from continuity.agent.tools import build_function_tools
from continuity.analysis.detect import detect
from continuity.analysis.slices import Slice
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

APP_NAME = "continuity-agent-stream"

_STAGE_LABELS = {
    "investigate": "Isolating the blast radius",
    "correlate": "Finding what changed",
    "quantify": "Quantifying impact",
    "brief": "Writing the brief",
}


def _sse(event: str, data: dict) -> str:
    """One Server-Sent Event frame. Duplicated from investigate_stream.py rather than
    shared -- it is three lines of wire format, and the same deliberate duplication
    cli.py and report_schema.py already apply to `_parse_bucket_datetime`."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _tool_call_frame(entry: AuditLogEntry, *, elapsed_ms: float) -> dict:
    """One measurement, as the UI needs it: what was asked, what came back, and the
    query that produced it -- the `audit_index` here is the same value a later brief
    claim cites, so the UI can link a figure in the brief to the step that measured it.
    """
    result = {k: v for k, v in entry.result.items() if not k.lower().endswith("sql")}
    return {
        "audit_index": entry.index,
        "tool": entry.tool_name,
        "arguments": dict(entry.arguments),
        "sql": entry.sql,
        "result": result,
        "elapsed_ms": round(elapsed_ms, 1),
    }


async def stream_agent_investigation(
    gateway: ClickHouseMCPGateway,
    *,
    metric_name: str,
    window: tuple[datetime, datetime],
    description: str,
    model: Model = DEFAULT_MODEL_ID,
) -> AsyncIterator[str]:
    """Yield SSE frames for one agent investigation: `detect`, then `tool_call` per
    measurement and `stage` per stage boundary, then `done` (or `error`).

    DETECT runs first and without the model, exactly as `scripts/run_agent.py` does it,
    and the agent is handed the FULL extent of every detected window rather than the
    single worst one -- handing over one narrow peak window cost a real run its answer
    (see that script's comment). Keeping the two paths identical is what makes the
    head-to-head in `scripts/compare_arms.py` describe what the product actually does.
    """
    started = time.perf_counter()
    detection = await detect(gateway, Slice(), metric_name, window[0], window[1])
    if not detection.windows:
        yield _sse(
            "done",
            {
                "detected": False,
                "message": "No anomaly window breached threshold over this range.",
                "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return

    worst = max(detection.windows, key=lambda w: abs(w.peak_z))
    span_start = min(w.start for w in detection.windows)
    span_end = max(w.end for w in detection.windows)
    yield _sse(
        "detect",
        {
            "description": description,
            "metric": metric_name,
            "windows_found": len(detection.windows),
            "span": {"start": span_start.isoformat(), "end": span_end.isoformat()},
            "peak_z": worst.peak_z,
            "sql": detection.sql,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )

    tools = build_function_tools(gateway)
    pipeline, audit_log = build_investigation_pipeline(tools, model=model)
    pending: list[AuditLogEntry] = []
    audit_log.observer = pending.append

    runner = InMemoryRunner(agent=pipeline, app_name=APP_NAME)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id="web")
    prompt = (
        f"A population-level anomaly was detected on metric '{metric_name}'.\n"
        f"Anomaly window: {span_start.isoformat()} to {span_end.isoformat()}.\n"
        f"Within it, {len(detection.windows)} separate burst(s) breached threshold; "
        f"the worst peaked at robust z {worst.peak_z:.1f} "
        f"({worst.start.isoformat()} to {worst.end.isoformat()}).\n"
        f"Investigate where this is concentrated and produce the full brief."
    )

    seen_stages: set[str] = set()
    last_tool_at = time.perf_counter()
    async for event in runner.run_async(
        user_id="web",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        stage = getattr(event, "author", None)
        if stage in _STAGE_LABELS and stage not in seen_stages:
            seen_stages.add(stage)
            yield _sse("stage", {"stage": stage, "label": _STAGE_LABELS[stage]})
        while pending:
            entry = pending.pop(0)
            now = time.perf_counter()
            yield _sse("tool_call", _tool_call_frame(entry, elapsed_ms=(now - last_tool_at) * 1000))
            last_tool_at = now

    while pending:  # anything recorded after the final event still gets reported
        entry = pending.pop(0)
        yield _sse("tool_call", _tool_call_frame(entry, elapsed_ms=0.0))

    final_session = await runner.session_service.get_session(
        app_name=APP_NAME, user_id="web", session_id=session.id
    )
    try:
        result = extract_pipeline_result(final_session.state)
    except KeyError as exc:
        yield _sse("error", {"message": str(exc)})
        return

    citations_ok = True
    citation_error = None
    try:
        verify_brief_citations(result.brief, audit_log)
    except ValueError as exc:
        citations_ok = False
        citation_error = str(exc)

    yield _sse(
        "done",
        {
            "detected": True,
            "total_elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "tool_calls": len(audit_log.entries),
            "investigation": result.investigation.model_dump(),
            "correlation": result.correlation.model_dump(),
            "quantify": result.quantify.model_dump(),
            "brief": result.brief.model_dump(),
            "citations_verified": citations_ok,
            "citation_error": citation_error,
        },
    )
