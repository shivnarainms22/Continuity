"""SSE endpoint: runs the deterministic investigation and streams one event per stage.

Deliberately re-orchestrates the same primitives `continuity/analysis/cli.py`'s
`investigate_pipeline` composes (detect -> walk -> merge -> refine -> correlate ->
quantify), in the same order, timing stages the same way -- rather than calling
`investigate_pipeline` itself and streaming its result. `investigate_pipeline` is a
single coroutine that only returns once every stage has finished, so calling it here
would mean waiting out the whole ~14s investigation and then emitting every SSE event
back-to-back at the end: technically an SSE response, but with all the buffering SSE
exists to avoid. Streaming incrementally is the entire point of this endpoint (see
ADR-001), so each stage is awaited and yielded as its own event as soon as it completes.

`continuity/analysis/` is not modified -- every call below is one of its existing public
functions, used exactly as `investigate_pipeline` itself uses it. The final "done" event
still calls that module's own `render_brief`, so the plain-text brief text matches the CLI
byte-for-byte for the same inputs.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from continuity.analysis.cli import (
    DEFAULT_GROUND_TRUTH_PATH,
    DEFAULT_MERGE_GAP,
    DEFAULT_REFINE_PADDING,
    IncidentInvestigation,
    InvestigationInputError,
    InvestigationReport,
    StageTiming,
    merge_windows_into_incidents,
    refine_incident,
    render_brief,
    resolve_investigation,
)
from continuity.analysis.correlate import correlate_changes
from continuity.analysis.detect import AnomalyWindow, DetectionResult, detect
from continuity.analysis.impact import compute_impact
from continuity.analysis.slices import Slice
from continuity.analysis.walk import WalkResult
from continuity.analysis.walk import walk as run_walk
from continuity.api.report_schema import (
    fetch_incident_series,
    iso,
    serialize_report,
    slice_predicates,
)
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway, QueryError

router = APIRouter(prefix="/api/investigate")


def _sse(event: str, data: dict) -> str:
    """One SSE frame. `data` is JSON on a single line -- SSE's `data:` field forbids
    literal newlines, so this can never be `json.dumps(..., indent=...)`."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _stage_event(timing: StageTiming, payload: dict) -> str:
    return _sse(
        "stage",
        {"stage": timing.name, "elapsed_ms": round(timing.elapsed_ms, 1), "payload": payload},
    )


async def stream_investigation(
    gateway: ClickHouseMCPGateway,
    *,
    metric_name: str,
    window: tuple[datetime, datetime],
    description: str,
) -> AsyncIterator[str]:
    """Yield one `event: stage` SSE frame per pipeline stage as it completes, then one
    `event: done` frame carrying the full rendered brief. Raises QueryError on the first
    failing query, same as `investigate_pipeline` -- the caller turns that into an
    `event: error` frame rather than a raised exception, since the response has already
    started streaming and cannot switch to an HTTP error status at that point.
    """
    total_started = time.perf_counter()

    idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    await gateway.query("SELECT 1")
    startup_timing = StageTiming(
        "session_startup",
        (time.perf_counter() - stage_started) * 1000,
        tuple(gateway.query_log[idx:]),
    )
    yield _stage_event(startup_timing, {})

    idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    detection: DetectionResult = await detect(gateway, Slice(), metric_name, window[0], window[1])
    detect_timing = StageTiming(
        "detect", (time.perf_counter() - stage_started) * 1000, tuple(gateway.query_log[idx:])
    )
    yield _stage_event(
        detect_timing,
        {
            "total_buckets": detection.total_buckets,
            "anomalous_buckets": detection.anomalous_buckets,
            "windows_found": len(detection.windows),
            "windows": [
                {"start": iso(w.start), "end": iso(w.end), "peak_z": w.peak_z}
                for w in detection.windows
            ],
        },
    )

    idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    entries: list[tuple[AnomalyWindow, WalkResult]] = []
    for anomaly in detection.windows:
        walk_result = await run_walk(
            gateway, metric_name=metric_name, window=(anomaly.start, anomaly.end)
        )
        entries.append((anomaly, walk_result))
    walk_timing = StageTiming(
        "walk", (time.perf_counter() - stage_started) * 1000, tuple(gateway.query_log[idx:])
    )
    incidents = merge_windows_into_incidents(entries, merge_gap=DEFAULT_MERGE_GAP)
    yield _stage_event(
        walk_timing,
        {
            "anomaly_windows": len(entries),
            "incidents_after_merge": len(incidents),
            "walks": [
                {
                    "start": iso(anomaly.start),
                    "end": iso(anomaly.end),
                    "blast_radius": slice_predicates(walk_result.final_slice),
                    "stop_reason": walk_result.stop_reason.value,
                }
                for anomaly, walk_result in entries
            ],
        },
    )

    idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    refined_incidents = [
        await refine_incident(
            gateway, incident, metric_name=metric_name, refine_padding=DEFAULT_REFINE_PADDING
        )
        for incident in incidents
    ]
    refine_timing = StageTiming(
        "refine", (time.perf_counter() - stage_started) * 1000, tuple(gateway.query_log[idx:])
    )
    yield _stage_event(
        refine_timing,
        {
            "incidents_refined": len(refined_incidents),
            "refined": [
                {
                    "used_fallback": r.used_fallback,
                    "span": {"start": iso(r.span[0]), "end": iso(r.span[1])},
                    "typical_multiple": float(r.typical_deviation_ratio) + 1.0,
                }
                for r in refined_incidents
            ],
        },
    )

    idx = len(gateway.query_log)
    stage_started = time.perf_counter()
    incident_results: list[IncidentInvestigation] = []
    incident_series: list[dict] = []
    for refined in refined_incidents:
        correlation = await correlate_changes(
            gateway,
            blast_radius=refined.final_slice,
            anomaly_window=refined.span,
            metric_name=metric_name,
        )
        impact = await compute_impact(
            gateway,
            slice_=refined.final_slice,
            window=refined.span,
            qoe_delta_ratio=refined.typical_deviation_ratio,
        )
        series = await fetch_incident_series(
            gateway, slice_=refined.final_slice, metric_name=metric_name, span=refined.span
        )
        incident_series.append(series)
        incident_results.append(
            IncidentInvestigation(
                incident=refined,
                correlation=correlation,
                impact=impact,
                qoe_delta_ratio=refined.typical_deviation_ratio,
            )
        )
    cq_timing = StageTiming(
        "correlate_and_quantify",
        (time.perf_counter() - stage_started) * 1000,
        tuple(gateway.query_log[idx:]),
    )
    yield _stage_event(
        cq_timing,
        {
            "incidents": len(incident_results),
            "detail": [
                {
                    "blast_radius": slice_predicates(ir.incident.final_slice),
                    "top_cause": (
                        ir.correlation.candidates[0].description
                        if ir.correlation.candidates
                        else None
                    ),
                    "affected_subscribers": ir.impact.affected_subscribers,
                    "arr_at_risk_expected": float(ir.impact.arr_at_risk_expected),
                }
                for ir in incident_results
            ],
        },
    )

    total_elapsed_ms = (time.perf_counter() - total_started) * 1000
    report = InvestigationReport(
        metric_name=metric_name,
        window=window,
        description=description,
        detection=detection,
        incidents=tuple(incident_results),
        stage_timings=(startup_timing, detect_timing, walk_timing, refine_timing, cq_timing),
        total_elapsed_ms=total_elapsed_ms,
    )
    brief = render_brief(report, show_sql=False)
    structured_report = serialize_report(report, incident_series)
    yield _sse(
        "done",
        {
            "total_elapsed_ms": round(total_elapsed_ms, 1),
            "brief": brief,
            "report": structured_report,
        },
    )


@router.get("/{incident_id}/stream")
async def investigate_stream(incident_id: str, request: Request) -> StreamingResponse:
    gateway: ClickHouseMCPGateway | None = getattr(request.app.state, "gateway", None)
    if gateway is None:
        raise HTTPException(status_code=503, detail="ClickHouse gateway is not ready")

    ground_truth_path: Path = getattr(
        request.app.state, "ground_truth_path", DEFAULT_GROUND_TRUTH_PATH
    )
    try:
        metric_name, window, description = resolve_investigation(
            metric=None,
            start=None,
            end=None,
            incident=incident_id,
            ground_truth_path=ground_truth_path,
        )
    except InvestigationInputError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def event_source() -> AsyncIterator[str]:
        try:
            async for frame in stream_investigation(
                gateway, metric_name=metric_name, window=window, description=description
            ):
                yield frame
        except QueryError as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
