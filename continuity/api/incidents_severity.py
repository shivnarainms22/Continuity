"""`GET /api/incidents/{id}/severity` -- a real, queried ARR-at-risk preview for one
ground-truth incident, for the landing feed's "money at risk" column.

This is NOT the investigation pipeline. The investigation (`investigate_stream.py`)
always rediscovers the blast radius from scratch via `walk()`, exactly as CLAUDE.md's
hard constraints require, and never sees the ground-truth predicate. This endpoint is a
landing-page PREVIEW -- the same convenience `ground_truth.py`'s own predicate exposure
already grants the UI for display purposes (see its module docstring) -- computed by
handing the ground-truth predicate and window straight to `compute_impact`, the same
function the real investigation's own quantify stage calls. Every number here is still
a real ClickHouse query; nothing is fabricated, it is simply not blind the way the real
investigation must be.

`qoe_delta_ratio` for the preview is `abs(multiplier - 1)`, matching how
`continuity/data/generator.py`'s planted `multiplier` scales a metric relative to
baseline regardless of whether the metric's bad direction is "higher" (rebuffer, ~4.5x
of baseline) or "lower" (bitrate, ~0.45x of baseline) -- `compute_impact` only wants a
non-negative magnitude, exactly like the real pipeline's own severity ratio.

A decoy incident (`effects` empty) has no planted QoE degradation to quantify --
this returns a real, honest zero rather than fabricating a ratio for a metric that was
never actually affected.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from continuity.analysis.impact import compute_impact
from continuity.analysis.slices import Slice
from continuity.api.ground_truth import DEFAULT_GROUND_TRUTH_PATH, GroundTruthError
from continuity.gateway.mcp_gateway import ClickHouseMCPGateway

router = APIRouter(prefix="/api/incidents")


def load_raw_incident(path: Path, incident_id: str) -> dict[str, Any]:
    """Independent reader of the same ground_truth.json file continuity/analysis/cli.py
    and continuity/api/ground_truth.py each already read on their own -- this one needs
    `effects`, which ground_truth.py's own summary deliberately drops (see its
    docstring): "Ground truth's predicate... is exposed here for the UI to display".
    """
    if not path.exists():
        raise GroundTruthError(f"Ground truth file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GroundTruthError(f"Ground truth file {path} is not valid JSON: {exc}") from exc
    incidents = payload.get("incidents")
    if not isinstance(incidents, list):
        raise GroundTruthError(f"Ground truth file {path} has no 'incidents' list.")
    for row in incidents:
        if row.get("incident_id") == incident_id:
            return row
    raise GroundTruthError(f"Unknown incident {incident_id!r} in {path}")


def _slice_from_predicate(predicate: dict[str, str]) -> Slice:
    slice_ = Slice()
    for dimension, value in predicate.items():
        slice_ = slice_.refine(dimension, value)
    return slice_


async def compute_incident_severity(
    gateway: ClickHouseMCPGateway, row: dict[str, Any]
) -> dict[str, Any]:
    """One incident's real, queried ARR-at-risk preview -- see module docstring."""
    effects = row.get("effects") or []
    if not effects:
        return {
            "id": row["incident_id"],
            "affected_subscribers": 0,
            "arr_at_risk_low": 0.0,
            "arr_at_risk_expected": 0.0,
            "arr_at_risk_high": 0.0,
            "sql": None,
        }
    start = datetime.fromisoformat(row["start"]).replace(tzinfo=None)
    end = datetime.fromisoformat(row["end"]).replace(tzinfo=None)
    slice_ = _slice_from_predicate(row.get("predicate", {}))
    qoe_delta_ratio = abs(float(effects[0]["multiplier"]) - 1.0)
    impact = await compute_impact(
        gateway, slice_=slice_, window=(start, end), qoe_delta_ratio=qoe_delta_ratio
    )
    return {
        "id": row["incident_id"],
        "affected_subscribers": impact.affected_subscribers,
        "arr_at_risk_low": float(impact.arr_at_risk_low),
        "arr_at_risk_expected": float(impact.arr_at_risk_expected),
        "arr_at_risk_high": float(impact.arr_at_risk_high),
        "sql": impact.sql,
    }


@router.get("/{incident_id}/severity")
async def incident_severity(incident_id: str, request: Request) -> dict[str, Any]:
    gateway: ClickHouseMCPGateway | None = getattr(request.app.state, "gateway", None)
    if gateway is None:
        raise HTTPException(status_code=503, detail="ClickHouse gateway is not ready")
    ground_truth_path: Path = getattr(
        request.app.state, "ground_truth_path", DEFAULT_GROUND_TRUTH_PATH
    )
    try:
        row = load_raw_incident(ground_truth_path, incident_id)
    except GroundTruthError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return await compute_incident_severity(gateway, row)
