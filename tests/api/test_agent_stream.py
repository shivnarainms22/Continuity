"""Tests for continuity/api/agent_stream.py -- the SSE protocol of the live agent view.

Network-free and ClickHouse-free: `detect` and `build_investigation_pipeline` are
monkeypatched, and the pipeline under test is driven by `FakeLlm`, so what these assert
is the STREAMING CONTRACT -- which frames come out, in what order, and what each one
carries. The agent's own correctness is covered by tests/agent/test_agents.py and by
scripts/compare_arms.py against real data.

The ordering assertions matter more than they look: the view exists to show measurements
AS THEY HAPPEN, so a stream that emitted every tool_call at the end would satisfy a
naive "did we get the frames" check while being exactly the spinner this replaces.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from google.adk.tools import FunctionTool

from continuity.agent.agents import (
    AuditLog,
    build_brief_agent,
    build_correlate_agent,
    build_investigate_agent,
    build_quantify_agent,
)
from continuity.agent.fake_model import FakeLlm, scripted_final_text, scripted_function_calls
from continuity.analysis.detect import AnomalyWindow, DetectionResult
from continuity.analysis.slices import Slice
from continuity.api import agent_stream

_WINDOW = (datetime(2026, 2, 12, 12, 0), datetime(2026, 2, 12, 13, 0))

_INVESTIGATION_JSON = json.dumps(
    {
        "hypothesis": "roku devices are driving the deviation",
        "final_slice": [{"dimension": "device_type", "value": "roku"}],
        "metric": "rebuffer",
        "window_start": "2026-02-12 12:00:00",
        "window_end": "2026-02-12 13:00:00",
        "final_lift": 4.4,
        "stop_reason": "evidence_sufficient",
        "reasoning": "roku showed lift 4.4x",
        "source": {"tool_name": "split_all_dimensions", "audit_index": 0},
    }
)
_CORRELATION_JSON = json.dumps(
    {
        "candidates": [
            {
                "change_id": "chg-1",
                "rank": 1,
                "disconfirming_evidence_assessment": "shipped everywhere, only roku degraded",
                "still_plausible": True,
            }
        ],
        "disconfirming_evidence": "only roku degraded despite a global rollout",
        "confidence": "high",
        "corroborated": True,
        "top_candidate_change_id": "chg-1",
        "reasoning": "temporal and dimensional match",
        "source": {"tool_name": "find_changes", "audit_index": 1},
    }
)
_QUANTIFY_JSON = json.dumps(
    {
        "affected_subscribers": 1000,
        "arr_at_risk_low": "1.00",
        "arr_at_risk_expected": "2.00",
        "arr_at_risk_high": "3.00",
        "methodology_caveat": "coefficients are documented assumptions",
        "source": {"tool_name": "quantify_impact", "audit_index": 2},
    }
)
_BRIEF_JSON = json.dumps(
    {
        "summary": "roku incident driven by chg-1",
        "claims": [
            {
                "text": "roku device_type is the blast radius",
                "source": {"tool_name": "split_all_dimensions", "audit_index": 0},
            }
        ],
        "recommended_action": "roll back chg-1",
        "methodology_notes": "severity is an assumption-based heuristic",
        "unresolved": False,
    }
)


def split_all_dimensions(slice_json: dict, metric: str) -> dict:
    """Stub: one split, carrying a query the stream is expected to surface."""
    return {
        "splits": [{"dimension": "device_type", "value": "roku", "lift": 4.4}],
        "sql": "SELECT device_type FROM qoe_rollup_5m WHERE bucket IN ('a','b')",
    }


def find_changes(slice_json: dict, onset: str) -> dict:
    """Stub: one candidate change."""
    return {"candidates": [{"change_id": "chg-1"}], "sql": "SELECT * FROM change_log"}


def quantify_impact(slice_json: dict, window_start: str, window_end: str) -> dict:
    """Stub: an impact figure."""
    return {"affected_subscribers": 1000, "sql": "SELECT subscriber_id FROM subscribers"}


def _scripted_pipeline():
    audit_log = AuditLog()
    investigate = build_investigate_agent(
        [FunctionTool(split_all_dimensions)],
        model=FakeLlm(
            model="fake-investigate",
            responses=[
                scripted_function_calls(
                    ("split_all_dimensions", {"slice_json": {}, "metric": "rebuffer"})
                ),
                scripted_final_text(_INVESTIGATION_JSON),
            ],
        ),
        audit_log=audit_log,
    )
    correlate = build_correlate_agent(
        [FunctionTool(find_changes)],
        model=FakeLlm(
            model="fake-correlate",
            responses=[
                scripted_function_calls(
                    ("find_changes", {"slice_json": {}, "onset": "2026-02-12 12:00:00"})
                ),
                scripted_final_text(_CORRELATION_JSON),
            ],
        ),
        audit_log=audit_log,
    )
    quantify = build_quantify_agent(
        [FunctionTool(quantify_impact)],
        model=FakeLlm(
            model="fake-quantify",
            responses=[
                scripted_function_calls(
                    (
                        "quantify_impact",
                        {
                            "slice_json": {},
                            "window_start": "2026-02-12 12:00:00",
                            "window_end": "2026-02-12 13:00:00",
                        },
                    )
                ),
                scripted_final_text(_QUANTIFY_JSON),
            ],
        ),
        audit_log=audit_log,
    )
    brief = build_brief_agent(
        model=FakeLlm(model="fake-brief", responses=[scripted_final_text(_BRIEF_JSON)])
    )
    from google.adk.workflow import START, Workflow

    return Workflow(
        name="test", edges=[(START, investigate, correlate, quantify, brief)]
    ), audit_log


def _detection(windows: int = 1) -> DetectionResult:
    made = [
        AnomalyWindow(
            slice=Slice(),
            metric="rebuffer",
            start=_WINDOW[0],
            end=_WINDOW[1],
            peak_z=7.0,
            peak_value=0.09,
            expected_at_peak=0.02,
            bucket_count=12,
            sql="SELECT bucket -- detect",
        )
        for _ in range(windows)
    ]
    return DetectionResult(
        slice=Slice(),
        metric="rebuffer",
        windows=made,
        total_buckets=12,
        anomalous_buckets=len(made),
        unknown_buckets=0,
        sql="SELECT bucket -- detect",
    )


@pytest.fixture
def patched(monkeypatch):
    async def fake_detect(gateway, slice_, metric, start, end, **kwargs):
        return _detection()

    pipeline, audit_log = _scripted_pipeline()
    monkeypatch.setattr(agent_stream, "detect", fake_detect)
    monkeypatch.setattr(agent_stream, "build_function_tools", lambda gateway: [])
    monkeypatch.setattr(
        agent_stream,
        "build_investigation_pipeline",
        lambda tools, model=None: (pipeline, audit_log),
    )
    return audit_log


def _parse(frames: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for frame in frames:
        event = frame.split("\n")[0].removeprefix("event: ")
        data = json.loads(frame.split("data: ", 1)[1].strip())
        parsed.append((event, data))
    return parsed


async def _collect(**kwargs) -> list[tuple[str, dict]]:
    frames = [
        frame
        async for frame in agent_stream.stream_agent_investigation(
            object(), metric_name="rebuffer", window=_WINDOW, description="test", **kwargs
        )
    ]
    return _parse(frames)


async def test_every_tool_call_is_streamed_before_the_run_finishes(patched):
    """The property the whole view rests on: measurements arrive as they happen, so
    every tool_call must precede `done`. A stream that batched them at the end would
    still contain all the frames and still be the spinner this replaces."""
    events = await _collect()

    kinds = [name for name, _ in events]
    assert kinds[0] == "detect"
    assert kinds[-1] == "done"
    tool_positions = [i for i, k in enumerate(kinds) if k == "tool_call"]
    assert len(tool_positions) == 3, f"expected one frame per tool call, got {kinds}"
    assert max(tool_positions) < kinds.index("done")


async def test_each_tool_call_frame_carries_its_query_and_its_citable_index(patched):
    """The frame is what makes the claim 'every number traces to a query' visible: the
    SQL is shown in full, and `audit_index` is the same value a brief claim cites, so
    the UI can link a figure back to the measurement."""
    events = await _collect()
    calls = [data for name, data in events if name == "tool_call"]

    assert [c["tool"] for c in calls] == [
        "split_all_dimensions",
        "find_changes",
        "quantify_impact",
    ]
    assert [c["audit_index"] for c in calls] == [0, 1, 2]
    assert calls[0]["sql"] == "SELECT device_type FROM qoe_rollup_5m WHERE bucket IN ('a','b')"
    assert calls[0]["result"]["splits"][0]["lift"] == 4.4
    # The query text lives in `sql`, never duplicated inside `result`.
    assert "sql" not in calls[0]["result"]


async def test_stage_frames_mark_each_boundary_in_order(patched):
    events = await _collect()
    stages = [data["stage"] for name, data in events if name == "stage"]

    assert stages == ["investigate", "correlate", "quantify", "brief"]
    assert all(data["label"] for name, data in events if name == "stage")


async def test_done_carries_the_typed_result_and_the_citation_verdict(patched):
    events = await _collect()
    _name, done = events[-1]

    assert done["detected"] is True
    assert done["tool_calls"] == 3
    assert done["investigation"]["final_slice"] == [{"dimension": "device_type", "value": "roku"}]
    assert done["correlation"]["confidence"] == "high"
    assert done["quantify"]["affected_subscribers"] == 1000
    assert done["brief"]["recommended_action"] == "roll back chg-1"
    assert done["citations_verified"] is True
    assert done["citation_error"] is None


async def test_a_quiet_range_ends_the_stream_without_ever_calling_the_model(monkeypatch):
    """No anomaly means no investigation -- the decoy case. It must close the stream
    cleanly saying so, not hand an empty window to the agent and spend tokens finding
    nothing."""

    async def no_windows(gateway, slice_, metric, start, end, **kwargs):
        return DetectionResult(
            slice=Slice(),
            metric="rebuffer",
            windows=[],
            total_buckets=12,
            anomalous_buckets=0,
            unknown_buckets=0,
            sql="SELECT 1",
        )

    def explode(*args, **kwargs):
        raise AssertionError("the agent must not be built when nothing was detected")

    monkeypatch.setattr(agent_stream, "detect", no_windows)
    monkeypatch.setattr(agent_stream, "build_investigation_pipeline", explode)

    events = await _collect()

    assert [name for name, _ in events] == ["done"]
    assert events[0][1]["detected"] is False


# ---------------------------------------------------------------------------
# Concurrency guard on the agent endpoint
# ---------------------------------------------------------------------------


async def test_the_agent_slot_limits_concurrent_investigations():
    """A public demo URL must not let arbitrary traffic start unlimited investigations.

    Each one costs ~90k Gemini tokens and, more immediately, the gateway holds ONE
    mcp-clickhouse session, so simultaneous investigations serialise on it anyway and
    just make each other slower. The guard turns that into an explicit, fast refusal
    rather than a queue of requests all timing out together.

    Deliberately a slot count rather than a per-IP rate limit: the resource being
    protected is the single gateway session and the token budget, neither of which cares
    which address asked.
    """
    slots = agent_stream.AgentSlots(limit=2)

    async with slots.acquire(), slots.acquire():
        assert slots.in_use == 2
        with pytest.raises(agent_stream.TooManyInvestigations):
            async with slots.acquire():
                pass  # pragma: no cover - the acquire above must raise

    assert slots.in_use == 0, "slots must be released even after a refusal"


async def test_a_slot_is_released_when_an_investigation_fails():
    """A crashed investigation must not permanently consume a slot -- three failures
    would otherwise wedge the demo closed until the instance restarts."""
    slots = agent_stream.AgentSlots(limit=1)

    with pytest.raises(ValueError):
        async with slots.acquire():
            raise ValueError("investigation blew up")

    assert slots.in_use == 0
    async with slots.acquire():
        assert slots.in_use == 1
