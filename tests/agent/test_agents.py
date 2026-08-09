"""Tests for continuity.agent.agents: the stage builders, the Workflow
pipeline, the audit log, and the ACT approval gate.

Two kinds of test:

* CONSTRUCTION tests build agents (and the pipeline) and inspect them --
  which tools are attached to which stage, what output_schema and output_key
  each has, that the McpToolset escape hatch is read-only and only ever
  attached to INVESTIGATE. These never execute an agent, so the gateway
  handed to `continuity.agent.tools.build_function_tools` is a bare
  `object()` sentinel: `AnalysisTools.__init__` only stores it, and
  `function_tools()` only wraps bound methods -- neither ever calls it.
* DYNAMIC tests drive the real ADK function-calling and output_schema
  validation flow through `google.adk.runners.InMemoryRunner`, substituting
  `continuity.agent.fake_model.FakeLlm` (a `BaseLlm`) for every stage's
  model. `FakeLlm` never imports or calls `google.genai`'s network client, so
  these tests make zero network calls and need no credentials. The tools
  handed to agents in these tests are small test-local stubs (not the real
  `continuity.agent.tools` primitives) -- sub-project 2 already tests the
  primitives' own correctness; what these tests prove is the WIRING: tool
  registration, stage order, output_schema enforcement, and the audit log.

`build_investigation_pipeline` returns a `google.adk.workflow.Workflow`, not
the deprecated `SequentialAgent` -- `_stage_nodes` below reads the four
`LlmAgent` stage nodes back off `pipeline.graph.nodes` in declared order,
filtering out the graph's `START` sentinel, since `Workflow` has no
`sub_agents` attribute.
"""

from __future__ import annotations

import inspect
import json

import pydantic
import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.workflow import START, Workflow
from google.genai import types

from continuity.agent.agents import (
    CORRELATE_TOOL_NAMES,
    INVESTIGATE_TOOL_NAMES,
    QUANTIFY_TOOL_NAMES,
    ApprovalRequiredError,
    AuditLog,
    build_brief_agent,
    build_correlate_agent,
    build_investigate_agent,
    build_investigation_pipeline,
    build_quantify_agent,
    build_readonly_mcp_toolset,
    extract_pipeline_result,
    propose_action,
    select_tools,
    verify_brief_citations,
)
from continuity.agent.fake_model import FakeLlm, scripted_final_text, scripted_function_calls
from continuity.agent.schemas import (
    ActProposal,
    BriefResult,
    ClaimReference,
    CorrelationResult,
)
from continuity.agent.tools import build_function_tools
from continuity.config import ClickHouseConfig

# ---------------------------------------------------------------------------
# Construction fixtures. `object()` is never called -- see the module
# docstring -- so no real ClickHouse or MCP session is involved.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_tools() -> list[FunctionTool]:
    return build_function_tools(object())


def _fake_clickhouse_config() -> ClickHouseConfig:
    return ClickHouseConfig(
        host="localhost", port=8123, user="default", password="x", database="continuity",
        secure=False,
    )


def _stage_nodes(pipeline: Workflow) -> list[LlmAgent]:
    """The four LlmAgent stage nodes off `pipeline.graph.nodes`, in declared
    order, with the graph's `START` sentinel filtered out -- `Workflow` has no
    `sub_agents` attribute the way `SequentialAgent` did."""
    return [node for node in pipeline.graph.nodes if node.name != START.name]


# ---------------------------------------------------------------------------
# Construction: tool wiring
# ---------------------------------------------------------------------------


def test_investigate_agent_gets_exactly_detect_measure_and_split(real_tools):
    agent = build_investigate_agent(
        select_tools(real_tools, INVESTIGATE_TOOL_NAMES),
        model="gemini-3.6-flash",
        audit_log=AuditLog(),
    )

    assert [t.name for t in agent.tools] == list(INVESTIGATE_TOOL_NAMES)


def test_correlate_agent_gets_only_find_changes(real_tools):
    agent = build_correlate_agent(
        select_tools(real_tools, CORRELATE_TOOL_NAMES),
        model="gemini-3.6-flash",
        audit_log=AuditLog(),
    )

    assert [t.name for t in agent.tools] == list(CORRELATE_TOOL_NAMES)


def test_quantify_agent_gets_only_quantify_impact(real_tools):
    agent = build_quantify_agent(
        select_tools(real_tools, QUANTIFY_TOOL_NAMES),
        model="gemini-3.6-flash",
        audit_log=AuditLog(),
    )

    assert [t.name for t in agent.tools] == list(QUANTIFY_TOOL_NAMES)


def test_brief_agent_has_no_tools():
    agent = build_brief_agent(model="gemini-3.6-flash")

    assert agent.tools == []


def test_select_tools_raises_naming_every_available_tool_on_a_missing_name(real_tools):
    with pytest.raises(KeyError, match="detect_anomalies"):
        select_tools(real_tools, ["not_a_real_tool"])


# ---------------------------------------------------------------------------
# Construction: output schemas and output_key
# ---------------------------------------------------------------------------


def test_each_stage_has_its_own_typed_output_schema_and_key(real_tools):
    pipeline, _audit_log = build_investigation_pipeline(real_tools, model="gemini-3.6-flash")
    investigate, correlate, quantify, brief = _stage_nodes(pipeline)

    assert (investigate.output_schema.__name__, investigate.output_key) == (
        "InvestigationResult",
        "investigation_result",
    )
    assert (correlate.output_schema.__name__, correlate.output_key) == (
        "CorrelationResult",
        "correlation_result",
    )
    assert (quantify.output_schema.__name__, quantify.output_key) == (
        "QuantifyResult",
        "quantify_result",
    )
    assert (brief.output_schema.__name__, brief.output_key) == ("BriefResult", "brief_result")


# ---------------------------------------------------------------------------
# Construction: pipeline order and model injectability
# ---------------------------------------------------------------------------


def test_pipeline_declares_stages_in_the_documented_order(real_tools):
    pipeline, _audit_log = build_investigation_pipeline(real_tools, model="gemini-3.6-flash")

    assert isinstance(pipeline, Workflow)
    assert [a.name for a in _stage_nodes(pipeline)] == [
        "investigate",
        "correlate",
        "quantify",
        "brief",
    ]


def test_act_is_not_a_pipeline_sub_agent(real_tools):
    """ACT sits behind an explicit approval gate (`propose_action`) that the
    pipeline never crosses on its own -- it must not be one of the stage nodes
    that runs automatically."""
    pipeline, _audit_log = build_investigation_pipeline(real_tools, model="gemini-3.6-flash")

    assert "act" not in [a.name for a in _stage_nodes(pipeline)]


def test_model_is_injectable_as_a_fake_base_llm_on_every_stage(real_tools):
    fake = FakeLlm(model="fake-shared")
    pipeline, _audit_log = build_investigation_pipeline(real_tools, model=fake)

    assert all(agent.model is fake for agent in _stage_nodes(pipeline))


def test_pipeline_construction_needs_no_credentials_or_env_vars(real_tools, monkeypatch):
    for var in (
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_API_KEY",
        "CLICKHOUSE_HOST",
        "CLICKHOUSE_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)

    pipeline, audit_log = build_investigation_pipeline(real_tools, model="gemini-3.6-flash")

    assert isinstance(pipeline, Workflow)
    assert audit_log.entries == []


# ---------------------------------------------------------------------------
# Construction: the McpToolset read-only escape hatch
# ---------------------------------------------------------------------------


def test_readonly_mcp_toolset_filter_has_no_write_capable_tool_name():
    toolset = build_readonly_mcp_toolset(_fake_clickhouse_config())

    assert isinstance(toolset, McpToolset)
    assert toolset.tool_filter == ["run_query", "list_tables", "list_databases"]
    assert not any(
        verb in name.lower()
        for name in toolset.tool_filter
        for verb in ("insert", "write", "delete", "update", "alter", "drop", "create")
    )


def test_mcp_toolset_is_attached_only_to_investigate(real_tools):
    toolset = build_readonly_mcp_toolset(_fake_clickhouse_config())
    pipeline, _audit_log = build_investigation_pipeline(
        real_tools, model="gemini-3.6-flash", mcp_toolset=toolset
    )
    investigate, correlate, quantify, brief = _stage_nodes(pipeline)

    assert toolset in investigate.tools
    assert toolset not in correlate.tools
    assert toolset not in quantify.tools
    assert toolset not in brief.tools


def test_no_write_capable_tool_exists_anywhere_in_the_pipelines_tool_surface(real_tools):
    """The five analysis primitives plus the MCP escape hatch are the entire tool
    surface this project's agents can ever be given -- none of them may write."""
    write_verbs = ("insert", "write", "delete", "update", "alter", "drop", "create")
    tool_names = [t.name for t in real_tools] + list(build_readonly_mcp_toolset(
        _fake_clickhouse_config()
    ).tool_filter)

    assert not any(verb in name.lower() for name in tool_names for verb in write_verbs)


# ---------------------------------------------------------------------------
# Schema rigour: bad answers made impossible, not merely discouraged.
# ---------------------------------------------------------------------------


def test_correlation_result_requires_disconfirming_evidence_field():
    payload = dict(
        candidates=[], confidence="high", reasoning="x",
        source={"tool_name": "find_changes", "audit_index": 0},
    )

    with pytest.raises(pydantic.ValidationError, match="disconfirming_evidence"):
        CorrelationResult(**payload)


def test_correlation_result_requires_confidence_field():
    payload = dict(
        candidates=[], disconfirming_evidence="none returned", reasoning="x",
        source={"tool_name": "find_changes", "audit_index": 0},
    )

    with pytest.raises(pydantic.ValidationError, match="confidence"):
        CorrelationResult(**payload)


def test_claim_reference_rejects_a_tool_name_that_does_not_exist():
    with pytest.raises(pydantic.ValidationError):
        ClaimReference(tool_name="drop_table", audit_index=0)


def test_act_proposal_cannot_be_constructed_with_requires_human_approval_false():
    with pytest.raises(pydantic.ValidationError):
        ActProposal(
            proposed_action="a", rationale="b", requires_human_approval=False
        )


def test_act_proposal_cannot_be_constructed_as_already_executed():
    with pytest.raises(pydantic.ValidationError):
        ActProposal(proposed_action="a", rationale="b", executed=True)


def test_act_proposal_default_is_locked_to_unexecuted_and_approval_required():
    proposal = ActProposal(proposed_action="a", rationale="b")

    assert proposal.requires_human_approval is True
    assert proposal.executed is False


# ---------------------------------------------------------------------------
# ACT: proposal only, behind an explicit approval gate, no write capability.
# ---------------------------------------------------------------------------


def _brief() -> BriefResult:
    return BriefResult(
        summary="roku incident driven by chg-1",
        claims=[
            {
                "text": "roku is the blast radius",
                "source": {"tool_name": "split_on_dimension", "audit_index": 0},
            }
        ],
        recommended_action="roll back chg-1",
        methodology_notes="severity is an assumption-based heuristic",
    )


def test_propose_action_refuses_without_explicit_approval():
    with pytest.raises(ApprovalRequiredError):
        propose_action(_brief(), approved=False)


def test_propose_action_returns_a_locked_proposal_when_approved():
    proposal = propose_action(_brief(), approved=True)

    assert proposal.proposed_action == "roll back chg-1"
    assert proposal.requires_human_approval is True
    assert proposal.executed is False


def test_propose_action_performs_no_io():
    """Every I/O operation in this codebase is async (the gateway, every tool).
    propose_action being a plain synchronous function is a structural proof it
    cannot await a query, a write, or anything else that touches ClickHouse or
    the network -- not just an unenforced convention."""
    assert not inspect.iscoroutinefunction(propose_action)
    assert set(inspect.signature(propose_action).parameters) == {"brief", "approved"}


# ---------------------------------------------------------------------------
# Dynamic: drive the real ADK flow through InMemoryRunner + FakeLlm.
# ---------------------------------------------------------------------------


async def detect_anomalies(slice_json, metric, start, end):
    return {"sql": "SELECT detect", "windows": []}


async def measure_slice(slice_json, metric, window_start, window_end):
    return {"sql": "SELECT measure", "value": 1.0, "z": 4.0}


async def split_on_dimension(slice_json, metric, dimension, window_start, window_end, top_n=8):
    return {"sql": "SELECT split", "values": [{"value": "roku", "lift": 4.4}], "informative": True}


async def find_changes(slice_json, onset):
    return {
        "sql": "SELECT changes",
        "candidates": [{"change_id": "chg-1"}],
        "rejected": [],
    }


async def quantify_impact(slice_json, window_start, window_end, severity_ratio):
    return {
        "sql": "SELECT quantify",
        "affected_subscribers": 1000,
        "arr_at_risk_low": "1.00",
        "arr_at_risk_expected": "2.00",
        "arr_at_risk_high": "3.00",
        "methodology": {"notes": "assumption-based"},
    }


_INVESTIGATION_JSON = json.dumps(
    {
        "hypothesis": "roku devices are driving the deviation",
        "final_slice": [{"dimension": "device_type", "value": "roku"}],
        "metric": "rebuffer",
        "window_start": "2026-02-12 12:00:00",
        "window_end": "2026-02-12 13:00:00",
        "final_lift": 4.4,
        "stop_reason": "evidence_sufficient",
        "reasoning": "roku showed lift 4.4x, confirmed by measure_slice",
        "source": {"tool_name": "split_on_dimension", "audit_index": 0},
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
        "disconfirming_evidence": "only roku degraded despite a global rollout, supporting chg-1",
        "confidence": "high",
        "top_candidate_change_id": "chg-1",
        "reasoning": "temporal and dimensional match, corroborated by disconfirming evidence",
        "source": {"tool_name": "find_changes", "audit_index": 1},
    }
)

_QUANTIFY_JSON = json.dumps(
    {
        "affected_subscribers": 1000,
        "arr_at_risk_low": "1.00",
        "arr_at_risk_expected": "2.00",
        "arr_at_risk_high": "3.00",
        "methodology_caveat": "coefficients are documented assumptions, not measured",
        "source": {"tool_name": "quantify_impact", "audit_index": 2},
    }
)

_BRIEF_JSON = json.dumps(
    {
        "summary": "roku incident driven by chg-1",
        "claims": [
            {
                "text": "roku device_type is the blast radius",
                "source": {"tool_name": "split_on_dimension", "audit_index": 0},
            },
            {
                "text": "chg-1 is the probable cause",
                "source": {"tool_name": "find_changes", "audit_index": 1},
            },
            {
                "text": "1000 subscribers affected, $2.00 ARR at risk",
                "source": {"tool_name": "quantify_impact", "audit_index": 2},
            },
        ],
        "recommended_action": "roll back chg-1",
        "methodology_notes": "severity is an assumption-based heuristic",
    }
)


def _build_scripted_pipeline() -> tuple[Workflow, AuditLog, dict[str, FakeLlm]]:
    """A full four-stage pipeline wired to test-local stub tools and one FakeLlm
    per stage, scripted to a consistent, valid happy-path investigation."""
    audit_log = AuditLog()

    investigate_model = FakeLlm(
        model="fake-investigate",
        responses=[
            scripted_function_calls(
                (
                    "split_on_dimension",
                    {
                        "slice_json": {},
                        "metric": "rebuffer",
                        "dimension": "device_type",
                        "window_start": "2026-02-12 12:00:00",
                        "window_end": "2026-02-12 13:00:00",
                    },
                )
            ),
            scripted_final_text(_INVESTIGATION_JSON),
        ],
    )
    investigate = build_investigate_agent(
        [
            FunctionTool(detect_anomalies),
            FunctionTool(measure_slice),
            FunctionTool(split_on_dimension),
        ],
        model=investigate_model,
        audit_log=audit_log,
    )

    correlate_model = FakeLlm(
        model="fake-correlate",
        responses=[
            scripted_function_calls(
                (
                    "find_changes",
                    {"slice_json": {"device_type": "roku"}, "onset": "2026-02-12 12:00:00"},
                )
            ),
            scripted_final_text(_CORRELATION_JSON),
        ],
    )
    correlate = build_correlate_agent(
        [FunctionTool(find_changes)], model=correlate_model, audit_log=audit_log
    )

    quantify_model = FakeLlm(
        model="fake-quantify",
        responses=[
            scripted_function_calls(
                (
                    "quantify_impact",
                    {
                        "slice_json": {"device_type": "roku"},
                        "window_start": "2026-02-12 12:00:00",
                        "window_end": "2026-02-12 13:00:00",
                        "severity_ratio": 3.4,
                    },
                )
            ),
            scripted_final_text(_QUANTIFY_JSON),
        ],
    )
    quantify = build_quantify_agent(
        [FunctionTool(quantify_impact)], model=quantify_model, audit_log=audit_log
    )

    brief_model = FakeLlm(model="fake-brief", responses=[scripted_final_text(_BRIEF_JSON)])
    brief = build_brief_agent(model=brief_model)

    pipeline = Workflow(
        name="continuity_investigation_under_test",
        edges=[(START, investigate, correlate, quantify, brief)],
    )
    models = {
        "investigate": investigate_model,
        "correlate": correlate_model,
        "quantify": quantify_model,
        "brief": brief_model,
    }
    return pipeline, audit_log, models


async def _run_pipeline(pipeline: Workflow):
    runner = InMemoryRunner(agent=pipeline, app_name="test_app")
    session = await runner.session_service.create_session(app_name="test_app", user_id="u1")
    content = types.Content(role="user", parts=[types.Part(text="investigate")])
    events = [
        event
        async for event in runner.run_async(
            user_id="u1", session_id=session.id, new_message=content
        )
    ]
    final_session = await runner.session_service.get_session(
        app_name="test_app", user_id="u1", session_id=session.id
    )
    return events, final_session.state


async def test_pipeline_runs_stages_in_declared_order():
    pipeline, _audit_log, _models = _build_scripted_pipeline()

    events, _state = await _run_pipeline(pipeline)

    authors_in_order = [e.author for e in events]
    first_seen = {}
    for position, author in enumerate(authors_in_order):
        first_seen.setdefault(author, position)
    assert (
        first_seen["investigate"]
        < first_seen["correlate"]
        < first_seen["quantify"]
        < first_seen["brief"]
    )


async def test_audit_log_captures_tool_name_arguments_and_sql_in_call_order():
    pipeline, audit_log, _models = _build_scripted_pipeline()

    await _run_pipeline(pipeline)

    assert [e.tool_name for e in audit_log.entries] == [
        "split_on_dimension",
        "find_changes",
        "quantify_impact",
    ]
    split_call = audit_log.entries[0]
    assert split_call.arguments["dimension"] == "device_type"
    assert split_call.sql == "SELECT split"


async def test_correlate_stage_sees_the_investigation_result_via_state_templating():
    """BRIEF and CORRELATE have no tool of their own for reading a prior stage's
    output -- they must receive it through ADK's `{state_key}` instruction
    templating. If this ever regressed to literal, unsubstituted `{investigation_result}`
    text, CORRELATE would never see roku mentioned at all."""
    pipeline, _audit_log, models = _build_scripted_pipeline()

    await _run_pipeline(pipeline)

    correlate_requests = models["correlate"].requests
    assert any(
        "roku" in (part.text or "")
        for request in correlate_requests
        for content in request.contents
        for part in content.parts
    )


async def test_pipeline_produces_the_typed_result_the_eval_harness_expects():
    pipeline, _audit_log, _models = _build_scripted_pipeline()

    _events, state = await _run_pipeline(pipeline)
    result = extract_pipeline_result(state)

    assert [d.model_dump() for d in result.investigation.final_slice] == [
        {"dimension": "device_type", "value": "roku"}
    ]
    assert result.correlation.confidence == "high"
    assert result.quantify.affected_subscribers == 1000
    assert result.brief.recommended_action == "roll back chg-1"


async def test_every_brief_claim_traces_to_a_real_logged_tool_call():
    pipeline, audit_log, _models = _build_scripted_pipeline()

    _events, state = await _run_pipeline(pipeline)
    result = extract_pipeline_result(state)

    verify_brief_citations(result.brief, audit_log)  # must not raise


def test_verify_brief_citations_catches_a_fabricated_audit_index():
    audit_log = AuditLog()
    audit_log.record(tool_name="split_on_dimension", arguments={}, result={"sql": "SELECT 1"})
    brief = BriefResult(
        summary="s",
        claims=[
            {
                "text": "a number nobody logged",
                "source": {"tool_name": "quantify_impact", "audit_index": 99},
            }
        ],
        recommended_action="a",
        methodology_notes="m",
    )

    with pytest.raises(ValueError, match="audit_index"):
        verify_brief_citations(brief, audit_log)


def test_verify_brief_citations_catches_a_tool_name_mismatch_at_a_real_index():
    audit_log = AuditLog()
    audit_log.record(tool_name="split_on_dimension", arguments={}, result={"sql": "SELECT 1"})
    brief = BriefResult(
        summary="s",
        claims=[
            {
                "text": "misattributed to the wrong tool",
                "source": {"tool_name": "quantify_impact", "audit_index": 0},
            }
        ],
        recommended_action="a",
        methodology_notes="m",
    )

    with pytest.raises(ValueError, match="tool_name"):
        verify_brief_citations(brief, audit_log)


async def test_a_stage_output_that_fails_schema_validation_is_raised_not_propagated():
    """A response missing required InvestigationResult fields must blow up loudly
    at the INVESTIGATE stage, never silently flow into CORRELATE as if it were a
    valid answer."""
    audit_log = AuditLog()
    bad_model = FakeLlm(
        model="fake-bad-investigate",
        responses=[
            scripted_final_text(
                json.dumps(
                    {
                        "hypothesis": "incomplete answer",
                        "metric": "rebuffer",
                        "window_start": "x",
                        "window_end": "y",
                        "final_lift": 1.0,
                        "reasoning": "missing stop_reason and source",
                    }
                )
            )
        ],
    )
    investigate = build_investigate_agent(
        [
            FunctionTool(detect_anomalies),
            FunctionTool(measure_slice),
            FunctionTool(split_on_dimension),
        ],
        model=bad_model,
        audit_log=audit_log,
    )
    pipeline = Workflow(name="bad_pipeline", edges=[(START, investigate)])

    with pytest.raises(pydantic.ValidationError):
        await _run_pipeline(pipeline)
