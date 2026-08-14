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
from google.adk.models.google_llm import _ResourceExhaustedError
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.workflow import START, RetryConfig, Workflow
from google.genai import types
from google.genai.errors import ClientError

from continuity.agent.agents import (
    CORRELATE_TOOL_NAMES,
    INVESTIGATE_TOOL_NAMES,
    QUANTIFY_TOOL_NAMES,
    RATE_LIMIT_RETRY,
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
# Construction: retry on a rate-limited model
# ---------------------------------------------------------------------------


def test_pipeline_retries_only_on_a_rate_limit_never_on_a_logic_error(real_tools):
    """A full investigation costs ~219k tokens across ~11 model turns, so a handful of
    them back to back exhausts a per-minute Vertex quota and returns 429 -- observed
    failing two of three incidents in one comparison run, and equally capable of killing
    a live demo. Per-minute quota recovers on its own, so the answer is to wait and
    retry rather than to fail the investigation.

    The retry must be scoped to exactly that error. Retrying broadly would re-run a
    stage that failed schema validation or hit a tool bug, quietly tripling the token
    cost of a real defect and blurring it into a flake.

    The assertion is on the GRAPH NODES, not on the `Workflow`, because that is what
    the retry actually runs off: `_node_runner` consults `self._node.retry_config`, and
    a `Workflow`-level `retry_config` does NOT propagate down to the nodes it builds --
    setting only that one leaves every stage with `retry_config=None` and buys nothing
    while looking configured.
    """
    pipeline, _audit_log = build_investigation_pipeline(real_tools, model="gemini-3.6-flash")

    stages = _stage_nodes(pipeline)
    assert stages, "expected the pipeline to have stage nodes to check"
    for stage in stages:
        retry = stage.retry_config
        assert retry is not None, f"{stage.name} would abandon the run on a 429"
        assert retry.exceptions == ["_ResourceExhaustedError"]
        assert retry.max_attempts >= 3
        # A per-minute quota window needs a wait on that order to clear.
        assert retry.max_delay >= 60.0


def test_the_retried_exception_name_still_exists_in_the_installed_adk():
    """`RetryConfig.exceptions` matches on `type(exc).__name__` as a STRING, so the name
    above is a literal with no import to break if ADK renames or relocates the class --
    the retry would simply stop happening, silently, and only show up as a demo dying on
    a 429. This test is the tripwire: it fails loudly on upgrade instead.

    It also pins WHY that name is the right one -- ADK raises it only for HTTP 429, so
    scoping the retry to it cannot accidentally catch other client errors.
    """
    from google.adk.models.google_llm import _ResourceExhaustedError

    assert _ResourceExhaustedError.__name__ == "_ResourceExhaustedError"
    assert issubclass(_ResourceExhaustedError, ClientError)


class _TinyResult(pydantic.BaseModel):
    value: str


class _RateLimitedThenFine(FakeLlm):
    """Raises a REAL `_ResourceExhaustedError` (the 429 ADK raises) for the first
    `fail_times` calls, then behaves like a normal `FakeLlm`.

    Constructed from a real `ClientError(429, ...)` rather than a look-alike, because
    the whole retry hinges on `type(exc).__name__` and a stand-in named the same thing
    would pass this test while proving nothing about the class ADK actually raises.
    """

    fail_times: int = 0
    _failures: int = pydantic.PrivateAttr(default=0)

    @property
    def failures(self) -> int:
        return self._failures

    async def generate_content_async(self, llm_request, stream: bool = False):
        if self._failures < self.fail_times:
            self._failures += 1
            raise _ResourceExhaustedError(
                ClientError(
                    429,
                    {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}},
                    None,
                )
            )
        async for response in super().generate_content_async(llm_request, stream=stream):
            yield response


def _one_stage_workflow(model: FakeLlm, retry: RetryConfig) -> Workflow:
    """A single-node graph, so the retry under test is the only moving part."""
    stage = LlmAgent(
        name="flaky",
        model=model,
        instruction="reply with JSON",
        output_schema=_TinyResult,
        output_key="tiny_result",
    )
    stage.retry_config = retry
    return Workflow(name="retry_probe", edges=[(START, stage)])


async def _run_workflow(pipeline: Workflow) -> str:
    runner = InMemoryRunner(agent=pipeline, app_name="retry_probe")
    session = await runner.session_service.create_session(app_name="retry_probe", user_id="u")
    errors = []
    async for event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="go")]),
    ):
        if getattr(event, "error_code", None):
            errors.append(event.error_code)
    return errors


async def test_a_rate_limited_stage_is_actually_retried_and_then_succeeds():
    """The behavioural half of the retry: config presence proves nothing on its own --
    the first version of this fix set `retry_config` on the `Workflow`, satisfied an
    assertion that it was configured, and retried nothing, because the node runner reads
    it off the NODE. So drive a stage that raises a real 429 twice and assert the model
    was called a third time and the run recovered.

    Delays are overridden to ~0 here; the production values are sized for a real
    per-minute quota window and are asserted separately above.
    """
    model = _RateLimitedThenFine(
        model="fake-flaky",
        fail_times=2,
        responses=[scripted_final_text(json.dumps({"value": "recovered"}))],
    )
    fast_retry = RetryConfig(
        max_attempts=RATE_LIMIT_RETRY.max_attempts,
        initial_delay=0.001,
        max_delay=0.002,
        backoff_factor=1.0,
        jitter=0.0,
        exceptions=RATE_LIMIT_RETRY.exceptions,
    )

    await _run_workflow(_one_stage_workflow(model, fast_retry))

    assert model.failures == 2, "expected both rate-limit failures to be exercised"
    assert model.call_count == 1, "expected the stage to reach the model once it recovered"


async def test_a_non_rate_limit_failure_is_not_retried():
    """The scoping half: a stage that fails for any other reason must fail once, not
    burn three more full stages' worth of tokens pretending a real defect is a flake.
    `FakeLlm` raises `AssertionError` when asked for an unscripted turn, so a model
    scripted with zero responses fails for a reason that is emphatically not a 429.
    """
    model = _RateLimitedThenFine(model="fake-broken", fail_times=0, responses=[])
    pipeline = _one_stage_workflow(
        model,
        RetryConfig(
            max_attempts=4,
            initial_delay=0.001,
            max_delay=0.002,
            exceptions=RATE_LIMIT_RETRY.exceptions,
        ),
    )

    # The non-429 failure surfaces rather than being swallowed -- which is the point:
    # it must reach the caller as a failure, not be quietly retried into a slow one.
    with pytest.raises(AssertionError, match="only scripted with 0 response"):
        await _run_workflow(pipeline)

    assert model.call_count == 1, (
        f"a non-429 failure was retried {model.call_count} times -- retrying broadly "
        "turns a real defect into an expensive intermittent one"
    )


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


def test_correlation_result_requires_corroborated_field():
    payload = dict(
        candidates=[], disconfirming_evidence="none returned", confidence="low",
        reasoning="x", source={"tool_name": "find_changes", "audit_index": 0},
    )

    with pytest.raises(pydantic.ValidationError, match="corroborated"):
        CorrelationResult(**payload)


def test_correlation_result_rejects_high_confidence_with_zero_candidates():
    """DEFECT 2: the central fix -- absence of a correlating change is evidence
    AGAINST the blast radius, so high confidence must be structurally unrepresentable
    when there is nothing to corroborate with."""
    payload = dict(
        candidates=[], disconfirming_evidence="nothing in change_log near onset",
        confidence="high", corroborated=False, reasoning="x",
        source={"tool_name": "find_changes", "audit_index": 0},
    )

    with pytest.raises(pydantic.ValidationError, match="confidence"):
        CorrelationResult(**payload)


def test_correlation_result_rejects_corroborated_true_with_zero_candidates():
    payload = dict(
        candidates=[], disconfirming_evidence="nothing in change_log near onset",
        confidence="low", corroborated=True, reasoning="x",
        source={"tool_name": "find_changes", "audit_index": 0},
    )

    with pytest.raises(pydantic.ValidationError, match="corroborated"):
        CorrelationResult(**payload)


def test_correlation_result_rejects_high_confidence_when_not_corroborated_even_with_candidates():
    payload = dict(
        candidates=[
            {
                "change_id": "chg-9",
                "rank": 1,
                "disconfirming_evidence_assessment": "shipped everywhere, everything degraded",
                "still_plausible": False,
            }
        ],
        disconfirming_evidence="the only candidate is not still_plausible",
        confidence="high", corroborated=False, reasoning="x",
        source={"tool_name": "find_changes", "audit_index": 0},
    )

    with pytest.raises(pydantic.ValidationError, match="confidence"):
        CorrelationResult(**payload)


def test_correlation_result_accepts_low_confidence_with_zero_candidates_and_not_corroborated():
    """The honest, unresolved answer must remain representable."""
    payload = dict(
        candidates=[], disconfirming_evidence="nothing in change_log near onset",
        confidence="low", corroborated=False, reasoning="x",
        source={"tool_name": "find_changes", "audit_index": 0},
    )

    result = CorrelationResult(**payload)

    assert result.corroborated is False
    assert result.candidates == []


def test_brief_result_requires_unresolved_field():
    with pytest.raises(pydantic.ValidationError, match="unresolved"):
        BriefResult(
            summary="s",
            claims=[
                {
                    "text": "t",
                    "source": {"tool_name": "quantify_impact", "audit_index": 0},
                }
            ],
            recommended_action="a",
            methodology_notes="m",
        )


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
        unresolved=False,
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


async def split_all_dimensions(slice_json, metric, window_start, window_end):
    return {
        "sql": "SELECT split_all",
        "dimensions": [
            {"dimension": "device_type", "top_value": "roku", "lift": 4.4, "informative": True}
        ],
    }


async def refine_incident_span(slice_json, metric, approx_start, approx_end):
    return {
        "sql": "SELECT refine",
        "refined": True,
        "start": "2026-02-12 12:00:00",
        "end": "2026-02-12 13:00:00",
        "buckets_breached": 6,
        "typical_severity_ratio": 2.1,
        "peak_severity_ratio": 3.4,
    }


async def find_changes(slice_json, onset):
    return {
        "sql": "SELECT changes",
        "candidates": [{"change_id": "chg-1"}],
        "rejected": [],
    }


async def quantify_impact(slice_json, metric, window_start, window_end):
    return {
        "sql": "SELECT quantify",
        "typical_severity_ratio": 2.1,
        "affected_subscribers": 1000,
        "arr_at_risk_low": "1.00",
        "arr_at_risk_expected": "2.00",
        "arr_at_risk_high": "3.00",
        "methodology": {"notes": "assumption-based"},
    }


# Tool call order the scripted INVESTIGATE model below drives: split_all_dimensions (0)
# -> refine_incident_span (1); then CORRELATE's find_changes (2); then QUANTIFY's
# quantify_impact (3). Every `source.audit_index` below must match this exactly.

_INVESTIGATION_JSON = json.dumps(
    {
        "hypothesis": "roku devices are driving the deviation",
        "final_slice": [{"dimension": "device_type", "value": "roku"}],
        "metric": "rebuffer",
        "window_start": "2026-02-12 12:00:00",
        "window_end": "2026-02-12 13:00:00",
        "final_lift": 4.4,
        "stop_reason": "evidence_sufficient",
        "reasoning": "roku showed lift 4.4x, confirmed and refined by refine_incident_span",
        "source": {"tool_name": "refine_incident_span", "audit_index": 1},
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
        "corroborated": True,
        "top_candidate_change_id": "chg-1",
        "reasoning": "temporal and dimensional match, corroborated by disconfirming evidence",
        "source": {"tool_name": "find_changes", "audit_index": 2},
    }
)

_QUANTIFY_JSON = json.dumps(
    {
        "affected_subscribers": 1000,
        "arr_at_risk_low": "1.00",
        "arr_at_risk_expected": "2.00",
        "arr_at_risk_high": "3.00",
        "methodology_caveat": "coefficients are documented assumptions, not measured",
        "source": {"tool_name": "quantify_impact", "audit_index": 3},
    }
)

_BRIEF_JSON = json.dumps(
    {
        "summary": "roku incident driven by chg-1",
        "claims": [
            {
                "text": "roku device_type is the blast radius",
                "source": {"tool_name": "split_all_dimensions", "audit_index": 0},
            },
            {
                "text": "chg-1 is the probable cause",
                "source": {"tool_name": "find_changes", "audit_index": 2},
            },
            {
                "text": "1000 subscribers affected, $2.00 ARR at risk",
                "source": {"tool_name": "quantify_impact", "audit_index": 3},
            },
        ],
        "recommended_action": "roll back chg-1",
        "methodology_notes": "severity is an assumption-based heuristic",
        "unresolved": False,
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
                    "split_all_dimensions",
                    {
                        "slice_json": {},
                        "metric": "rebuffer",
                        "window_start": "2026-02-12 12:00:00",
                        "window_end": "2026-02-12 13:00:00",
                    },
                )
            ),
            scripted_function_calls(
                (
                    "refine_incident_span",
                    {
                        "slice_json": {"device_type": "roku"},
                        "metric": "rebuffer",
                        "approx_start": "2026-02-12 12:00:00",
                        "approx_end": "2026-02-12 13:00:00",
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
            FunctionTool(split_all_dimensions),
            FunctionTool(refine_incident_span),
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
                        "metric": "rebuffer",
                        "window_start": "2026-02-12 12:00:00",
                        "window_end": "2026-02-12 13:00:00",
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
        "split_all_dimensions",
        "refine_incident_span",
        "find_changes",
        "quantify_impact",
    ]
    split_call = audit_log.entries[0]
    assert split_call.arguments["metric"] == "rebuffer"
    assert split_call.sql == "SELECT split_all"
    refine_call = audit_log.entries[1]
    assert refine_call.arguments["approx_start"] == "2026-02-12 12:00:00"
    assert refine_call.sql == "SELECT refine"


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
        unresolved=False,
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
        unresolved=False,
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
