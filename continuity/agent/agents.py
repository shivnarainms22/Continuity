"""The agent definitions: one ``LlmAgent`` per stage, wired into a ``Workflow``.

DETECT is deterministic (``continuity.analysis.detect``, no LLM -- see CLAUDE.md hard
constraint 4) and is not built here; it runs before this pipeline and its result (a
metric, window, and slice) is the pipeline's input. This module builds the four
judgement stages -- INVESTIGATE, CORRELATE, QUANTIFY, BRIEF -- as one linear
``google.adk.workflow.Workflow`` graph (``START -> investigate -> correlate ->
quantify -> brief``), plus ``propose_action``, the ACT stage's approval gate.

WHY ``Workflow`` AND NOT ``SequentialAgent``. ``google.adk.agents.SequentialAgent``
is deprecated in ADK 2.6.3 ("SequentialAgent is deprecated in favor of Workflow and
will be removed in a future version") and its own module has been verified to still
support an ``LlmAgent`` as a workflow node end to end (tool-calling loop,
``output_schema`` validation, and ``after_tool_callback`` all work identically inside
a ``Workflow`` node). All four of THIS module's stages remain ``LlmAgent`` --
none of them are deterministic. The two deterministic computations the hackathon
brief is thinking of when it says "measurement nodes and judgement nodes" already
live outside this module's stage graph: DETECT is a separate caller-driven step (see
above) in ``continuity.analysis``, and every number QUANTIFY reports comes from the
``quantify_impact`` ``FunctionTool`` (``continuity.agent.tools``) that the QUANTIFY
``LlmAgent`` calls -- not a pipeline stage of its own. So no stage in
``build_investigation_pipeline`` becomes a ``FunctionNode``: doing so here would mean
either inventing a node that duplicates logic that already lives in ``tools.py``
(out of this module's scope to touch) or replacing a stage's LLM judgement with code,
which is not what CORRELATE/QUANTIFY/BRIEF do.

Every stage's tools come from ``continuity.agent.tools.build_function_tools`` (the
five analysis primitives) -- this module only decides which of those five tools go to
which stage, never adds new ones and never adds a write capability. INVESTIGATE
additionally gets a construction-only, read-only ``McpToolset`` escape hatch (see
``build_readonly_mcp_toolset``); nothing in this module ever calls it.

MODEL SUBSTITUTION. ``google.adk.agents.llm_agent.LlmAgent.model`` is typed
``Union[str, BaseLlm]`` -- this is ADK's own supported extension point for swapping the
model, not a workaround. Every builder function below accepts ``model: str | BaseLlm``
and passes it straight through, so ``continuity.agent.fake_model.FakeLlm`` (a
``BaseLlm`` subclass) is a drop-in substitute for the real Gemini model id string in
every test in this package. Constructing any agent here -- with either a model id
string or a ``FakeLlm`` -- never imports ``google.genai``'s network client and never
requires credentials.

AUDIT LOG. ``AuditLog`` is wired onto every tool-bearing agent via ``after_tool_callback``,
which ADK calls once per tool invocation with the tool, its arguments and its already-
resolved result (see ``google.adk.flows.llm_flows.functions._execute_single_function_call_async``).
It records tool name, arguments and the SQL the tool's result carried (when it carried
one -- an ``"invalid_input"`` or ``"no_data"`` error has no SQL, and that absence is
itself recorded, not papered over), and hands the tool's own result back to the model
with an ``audit_index`` stamped onto it so every later stage that must cite a tool call
(see ``continuity/agent/schemas.py``'s ``ClaimReference``) has an exact index to copy,
never one it invents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.workflow import START, Workflow
from mcp import StdioServerParameters

from continuity.agent.schemas import (
    ActProposal,
    BriefResult,
    CorrelationResult,
    InvestigationResult,
    QuantifyResult,
)
from continuity.config import ClickHouseConfig
from continuity.gateway.mcp_gateway import _server_executable

Model = str | BaseLlm
"""What every builder in this module accepts for ``model`` -- a Gemini model id
string for production, or a ``BaseLlm`` (e.g. ``FakeLlm``) for tests. Never resolved
or validated at construction time; ADK only touches it when a real run actually calls
the model."""

DEFAULT_MODEL_ID = "gemini-3.6-flash"
"""Matches ``CONTINUITY_MODEL`` in ``.env.example``. Only a fallback -- production
wiring should pass the configured model explicitly rather than rely on this."""

# The MCP server's own read-only tool surface (verified against mcp-clickhouse 0.4.1,
# see CLAUDE.md). No write-capable tool name is ever listed here, and this filter is a
# construction-time restriction on McpToolset itself, not a prompt instruction the
# model could be talked out of.
_MCP_READONLY_TOOL_FILTER = ["run_query", "list_tables", "list_databases"]


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditLogEntry:
    """One tool call: what it was asked, and the SQL its result carried."""

    index: int
    tool_name: str
    arguments: Mapping[str, Any]
    sql: str | None
    result: Mapping[str, Any]


@dataclass
class AuditLog:
    """Every tool call made across one investigation, in call order.

    Shared across every stage of one pipeline run (all four LLM agents are wired to
    the SAME ``AuditLog`` instance by ``build_investigation_pipeline``), so
    ``audit_index`` values referenced in a later stage's ``ClaimReference`` are unique
    and stable across the whole investigation, not just within one stage.
    """

    entries: list[AuditLogEntry] = field(default_factory=list)

    def record(
        self, *, tool_name: str, arguments: Mapping[str, Any], result: Mapping[str, Any]
    ) -> int:
        """Append one entry and return its index -- the value stamped onto the tool's
        result as ``audit_index`` for the model to cite later."""
        index = len(self.entries)
        sql = result.get("sql") if isinstance(result, Mapping) else None
        self.entries.append(
            AuditLogEntry(
                index=index,
                tool_name=tool_name,
                arguments=dict(arguments),
                sql=sql,
                result=dict(result) if isinstance(result, Mapping) else {"value": result},
            )
        )
        return index

    def after_tool_callback(self, *, tool: Any, args: dict, tool_context: Any, tool_response: Any):
        """An ADK ``after_tool_callback``: records the call, then hands the model back
        its own result with ``audit_index`` stamped on -- never a mutated original, ADK
        callbacks must return a new object or ``None``, never mutate `tool_response` in
        place (other callbacks may still hold a reference to it)."""
        index = self.record(tool_name=tool.name, arguments=args, result=tool_response)
        if isinstance(tool_response, Mapping):
            return {**tool_response, "audit_index": index}
        return None


# ---------------------------------------------------------------------------
# INVESTIGATE
# ---------------------------------------------------------------------------

INVESTIGATE_TOOL_NAMES = ("detect_anomalies", "measure_slice", "split_on_dimension")

INVESTIGATE_INSTRUCTION = """\
You are investigating a suspected quality-of-experience incident on a streaming video
platform. A population-level anomaly has already been detected; your job is to find
WHERE it is concentrated -- the smallest sub-population (the "blast radius") whose own
deviation explains it -- using only the tools below. You must never invent a number:
every figure in your final answer must come from a tool call you actually made in this
investigation.

Your tools:
- detect_anomalies: confirms WHEN and how severely a slice deviated from its own
  baseline over a window. Use it to check whether narrowing to a candidate slice
  sharpened the signal -- a real fault reads as a LARGER z-score once isolated from
  the diluted whole-population signal.
- measure_slice: one clean baseline comparison (value, baseline, z) for a specific
  slice and window. Use it to confirm a candidate's own deviation before, or after,
  you commit to it.
- split_on_dimension: breaks a slice down by one dimension (device_type, app_version,
  os_version, cdn, pop, isp, country, region, title_id) and ranks each value by
  contribution AND lift. LIFT IS THE SIGNAL TO ACT ON, not raw share or raw
  contribution: lift around 1.0 means a value is just a big population segment, not a
  cause; lift meaningfully above 1.0 (roughly above 1.5) means that value is genuinely
  worse than its size predicts and is worth descending into. Lift below 1.0, or null,
  means do not chase that value no matter how large its share looks.

Procedure:
1. Form a hypothesis about which dimension might explain the deviation.
2. Call split_on_dimension for that dimension on your current slice (start from the
   whole population, i.e. an empty slice).
3. Read the lift of the top value. If it is meaningfully above 1.0, refine your slice
   to pin that value and repeat from step 1 with a DIFFERENT dimension -- a dimension
   already pinned in your slice cannot be split again.
4. Stop descending as soon as any of these holds, and record which one as
   stop_reason:
   - low_lift: the best remaining candidate's lift is not meaningfully above 1.0.
   - low_share: no remaining dimension explains enough of the deviation to justify
     descending further.
   - single_value: every remaining dimension has only one usable value left --
     nothing left to compare.
   - max_depth: you have already descended three levels; stop regardless of lift to
     keep the investigation bounded.
   - evidence_sufficient: one more level of splitting no longer meaningfully raises
     lift over your current slice -- the evidence has stopped improving.
5. Before finalizing, call measure_slice or detect_anomalies on your final slice to
   confirm its own deviation is real -- do not stop on a split_on_dimension result
   alone without this confirmation, and cite that confirming call as `source`.

Report the final slice as exact dimension/value pairs (empty if no split ever showed
meaningful lift), the metric and window you investigated, the lift you read at the
point you stopped (null if final_slice is empty), your stop_reason, your reasoning,
and the `source` tool call that confirmed the final slice. Every tool result you
receive carries its own audit_index -- copy it exactly into `source`, never guess.
"""


def build_investigate_agent(
    tools: Sequence[FunctionTool],
    *,
    model: Model,
    audit_log: AuditLog,
    mcp_toolset: McpToolset | None = None,
    name: str = "investigate",
) -> LlmAgent:
    """The core stage: forms a hypothesis, splits on a dimension, reads lift, and
    decides whether to descend -- see the module docstring for how `model`
    substitution and the audit log work.

    `tools` must be exactly the three tools named in `INVESTIGATE_TOOL_NAMES`
    (`detect_anomalies`, `measure_slice`, `split_on_dimension`) -- `find_changes` and
    `quantify_impact` belong to later stages and are never handed to INVESTIGATE.
    `mcp_toolset`, when given, is the read-only raw-SQL escape hatch (construction
    only; this function never calls it) for when the three primitives above do not
    anticipate something.
    """
    agent_tools: list[Any] = list(tools)
    if mcp_toolset is not None:
        agent_tools.append(mcp_toolset)
    return LlmAgent(
        name=name,
        model=model,
        instruction=INVESTIGATE_INSTRUCTION,
        tools=agent_tools,
        output_schema=InvestigationResult,
        output_key="investigation_result",
        after_tool_callback=audit_log.after_tool_callback,
    )


# ---------------------------------------------------------------------------
# CORRELATE
# ---------------------------------------------------------------------------

CORRELATE_TOOL_NAMES = ("find_changes",)

CORRELATE_INSTRUCTION = """\
INVESTIGATE has localized the anomaly to a specific slice (blast radius) with a known
onset. Your job is to judge, from find_changes, which recorded change most plausibly
caused it -- and you must engage with the disconfirming evidence the tool gives you
for each candidate, not just its score.

The investigation so far:
{investigation_result}

Call find_changes once, with that slice and its onset. For every candidate it
returns:
- Read disconfirming_evidence.note and the sibling counts (siblings_checked,
  siblings_degraded, siblings_not_degraded). A change that shipped everywhere but
  only THIS slice degraded is much stronger evidence than one that shipped
  everywhere and everything degraded -- the latter's disconfirming evidence should
  lower your confidence in that candidate even if its raw score is high.
- Decide whether the candidate is still_plausible after weighing that evidence, and
  write a one-sentence disconfirming_evidence_assessment explaining why, as
  disconfirming_evidence_assessment on that candidate.

Do not just report the top-scored candidate uncritically. Rank every candidate
find_changes returned (rank starts at 1), state your overall confidence (low, medium,
or high) given how the disconfirming evidence came out across all of them, and
summarize that reasoning in the required disconfirming_evidence field -- fill this in
even if no candidate survives scrutiny or find_changes returned nothing. Cite the
find_changes call as `source`, using the audit_index its result carried.
"""


def build_correlate_agent(
    tools: Sequence[FunctionTool],
    *,
    model: Model,
    audit_log: AuditLog,
    name: str = "correlate",
) -> LlmAgent:
    """Ranks `find_changes` candidates and must engage with the disconfirming
    evidence each one carries -- `CorrelationResult.disconfirming_evidence` and
    `.confidence` are required fields, so a response that skips this fails schema
    validation rather than passing quietly.

    `tools` must be exactly `find_changes` (`CORRELATE_TOOL_NAMES`).
    """
    return LlmAgent(
        name=name,
        model=model,
        instruction=CORRELATE_INSTRUCTION,
        tools=list(tools),
        output_schema=CorrelationResult,
        output_key="correlation_result",
        after_tool_callback=audit_log.after_tool_callback,
    )


# ---------------------------------------------------------------------------
# QUANTIFY
# ---------------------------------------------------------------------------

QUANTIFY_TOOL_NAMES = ("quantify_impact",)

QUANTIFY_INSTRUCTION = """\
The blast radius, its window, and the severity of its deviation are already known
from the investigation below. Call quantify_impact once with that slice, window, and
a severity_ratio derived from the investigation's own measured deviation (never a
number you invent). Do not compute or estimate any figure yourself --
affected_subscribers and the ARR-at-risk band in your answer must equal exactly what
the tool returned.

The investigation so far:
{investigation_result}

Your job is the methodology_caveat: read the tool's methodology (every coefficient is
a stated ASSUMPTION, not a measured or trained value -- there is no churn-event ground
truth in this dataset to calibrate against) and write a plain-language caveat about
what these figures do and do not support. Cite the quantify_impact call as `source`,
using the audit_index its result carried.
"""


def build_quantify_agent(
    tools: Sequence[FunctionTool],
    *,
    model: Model,
    audit_log: AuditLog,
    name: str = "quantify",
) -> LlmAgent:
    """The tool computes every number; this agent only writes the methodology
    caveat, and must cite the exact `quantify_impact` call every figure came from.

    `tools` must be exactly `quantify_impact` (`QUANTIFY_TOOL_NAMES`).
    """
    return LlmAgent(
        name=name,
        model=model,
        instruction=QUANTIFY_INSTRUCTION,
        tools=list(tools),
        output_schema=QuantifyResult,
        output_key="quantify_result",
        after_tool_callback=audit_log.after_tool_callback,
    )


# ---------------------------------------------------------------------------
# BRIEF
# ---------------------------------------------------------------------------

BRIEF_INSTRUCTION = """\
Compose the final incident brief from the three prior stages below. You have no
tools -- do not attempt to call one. Use only the figures already present in these
stage outputs; do not introduce a number that is not in them.

Investigation:
{investigation_result}

Correlation:
{correlation_result}

Quantification:
{quantify_result}

Write a short summary, then one claim per material figure or finding (the blast
radius, the probable cause and its disconfirming evidence, the subscriber/ARR
impact). EVERY claim's `source` must be copied exactly -- same tool_name, same
audit_index -- from the `source` field of the stage output that figure came from.
Never invent an audit_index. State a recommended_action: this is a PROPOSAL only, it
will not be executed by you or by anyone without an explicit human approval step
afterward. Include methodology_notes summarizing the caveats already stated above.
"""


def build_brief_agent(*, model: Model, name: str = "brief") -> LlmAgent:
    """Composes the final document from the three prior stages' already-validated,
    already-cited outputs (available via ADK's `{state_key}` instruction templating,
    populated by each prior stage's `output_key`). Deliberately has NO tools: every
    figure it can possibly report already carries a `source` on the object it is
    reading, so it only ever copies a citation forward, never invents one.
    """
    return LlmAgent(
        name=name,
        model=model,
        instruction=BRIEF_INSTRUCTION,
        tools=[],
        output_schema=BriefResult,
        output_key="brief_result",
    )


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def select_tools(tools: Sequence[FunctionTool], names: Sequence[str]) -> list[FunctionTool]:
    """The subset of `tools` named in `names`, in that order. Raises `KeyError` naming
    every tool actually available if `names` asks for one that is not -- a stage must
    never silently end up with fewer tools than it was supposed to have."""
    by_name = {tool.name: tool for tool in tools}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(
            f"Tool(s) {missing} not found among available tools {sorted(by_name)}."
        )
    return [by_name[name] for name in names]


def build_investigation_pipeline(
    tools: Sequence[FunctionTool],
    *,
    model: Model = DEFAULT_MODEL_ID,
    mcp_toolset: McpToolset | None = None,
    name: str = "continuity_investigation",
) -> tuple[Workflow, AuditLog]:
    """Wires INVESTIGATE -> CORRELATE -> QUANTIFY -> BRIEF into one linear `Workflow`
    graph (`START -> investigate -> correlate -> quantify -> brief`), sharing one
    `AuditLog` across all of them.

    `tools` is the five-tool list `continuity.agent.tools.build_function_tools(gateway)`
    returns; this function only decides which of those five go to which stage
    (`select_tools`), it never constructs `AnalysisTools` itself -- callers own the
    `ClickHouseMCPGateway` binding (production wiring in sub-project 4; a placeholder
    gateway in tests, since binding a gateway never calls it). ACT is deliberately NOT
    included here: it sits behind an explicit approval gate (`propose_action`) that
    this pipeline never crosses on its own.

    `Workflow` (not the deprecated `SequentialAgent`) is what actually runs an
    `LlmAgent` per stage -- see the module docstring for why no stage here becomes a
    `FunctionNode`. Each `LlmAgent` is cloned when the graph is built (ADK's own
    `Workflow` construction behavior for an `LlmAgent` node), but `tools`, `model`,
    `output_schema` and `output_key` are preserved by reference on the clone, so every
    invariant the four `build_*_agent` functions establish still holds on the graph's
    nodes.
    """
    audit_log = AuditLog()
    investigate = build_investigate_agent(
        select_tools(tools, INVESTIGATE_TOOL_NAMES),
        model=model,
        audit_log=audit_log,
        mcp_toolset=mcp_toolset,
    )
    correlate = build_correlate_agent(
        select_tools(tools, CORRELATE_TOOL_NAMES), model=model, audit_log=audit_log
    )
    quantify = build_quantify_agent(
        select_tools(tools, QUANTIFY_TOOL_NAMES), model=model, audit_log=audit_log
    )
    brief = build_brief_agent(model=model)
    pipeline = Workflow(name=name, edges=[(START, investigate, correlate, quantify, brief)])
    return pipeline, audit_log


def build_readonly_mcp_toolset(config: ClickHouseConfig) -> McpToolset:
    """The read-only raw-SQL escape hatch: an `McpToolset` restricted, by
    construction, to `run_query`, `list_tables` and `list_databases` -- no
    write-capable MCP tool is ever in this filter. Construction only: this talks to
    no process and no network; the MCP subprocess is spawned lazily on first real
    use, which this project's tests never trigger (see the module docstring).
    """
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=_server_executable(),
                args=[],
                env={
                    "CLICKHOUSE_HOST": config.host,
                    "CLICKHOUSE_PORT": str(config.port),
                    "CLICKHOUSE_USER": config.user,
                    "CLICKHOUSE_PASSWORD": config.password,
                    "CLICKHOUSE_DATABASE": config.database,
                    "CLICKHOUSE_SECURE": "true" if config.secure else "false",
                },
            ),
        ),
        tool_filter=list(_MCP_READONLY_TOOL_FILTER),
    )


# ---------------------------------------------------------------------------
# Typed results for the eval harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """The whole investigation, typed -- what the eval harness compares against
    ground truth and against the deterministic walker's own result."""

    investigation: InvestigationResult
    correlation: CorrelationResult
    quantify: QuantifyResult
    brief: BriefResult


_STAGE_STATE_KEYS = (
    "investigation_result",
    "correlation_result",
    "quantify_result",
    "brief_result",
)


def extract_pipeline_result(state: Mapping[str, Any]) -> PipelineResult:
    """Re-hydrates the four stage outputs ADK stored as plain dicts in session
    state (each under its stage's `output_key`) back into their Pydantic
    schemas -- the typed contract the eval harness expects, not a bag of dicts.

    Raises `KeyError` naming every missing stage if the pipeline did not run to
    completion; never returns a partially-filled result.
    """
    missing = [key for key in _STAGE_STATE_KEYS if key not in state]
    if missing:
        raise KeyError(f"Pipeline did not complete: missing state key(s) {missing}.")
    return PipelineResult(
        investigation=InvestigationResult.model_validate(state["investigation_result"]),
        correlation=CorrelationResult.model_validate(state["correlation_result"]),
        quantify=QuantifyResult.model_validate(state["quantify_result"]),
        brief=BriefResult.model_validate(state["brief_result"]),
    )


def verify_brief_citations(brief: BriefResult, audit_log: AuditLog) -> None:
    """The mechanical check behind "every number in a generated brief traces to a
    logged tool call": raises `ValueError` naming the offending claim if any
    `BriefClaim.source` cites an `audit_index` that was never recorded, or cites
    the wrong `tool_name` for the entry actually at that index. Never silently
    drops or ignores a bad citation -- a fabricated-looking figure must fail
    loudly, not be filtered out quietly.
    """
    for claim in brief.claims:
        index = claim.source.audit_index
        if index >= len(audit_log.entries):
            raise ValueError(
                f"Claim {claim.text!r} cites audit_index={index}, but the audit log "
                f"only has {len(audit_log.entries)} entries."
            )
        entry = audit_log.entries[index]
        if entry.tool_name != claim.source.tool_name:
            raise ValueError(
                f"Claim {claim.text!r} cites tool_name={claim.source.tool_name!r} at "
                f"audit_index={index}, but that entry is actually tool_name="
                f"{entry.tool_name!r}."
            )


# ---------------------------------------------------------------------------
# ACT -- proposal only, behind an explicit approval gate
# ---------------------------------------------------------------------------


class ApprovalRequiredError(PermissionError):
    """Raised by `propose_action` when `approved` is not `True`. ACT never runs
    without an explicit, affirmative human approval -- there is no default that
    lets it through."""


def propose_action(brief: BriefResult, *, approved: bool) -> ActProposal:
    """The ACT stage: turns BRIEF's already-decided `recommended_action` into a
    locked `ActProposal`, and only if a human has explicitly approved it first.

    This is deliberately NOT an `LlmAgent`. The judgement already happened in
    BRIEF, which authored `recommended_action`; ACT's only job is enforcing the
    approval gate and producing a proposal record, neither of which needs a model
    call. It performs no I/O whatsoever -- no tool, no gateway, no MCP session, no
    write of any kind -- so "nothing may write anywhere" holds structurally, not
    by convention: there is no code path here that could.
    """
    if not approved:
        raise ApprovalRequiredError(
            "ACT requires explicit human approval. Call propose_action(brief, "
            "approved=True) only after a human has reviewed the brief -- never "
            "default to approving it."
        )
    return ActProposal(
        proposed_action=brief.recommended_action,
        rationale=brief.summary,
    )
