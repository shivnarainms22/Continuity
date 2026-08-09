"""Typed Pydantic output contracts between pipeline stages.

Every ``LlmAgent`` in ``continuity.agent.agents`` is given one of these as its
``output_schema``. This is where the project's central rule -- "Gemini decides,
tools measure, and a number that did not come from a tool call must be
impossible to publish quietly" -- gets enforced mechanically rather than by
convention. Wherever a schema CAN make a bad answer a validation error rather
than merely a discouraged one, it does:

* ``InvestigationResult.final_slice`` is a list of ``SliceDimension`` pairs,
  never prose, so the eval harness can compare the model's blast radius
  against ground truth and against the deterministic walker's own
  ``Slice.predicates`` mechanically (dict equality), not by parsing English.
* ``CorrelationResult.disconfirming_evidence`` and ``.confidence`` are
  required fields with no default -- a response that never engages with the
  disconfirming evidence ``find_changes`` returned fails Pydantic validation
  at the ADK layer instead of silently passing through as a confident,
  unexamined answer.
* ``ClaimReference`` restricts ``tool_name`` to the five known analysis
  primitives (``Literal``, not ``str``) -- a claim cannot cite a tool that
  does not exist, and every ``BriefClaim`` requires one.
* ``ActProposal.requires_human_approval`` and ``.executed`` are ``Literal``
  singletons (``True`` and ``False`` respectively) with fixed defaults -- no
  value Gemini could produce can flip either invariant. This mirrors the
  ACT stage itself: ``continuity.agent.agents.propose_action`` performs no
  I/O and cannot write anywhere; the schema makes the same guarantee at the
  type level.

None of this module talks to a model, a database, or the network. It is pure
``pydantic.BaseModel`` definitions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The five analysis primitives from ``continuity/agent/tools.py``, and nothing
# else -- a ``ClaimReference`` citing anything outside this set fails Pydantic
# validation before an eval harness ever has to check it.
ToolName = Literal[
    "detect_anomalies",
    "measure_slice",
    "split_on_dimension",
    "find_changes",
    "quantify_impact",
]

StopReason = Literal[
    "low_lift",
    "low_share",
    "single_value",
    "max_depth",
    "evidence_sufficient",
]


class SliceDimension(BaseModel):
    """One dimension pinned to one value, e.g. ``device_type`` = ``roku``.

    A structured pair, not a sentence -- this is what lets the eval harness
    diff the model's blast radius against ``ground_truth.json`` and against
    ``walk.WalkResult.final_slice.predicates`` with a dict comparison instead
    of parsing prose.
    """

    dimension: str = Field(min_length=1)
    value: str = Field(min_length=1)


class ClaimReference(BaseModel):
    """Points a figure at the exact tool call that produced it.

    ``tool_name`` is restricted to the five known primitives -- citing a tool
    that does not exist is a validation error, not a plausible-looking typo.
    ``audit_index`` must match an entry actually recorded in this
    investigation's audit log (`continuity.agent.agents.AuditLog`); the
    schema cannot enforce that on its own (it has no access to the log at
    validation time), so the eval harness/tests check it mechanically against
    the log after the fact. Every stage that calls a tool threads its own
    ``source`` through to its output so a later stage (BRIEF has no tools of
    its own) can cite an upstream tool call without guessing its index.
    """

    tool_name: ToolName
    audit_index: int = Field(
        ge=0,
        description="Index into this investigation's audit log identifying exactly "
        "which tool call produced the referenced figure. Every tool result you "
        "receive carries its own audit_index -- copy it exactly, never guess.",
    )


class InvestigationResult(BaseModel):
    """The INVESTIGATE stage's output: the final blast-radius slice, reached by
    reading ``lift`` from ``split_on_dimension`` and deciding when to stop
    descending, plus the reasoning that produced it.

    ``final_slice`` is empty when no dimension explained the deviation better
    than the whole population -- that is a real finding (`stop_reason` will
    usually be ``"low_lift"`` or ``"single_value"``), not a missing answer.
    """

    hypothesis: str = Field(
        min_length=1,
        description="The hypothesis being tested about what is driving the deviation.",
    )
    final_slice: list[SliceDimension] = Field(
        default_factory=list,
        description="The localized blast radius, as dimension/value pairs in descent "
        "order. Empty means the whole population -- no split explained the deviation.",
    )
    metric: str = Field(min_length=1)
    window_start: str = Field(min_length=1, description="ISO-8601 datetime.")
    window_end: str = Field(min_length=1, description="ISO-8601 datetime.")
    final_lift: float | None = Field(
        description="The lift split_on_dimension reported for the last value descended "
        "into, from measure_slice/split_on_dimension. null if final_slice is empty -- "
        "there is no lift for a value that was never descended into."
    )
    stop_reason: StopReason = Field(
        description="Why descent stopped -- must be one of the known reasons, not free text."
    )
    reasoning: str = Field(min_length=1)
    source: ClaimReference = Field(
        description="The tool call (measure_slice or split_on_dimension) that confirmed "
        "this final slice's own deviation -- not merely the last split() that suggested it."
    )


class RankedCandidate(BaseModel):
    """One ``find_changes`` candidate, ranked and weighed against its own
    disconfirming evidence -- never just its raw score."""

    change_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    disconfirming_evidence_assessment: str = Field(
        min_length=1,
        description="How THIS candidate's own disconfirming_evidence (sibling counts and "
        "note) from find_changes affected its rank -- required so a candidate cannot be "
        "ranked without engaging with the evidence against it.",
    )
    still_plausible: bool


class CorrelationResult(BaseModel):
    """The CORRELATE stage's output.

    ``disconfirming_evidence`` and ``confidence`` are REQUIRED with no default
    -- a model that ranks candidates without engaging with the disconfirming
    evidence ``find_changes`` returned fails schema validation, it does not
    quietly produce an unexamined answer.
    """

    candidates: list[RankedCandidate] = Field(default_factory=list)
    disconfirming_evidence: str = Field(
        min_length=1,
        description="Required synthesis of the disconfirming evidence find_changes "
        "returned across all candidates, and how it moved (or did not move) confidence. "
        "Must be filled in even when no candidates were returned.",
    )
    confidence: Literal["low", "medium", "high"]
    top_candidate_change_id: str | None = Field(
        default=None, description="null when candidates is empty."
    )
    reasoning: str = Field(min_length=1)
    source: ClaimReference = Field(description="The find_changes call all candidates came from.")


class QuantifyResult(BaseModel):
    """The QUANTIFY stage's output. Every number here must equal what
    ``quantify_impact`` returned -- the model's only job is the methodology
    caveat, never a computation."""

    affected_subscribers: int = Field(ge=0)
    arr_at_risk_low: str = Field(min_length=1)
    arr_at_risk_expected: str = Field(min_length=1)
    arr_at_risk_high: str = Field(min_length=1)
    methodology_caveat: str = Field(
        min_length=1,
        description="Plain-language caveat about what the heuristic's assumptions do "
        "and do not support -- prose, never a number.",
    )
    source: ClaimReference


class BriefClaim(BaseModel):
    """One figure or finding in the brief, with the tool call it came from."""

    text: str = Field(min_length=1)
    source: ClaimReference


class BriefResult(BaseModel):
    """The BRIEF stage's output: the composed incident document.

    ``claims`` requires at least one entry, and every entry carries a
    ``ClaimReference`` -- a brief with zero traceable claims is a validation
    error, not an empty-but-valid document.
    """

    summary: str = Field(min_length=1)
    claims: list[BriefClaim] = Field(min_length=1)
    recommended_action: str = Field(
        min_length=1,
        description="The proposed remediation. This is a PROPOSAL -- it is never "
        "executed by this stage; continuity.agent.agents.propose_action turns it into "
        "an ActProposal only behind an explicit human approval gate.",
    )
    methodology_notes: str = Field(min_length=1)


class ActProposal(BaseModel):
    """The ACT stage's output: a proposal, never an execution.

    ``requires_human_approval`` and ``executed`` are ``Literal`` singletons --
    no value this schema can hold ever claims the action ran or skips human
    approval. There is no write tool anywhere in this project's tool surface
    (see ``continuity/agent/tools.py``), and ``propose_action`` performs no
    I/O at all, so this invariant is enforced twice: once by the type system
    here, once by the absence of any capability to act on it.
    """

    proposed_action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    requires_human_approval: Literal[True] = True
    executed: Literal[False] = False
