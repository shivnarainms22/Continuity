# Continuity Sub-project 3: Agent Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development.

**Goal:** A Gemini-driven investigator built on ADK that *decides what to investigate* while every number it reports comes from SQL — plus the harness that measures whether the model actually beats the deterministic walker.

**Tech Stack:** `google-adk` 2.6.3, `google-genai`, Gemini 3.6 Flash on Vertex (`global`), `mcp-clickhouse` via `McpToolset`, Pydantic typed contracts.

---

## The decision this sub-project turns on

Sub-project 2 deliberately contains no LLM. If sub-project 3 simply wraps it in prose, a judge can fairly ask what the AI contributes — and *Technological Implementation* is a quarter of the score.

So the split is:

| | Owns |
|---|---|
| **Tools (sub-project 2)** | every measurement. Baselines, deviations, contributions, impact. All SQL, all logged, all re-runnable. |
| **Gemini** | every judgement. Which hypothesis to test, which branch to descend, when the evidence suffices, what disconfirming evidence to seek, what it means for the business. |

Gemini never computes a number and never sees a number it did not request through a tool.

### Why this is stronger than either extreme

A pure-LLM agent hallucinates figures and cannot be trusted with a revenue claim. A pure-algorithm pipeline is a report generator with no judgement. The interesting middle is an investigator that reasons over evidence it must go and fetch.

And critically: **the walker gives us a control arm.** Sub-project 2's greedy drill-down solves the same problem on the same primitives. So we can *measure* whether Gemini adds value rather than asserting it.

---

## Architecture

```
                    Gemini (ADK SequentialAgent)
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
   FunctionTools         McpToolset            typed output
   (analysis core)     (raw ClickHouse,      (Pydantic schemas
   measure / split      read-only filter)     per stage)
   detect / changes
   quantify
```

### The tool surface

Analysis primitives exposed as ADK `FunctionTool`s. Each returns **typed, grounded data with the SQL that produced it**:

| Tool | Returns |
|---|---|
| `detect_anomalies(slice, metric, start, end)` | anomaly windows, peak z, unknown fraction |
| `measure_slice(slice, metric, window)` | value, baseline, z, sample size |
| `split_on_dimension(slice, metric, dimension, window)` | per-value contribution, share, **lift** |
| `find_changes(slice, window)` | ranked candidates + **disconfirming evidence** + rejections |
| `quantify_impact(slice, window, severity)` | subscribers, ARR band, methodology |

Plus `McpToolset` with `tool_filter=["run_query", "list_tables", "list_databases"]` — read-only by construction — so the model can look at something the primitives did not anticipate. This also makes the mandated ClickHouse MCP integration visible in the agent itself, not only in the plumbing.

**`split_on_dimension` returning `lift` is what makes the model a good investigator.** "Roku explains 4.4× more than its size predicts" is a fact a model can reason about; a raw share is not.

### The stages

```
DETECT → INVESTIGATE → CORRELATE → QUANTIFY → BRIEF → [approval] → ACT
```

- **Detect** — no LLM. Deterministic scan, same as sub-project 2.
- **Investigate** — *the* stage. Gemini forms a hypothesis, calls `split_on_dimension`, reads lift, decides whether to descend, and stops when the evidence stops improving. Output: a `Slice` and its reasoning.
- **Correlate** — Gemini ranks candidates from `find_changes`, states confidence, and **must** name the disconfirming evidence it considered. The tool surfaces it; the model must engage with it.
- **Quantify** — tool computes; Gemini writes the methodology caveat.
- **Brief** — Gemini writes the document. Every figure carries its query id.
- **Act** — proposal only, behind a human approval gate.

Each stage has a Pydantic `output_schema`. Verified working against Vertex on 2026-08-08.

---

## Non-negotiables

1. **Zero non-Google AI packages.** `google-adk`, `google-genai` only. Check `pyproject.toml` before adding anything.
2. **`GOOGLE_CLOUD_LOCATION=global`.** Every Gemini 3.x model 404s in `us-central1`.
3. **The model never sources a number.** If a figure appears in a brief that no tool returned, that is a bug, and the eval harness must be able to catch it.
4. **`McpToolset` stays read-only** via `tool_filter`. Not a prompt instruction — a construction constraint.
5. **Fewer, richer model calls.** Measured: SQL is 21–43 ms, a full investigation ~14 s, and a Gemini call is seconds. Model round-trips dominate; optimising query count would be optimising the wrong thing by an order of magnitude.

---

## Tasks

| # | Task | Notes |
|---|---|---|
| 1 | Add `google-adk[mcp]`, wire config | model id, location, project from env |
| 2 | **Tool layer** — analysis primitives as typed `FunctionTool`s | every return carries its SQL |
| 3 | **Investigator agent** | the core; Gemini drives the drill-down |
| 4 | Correlate + Quantify + Brief agents | typed schemas, disconfirming evidence mandatory |
| 5 | `SequentialAgent` pipeline + approval gate | audit log of every tool call |
| 6 | **Agent vs walker comparison** | same incidents, same primitives, scored |
| 7 | Acceptance gate | must beat or match the walker, and never invent a number |

---

## Acceptance criteria

- [ ] The agent isolates `roku` + `8.2.0` and `cdn_northwind` + `nw-atl-2` from a cold start
- [ ] Every number in a generated brief traces to a logged tool call — **verified mechanically**, not by reading
- [ ] The agent engages with disconfirming evidence rather than ignoring it
- [ ] The decoy produces no incident
- [ ] Head-to-head against the walker on all three incidents, with the result reported honestly **whichever way it goes**
- [ ] An investigation completes in a time that will not stall a live demo
- [ ] `pyproject.toml` contains no non-Google AI package
