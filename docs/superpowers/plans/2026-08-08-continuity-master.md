# Continuity — Master Decomposition

**Spec:** `docs/superpowers/specs/2026-08-08-continuity-design.md`
**Deadline:** 2026-09-09, 2:00pm PDT. **Judging Period:** 2026-09-23 to 2026-10-07.
(Corrected 2026-08-28; this plan was written against a 2026-09-07 deadline that was never right.)

Five sub-projects. Each produces working, independently testable software. Each gets its own plan written *after* the previous one lands, so later plans are informed by real outcomes rather than guesses.

---

## Sub-project map

| # | Name | Produces | Depends on | Budget |
|---|---|---|---|---|
| 1 | **Data foundation** | A ClickHouse cluster holding realistic streaming telemetry with deliberately planted incidents, plus an off-database ground-truth manifest | — | 5 days |
| 2 | **Analysis core** | Pure-Python + SQL deterministic engine: detection, hierarchical drill-down, correlation, impact. A CLI that finds the planted incidents with zero LLM involvement | 1 | 7 days |
| 3 | **Agent pipeline** | ADK `SequentialAgent` wrapping the analysis core, ClickHouse reached via `mcp-clickhouse`, typed Pydantic contracts between stages | 2 | 6 days |
| 4 | **Product surface** | FastAPI + SSE backend, React UI, incident feed, live investigation view, approval gate, audit log | 3 | 6 days |
| 5 | **Eval, deploy, submit** | Scoring harness against ground truth, Cloud Run deploy, README, 3-min video, Devpost form | 4 | 6 days |

Ordering is strict — each depends on the one before. The 30-day budget above sums to 30 with no slack, which is wrong on purpose: sub-projects 1 and 2 are the ones with unknown unknowns, and 5 is compressible. Real slack comes from cutting scope in 4, not from hoping 1 and 2 go fast.

---

## The one architectural decision that governs everything

**The deterministic analysis engine (sub-project 2) contains no LLM calls.**

Detection, drill-down, correlation and impact are statistics. They run in ClickHouse and in pure Python. Gemini's job in sub-project 3 is to *choose what to investigate*, *narrate findings*, and *write the brief* — never to source a number.

Three things follow, and all three matter:

1. **Tests are fast, free and deterministic.** The hard logic is unit-testable with no API calls and no flakiness.
2. **The eval harness measures the right thing.** When attribution accuracy drops, it's a statistics bug or a prompt bug, and the layering tells you which.
3. **It's the honest answer to hallucination.** Every number in the final brief traces to a stored, re-runnable query. That's a demo asset *and* an engineering position.

## The second decision: how ClickHouse is accessed

The track rules require ClickHouse be used **at runtime via the official `mcp-clickhouse` server**. So:

- **Agent runtime** (sub-projects 2, 3) — *all* reads go through `mcp-clickhouse`. Two ways, both legitimate: the deterministic core issues its fixed queries through an MCP session, and `McpToolset` exposes MCP tools directly to Gemini for exploratory work inside the Scope stage.
- **Bulk data loading** (sub-project 1) — uses `clickhouse-connect` directly. This is build-time ops, not agent runtime, and `mcp-clickhouse` is read-only by default. **This distinction gets stated explicitly in the README** so a judge reading the code sees a deliberate choice rather than a dodged requirement.

Tests run against the real `mcp-clickhouse` server pointed at local Docker. No mocking the gateway — mocking the one thing the track is graded on would be self-defeating.

## The third decision: ground truth lives outside the database

Planted incident truth is written to `data/ground_truth.json`, never to ClickHouse. The agent therefore *cannot* read the answers, accidentally or otherwise. Only the eval harness (sub-project 5) opens that file.

---

## Verified environment facts (checked 2026-08-08, not assumed)

| Fact | Value |
|---|---|
| ADK MCP import | `from google.adk.tools.mcp_tool import McpToolset` |
| Connection params | `StdioConnectionParams`, `StreamableHTTPConnectionParams` from `google.adk.tools.mcp_tool.mcp_session_manager` |
| Server params | `StdioServerParameters` from `mcp` |
| Security filter | `McpToolset(tool_filter=[...])` — restrict Gemini to read-only tools |
| ADK gotcha | Agent + `McpToolset` **must be defined synchronously** in `agent.py` for deployment |
| ClickHouse MCP package | `mcp-clickhouse`; tools `run_query`, `list_databases`, `list_tables`; env `CLICKHOUSE_HOST/PORT/USER/PASSWORD/SECURE` |
| Gemini models on Vertex | 3.1 Pro, 3.6 Flash, 3.5 Flash, 3.5/3.1 Flash-Lite. **Pin exact IDs by listing from the live API in sub-project 3** — do not hardcode from documentation |
| Allowed AI packages | `google-adk`, `google-genai`, `google-generativeai`, `google-cloud-aiplatform` — and nothing else, any vendor |
| Local toolchain | Python 3.13.1, uv 0.10.2, Docker 29.4.3, Node 23.6.1, gcloud 579.0.0 |

---

## Calendar commitments

| Date | Action | Why |
|---|---|---|
| ~~2026-08-24~~ **actually 2026-08-15** | Create ClickHouse Cloud account | Intent was a trial still live during judging. Missed: signed up 9 days early, trial ends ~2026-09-14, judging starts 2026-09-23. Needs a card. |
| 2026-08-31 | Feature freeze | A week for deploy, video, README, submission |
| ~~2026-09-05~~ **done 2026-08-28** | Submit to Devpost | Never submit on deadline day |
| 2026-09-22 | Enable `--min-instances=1` and disable ClickHouse idle-suspend | Day before judging opens; earlier just bills an idle instance |
| 2026-09-09 14:00 PDT | Hard deadline | Was recorded as 2026-09-07 until 2026-08-28 |

---

## Standing risks

| Risk | Trigger to watch | Response |
|---|---|---|
| ClickHouse Cloud trial expires mid-judging | Signup date drifts past Aug 24 | Self-host on GCE; rules permit it and it cannot expire |
| Drill-down too slow for a live demo | Any stage >10s on demo data | Pre-aggregate harder in the rollup MV; the SSE stream keeps it *visible* while it works |
| Gemini output drifts off the typed schema | Any schema-validation retry in logs | Pydantic `output_schema` + repair retry; eval suite catches regressions |
| Scope creep in the UI | Sub-project 4 running past 6 days | Cut features, not the eval harness — the harness is the differentiator |
