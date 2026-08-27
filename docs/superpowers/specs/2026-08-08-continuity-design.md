# Continuity — Design Spec

**Date:** 2026-08-08
**Hackathon:** Agentic Cinema: The Blockbuster Hackathon (Devpost)
**Track:** ClickHouse
**Deadline:** 2026-09-07, 2:00pm PDT
**Status:** DRAFT — awaiting approval

---

## 1. Name

**Continuity** — "The agent that keeps the show running."

Double meaning, deliberately:
- In film production, *continuity* is the crew discipline that ensures nothing breaks between shots.
- In streaming, *playback continuity* is literally the QoE metric domain (rebuffering, startup, bitrate stability).

---

## 2. The problem

A mid-size streaming operator (SVOD or FAST) has three teams that do not talk to each other:

| Team | Watches | Blind to |
|---|---|---|
| Streaming ops | rebuffer ratio, startup time, error rate | who churned |
| Growth / retention | churn, engagement | why they left |
| Content / programming | title performance | whether the title underperformed or the *delivery* did |

When a premiere night goes badly, the postmortem takes 3–5 days of senior analyst time. By the time it lands, the affected subscribers have already churned.

The reason it is slow is specific and structural:

1. The evidence lives in a heartbeat-level playback event table with 10^8–10^9 rows that maybe two people in the company can query well.
2. The question is not one query. It is a *multi-step argument*: isolate the anomaly → find which dimension slice explains it → find what changed at that moment → convert affected sessions into subscribers → convert subscribers into money.
3. Nobody has time to build that argument for every incident, so most incidents are never properly explained. They get a Slack thread and a shrug.

**This is not a dashboard problem.** Dashboards already show the rebuffer spike. The missing artifact is the *reasoned, evidence-backed causal chain from a telemetry anomaly to a dollar figure and a specific action* — and that is exactly the shape of work an agent can do and a dashboard cannot.

---

## 3. What Continuity is

A **deterministic multi-stage agent system** that, on a trigger (nightly schedule, or on-demand for a title/time window), runs a fixed investigative pipeline and produces a **Release Continuity Report**: an evidence-backed brief going from anomaly → blast radius → probable cause → revenue/churn impact → recommended action, where *every claim links to the exact SQL that produced it*.

The deliverable is not a chat transcript. It is a document a VP can forward.

### 3.1 Why "deterministic multi-step" (the hackathon asks for this explicitly)

The pipeline is a **fixed DAG with typed contracts between stages**, not a free-form ReAct loop. Gemini decides *what to look at within a stage*; it does not decide *what the stages are*. Statistics are computed in ClickHouse, never estimated by the model.

```
  [ DETECT ] → [ SCOPE ] → [ CORRELATE ] → [ QUANTIFY ] → [ BRIEF ] → (human approval gate) → [ ACT ]
     ↓            ↓             ↓               ↓
  Anomaly[]   BlastRadius  CandidateCause[]  ImpactEstimate      ← typed Pydantic schemas
```

| Stage | What is deterministic (code/SQL) | What Gemini does |
|---|---|---|
| **Detect** | Seasonality-aware baseline + statistical test over QoE metrics. Fixed thresholds. | Nothing. Pure SQL. Model never sees this decision. |
| **Scope** | Hierarchical dimension decomposition: walk CDN → PoP → ISP → device → OS → app version → title, computing each level's contribution-to-deviation, descending only where contribution exceeds threshold. Fixed algorithm. | Names the finding, writes the human-readable characterization of the affected cohort. |
| **Correlate** | Time-windowed join against `change_log` (deploys, encode ladder changes, CDN config, rights windows). Fixed window. | Ranks candidates, states confidence, and — required by the output schema — names the *disconfirming evidence it checked*. |
| **Quantify** | Affected sessions → subscriber cohort → churn-risk scoring → ARR at risk + ad impressions lost. Fixed model, published methodology. | Writes the methodology caveat and confidence interval narrative. |
| **Brief** | Assembles evidence bundle; every claim carries its source query id. | Writes the brief: exec summary, technical detail, recommended action. |
| **Act** | Proposed action is *proposed only*. Requires explicit human approval before any write. | Drafts the remediation proposal and the customer-comms note. |

The adaptive drill-down in **Scope** is the technically interesting core, and it is the thing that would take an analyst two days by hand.

### 3.2 Anti-hallucination posture

Every number in the final brief traces to a query that is stored, displayed in the UI, and re-runnable by the judge. The model narrates evidence; it never sources facts. This is both an engineering position and a demo asset — "click any number to see the SQL behind it" is a strong 15 seconds of the video.

---

## 4. Architecture

```
                    ┌─────────────────────────────────────────┐
   Browser  ──────► │  Continuity UI (React + Vite + Tailwind) │
                    │  incident feed · live DAG · brief · gate │
                    └───────────────┬─────────────────────────┘
                                    │ SSE (stage-by-stage progress)
                    ┌───────────────▼─────────────────────────┐
                    │  FastAPI service            (Cloud Run)  │
                    │  ┌───────────────────────────────────┐   │
                    │  │  ADK SequentialAgent "Continuity" │   │
                    │  │   Detect → Scope → Correlate →    │   │
                    │  │   Quantify → Brief → Act          │   │
                    │  │        (Gemini via Vertex AI)     │   │
                    │  └──────────────┬────────────────────┘   │
                    └─────────────────┼────────────────────────┘
                                      │ ADK MCPToolset
                    ┌─────────────────▼────────────────────────┐
                    │  mcp-clickhouse  (official MCP server)   │
                    └─────────────────┬────────────────────────┘
                                      │ HTTPS :8443
                    ┌─────────────────▼────────────────────────┐
                    │  ClickHouse Cloud                        │
                    │  playback_events · sessions_mv · titles  │
                    │  subscribers · change_log                │
                    └──────────────────────────────────────────┘
```

Supporting Google Cloud: **Secret Manager** (ClickHouse credentials), **Cloud Scheduler** (nightly "dailies" run), **Artifact Registry** (images), dedicated **service account with least-privilege IAM** (the governance story the hackathon brief asks for).

### 4.1 Google Cloud usage — depth, not decoration

- **ADK** (`google-adk`): `SequentialAgent` composing `LlmAgent` stages, with `ParallelAgent` for independent sub-investigations inside Scope. Typed Pydantic `output_schema` per stage — this is what makes the DAG contracts real rather than prompt-hopeful.
- **Gemini via Vertex AI** (`google-genai` / `google-cloud-aiplatform`): reasoning stages. Exact model pinned at build time against what is current on Vertex.
- **Cloud Run**: hosted public URL (a submission requirement).
- **Secret Manager + IAM**: no credentials in the repo; agent service account has read-only DB role.

### 4.2 ClickHouse usage — depth, not decoration

Required by the track rules: access **must** go through the official `mcp-clickhouse` server. Wired into ADK as an `MCPToolset`. No direct `clickhouse-connect` calls in the agent path.

The schema is designed to look like a real streaming telemetry warehouse, not a demo table:

| Table | Shape | Notes |
|---|---|---|
| `playback_events` | 50–200M rows, heartbeat-level | `MergeTree`, ordering key tuned for the drill-down access pattern |
| `sessions_mv` | session rollup | `AggregatingMergeTree` materialized view — pre-aggregates so drill-down stays interactive |
| `titles` | catalog | genre, release window, rights window |
| `subscribers` | plan, tenure, ARPU | for the impact join |
| `change_log` | deploys, encode ladder changes, CDN config | for the correlate stage |

Uses features that signal genuine understanding to a ClickHouse-engineer judge: proper ordering keys, materialized views, `quantileTDigest` for p95 startup, aggregate-function state columns, and projections where the drill-down needs a second access path.

### 4.3 Synthetic data with planted ground truth

A generator produces realistic telemetry with **deliberately planted incidents** whose true cause is recorded out-of-band:

- a bad app version rolled to 8% of Roku devices
- a degraded CDN PoP in one region
- an encode ladder regression on one title's 1080p rendition
- a decoy: an organic traffic spike that looks like an incident but is not

This makes the demo reproducible *and* makes the next item possible.

---

## 5. The differentiator: an evaluation harness

Because incidents are planted with known ground truth, Continuity ships an **eval suite** that scores the agent end to end:

- **Detection**: did it find the planted incidents, and did it stay quiet on the decoy? (precision/recall)
- **Localization**: was the identified blast radius the true one? (dimension-set F1)
- **Attribution**: was the top-ranked cause the true cause? (top-1 / top-3 accuracy)
- **Quantification**: was ARR-at-risk within tolerance of ground truth?

Reported as a table in the README, re-runnable with one command.

Almost nobody ships this at a hackathon. It is the single strongest available answer to "is this a product or a proof of concept," which is one of the four judging criteria stated verbatim. 30 days is enough time to do it properly.

---

## 6. Product experience

Explicitly **not a chat box** — a chat box reads as PoC and costs points on Design.

1. **Incident feed** — triaged, severity-ranked, each with a one-line "what and how much."
2. **Investigation view** — the DAG executing live over SSE; each stage expands to show its evidence, its chart, and its SQL.
3. **The brief** — exec summary → blast radius → probable cause with confidence and disconfirming evidence → impact with methodology → recommended action.
4. **Approval gate** — the recommended action sits behind an explicit human approve/reject. Nothing writes without it. Every agent action is audit-logged.
5. **Ground-truth toggle** (demo mode) — reveal the planted truth next to the agent's finding. Judges love verifiable claims.

---

## 7. Scope boundaries

**In:** the six-stage pipeline; ClickHouse schema + generator + planted incidents; MCP integration; web UI; eval harness; Cloud Run deploy; governance/audit layer; README + video.

**Out (explicitly):** real CDN/player integrations; auth/multi-tenancy beyond a demo login; a trained ML churn model (a transparent, documented heuristic is *better* here — it is auditable and the point is the pipeline, not the model); writing back to any real system; Grafana (a tempting second partner, but it earns nothing in the ClickHouse track and costs setup time).

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| ClickHouse Cloud trial expires or is throttled before judging | Self-hosted fallback on a GCE VM is permitted by the rules; keep the deploy path parameterized from day one |
| MCP tool-call latency makes the UI feel slow | Pre-aggregate via materialized views; stream progress over SSE so the pipeline is *visible* while it works |
| Gemini output drifts from the typed schema | Pydantic `output_schema` per stage + a retry-with-repair path; the eval suite catches regressions |
| Data generation at 200M rows is slow/expensive | Parameterize volume; generate at 50M for dev, scale up once for the demo cluster |
| Judge cannot run it locally | Docker Compose path with a small seeded dataset, verified from a clean clone |

---

## 9. Accounts the user must create (blocking, one-time)

1. **Google Cloud** account + billing enabled (new accounts get $300 free credit). Enable Vertex AI, Cloud Run, Secret Manager, Artifact Registry.
2. **ClickHouse Cloud** account (free trial). A self-hosted Docker cluster works for development and is rules-permitted as a fallback.
3. **GitHub** public repo with a detectable OSS license (Apache-2.0 or MIT) — the rules require the license to be visible in the repo's About section.
4. **YouTube** or Vimeo, for the 3-minute demo video.

No other paid services required.

---

## 10. Compliance checklist (verified against the official rules)

- [x] Track requirement: ClickHouse accessed at runtime via official `mcp-clickhouse` server
- [x] Powered by Gemini + Google Cloud Agent Builder / Gemini Enterprise Agent Platform (ADK)
- [x] Only permitted AI packages: `google-adk`, `google-genai`, `google-cloud-aiplatform`. **Zero Anthropic/OpenAI/AWS/Microsoft AI runtime dependencies.**
- [x] Deterministic, multi-step agent solving enterprise friction
- [x] Media & entertainment workflow
- [x] Newly created within the contest period (started 2026-08-08)
- [ ] Hosted public URL — Cloud Run
- [ ] Public repo with detectable OSS license
- [ ] 3-minute demo video, English
- [ ] Devpost submission form
