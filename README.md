# Continuity

**The agent that keeps the show running.**

An agentic incident-investigation system for streaming video. Continuity detects a quality-of-experience regression in billion-row playback telemetry, isolates which slice of the audience it affects, works out what change caused it, and quantifies the subscriber churn and revenue at risk — producing an evidence-backed brief where every number links to the SQL that produced it.

Built for the **Agentic Cinema** hackathon, ClickHouse track. Powered by Gemini on the Gemini Enterprise Agent Platform, with ClickHouse reached at runtime through the official `mcp-clickhouse` server.

> Status: in development. Data foundation, deterministic analysis core, and the
> Gemini agent pipeline are built and merged; the agent drives the product surface and
> streams each measurement live. Remaining: deploy, and the head-to-head write-up below
> kept in step with `results/comparison.json`.
>
> Measured head-to-head on the planted incidents, zero errors in either arm
> (`results/comparison.json`, reproduce with `uv run python scripts/compare_arms.py`):
>
> | arm | exact blast radius | attribution | decoy ignored | cost |
> |---|---|---|---|---|
> | deterministic walker | 2/3 | 2/3 | 1/1 | $0, no model calls |
> | Gemini agent | **3/3** | 2/3 | 1/1 | 655k tokens, 293s |
>
> The agent wins on localisation; the arms tie on attribution. Both numbers are
> reported because the walker is a genuinely strong baseline, and a comparison that
> only showed the flattering half would not be worth running.

---

## The problem

At a streaming operator, three teams watch three different numbers and never talk to each other. Streaming ops watches rebuffer ratio. Growth watches churn. Programming watches title performance. When a premiere night goes badly, the postmortem takes three to five days of senior analyst time — and by then the affected subscribers have already cancelled.

It is slow for a structural reason. The answer is not one query, it is a *multi-step argument*: isolate the anomaly, find which dimension slice explains it, find what changed at that moment, turn affected sessions into subscribers, turn subscribers into money. Each step is a different query shape over a table with hundreds of millions of rows.

This is not a dashboard problem. The dashboard already shows the rebuffer spike. The missing artifact is the reasoned causal chain from telemetry to a dollar figure to a specific action.

---

## How it works

A **deterministic six-stage pipeline** — a fixed DAG with typed contracts between stages, not a free-form agent loop.

```
DETECT → SCOPE → CORRELATE → QUANTIFY → BRIEF → [human approval] → ACT
```

| Stage | Deterministic (SQL / Python) | Gemini |
|---|---|---|
| **Detect** | Seasonality-aware baseline and statistical test over QoE metrics | nothing — pure SQL |
| **Scope** | Hierarchical drill-down through CDN → PoP → ISP → device → OS → app version, descending only where a dimension explains enough of the deviation | names the finding, characterises the affected cohort |
| **Correlate** | Time-windowed join against the deploy / encode / CDN change log | ranks candidates, states confidence, names the disconfirming evidence it checked |
| **Quantify** | Affected sessions → subscriber cohort → churn risk → ARR at risk | writes the methodology caveat |
| **Brief** | Assembles the evidence bundle; every claim carries its query id | writes the brief |
| **Act** | Proposal only — nothing writes without human approval | drafts the remediation and comms note |

**Gemini never sources a number.** Detection, drill-down, correlation and impact are statistics computed in ClickHouse. The model chooses what to investigate and writes the prose. Every figure in a brief traces to a stored, re-runnable query.

---

## Architecture

```
Browser ──► Continuity UI (React)
              │ SSE — live stage-by-stage progress
            FastAPI service (Cloud Run)
              │
            ADK SequentialAgent — Gemini via Vertex AI
              │ McpToolset
            mcp-clickhouse  (official ClickHouse MCP server)
              │ HTTPS
            ClickHouse
```

### How ClickHouse is accessed, and why it differs by path

- **Agent runtime** — every read goes through the official **`mcp-clickhouse`** server, as the ClickHouse track requires. Two paths, both through MCP: the deterministic engine issues its fixed queries over an MCP session, and `McpToolset` exposes MCP tools directly to Gemini for exploratory work during the Scope stage.
- **Bulk data loading** — uses `clickhouse-connect` directly. This is build-time ops rather than agent runtime, and `mcp-clickhouse` is read-only by default. The distinction is deliberate and is called out here so it is not mistaken for a shortcut.

### Schema notes

`playback_events` is a `MergeTree` partitioned by day, ordered `(event_time, cdn, device_type, app_version)` — every drill-down query filters a narrow time window first, so time leads the key.

`qoe_rollup_5m` is an `AggregatingMergeTree` materialized view holding `uniqState`, `quantilesTDigestState` and conditional aggregates, so drill-down stays interactive without touching raw events. `title_id` is deliberately **excluded** from the rollup: with hundreds of titles it would multiply group cardinality by two orders of magnitude and make the rollup larger than the raw table. Title-level analysis queries `playback_events` directly over a narrow window, which the partitioning already makes cheap.

---

## Evaluation

The synthetic telemetry contains **deliberately planted incidents** with known blast radii, plus a decoy that looks like an incident (a large organic traffic spike) but has entirely healthy QoE. Ground truth is written to `data/ground_truth.json` and **never loaded into ClickHouse**, so the agent cannot read the answers.

That makes the system measurable rather than merely demonstrable:

| Metric | What it measures |
|---|---|
| Detection precision / recall | found the real incidents, stayed quiet on the decoy |
| Localisation F1 | identified blast radius vs. the true one |
| Attribution top-1 | ranked the true cause first |
| Impact error | ARR-at-risk estimate vs. ground truth |

### Dataset acceptance, verified on 59.8M events

Before any agent work began, the dataset itself had to prove it can support the product.
`scripts/acceptance_check.py` reads it the way an analyst would — through the MCP gateway,
so it also exercises the runtime path at full scale:

```
Dataset: 59,802,205 events, 5,501,034 sessions, 3 change-log entries

  [+] INC-APP-ROKU-820   rebuffer 0.003748 inside vs 0.001171 control  (3.20x, planted 4.5x)
  [+] INC-POP-NW-ATL-2   p95 startup 10,190ms vs 3,368ms control       (3.03x, planted 3.2x)
  [+] INC-ENCODE-1       bitrate 1,127kbps vs 2,322kbps control        (0.49x, planted 0.45x)
  [+] DECOY-PREMIERE-3   volume 3.6x, rebuffer 0.87x  -- a spike, not a fault
  [+] naive threshold detector: 100% of alerts fall in 18:00-23:00
  [+] no ClickHouse table contains incident ground truth
6/6 checks passed
```

Each incident is compared against **the same dimension slice at the same hours on other
days**, holding both the audience segment and the time of day fixed — so a deviation
cannot be explained by "Roku users rebuffer more" or "everything is worse at 9pm".

The fifth line is the one that justifies the architecture. Because the generator couples
QoE to concurrency, a fixed-threshold detector fires *every night at peak*. That is the
false-positive problem real ops teams have, and it is why the Detect stage needs a
seasonality-aware baseline rather than a threshold.

---

## Running it

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
cp .env.example .env
docker compose up -d          # local ClickHouse on :8123
uv sync
uv run pytest -m "not integration"   # fast unit tests, no Docker needed
uv run pytest -m integration         # needs ClickHouse running
```

---

## License

MIT. See [LICENSE](LICENSE).
