# Continuity

**The agent that keeps the show running.**

An agentic incident-investigation system for streaming video. Continuity detects a quality-of-experience regression in billion-row playback telemetry, isolates which slice of the audience it affects, works out what change caused it, and quantifies the subscriber churn and revenue at risk — producing an evidence-backed brief where every number links to the SQL that produced it.

Built for the **Agentic Cinema** hackathon, ClickHouse track. Powered by Gemini on the Gemini Enterprise Agent Platform, with ClickHouse reached at runtime through the official `mcp-clickhouse` server.

### Live: https://continuity-609752596743.us-central1.run.app

Cloud Run (`us-central1`) against ClickHouse Cloud (`us-central1`, ClickHouse 26.2),
63.8M events. Pick an incident and watch the agent work — one frame per measurement,
each expanding to the ClickHouse query behind it. A full investigation takes **40-100s**
(measured across runs: 40s best, ~65s typical on a warm instance, ~100s on a cold one).
The spread is Gemini latency on shared capacity, not query time -- ClickHouse accounts for
0.2s of it. The range is quoted rather than the best number because a judge will time it.

**Head-to-head against the deployed database**, zero errors in either arm
(`results/comparison_cloud.json`, reproduce with `uv run python scripts/compare_arms.py`):

| arm | exact blast radius | attribution | decoy ignored | cost |
|---|---|---|---|---|
| deterministic walker | 2/3 | 2/3 | 1/1 | $0, no model calls |
| Gemini agent | **3/3** | 2/3 | 1/1 | 221k tokens |

Timings are reported separately below because they depend on where the client runs, and
the accuracy figures do not.

The agent wins on localisation and ties on attribution. Both are reported because the
walker is a genuinely strong baseline and a comparison showing only the flattering half
would not be worth running.

Two things the table cannot say. The agent's win is `INC-ENCODE-1`, a per-title encode
fault: the walker's drill-down excludes `title_id` by default, lands on
`pop`/`app_version`/`cdn`, and understates revenue at risk by 87%. And on that same
incident the agent could not corroborate a cause, **said so**, and marked its own impact
figure unreliable rather than attaching the nearest plausible change — scoring zero for
attribution, identically to the walker's confidently wrong answer. Those two failures are
not equivalent and the scoring cannot see the difference.

**On speed, the walker wins clearly, and where you measure from changes the margin.**
It issues ~79 queries per investigation, so its wall time is dominated by round trips to
the database, while the agent is bound by model latency and barely moves:

| measured from | walker | agent |
|---|---|---|
| deployed service (Cloud Run → ClickHouse Cloud, same region) | **9–11s** | 40–100s |
| the comparison harness (laptop → ClickHouse Cloud) | 25–33s | 51–54s |
| laptop → local Docker ClickHouse | ~5s | 45–56s |

The deployed row is the one a visitor experiences, and there the agent is roughly 4–10x
slower. An earlier draft of this README quoted only the middle row and concluded the gap
narrows to ~1.7x on real infrastructure. That was measured with the client a long way from
the database, which penalises the query-bound arm and flatters the model-bound one — the
opposite of what happens on the deployment itself.

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
              │ SSE — one frame per MEASUREMENT, as the agent makes it
            FastAPI service (Cloud Run)
              │
            ADK Workflow graph — Gemini via Vertex AI (global endpoint)
              │ FunctionTools (analysis primitives) + McpToolset
            mcp-clickhouse  (official ClickHouse MCP server)
              │ HTTPS
            ClickHouse Cloud
```

`Workflow`, not the deprecated `SequentialAgent`: ADK 2.6.3 deprecates the latter and
will remove it. The graph API also gives per-node retry, which is what absorbs Vertex's
intermittent 429s from shared serving capacity.

The stream is per tool call rather than per stage on purpose. The INVESTIGATE stage alone
runs ~28s, so stage-level frames would leave a spinner for most of the run. Streaming each
measurement turns the wait into the product's actual claim — you watch it form a
hypothesis, split the population, read the lift, decide whether to descend, and stop —
and it puts the ClickHouse query behind every step on screen while it runs.

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

### Dataset acceptance, verified on 63.8M events in ClickHouse Cloud

Before any agent work began, the dataset itself had to prove it can support the product.
`scripts/acceptance_check.py` reads it the way an analyst would — through the MCP gateway,
so it also exercises the runtime path at full scale. This run is against the deployed
ClickHouse Cloud service (GCP `us-central1`, ClickHouse 26.2), not a laptop:

```
Dataset: 63,847,247 events, 5,897,702 sessions, 3 change-log entries

  [+] INC-APP-ROKU-820   rebuffer 0.003894 inside vs 0.001142 control  (3.41x, planted 4.5x)
  [+] INC-POP-NW-ATL-2   p95 startup 11,610ms vs 3,407ms control       (3.41x, planted 3.2x)
  [+] INC-ENCODE-1       bitrate 1,171kbps vs 2,321kbps control        (0.50x, planted 0.45x)
  [+] DECOY-PREMIERE-3   volume 5.2x, rebuffer 1.10x  -- a spike, not a fault
  [+] naive threshold detector: 867 alerts, 100% fall in 18:00-23:00
  [+] no ClickHouse table contains incident ground truth
6/6 checks passed          MCP queries executed: 44, slowest 2,088 ms
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
