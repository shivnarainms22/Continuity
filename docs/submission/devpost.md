# Devpost submission copy

Paste-ready. Every number here is measured and reproducible from this repo — nothing is
rounded up for effect, because the whole pitch is that this system does not invent figures.

---

## Elevator pitch (144 chars)

Three days of postmortem in 40 seconds: the audience slice, the change that broke it, the revenue at risk — each backed by a re-runnable query.

---

## Inspiration

At a streaming operator, three teams watch three different numbers and never talk to each
other. Streaming ops watches rebuffer ratio, growth watches churn, programming watches title
performance. When a premiere night goes badly the postmortem takes three to five days of
senior analyst time, and by then the affected subscribers have already cancelled.

It is slow for a structural reason, not a tooling one. The dashboard already shows the spike.
What is missing is the *argument*: isolate the anomaly, find which slice of the audience
explains it, find what changed at that moment, turn affected sessions into subscribers, turn
subscribers into money. Each link is a different query shape over hundreds of millions of
rows, and each one depends on the answer to the last.

That shape — sequential decisions, each needing the previous result — is exactly what an
agent is for, and exactly what a dashboard cannot do.

## What it does

Continuity turns a quality-of-experience alert into an evidence-backed incident brief, live,
in 40–100 seconds.

A real run from the deployed system, starting from a rebuffer spike across the whole
audience. The agent splits the population and finds Roku devices explain 4.2x more of the
deviation than their share predicts. It splits again inside Roku and finds app version 8.2.0
at 2.9x. It checks whether anything deeper explains more, finds nothing, and stops. It
searches the change log and finds the 8.2.0 rollout three hours before onset. Then it checks
the evidence *against* itself: 8.2.0 also shipped to three other device types and none of
them degraded, so this is Roku-specific rather than a bad release generally. It quantifies
3,607 affected subscribers and roughly $35,000 of annual revenue at risk, drafts a rollback,
and stops at a human approval gate.

Every step appears on screen as it happens, and every one expands to the exact ClickHouse
query behind it. **Gemini never sources a number.** It decides what to investigate and writes
the prose; detection, drill-down, correlation and impact are statistics computed in
ClickHouse. A mechanical check fails the run if any claim in a brief cites a tool call that
was never made.

## How we built it

- **ClickHouse** holds 63.8M synthetic playback events (56 days, 5.9M sessions) with
  deliberately planted incidents plus a decoy that looks like a fault but has entirely
  healthy QoE. Ground truth lives in a JSON file that is never loaded into the database, so
  the agent cannot read the answers.
- **A deterministic analysis engine** performs every measurement: seasonality-aware
  detection, hierarchical drill-down, change correlation with disconfirming evidence, and an
  auditable revenue-impact model. No LLM anywhere in it.
- **Gemini 3.6 Flash on Vertex**, orchestrated with the ADK `Workflow` graph API, supplies the
  judgement: which dimension to descend, when the evidence stops improving, which change to
  believe, and what counts against it.
- **`mcp-clickhouse`** is the only runtime path to the database, as the track requires. Bulk
  loading uses `clickhouse-connect` directly — build-time ops, stated explicitly rather than
  quietly.
- **FastAPI + SSE + React** on Cloud Run, streaming one frame per measurement.

## Challenges we ran into

**A phantom memory limit that looked like flaky tests.** The suite failed roughly one run in
two with ClickHouse `MEMORY_LIMIT_EXCEEDED`, always on a different test. Sampling tracked
memory once a second through a full run showed it peaking at 5.77 GiB against a 5.97 GiB cap
while *resident* memory was 1.71 GiB and the largest query in that window was 221 MiB. The
tracker was over-counting by roughly 4 GiB, reaching the cap, and OvercommitTracker was
stopping whichever query happened to be in flight — so every victim was an innocent 20–60 MiB
bystander. The failing test was never the cause, only the unlucky one.

**Diagnosing a 429 twice from the shape of the failure, and being wrong twice.** Vertex
returned `RESOURCE_EXHAUSTED` mid-comparison. We assumed a tokens-per-minute quota and retried
the stage; then assumed a quota the retry could not drain and paced the harness. The next run
failed on the *first* item, which no accumulation theory explains. One `gcloud quotas describe`
showed 4,000,000 input tokens per minute against a workload using 216k — two orders of
magnitude inside the only limit that exists. It was shared serving capacity, and the fix
belonged on the HTTP request rather than the enclosing stage: retrying a stage re-runs it from
turn one and re-spends tokens against the very limit it is waiting on.

**Configuration that was unused and wrong.** `.env` said the dataset was 21 days; every
committed artefact came from 56; nothing in the code read the variable at all. We hit the same
class of bug one layer deeper while loading ClickHouse Cloud — the loader's own default said
250,000 sessions per day while the database everything had been measured against held 100,000.
Ground truth pins the seed and the planted incidents but not the traffic volume, so a reload at
2.5x scale would have reproduced the same incidents over different traffic and passed every
predicate-based check while every impact figure silently moved.

## Accomplishments that we are proud of

**We measured the agent against a control arm.** A second, non-AI implementation solves the
same problem with pure statistics over the same primitives. Both run against the same planted
incidents and are scored by the same code. On the deployed stack the agent found the exact
affected slice 3/3 against the walker's 2/3, and the two tie on attribution — with zero errors
in either arm.

**We report the half that does not flatter us.** The walker is fast, free and deterministic,
and on two of three incidents it is simply right. On the deployed service it answers in 9-11
seconds against the agent's 40-100 — so the agent is four to ten times slower and costs real
tokens for a one-incident advantage. An earlier draft of our README quoted a measurement taken
from a laptop, where the walker's ~79 queries each pay a long round trip, and concluded the gap
was only 1.7x. That flattered the agent by measuring from the one place that penalises the
other arm, and it is corrected.

**The agent refused to guess.** On the incident it won, it could not corroborate a cause, said
so, and marked its own revenue estimate unreliable rather than attaching the nearest plausible
change. It scored zero for attribution — the same score as the walker's confidently wrong
answer. Those two failures are not equivalent, and the scoring cannot see the difference.

## What we learned

Cloud rate limits are *readable*, and inferring one from which request happened to fail cost
two wrong diagnoses and a commit each. A health endpoint that returns 200 regardless of state
is decoration — ours told Cloud Run a cold instance was ready while its database session was
still connecting. And a benchmark with a non-zero error count is not a measurement, however
much it renders like a scoreboard.

## What's next for Continuity

Wire the remediation proposal to a real deployment system behind the approval gate. Widen the
drill-down so it learns which dimensions matter for each incident type rather than following a
fixed hierarchy. And run the evaluation continuously, so a prompt change that costs accuracy
fails loudly instead of being discovered during a demo.

## Built with

`clickhouse` · `clickhouse-cloud` · `mcp` · `mcp-clickhouse` · `google-adk` · `gemini` ·
`vertex-ai` · `python` · `fastapi` · `sse` · `react` · `typescript` · `tailwind` · `vite` ·
`google-cloud-run` · `docker` · `uv` · `pytest`

## Try it out

- Live demo: https://continuity-609752596743.us-central1.run.app
- Source: https://github.com/shivnarainms22/Continuity
