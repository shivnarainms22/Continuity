# Devpost submission copy

Paste-ready. Every number below is measured and reproducible from this repo. Nothing is
rounded up for effect, because the whole pitch is that this system does not invent figures.
Where a figure came from a specific run, the run is named.

---

## Elevator pitch

> An agent that turns a streaming video quality alert into an evidence-backed incident brief in about a minute: who is affected, what changed, what it costs. Every number links to its ClickHouse query.

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

That shape, sequential decisions where each one needs the previous result, is exactly what an
agent is for and exactly what a dashboard cannot do.

## What it does

Continuity turns a quality-of-experience alert into an evidence-backed incident brief, live,
in 40 to 100 seconds.

Here is a real run against the deployed system, recorded on 2026-08-28, starting from a
rebuffer spike across the whole audience:

- It splits the entire population across every dimension at once and finds Roku devices carry
  **4.16x** more of the deviation than their audience share predicts.
- It splits again inside Roku and finds app version **8.2.0** at **2.89x**.
- It splits a third time, sees nothing that explains more, and stops.
- It refines the incident span from the detector's rough window to 18:10 through 01:55, where
  the slice runs at **3.5x** its own seasonal baseline and peaks at **10x**.
- It searches the change log and finds the Roku 8.2.0 rollout, shipped **3 hours 10 minutes**
  before onset.
- Then it checks that evidence *against* itself. 8.2.0 also went to three other device types,
  and the tool reports that **0 of those 3 degraded**. So this is Roku-specific rather than a
  bad release generally, and the agent says so instead of quietly banking the correlation.
- It quantifies **3,607 affected subscribers** and **$35,269** of annual revenue at risk (band
  $21,161 to $49,376), publishes the churn coefficients alongside the number, drafts a
  rollback, and stops at a human approval gate.

That run took **48.8 seconds** end to end, across 7 tool calls, and its citation check
passed.

**The product is not a chat box.** It is a triaged incident feed, an investigation view where
each measurement lands as its own frame over SSE as the agent makes it, a brief where every
claim carries the id of the query that produced it, and an approval gate that nothing writes
past. Every step on screen expands to the exact ClickHouse SQL behind it.

**Gemini never sources a number.** It decides what to investigate and writes the prose.
Detection, drill-down, correlation and impact are statistics computed in ClickHouse. A
mechanical citation check fails the run if any claim in a brief cites a tool call that was
never made.

## How we built it

**ClickHouse, used properly rather than as a bucket.** 63,847,247 synthetic playback events
across 5,897,702 sessions and 56 days. `playback_events` is a `MergeTree` partitioned by day
and ordered `(event_time, cdn, device_type, app_version)`, because every drill-down filters a
narrow time window first. On top of it sits `qoe_rollup_5m`, an `AggregatingMergeTree`
materialized view holding `uniqState` and `quantilesTDigestState` columns, which is what keeps
a six-level drill-down interactive without touching raw events. Every drill-down aggregate is
merge-invariant on purpose: `uniqMerge`, `quantilesTDigestMerge`, `sum` over
`SimpleAggregateFunction`. A plain `count()` on that view returns however many unmerged parts
happen to exist at query time, so a metric built on one would drift between identical queries
for no visible reason. `title_id` is deliberately excluded from the rollup, because hundreds
of titles would multiply group cardinality by two orders of magnitude and make the rollup
larger than the table it summarises. Title-level questions go to `playback_events` over a
narrow window, which the partitioning already makes cheap.

**`mcp-clickhouse` is the only runtime path to the database**, as the track requires, and it
carries both halves of the work: the deterministic engine issues its fixed queries over an MCP
session, and `McpToolset` exposes MCP tools directly to Gemini for exploratory work. The
toolset is filtered to `run_query`, `list_tables` and `list_databases`, so the agent is
read-only by construction rather than by policy. Bulk loading uses `clickhouse-connect`
directly. That is build-time ops rather than agent runtime, and we state the distinction
explicitly rather than hoping nobody reads the loader.

**A deterministic analysis engine performs every measurement**: seasonality-aware detection,
hierarchical drill-down, change correlation that actively seeks disconfirming evidence, and an
auditable revenue-impact model whose every coefficient is a named, documented assumption. No
LLM anywhere in it.

**Gemini 3.6 Flash on Vertex**, orchestrated with the ADK `Workflow` graph API, supplies the
judgement: which dimension to descend, when the evidence stops improving, which change to
believe, and what counts against it. We use `Workflow` rather than `SequentialAgent` because
ADK 2.6.3 deprecates the latter, and because per-node retry is what absorbs Vertex's
intermittent 429s from shared serving capacity.

**FastAPI, SSE and React on Cloud Run**, streaming one frame per measurement, with concurrent
investigations bounded so a public URL cannot be turned into a token bill.

## Challenges we ran into

**A phantom memory limit that looked like flaky tests.** The suite failed roughly one run in
two with ClickHouse `MEMORY_LIMIT_EXCEEDED`, always on a different test. Sampling tracked
memory once a second through a full run showed it peaking at 5.77 GiB against a 5.97 GiB cap
while *resident* memory was 1.71 GiB and the largest query in that window was 221 MiB. The
tracker was over-counting by roughly 4 GiB, reaching the cap, and OvercommitTracker was
stopping whichever query happened to be in flight, so every victim was an innocent 20 to 60
MiB bystander. The failing test was never the cause, only the unlucky one.

**Diagnosing a 429 twice from the shape of the failure, and being wrong twice.** Vertex
returned `RESOURCE_EXHAUSTED` mid-comparison. We assumed a tokens-per-minute quota and retried
the stage, then assumed a quota the retry could not drain and paced the harness. The next run
failed on the *first* item, which no accumulation theory explains. One `gcloud quotas describe`
showed a limit of 4,000,000 input tokens per minute against an entire three-incident run that
spends 215,376, roughly nineteen times inside the only limit that exists. It was shared serving
capacity, and the fix belonged on the HTTP request rather than the enclosing stage: retrying a
stage re-runs it from turn one and re-spends tokens against the very limit it is waiting on.

**Configuration that was unused and wrong.** `.env` said the dataset was 21 days; every
committed artefact came from 56; nothing in the code read the variable at all. We hit the same
class of bug one layer deeper while loading ClickHouse Cloud, where the loader's own default
said 250,000 sessions per day while the database everything had been measured against held
100,000. Ground truth pins the seed and the planted incidents but not the traffic volume, so a
reload at 2.5x scale would have reproduced the same incidents over different traffic, passed
every predicate-based check, and silently moved every impact figure.

**A health check that lied.** Cloud Run was told a cold instance was ready while its ClickHouse
session was still connecting, so the first request after a scale-up reached a service that had
just reported itself healthy and could not answer. The endpoint now runs a real query before
it returns 200.

## Accomplishments that we are proud of

**We measured the agent against a control arm.** A second, non-AI implementation solves the
same problem with pure statistics over the same primitives. Both run against the same planted
incidents and are scored by the same code, on the deployed stack:

| arm | exact blast radius | attribution | decoy ignored | errors | cost |
|---|---|---|---|---|---|
| deterministic walker | 2 of 3 | 2 of 3 | 1 of 1 | 0 | $0, no model calls |
| Gemini agent | **3 of 3** | 2 of 3 | 1 of 1 | 0 | 221,166 tokens |

Almost no hackathon entry can answer "is the LLM actually adding anything" with a number
rather than an assertion. This one can, and the number is not a clean sweep.

**We report the half that does not flatter us.** The walker is fast, free and deterministic,
and on two of three incidents it is simply right. On the deployed service it answers in 9 to
11 seconds against the agent's 40 to 100, so the agent is four to ten times slower and costs
real tokens for a one-incident advantage. An earlier draft of our README quoted a measurement
taken from a laptop, where the walker's roughly 79 queries each pay a long round trip, and
concluded the gap was only 1.7x. That flattered the agent by measuring from the one place that
penalises the other arm. It is corrected, and the correction is stated in the README rather
than buried.

**The agent's one win is the one that would cost real money.** `INC-ENCODE-1` is a bad encode
on a single title. The walker's drill-down excludes `title_id` by default, so it lands on
`cdn_solstice / sol-yyz-1 / 8.1.4`, a slice that is simply wrong, and understates the revenue
at risk by **87%**. The agent finds `title_id = 1` exactly.

**The agent refused to guess, and we scored it down for that anyway.** On that same incident
neither arm identified the true cause, so both score zero for attribution. Only one of them
says so. The agent marked the investigation unresolved and its own revenue figure low
confidence rather than attaching the nearest plausible change; the walker returned its wrong
slice with no uncertainty attached to it at all. Those two failures are not equivalent, and
the scoring cannot see the difference. We are reporting that rather than tuning it away.

## What we learned

Cloud rate limits are *readable*, and inferring one from which request happened to fail cost
two wrong diagnoses and a commit each. A health endpoint that returns 200 regardless of state
is decoration, not a check. Evidence that cannot fail is worse than no evidence, because it
stops anyone from looking: we caught ourselves running a `git diff` against a file that was
never tracked and reporting the silence as proof. And a benchmark with a non-zero error count
is not a measurement, however much it renders like a scoreboard.

## What's next for Continuity

Wire the remediation proposal to a real deployment system behind the approval gate. Widen the
drill-down so it learns which dimensions matter for each incident type instead of following a
fixed hierarchy, which is precisely the gap that cost the walker `INC-ENCODE-1`. And run the
evaluation continuously, so a prompt change that costs accuracy fails loudly instead of being
discovered during a demo.

## Built with

clickhouse, clickhouse-cloud, mcp, mcp-clickhouse, google-adk, gemini, vertex-ai, python,
fastapi, sse, react, typescript, tailwind, vite, google-cloud-run, docker, uv, pytest

## Try it out

- Live demo: https://continuity-609752596743.us-central1.run.app
- Source (MIT): https://github.com/shivnarainms22/Continuity
- Video: https://youtu.be/xFN79wbc28M

---

## Notes for whoever fills in the form (not part of the submission)

- Partner track: **ClickHouse**.
- The demo video goes in Devpost's own video field, not in "Try it out":
  https://youtu.be/xFN79wbc28M (uploaded 2026-08-28, Public, 2:01). The rules require it be
  "made publicly visible" on YouTube or Vimeo, so Public and not Unlisted.
- The elevator pitch field caps at 200 characters. The one above is 199, so it fits with one
  character to spare. If the form counts differently, drop "streaming " from "streaming video
  quality alert" to land at 189.
- First load after the service has been idle pays a Cloud Run cold start before anything
  renders. Warm it before any judged walkthrough.
