# Continuity Sub-project 2: Analysis Core — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A deterministic analysis engine — seasonality-aware detection, dimensional decomposition, change correlation and revenue impact — containing **zero LLM calls**, exposed as composable primitives that sub-project 3's Gemini agent will drive.

**Architecture:** Every measurement is SQL through the MCP gateway. Two consumers sit on top of the same primitives: a deterministic greedy walker (built here) and a Gemini-driven investigator (built in sub-project 3). Because both produce the same typed result, the eval harness can score them against each other.

**Tech Stack:** Python 3.13, ClickHouse via `mcp-clickhouse`, NumPy, Pydantic. No AI packages.

---

## The design change, and why

Sub-project 1's plan had the drill-down as a fixed algorithm with Gemini narrating the result. That is too thin, for a reason that is about winning rather than about engineering:

> *Technological Implementation — how effectively does it use Google Cloud and the Partner services?*

If SQL does all the analysis and the model only writes prose, a judge can reasonably ask what the AI is contributing. "Gemini never sources a number" is the right engineering position and a good demo line, but on its own it argues *against* our own score.

**The resolution is not to weaken determinism.** It is to split responsibilities along the line where each side is actually good:

| | Owns |
|---|---|
| **Code (this sub-project)** | every *measurement* — baselines, deviations, decompositions, impact. All in SQL, all reproducible, all logged with the query that produced them. |
| **Gemini (sub-project 3)** | every *judgement* — which hypothesis to test, which branch is worth descending, when the evidence is sufficient, what disconfirming evidence to seek, what it means. |

So this sub-project ships **primitives, not a pipeline**. `measure`, `baseline`, `split`, `candidate_changes`, `impact`. The agent composes them.

### The greedy walker is not throwaway

A deterministic greedy drill-down is built here as a first consumer of those primitives. It serves three purposes, and the third is the valuable one:

1. It proves the primitives compose into a working investigation with no LLM at all.
2. It is a fallback if the agent misbehaves during a demo.
3. **It is the control arm in the evaluation.** The eval harness can score *fixed algorithm* vs *Gemini-driven investigation* vs *ground truth*, on the same data, with the same primitives.

That last point turns "is the LLM actually adding value?" from an argument into a measurement. Almost no hackathon entry can answer that question with a number. If Gemini wins, we have evidence. If it ties, we have honesty and a story about where judgement matters. Either is stronger than an assertion.

---

## File structure

```
continuity/analysis/
├── __init__.py
├── slices.py        # Slice: an immutable dimension predicate + SQL rendering
├── metrics.py       # Metric definitions: SQL expressions over rollup vs raw events
├── baseline.py      # seasonality-aware expected value + robust spread
├── detect.py        # anomaly detection over time for a slice
├── split.py         # contribution-to-deviation decomposition on one dimension
├── correlate.py     # change_log matching within a time window
├── impact.py        # affected subscribers -> churn risk -> ARR at risk
├── walk.py          # deterministic greedy drill-down (the control arm)
└── cli.py           # `python -m continuity.analysis.cli investigate ...`
tests/analysis/      # unit tests (pure maths, no DB)
tests/integration/   # against the loaded 59.8M-row dataset
```

---

## The maths that must be right

Three things here are easy to get subtly wrong and impossible to notice later.

### 1. Baselines must be seasonality-aware and robust

Measured in sub-project 1: rebuffer ratio is **0.00133 at 21:00 and 0.00039 at 09:00** — a 3.4× swing from time of day alone. Any fixed threshold fires nightly. Verified: a mean+2σ detector put **100% of its alerts in 18:00–23:00**, all false.

Baseline for a bucket = the **median of the same time-of-day bucket over the trailing 7 days**, excluding the day under test. Spread = **MAD**, converted with the 1.4826 factor. Robust z = `(actual − median) / (1.4826 × MAD)`.

Median and MAD rather than mean and σ, deliberately: a real incident in the trailing window would inflate σ and mask the next one. That is self-defeating for an incident detector.

Guard: when MAD is 0 (a perfectly flat slice, common in thin slices), the z-score is undefined. It must return "insufficient data", never infinity and never zero. A silent zero would make thin slices permanently invisible.

### 2. Ratio metrics do not decompose like sums

`rebuffer_ratio = sum(rebuffer_ms) / sum(watched_ms)`. The parent's deviation is **not** the sum of its children's ratio deviations — children carry different weights.

For dimension values *v* with watch-time share *wᵥ* and ratio *rᵥ*:

```
parent_ratio          = Σ wᵥ · rᵥ
contribution of v     = wᵥ · (rᵥ − rᵥ_baseline)
share of deviation    = contribution_v / Σ contribution
```

A naive `rᵥ − r̄` ranking promotes tiny slices with wild ratios and buries the real cause. **Test with a constructed case**: one large slice moderately degraded and one tiny slice wildly degraded, asserting the large one ranks first.

### 3. Impact must be auditable, not clever

No trained model. A transparent, documented heuristic:

```
churn_risk(subscriber) = base_rate
                       × tenure_multiplier(tenure_days)     # newer subscribers churn more
                       × severity_multiplier(sessions_affected, qoe_delta)
ARR_at_risk = Σ churn_risk × monthly_arpu × 12
```

Every coefficient is a named constant with a comment stating its source and that it is an assumption. The Quantify stage must emit its methodology alongside its number. An impact figure a judge cannot interrogate is worth nothing; one they can argue with is credible.

Use `Decimal` throughout — `monthly_arpu` is already `Decimal(8,2)`.

---

## Query performance is a live constraint

Sub-project 1 measured the slowest MCP query at **21s** (one-time server start; steady-state queries were fast). An investigation issues many queries, so this needs measuring before the walker's shape is fixed.

**Task 1 is a benchmark, not a feature.** Measure, against the real 59.8M-row dataset: a single-slice metric over a 6h window; a `GROUP BY` split on each dimension; a full 8-level walk. If a split exceeds ~2s, the rollup needs a projection or the walker needs to batch splits into one query. Decide from numbers, not intuition.

### Result (measured 2026-08-08, 59.8M events / 10.4M rollup rows)

| Query | Median |
|---|---|
| Whole-population metric, 8h window | 24 ms |
| 2-dimension slice metric, 8h window | 23 ms |
| 7-day trailing baseline series (2,016 buckets) | 41 ms |
| One split per dimension, ×8 | 183 ms |
| All 8 dimensions batched in one `UNION ALL` | **43 ms** |
| Raw-events split on `title_id` | 28 ms |

**SQL is not the constraint.** A full 8-level drill-down costs ~350 ms batched, ~1.5 s
unbatched. The rollup and its ordering key are doing their job.

Two design consequences, both from the numbers rather than from taste:

1. **Batch splits per level anyway** — 4.3× for a one-line change is free, and it keeps
   headroom if the demo dataset grows.
2. **The LLM dominates the budget, not the database.** A Gemini call is seconds; the whole
   investigation's SQL is under half a second. Sub-project 3 should therefore minimise
   *model round-trips*, not queries, and can afford to hand the model generous evidence
   per turn — which is exactly what makes it a better investigator rather than a narrator.
   Optimising query count would be optimising the wrong thing by an order of magnitude.

---

## Task list

| # | Task | Depends on |
|---|---|---|
| 1 | **Query benchmark** against the loaded dataset — establishes the performance envelope | — |
| 2 | `slices.py` + `metrics.py` — immutable predicates, SQL rendering, injection-safe | — |
| 3 | `baseline.py` — median/MAD seasonality-aware baseline, insufficient-data handling | 2 |
| 4 | `detect.py` — scan for anomalous buckets; must be silent on nightly peaks and loud on planted incidents | 3 |
| 5 | `split.py` — weighted contribution-to-deviation, correct for ratio metrics | 2, 3 |
| 6 | `correlate.py` — change_log matching, ranked, with explicit non-matches | 2 |
| 7 | `impact.py` — churn heuristic, ARR, published methodology | 2 |
| 8 | `walk.py` — greedy drill-down composing 3–5, the control arm | 3–5 |

**Carried forward into Task 8 — split baselines must be robust.** `split.py` compares
against a *single* prior window (same hours, N days earlier). `detect.py` uses a median
across trailing days. The single-window form is fragile for the same reason mean/σ was
rejected in the baseline: if that one comparison window happens to contain an incident,
the baseline is corrupted and every contribution computed from it is wrong. With three
planted incidents in a 21-day dataset that is a live possibility, not a hypothetical.
Measured shares (91.6% and 96.0%) were decisive enough to survive it here, which is luck
rather than design. The walker must either pass a median-of-trailing-windows baseline into
`split`, or `split` must accept a sequence of comparison windows and take their median.
Add a test that plants a synthetic incident inside the comparison window and asserts the
ranking still holds.
| 9 | `cli.py` — run a full investigation with no LLM | 3–8 |
| 10 | **Acceptance:** find all three planted incidents from a cold start, locate the true blast radius, attribute the true change, stay silent on the decoy and on nightly peaks | all |

---

## Acceptance criteria for sub-project 2

Against the real 59.8M-row dataset, with no LLM involved:

- [ ] Detects all three planted incidents
- [ ] Raises **zero** alerts on nightly peaks (the naive detector raised 353)
- [ ] Stays silent on the decoy
- [ ] For `INC-APP-ROKU-820`, isolates `device_type=roku AND app_version=8.2.0` — **both** dimensions, since neither alone identifies it
- [ ] Ranks the true `change_log` entry first for each real incident
- [ ] Produces an ARR-at-risk figure with its methodology
- [ ] A full investigation completes in a time that will not stall a live demo
- [ ] Every number carries the SQL that produced it
