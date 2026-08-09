# Continuity — Project Instructions

Agentic incident investigation for streaming video quality of experience.
Built for the Agentic Cinema hackathon, **ClickHouse track**. Deadline **2026-09-07, 2:00pm PDT**.

- Design spec: `docs/superpowers/specs/2026-08-08-continuity-design.md`
- Master plan: `docs/superpowers/plans/2026-08-08-continuity-master.md`
- Setup checklist: `docs/SETUP.md`

---

## Hard constraints — violating any of these disqualifies the submission

1. **Only Google AI packages may be used at runtime.** Permitted: `google-adk`, `google-genai`,
   `google-generativeai`, `google-cloud-aiplatform`. AI models, agent frameworks and AI APIs from
   every other vendor (OpenAI, Anthropic, AWS, Microsoft, …) are prohibited. Non-AI third-party
   libraries are fine. Check `pyproject.toml` before adding any dependency.
2. **ClickHouse must be accessed at runtime through the official `mcp-clickhouse` server.**
   Agent-runtime reads go through `continuity/gateway/mcp_gateway.py` and nothing else.
   Bulk loading (`continuity/data/load.py`) uses `clickhouse-connect` directly — that is build-time
   ops, not agent runtime, and the distinction is stated explicitly in the README.
3. **Ground truth never enters ClickHouse.** Planted-incident truth lives in `data/ground_truth.json`
   and is read only by the eval harness. The agent must not be able to discover the answers.
4. **The deterministic analysis engine contains no LLM calls.** Detection, drill-down, correlation
   and impact are statistics. Gemini decides what to investigate and writes prose; it never
   sources a number.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.13.1 |
| Package manager | uv 0.10.2 |
| Database | ClickHouse 25.8 (Docker locally; Cloud or self-hosted GCE for the demo) |
| Runtime DB access | `mcp-clickhouse` via `mcp` client |
| Bulk load | `clickhouse-connect` |
| Agent framework | `google-adk` (added in sub-project 3) |
| Model | `gemini-3.6-flash` via Vertex AI on the **`global`** endpoint |
| API | FastAPI + SSE (sub-project 4) |
| Frontend | React + Vite + Tailwind (sub-project 4) |
| Deploy | Cloud Run, project `agentic-hackathon-504919`, region `us-central1` |

### Model availability — verified against the live API, 2026-08-08

`GOOGLE_CLOUD_LOCATION` must be **`global`**, not a region. Every Gemini 3.x model returns
404 in `us-central1`; they serve only from the global endpoint. `us-central1` offers
nothing newer than the 2.5 generation.

| Model | `global` | `us-central1` |
|---|---|---|
| `gemini-3.6-flash` | yes | no |
| `gemini-3.5-flash`, `gemini-3.5-flash-lite` | yes | no |
| `gemini-flash-latest` | yes | no |
| `gemini-2.5-pro` | yes | yes |
| `gemini-2.5-flash` | — | yes |
| `gemini-3.1-pro` / `3.5-pro` / `3.6-pro` | **404 — do not exist for this project** | no |

There is no Gemini 3.x Pro tier here, so the design cannot assume one. Default to
`gemini-3.6-flash`; `gemini-2.5-pro` is the only Pro-tier fallback if a stage proves too
hard for Flash. Note the Vertex location (`global`) is independent of the Cloud Run
deploy region (`us-central1`) — keep them as separate settings.

### ADK runtime — proven end to end, 2026-08-08

All four verified against the real project and real credentials, not documentation:

| Fact | Verified |
|---|---|
| `google-adk` 2.6.3 declares `mcp>=1.24,<2` under its `mcp` extra | matches the pin already forced by the fastmcp break — no conflict |
| `google-adk` + `mcp-clickhouse` coexist | adk 2.6.3, google-genai 2.17.0, mcp 1.29.0, fastmcp 2.14.7 |
| Import paths | `from google.adk.agents import LlmAgent, SequentialAgent`; `from google.adk.tools.mcp_tool import McpToolset`; `from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams`; `from mcp import StdioServerParameters` |
| `McpToolset(tool_filter=[...])` | constructs against the real server; restrict the agent to `run_query`, `list_tables`, `list_databases` — read-only by construction |
| `LlmAgent` → `gemini-3.6-flash` on Vertex | replies correctly with `GOOGLE_CLOUD_LOCATION=global` |
| Pydantic `output_schema` | round-trips as valid JSON, so typed stage contracts are real |

Install with the extra: `google-adk[mcp]`.

**`GOOGLE_CLOUD_LOCATION` must be `global`.** With `us-central1` every call returns
404 NOT_FOUND for a model that only serves globally. `.env` is gitignored, so
`tests/test_env_completeness.py` guards against it drifting behind `.env.example`.

---

## Commands

```bash
# Environment
uv sync                                    # install deps
docker compose up -d                       # start local ClickHouse
docker compose ps                          # expect status "healthy"

# Test
uv run pytest -m "not integration"         # fast, no Docker needed
uv run pytest -m integration               # needs ClickHouse running
uv run pytest                              # everything

# Lint / format
uv run ruff check .
uv run ruff format .

# Data
uv run python -m continuity.data.load --days 56
```

`gcloud` is installed at user scope. If it is not on PATH, use:
`"$LOCALAPPDATA/Google/Cloud SDK/google-cloud-sdk/bin/gcloud"`

---

## Conventions

- **TDD.** Failing test first, then minimal implementation. Tests are named for behaviour, not
  implementation.
- **Never work on `main`.** Branch per sub-project: `feat/data-foundation`, `feat/analysis-core`, …
- **Pure logic stays pure.** `seasonality.py`, `topology.py`, `incidents.py` have no I/O so the
  subtle logic is testable without Docker. Keep it that way.
- **Errors are never swallowed.** A failed query must raise, never degrade into an empty result.
  A silent partial failure is invisible by construction and will make every downstream stage
  confidently wrong.
- **Every query is recorded.** The gateway logs SQL and duration so each claim in a generated brief
  can link back to the query that produced it. This is a product feature, not debug logging.
- **Never use `count()` on `qoe_rollup_5m`.** It is an `AggregatingMergeTree`, so a plain `count()`
  returns however many unmerged parts happen to exist at query time — a background-merge artifact,
  not a logical row count. It changes between identical queries. Use merge-invariant aggregates
  instead: `sum(...)` over `SimpleAggregateFunction` columns, `uniqMerge(sessions)`,
  `quantilesTDigestMerge(...)(startup_q)`, `avgMerge(bitrate_avg)`. These are associative and
  therefore deterministic regardless of merge state. This applies to every drill-down query in
  sub-project 2 — a metric built on `count()` would drift for no visible reason.
- **Convert numpy columns with `.tolist()` before `clickhouse-connect` inserts.** It expects native
  Python types; passing arrays fails deep inside the driver with an opaque
  `'numpy.datetime64' object has no attribute 'timestamp'`.
- **No secrets in code or logs.** `ClickHouseConfig.__repr__` redacts the password; keep it that way.
- Line length 100. `ruff` governs style — do not hand-format around it.

---

## Environment notes

- Platform is Windows 11; the shell is PowerShell, with Git Bash also available.
- Docker Desktop must be running before any integration test.
- Fixed generation seed (`CONTINUITY_SEED`) keeps datasets reproducible. The eval harness depends
  on regeneration being byte-identical — do not introduce unseeded randomness.
