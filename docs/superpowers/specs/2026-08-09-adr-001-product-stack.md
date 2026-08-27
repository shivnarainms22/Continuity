# ADR-001: Product surface tech stack

**Date:** 2026-08-09
**Status:** Accepted — do not relitigate without a measured reason
**Context:** Sub-project 4 builds the hosted product. The submission requires a public URL, and *Design* is one of four equally weighted judging criteria. Reversing this later costs days we do not have.

---

## Decision

| Layer | Choice |
|---|---|
| Backend | **FastAPI** (Python 3.13, async) |
| Streaming | **SSE** via `StreamingResponse` |
| Frontend | **React 19 + TypeScript + Vite** |
| Styling | **Tailwind CSS v4** + **shadcn/ui** primitives |
| Charts | **Recharts** |
| Data fetching | **TanStack Query** |
| Packaging | **Single container** — Vite builds to static, FastAPI serves it |
| Deploy | **Cloud Run**, `--no-cpu-throttling`, `--min-instances=1` |

---

## The two constraints that actually drove this

### 1. Cloud Run throttles CPU between requests. We hold a subprocess.

`ClickHouseMCPGateway` owns a long-lived `mcp-clickhouse` **subprocess**, deliberately: establishing a session costs ~21 s, and the integration suite went from 301 s to 24 s once it was reused.

Cloud Run's default request-based CPU allocation throttles the container the instant a response is sent. Background threads and child processes freeze between requests. A held MCP session would stall and likely die.

**Therefore `--no-cpu-throttling` (CPU always allocated) is mandatory, not an optimisation.** Paired with `--min-instances=1` so no judge ever eats a 21 s cold start. Always-allocated CPU is also billed ~25% lower per second, which offsets some of the idle cost.

The alternative — a fresh MCP session per request — would put 21 s in front of every investigation. Unusable for a demo.

### 2. SSE, not WebSockets

The investigation streams stage-by-stage progress: strictly server → client. Cloud Run supports HTTP server streaming, so SSE works with `Content-Type: text/event-stream` plus `X-Accel-Buffering: no` and `Cache-Control: no-cache` to defeat intermediary buffering. WebSockets are also supported but bidirectional machinery we do not need, and they complicate reconnection.

A full investigation runs ~14 s, far inside any Cloud Run request timeout.

---

## Why single-container

One Cloud Run service serves both the API and the built frontend:

- **One URL** for the Devpost submission
- **No CORS**, no second deployment, no origin configuration
- **One thing to keep warm** — with `min-instances=1`, two services would double idle cost
- A judge runs `docker compose up` and gets the whole product

Multi-stage Dockerfile: Node builds the frontend, the Python image serves it. The Node toolchain never ships.

---

## Frontend choices

**React + Vite over Next.js.** Next.js brings SSR, routing and a Node runtime we do not need — the app is one authenticated-free dashboard talking to a Python API. Vite builds to static files FastAPI can serve directly, keeping the single-container story simple.

**Tailwind + shadcn/ui.** shadcn is copy-in components, not a runtime dependency, so we get professional primitives without inheriting a design system's look. This matters: *Design* is 25% of the score, and a stock component library reads as templated.

**Recharts.** The three visualisations needed are a QoE time series with a baseline band and highlighted anomaly windows, a contribution/lift bar chart, and small sparklines in the incident feed. Recharts covers all three declaratively (`ReferenceArea` handles the anomaly overlay). visx offers more control at meaningfully more effort; ECharts looks like a stock enterprise dashboard. If the hero chart needs more, drop to custom SVG for that one chart only.

**Rejected: Streamlit / Gradio.** Fastest to build and instantly recognisable as a prototype. Given Design is a quarter of the score, that trade is backwards.

---

## What this does NOT decide

Whether the investigation is driven by the deterministic pipeline or the Gemini agent. Both produce the same typed result, deliberately, so the UI is written against that contract and the swap is one seam. That is why the walker was built as a control arm.

---

## Cost note

`min-instances=1` with always-allocated CPU bills continuously. Acceptable under credits, but **it must not be enabled before deploy week** — there are no credits yet. Local development and the judge's `docker compose` path cost nothing.

---

## Verification — done 2026-08-09, ADR stands

A walking skeleton proved every risky assumption in one container. Measurements, not assurances:

| Claim | Evidence |
|---|---|
| Frontend builds | React 19.2.8, Vite 8.2.1, TypeScript 6.0.2, Tailwind 4.3.3, Recharts 3.10.1 on Node 23.6.1. No incompatibilities. |
| SPA deep links | `/some/deep/route` → 200 index.html; `/api/nonexistent` → 404. Fallback correctly excludes `/api/*`. |
| SSE streams incrementally | Events at t = 21.5, 23.0, 26.2, 27.9, 29.0 s — inter-event gaps of 0.8–3.3 s. |
| MCP session persists | `queries_run` climbed 70 → 78 across separate client processes and never reset. |
| Single container | 521 MB, Node toolchain absent from the runtime layer. |

### Amendments

**TanStack Query is not yet used.** The skeleton needed one fetch-on-mount. Add it when the real UI has caching or polling needs; do not add it speculatively.

**Never test SSE through Starlette's `TestClient`.** It buffers `StreamingResponse` for a
synchronous caller: all six frames arrived within microseconds, which is indistinguishable
from broken streaming. Tests run against a real `uvicorn` subprocess instead. The dangerous
direction is the opposite one — a buffering client can PASS while real streaming is broken,
and that failure would surface on the deployed URL during judging.

**Check for build artifacts at request time, not import time.** Gating route registration on
`web/dist/` existing makes the route vanish into a plain 404 before the frontend is built,
and makes it untestable without producing that artifact first.

### The ~21 s spike — RESOLVED 2026-08-10, and it was not CPU throttling

A ~21 s stall on the first operation after an idle gap was logged here as a possible
Cloud Run CPU-throttling symptom. It was not. It was **DNS**.

`localhost` resolves to IPv6 `::1` before IPv4 on this host, and Docker's published port
binds IPv4 only, so every first connection waited for the IPv6 attempt to time out:

| Host | First connection | First gateway query |
|---|---|---|
| `localhost` | 21,612 ms | 21,194 ms |
| `127.0.0.1` | 221 ms | 118 ms |

Switching `CLICKHOUSE_HOST` to `127.0.0.1` took `session_startup` from 21,122 ms to
**64.9 ms** and a full investigation from 30.8 s to **16.4 s**.

Worth recording as a reasoning failure as much as a fix: a plausible architectural story
(serverless CPU throttling) was available, matched the symptom, and was wrong. The actual
cause was one layer below and took a two-line experiment to disprove.

`--no-cpu-throttling` is still required for the held MCP subprocess — that reasoning is
independent and stands. It just is not what caused this.
