# Setup Checklist — Continuity (Agentic Cinema Hackathon, ClickHouse track)

Submission deadline: **2026-09-09, 2:00pm PDT**. Judging Period: **2026-09-23 to 2026-10-07**.
(Verified 2026-08-28 against https://agentic-cinema.devpost.com/ and its rules page. This file
previously said 2026-09-07 and assumed judging followed straight after the deadline. Both wrong.)

**Status 2026-08-28: submitted.** Everything below is done except two boxes nobody can verify
from this machine, and one that is genuinely outstanding and matters:

> ### Outstanding: attach a payment method to ClickHouse Cloud
>
> The trial expires around **2026-09-14**. Judging opens **2026-09-23**. Without a card the
> hosted URL is dead for the entire evaluation window and the rest of this checklist is moot.
> See the ClickHouse Cloud section below for how the timing went wrong.

Boxes below are ticked only where there is evidence, and the evidence is named. Where it could
not be checked from the repo or the cloud APIs, the box is left open and says so rather than
being ticked on assumption.

---

## Do now — blocking (~30 min)

- [ ] **Hackathon credit form** — https://forms.gle/XPe837tzogh8L5sX6
      Free Google Cloud credits for participants. Submit first; lead time is unpredictable and allocations can run out.
      *Not verifiable from here. Tick it yourself if you submitted it.*

- [x] **Register on Devpost** — https://agentic-cinema.devpost.com/
      Click "Join hackathon". Required for eligibility.
      *Done: the submission form was completed 2026-08-28, which requires having joined.*

- [x] **Google Cloud account + billing enabled** — https://cloud.google.com/free
      $300 / 90 days for new accounts. Billing must be *enabled* even while on credits or Vertex AI will not serve.
      *Done: Vertex AI serves `gemini-3.6-flash` in production, which it would not do otherwise.*

- [x] **Create GCP project** — https://console.cloud.google.com/projectcreate
      Record the project ID; it goes in `.env`.
      *Done: `agentic-hackathon-504919`.*

- [x] **Enable APIs** (click each, select the project, press Enable):
    - [x] Vertex AI — `aiplatform.googleapis.com`, in use for every agent call
    - [x] Cloud Run — `run.googleapis.com`, hosts the `continuity` service in `us-central1`
    - [x] Secret Manager — `secretmanager.googleapis.com`, holds `clickhouse-password`, mounted as `CLICKHOUSE_PASSWORD`
    - [x] Artifact Registry — `artifactregistry.googleapis.com`, hosts `us-central1-docker.pkg.dev/agentic-hackathon-504919/continuity/continuity:v2`
    - [x] Cloud Scheduler — `cloudscheduler.googleapis.com`, enabled but **zero jobs**. The nightly "dailies" run was scoped and never built. Do not list it as a product used.

- [x] **Public GitHub repo** — https://github.com/new
      Public. Select **Apache-2.0** or **MIT** in the "Add a license" dropdown *at creation time* — that is what makes the license appear in the About sidebar, which the rules require. A hand-added LICENSE file is often not detected.
      *Done: github.com/shivnarainms22/Continuity, `visibility: PUBLIC`, GitHub's own detection returns `{"key": "mit"}`, so it renders in the About sidebar as the rules require.*

- [ ] **Join the Discord** — https://discord.gg/7Dqk5ebCD4
      Partner engineers answer questions here.
      *Not verifiable from here. Tick it yourself if you joined.*

---

## Do on ~2026-08-24 — deliberately delayed

- [x] **ClickHouse Cloud** — https://console.clickhouse.cloud/signup
      *Account created on or before 2026-08-15. Service `uvxjinv6pj.us-central1.gcp.clickhouse.cloud`,
      ClickHouse 26.2, database `continuity`, 63,847,247 events loaded.*
- [ ] **Attach a payment method.** Outstanding, and the one item that can still sink the submission.

**This plan did not survive contact, and the gap is not closed.** The trial is **30 days** with
$300 credits, and the intent was to delay signup so it stayed live through judging. In the event
the account was created on or before **2026-08-15** (`results/comparison_cloud.json` is stamped
that day), nine days earlier than planned, so the trial expires around **2026-09-14**. Judging
does not begin until **2026-09-23**. The hosted demo is therefore dark for the whole judging
window unless a payment method is attached to the ClickHouse Cloud account. Note that trial
credits expire at day 30 even with a card attached, per ClickHouse's own docs, so the card is
the question and not the credits.

Until then, development runs against local Docker ClickHouse.

**Fallback if timing still looks tight:** self-hosted ClickHouse on a GCE VM funded by GCP credits. Explicitly permitted by the rules ("ClickHouse Cloud or self-hosted cluster") and cannot expire mid-judging.

---

## Local dev tools

All four verified present on this machine 2026-08-28.

- [x] **Docker Desktop** — https://www.docker.com/products/docker-desktop/ (needs WSL2 on Windows 11) — Docker 29.4.3
- [x] **gcloud CLI** — https://cloud.google.com/sdk/docs/install — installed at user scope
- [x] **Node.js LTS** — https://nodejs.org — v23.6.1, `web/dist/` builds
- [x] **uv** (Python env manager) — https://docs.astral.sh/uv/getting-started/installation/ — 0.10.2

---

## Submission artifacts (end of project)

- [x] Hosted public URL (Cloud Run) — https://continuity-609752596743.us-central1.run.app
- [x] Public repo with detectable OSS license — MIT, detected in the About sidebar
- [x] 3-minute demo video, English, public on YouTube or Vimeo — https://youtu.be/xFN79wbc28M
      *2:01, Public not Unlisted (the rules require "made publicly visible"), English narration plus English captions.*
- [x] Partner track selected: **ClickHouse**
- [x] Devpost submission form completed — 2026-08-28

---

## After submission

- [ ] **2026-09-22: remove the cold start**, the day before judging opens. Warm is 0.27s, cold is
      35 to 63s across three measurements. Both halves are needed, because if ClickHouse is asleep
      then `min-instances` alone cannot help:

      gcloud run services update continuity --region us-central1 \
        --project agentic-hackathon-504919 --min-instances=1 --no-cpu-throttling

      and disable idle-suspend on the ClickHouse Cloud service.

      Left until then on purpose: an idle instance bills continuously and nothing is looked at
      for four weeks after submission.

---

## Reference

| What | Link |
|---|---|
| ClickHouse MCP server (mandated integration) | https://github.com/ClickHouse/mcp-clickhouse |
| Gemini Enterprise Agent Platform docs | https://docs.cloud.google.com/gemini-enterprise-agent-platform |
| ADK on Agent Platform | https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk |
| Deploying ADK agents (notebook) | https://github.com/GoogleCloudPlatform/generative-ai/blob/main/agents/agent_engine/tutorial_deploy_your_first_adk_agent_on_agent_engine.ipynb |
| MCP Database Toolbox (notebook) | https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/agent-engine/tutorial_mcp_toolbox_for_databases.ipynb |
| Cloud Run quickstart | https://cloud.google.com/run/docs/quickstarts |
| Hackathon rules | https://agentic-cinema.devpost.com/rules |
| Hackathon resources | https://agentic-cinema.devpost.com/resources |
