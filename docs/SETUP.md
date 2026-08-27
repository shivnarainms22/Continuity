# Setup Checklist — Continuity (Agentic Cinema Hackathon, ClickHouse track)

Submission deadline: **2026-09-07, 2:00pm PDT**

---

## Do now — blocking (~30 min)

- [ ] **Hackathon credit form** — https://forms.gle/XPe837tzogh8L5sX6
      Free Google Cloud credits for participants. Submit first; lead time is unpredictable and allocations can run out.

- [ ] **Register on Devpost** — https://agentic-cinema.devpost.com/
      Click "Join hackathon". Required for eligibility.

- [ ] **Google Cloud account + billing enabled** — https://cloud.google.com/free
      $300 / 90 days for new accounts. Billing must be *enabled* even while on credits or Vertex AI will not serve.

- [ ] **Create GCP project** — https://console.cloud.google.com/projectcreate
      Record the project ID; it goes in `.env`.

- [ ] **Enable APIs** (click each, select the project, press Enable):
    - [ ] Vertex AI — https://console.cloud.google.com/apis/library/aiplatform.googleapis.com
    - [ ] Cloud Run — https://console.cloud.google.com/apis/library/run.googleapis.com
    - [ ] Secret Manager — https://console.cloud.google.com/apis/library/secretmanager.googleapis.com
    - [ ] Artifact Registry — https://console.cloud.google.com/apis/library/artifactregistry.googleapis.com
    - [ ] Cloud Scheduler — https://console.cloud.google.com/apis/library/cloudscheduler.googleapis.com

- [ ] **Public GitHub repo** — https://github.com/new
      Public. Select **Apache-2.0** or **MIT** in the "Add a license" dropdown *at creation time* — that is what makes the license appear in the About sidebar, which the rules require. A hand-added LICENSE file is often not detected.

- [ ] **Join the Discord** — https://discord.gg/7Dqk5ebCD4
      Partner engineers answer questions here.

---

## Do on ~2026-08-24 — deliberately delayed

- [ ] **ClickHouse Cloud** — https://console.clickhouse.cloud/signup

**Why delayed:** the trial is **30 days** with $300 credits. Signing up on Aug 8 expires it ~Sept 7 — the submission deadline — and judges test the hosted URL *after* that. Signing up Aug 24 covers judging through ~Sept 23.

Until then, development runs against local Docker ClickHouse.

**Fallback if timing still looks tight:** self-hosted ClickHouse on a GCE VM funded by GCP credits. Explicitly permitted by the rules ("ClickHouse Cloud or self-hosted cluster") and cannot expire mid-judging.

---

## Local dev tools

- [ ] **Docker Desktop** — https://www.docker.com/products/docker-desktop/ (needs WSL2 on Windows 11)
- [ ] **gcloud CLI** — https://cloud.google.com/sdk/docs/install
- [ ] **Node.js LTS** — https://nodejs.org
- [ ] **uv** (Python env manager) — https://docs.astral.sh/uv/getting-started/installation/

---

## Submission artifacts (end of project)

- [ ] Hosted public URL (Cloud Run)
- [ ] Public repo with detectable OSS license
- [ ] 3-minute demo video, English, public on YouTube or Vimeo — https://youtube.com/upload
- [ ] Partner track selected: **ClickHouse**
- [ ] Devpost submission form completed

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
