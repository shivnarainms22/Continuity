# syntax=docker/dockerfile:1
#
# Walking-skeleton container for the Continuity product surface (ADR-001): one image,
# one process, serving both the FastAPI backend and the built React frontend. The Node
# toolchain used to build the frontend never reaches the final image -- only the built
# static files are copied across the stage boundary.
#
# ClickHouse host/credentials are supplied entirely through environment variables at
# `docker run` time (CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
# CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE, CLICKHOUSE_SECURE -- see
# continuity/config.py::ClickHouseConfig.from_env), so the same image points at a local
# `docker compose` ClickHouse or a Cloud ClickHouse instance without a rebuild.
#
# The agent path additionally needs GOOGLE_CLOUD_PROJECT, GOOGLE_GENAI_USE_ENTERPRISE=1,
# GOOGLE_CLOUD_LOCATION=global (a region 404s for every Gemini 3.x model -- see
# CLAUDE.md) and CONTINUITY_MODEL. Credentials come from the runtime service account via
# ADC; no key material is ever baked into this image.
#
# Two things to get right when deploying to Cloud Run:
#   * PORT is injected by the platform, so the server must bind whatever it is given
#     rather than a hardcoded 8080 (see CMD below).
#   * Startup spawns the mcp-clickhouse subprocess and its first connection, which costs
#     ~20s. Give the service a startup probe with enough failureThreshold to cover that,
#     or the first revision is killed before it ever becomes healthy.

# ---- Stage 1: build the frontend --------------------------------------------------
FROM node:23.6.1-slim AS frontend-builder
WORKDIR /app/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

# ---- Stage 2: python runtime ------------------------------------------------------
FROM python:3.13-slim AS runtime
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Dependencies first, in their own layer, so editing continuity/ source does not
# invalidate (and re-resolve/re-download) the whole dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY continuity/ ./continuity/
COPY data/ ./data/
COPY README.md ./
RUN uv sync --frozen --no-dev

# The Node toolchain and node_modules from stage 1 are never copied -- only the
# built static output is.
COPY --from=frontend-builder /app/web/dist ./web/dist

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENV PORT=8080
EXPOSE 8080

# Both halves of this matter and they pull against each other.
#
# Cloud Run injects PORT, and a plain exec-form CMD cannot expand it -- a hardcoded port
# silently fails to bind whenever the platform assigns anything else. Plain shell form
# fixes that but makes /bin/sh PID 1, so SIGTERM on shutdown goes to the shell and never
# reaches uvicorn: no graceful drain, and the mcp-clickhouse subprocess the gateway owns
# is killed rather than closed.
#
# `exec` gets both. The shell expands PORT, then replaces itself with uvicorn, so uvicorn
# is PID 1 and receives signals directly.
CMD ["/bin/sh", "-c", "exec uvicorn continuity.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
