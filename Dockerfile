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

EXPOSE 8080

CMD ["uvicorn", "continuity.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
