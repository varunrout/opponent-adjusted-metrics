# Cloud Run image for the FastAPI backend (src/opponent_adjusted/api/main.py).
# Build context is the repo root (this file lives next to pyproject.toml).
#
# Two stages: `builder` resolves poetry.lock into a venv (Poetry 2.x has no
# built-in `export` — that's a separate plugin — so this uses `poetry install`
# directly into an in-project venv instead of exporting to requirements.txt),
# `runtime` copies just that venv + source into a slim image. Keeps the final
# image free of Poetry itself and any build-only tooling.

FROM python:3.11-slim AS builder

ENV POETRY_VERSION=2.4.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR /app

# Dependencies first, in their own layer, so an app-code-only change doesn't
# invalidate the dependency-install cache.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# Now the actual application code, then install the project itself (editable,
# into the same venv) so `opponent_adjusted` is importable. README.md is
# needed too: pyproject.toml declares it as the package readme, and Poetry's
# root install (no --no-root this time) reads it — omitting it fails the
# build with "Readme path /app/README.md does not exist."
COPY src ./src
COPY README.md ./
RUN poetry install --only main


FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

USER appuser

# Cloud Run injects $PORT at runtime (default 8080 locally/for `docker run`);
# never hardcode a port here. Shell-form CMD + `exec` so uvicorn becomes PID 1
# and receives Cloud Run's SIGTERM directly on scale-down/deploy, rather than
# an intermediate shell swallowing it.
EXPOSE 8080
CMD exec uvicorn opponent_adjusted.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
