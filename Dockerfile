# syntax=docker/dockerfile:1

# Build stage: resolve locked dependencies and package the pinned model bundle.
# Network access happens here only; runtime startup never downloads weights.
FROM python:3.14-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
# --no-editable: the runtime stage keeps only the virtualenv, not the source tree.
RUN uv sync --frozen --no-dev --no-editable
RUN .venv/bin/python -c "import sentiment_service.api"

# Model identity comes from the delivered manifest, just as in `just model`.
COPY model_manifest.json ./
RUN MODEL_ID=$(.venv/bin/python -c "import json; print(json.load(open('model_manifest.json'))['model_name'])") \
    && MODEL_REVISION=$(.venv/bin/python -c "import json; print(json.load(open('model_manifest.json'))['source_revision'])") \
    && MODEL_FILES=$(.venv/bin/python -c "from sentiment_service.artifacts import PACK_FILES; print(*PACK_FILES)") \
    && .venv/bin/hf download "$MODEL_ID" $MODEL_FILES \
        --revision "$MODEL_REVISION" --local-dir /app/models/sentiment \
    && rm -rf /app/models/sentiment/.cache \
    && .venv/bin/python -m sentiment_service.artifacts \
        --model-id "$MODEL_ID" --revision "$MODEL_REVISION" --destination /app/models/sentiment


# Runtime stage: virtualenv, verified bundle and source only.
FROM python:3.14-slim AS runtime

# The image pins only what the container must know: where the bundle lives and
# how to reach the app. Runtime policy defaults live in Settings, the single
# source; override any of them with `-e` at run time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    MODEL_DIR=/app/models/sentiment

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=12 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

CMD ["python", "-m", "sentiment_service"]
