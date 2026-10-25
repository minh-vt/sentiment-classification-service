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

ARG MODEL_REVISION=714eb0fa89d2f80546fda750413ed43d93601a13
# Downloading is `hf download`'s job; the packaging CLI only writes the manifest.
RUN .venv/bin/hf download \
        distilbert/distilbert-base-uncased-finetuned-sst-2-english \
        config.json model.safetensors tokenizer_config.json vocab.txt \
        --revision "${MODEL_REVISION}" \
        --local-dir /app/models/sentiment
# The bundle is the four runtime files plus its manifest, nothing else.
RUN rm -rf /app/models/sentiment/.cache
RUN .venv/bin/python -m sentiment_service.artifacts \
        --revision "${MODEL_REVISION}" \
        --destination /app/models/sentiment


# Runtime stage: virtualenv, verified bundle and source only.
FROM python:3.14-slim AS runtime

# Runtime defaults. Every value here can be overridden at run time; none of them
# decides which weights the image contains (that is MODEL_REVISION at build time).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    MODEL_DIR=/app/models/sentiment \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_BATCH_SIZE=32 \
    MAX_POSITIONS=512 \
    MAX_CONCURRENCY=4

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/models /app/models

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=12 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

CMD ["python", "-m", "sentiment_service"]
