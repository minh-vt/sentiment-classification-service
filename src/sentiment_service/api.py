"""HTTP boundary: liveness, readiness, prediction and request validation.

The model load runs in a background thread so that liveness and readiness stay
observable while weights are being read; nothing here downloads artifacts.
Inference runs in a worker thread too, so a long prediction cannot stall the
event loop that answers health and metrics.
"""

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.concurrency import run_in_threadpool

from sentiment_service.inference import OverlongInput

logger = logging.getLogger(__name__)


class Rejected(Exception):
    """A client error with a bounded, metric-safe reason code."""

    def __init__(self, reason, detail, index=None):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.index = index


class _State:
    """Model lifecycle state shared with request handlers."""

    def __init__(self):
        self.model = None
        self.load_error = None


def not_ready(state):
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "reason": "model_load_failed" if state.load_error else "model_not_loaded",
            "detail": state.load_error or "model is still loading",
        },
    )


def rejection(reason, detail, index=None):
    error = {"reason": reason, "detail": detail}
    if index is not None:
        error["index"] = index
    return JSONResponse(status_code=400, content={"error": error})


def parse_texts(payload):
    """One field, scalar or batch; the response is always a list.

    Precedence is structural checks, then per-item blank, then batch size, then
    token count, so a report can attribute one reason per rejected request.
    """
    if not isinstance(payload, dict):
        raise Rejected("not_an_object", "the request body must be a JSON object")
    if "text" not in payload:
        raise Rejected("missing_text", "field 'text' is required")

    texts = payload["text"]
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        raise Rejected("wrong_type", "field 'text' must be a string or a list of strings")
    elif not texts:
        raise Rejected("empty_batch", "field 'text' must contain at least one sentence")

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise Rejected("wrong_type", "each item must be a string", index)
        if not text.strip():
            raise Rejected("blank_text", "each item must contain non-whitespace text", index)
    return texts


async def read_inputs(request, settings):
    try:
        payload = await request.json()
    except json.JSONDecodeError, UnicodeDecodeError:
        # JSON must be UTF-8; undecodable bytes are a client error, not a crash.
        raise Rejected("malformed_json", "the request body is not valid UTF-8 JSON") from None
    texts = parse_texts(payload)
    if len(texts) > settings.max_batch_size:
        raise Rejected(
            "batch_too_large", f"at most {settings.max_batch_size} sentences per request"
        )
    return texts


def create_app(settings, load_model):
    state = _State()
    # One CPU-bound model: bound concurrent inference so requests queue instead
    # of oversubscribing the threads PyTorch already uses.
    slots = asyncio.Semaphore(settings.max_concurrency)
    registry = CollectorRegistry()
    requests_total = Counter(
        "sentiment_requests_total", "HTTP requests", ["endpoint", "status"], registry=registry
    )
    duration = Histogram(
        "sentiment_request_duration_seconds", "Request duration", ["endpoint"], registry=registry
    )
    rejections_total = Counter(
        "sentiment_rejections_total", "Rejected requests", ["reason"], registry=registry
    )
    # Bounded label vocabulary: unknown paths collapse into one series.
    known_paths = {"/predict", "/health", "/ready", "/metrics"}

    def refuse(reason, detail, index=None):
        rejections_total.labels(reason).inc()
        return rejection(reason, detail, index)

    def load_in_background():
        try:
            state.model = load_model()
        except Exception as exc:  # a bad artifact must not kill the process
            state.load_error = str(exc)
            logger.exception("model load failed")

    @asynccontextmanager
    async def lifespan(_app):
        thread = threading.Thread(target=load_in_background, name="model-load", daemon=True)
        thread.start()
        yield
        thread.join(timeout=5)

    app = FastAPI(title="sentiment-service", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request, call_next):
        endpoint = request.url.path if request.url.path in known_paths else "other"
        try:
            with duration.labels(endpoint).time():
                response = await call_next(request)
        except Exception:
            # A crashing handler still served a request; count it or the counter
            # disagrees with the histogram exactly when things are broken.
            requests_total.labels(endpoint, "500").inc()
            raise
        requests_total.labels(endpoint, str(response.status_code)).inc()
        return response

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready")
    def ready():
        if state.model is not None:
            return {"status": "ready", "model_version": state.model.version}
        return not_ready(state)

    @app.get("/metrics")
    def metrics():
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict")
    async def predict(request: Request):
        if state.model is None:
            return not_ready(state)
        try:
            texts = await read_inputs(request, settings)
        except Rejected as exc:
            return refuse(exc.reason, exc.detail, exc.index)
        async with slots:
            try:
                predictions = await run_in_threadpool(
                    state.model.predict, texts, settings.max_positions
                )
            except OverlongInput as exc:
                return refuse(
                    "too_many_tokens",
                    f"at most {settings.max_positions} positions including special tokens",
                    exc.index,
                )
        return {"predictions": predictions}

    return app
