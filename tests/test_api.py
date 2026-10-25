"""HTTP boundary behaviour; the model is substituted so the suite stays fast.

The real tokenizer/weights are exercised by the container smoke test instead.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentiment_service.api import create_app
from sentiment_service.inference import OverlongInput
from sentiment_service.settings import Settings


class FakeModel:
    version = "fake-v1"

    def predict(self, texts, max_positions=512):
        # Controlled error for the HTTP test, not a tokenizer simulation.
        for index, text in enumerate(texts):
            if text == "overlong":
                raise OverlongInput(index, max_positions + 1, max_positions)
        return [{"label": "NEGATIVE", "score": 0.75} for _ in texts]


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def make_client(load_model, **overrides):
    settings = Settings(model_dir=Path("/nonexistent"), **overrides)
    return TestClient(create_app(settings, load_model=load_model))


@pytest.fixture
def ready_client():
    """A client whose model is loaded, torn down even when an assertion fails."""
    with make_client(lambda: FakeModel()) as client:
        assert wait_until(lambda: client.get("/ready").status_code == 200)
        yield client


def test_health_answers_200_while_the_model_is_still_loading():
    started = threading.Event()
    release = threading.Event()

    def load_model():
        started.set()
        assert release.wait(timeout=10)
        return FakeModel()

    with make_client(load_model) as client:
        assert started.wait(timeout=10)
        assert client.get("/health").status_code == 200
        release.set()


def test_ready_is_503_until_loaded_then_reports_the_model_version():
    release = threading.Event()

    def load_model():
        assert release.wait(timeout=10)
        return FakeModel()

    with make_client(load_model) as client:
        assert client.get("/ready").status_code == 503
        assert client.get("/ready").json()["reason"] == "model_not_loaded"
        release.set()
        assert wait_until(lambda: client.get("/ready").status_code == 200)
        assert client.get("/ready").json()["model_version"] == "fake-v1"


def test_load_failure_keeps_the_process_alive_but_permanently_unready():
    def load_model():
        raise ValueError("weights digest mismatch")

    with make_client(load_model) as client:
        assert wait_until(lambda: client.get("/ready").json().get("reason") == "model_load_failed")
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert "digest mismatch" in client.get("/ready").json()["detail"]


def test_predict_is_503_until_the_model_is_loaded():
    release = threading.Event()

    def load_model():
        assert release.wait(timeout=10)
        return FakeModel()

    with make_client(load_model) as client:
        response = client.post("/predict", json={"text": "a good film"})
        assert response.status_code == 503
        assert response.json()["reason"] == "model_not_loaded"
        release.set()


def test_one_sentence_returns_one_prediction(ready_client):
    response = ready_client.post("/predict", json={"text": "a good film"})
    assert response.status_code == 200
    assert response.json() == {"predictions": [{"label": "NEGATIVE", "score": 0.75}]}


def test_batch_predictions_keep_the_input_order():
    class EchoModel(FakeModel):
        def predict(self, texts, max_positions=512):
            return [{"label": text.upper(), "score": 0.5} for text in texts]

    with make_client(EchoModel) as client:
        assert wait_until(lambda: client.get("/ready").status_code == 200)
        response = client.post("/predict", json={"text": ["second", "first"]})
        assert [p["label"] for p in response.json()["predictions"]] == ["SECOND", "FIRST"]


def test_malformed_json_body_is_rejected_as_a_client_error(ready_client):
    response = ready_client.post(
        "/predict", content=b'{"text": "broken"', headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["reason"] == "malformed_json"


def test_invalid_utf8_body_is_rejected_as_a_client_error(ready_client):
    response = ready_client.post(
        "/predict", content=b'{"text": "\xff"}', headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["reason"] == "malformed_json"


@pytest.mark.parametrize(
    ("payload", "reason", "index"),
    [
        ([1, 2], "not_an_object", None),
        ({}, "missing_text", None),
        ({"text": None}, "wrong_type", None),
        ({"text": 5}, "wrong_type", None),
        ({"text": ["ok", 5]}, "wrong_type", 1),
        ({"text": []}, "empty_batch", None),
        ({"text": "   "}, "blank_text", 0),
        ({"text": ["ok", "\t\n"]}, "blank_text", 1),
        ({"text": ["a"] * 33}, "batch_too_large", None),
        ({"text": "overlong"}, "too_many_tokens", 0),
    ],
)
def test_invalid_input_is_rejected_with_400_and_a_defined_reason(
    ready_client, payload, reason, index
):
    response = ready_client.post("/predict", json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["reason"] == reason
    assert error.get("index") == index


def test_metrics_expose_counts_latency_histogram_and_rejections_by_reason(ready_client):
    assert ready_client.post("/predict", json={"text": "a good film"}).status_code == 200
    assert ready_client.post("/predict", json={"text": "  "}).status_code == 400
    metrics = ready_client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    body = metrics.text

    assert 'sentiment_requests_total{endpoint="/predict",status="200"} 1.0' in body
    assert 'sentiment_requests_total{endpoint="/predict",status="400"} 1.0' in body
    assert 'sentiment_rejections_total{reason="blank_text"} 1.0' in body
    assert 'sentiment_request_duration_seconds_bucket{endpoint="/predict",le=' in body


def test_a_crashing_request_is_counted_as_a_server_error():
    class CrashModel(FakeModel):
        def predict(self, texts, max_positions=512):
            raise RuntimeError("inference exploded")

    # Assert on the HTTP response instead of letting the exception escape.
    with TestClient(
        create_app(Settings(model_dir=Path("/nonexistent")), CrashModel),
        raise_server_exceptions=False,
    ) as client:
        assert wait_until(lambda: client.get("/ready").status_code == 200)
        assert client.post("/predict", json={"text": "hi"}).status_code == 500
        body = client.get("/metrics").text

    assert 'sentiment_requests_total{endpoint="/predict",status="500"} 1.0' in body


def test_health_keeps_answering_while_a_prediction_is_running():
    entered = threading.Event()
    release = threading.Event()
    health_answered = threading.Event()

    class SlowModel(FakeModel):
        def predict(self, texts, max_positions=512):
            entered.set()
            assert release.wait(timeout=10)
            return super().predict(texts, max_positions)

    with make_client(SlowModel) as client:
        assert wait_until(lambda: client.get("/ready").status_code == 200)

        def health():
            client.get("/health")
            health_answered.set()

        prediction = threading.Thread(target=lambda: client.post("/predict", json={"text": "hi"}))
        probe = threading.Thread(target=health)
        prediction.start()
        assert entered.wait(timeout=10), "prediction never started"
        probe.start()
        try:
            # The event loop must still answer while the model works.
            assert health_answered.wait(timeout=2), "health blocked behind the prediction"
        finally:
            release.set()
            prediction.join(timeout=10)
            probe.join(timeout=10)
