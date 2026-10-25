"""Smoke the real container over HTTP; used by CI and locally, stdlib only."""

import argparse
import json
import time
import urllib.error
import urllib.request


def call(base_url, path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url}{path}", data=data, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def wait_until_ready(base_url, timeout):
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            status, body = call(base_url, "/ready")
        except OSError as exc:
            last = str(exc)
        else:
            if status == 200:
                return json.loads(body)
            last = body
        time.sleep(1)
    raise SystemExit(f"service never became ready: {last}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)

    ready = wait_until_ready(args.base_url, args.timeout)
    print(f"ready: {ready}")
    if not ready.get("model_version"):
        raise SystemExit("ready did not report a model_version")

    status, body = call(args.base_url, "/predict", {"text": "a good film"})
    prediction = json.loads(body)["predictions"][0]
    print(f"predict: {status} {prediction}")
    if status != 200 or not prediction.get("label"):
        raise SystemExit("valid input did not return a prediction")

    status, body = call(args.base_url, "/predict", {"text": "   "})
    reason = json.loads(body)["error"]["reason"]
    print(f"rejected: {status} {reason}")
    if status != 400 or reason != "blank_text":
        raise SystemExit("invalid input was not rejected with 400 blank_text")

    status, body = call(args.base_url, "/metrics")
    for expected in ("sentiment_requests_total", "sentiment_rejections_total"):
        if expected not in body:
            raise SystemExit(f"{expected} missing from /metrics")
    print("metrics: required series present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
