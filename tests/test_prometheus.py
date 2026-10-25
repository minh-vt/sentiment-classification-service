"""The CI probe retries startup failures and fails when no scrape succeeds."""

import http.server
import json
import subprocess
import sys
import threading
from collections import deque
from contextlib import contextmanager
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_prometheus.py"


def target(value="1"):
    return json.dumps(
        {
            "status": "success",
            "data": {"result": [{"metric": {"job": "sentiment-service"}, "value": [0, value]}]},
        }
    ).encode()


@contextmanager
def prometheus(responses):
    replies = deque(responses)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            status, body = replies[0]
            if len(replies) > 1:
                replies.popleft()
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def run_probe(url, timeout="0.1"):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base-url", url, "--timeout", timeout],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_probe_retries_startup_then_succeeds():
    with prometheus([(200, b""), (503, b"starting"), (200, target())]) as url:
        result = run_probe(url, timeout="2")
    assert result.returncode == 0, result.stderr
    assert "target up" in result.stdout


def test_probe_times_out_when_the_target_stays_down():
    with prometheus([(200, target("0"))]) as url:
        result = run_probe(url)
    assert result.returncode != 0
    assert "target not up" in result.stderr


def test_probe_handles_connection_failure_without_traceback():
    with http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler
    ) as server:
        url = f"http://127.0.0.1:{server.server_port}"
    result = run_probe(url)
    assert result.returncode != 0
    assert "never scraped sentiment-service" in result.stderr
    assert "Traceback" not in result.stderr
