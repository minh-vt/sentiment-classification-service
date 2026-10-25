"""The CI probe retries startup failures and fails when no scrape succeeds."""

import http.server
import json
import os
import shutil
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


def probe(url):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--base-url", url, "--timeout", "0.1"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_monitor_retries_startup_without_uv(tmp_path):
    executable = shutil.which("just")
    assert executable, "the project's developer entrypoint requires just"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    (bin_dir / "sh").symlink_to(shutil.which("sh"))
    with prometheus([(200, b""), (503, b"starting"), (200, target())]) as url:
        result = subprocess.run(
            [executable, "monitor", url],
            cwd=SCRIPT.parent.parent,
            env={**os.environ, "PATH": str(bin_dir)},
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert result.returncode == 0, result.stderr
    assert "target up" in result.stdout


def test_probe_times_out_when_the_target_stays_down():
    with prometheus([(200, target("0"))]) as url:
        result = probe(url)
    assert result.returncode != 0
    assert "target not up" in result.stderr


def test_probe_handles_connection_failure_without_traceback():
    with http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), http.server.BaseHTTPRequestHandler
    ) as server:
        url = f"http://127.0.0.1:{server.server_port}"
    result = probe(url)
    assert result.returncode != 0
    assert "never scraped sentiment-service" in result.stderr
    assert "Traceback" not in result.stderr
