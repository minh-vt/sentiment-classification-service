"""The replay CLI is the seam: order, accounting and report come from a real run."""

import http.server
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay.py"


class Recorder(http.server.BaseHTTPRequestHandler):
    received: ClassVar[list] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        text = json.loads(body)["text"]
        Recorder.received.append(text)
        if text == "boom":
            self._reply(500, {})
        elif isinstance(text, str) and not text.strip():
            self._reply(400, {"error": {"reason": "blank_text", "detail": "blank"}})
        else:
            self._reply(200, {"predictions": [{"label": "POSITIVE", "score": 0.9}]})

    def _reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Server:
    def __enter__(self):
        Recorder.received = []
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def run_replay(tmp_path, log_text, base_url):
    log = tmp_path / "replay.jsonl"
    log.write_text(log_text, encoding="utf-8")
    report = tmp_path / "replay_report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log",
            str(log),
            "--base-url",
            base_url,
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed, json.loads(report.read_text()) if report.exists() else None


MIXED_LOG = (
    "\n".join(
        [
            json.dumps({"id": "a", "ts": "2026-10-15T09:00:05+07:00", "text": "third"}),
            json.dumps({"id": "b", "ts": "2026-10-15T09:00:01+07:00", "text": "first"}),
            json.dumps({"id": "b", "ts": "2026-10-15T09:00:01+07:00", "text": "first"}),
            "{broken",
            json.dumps({"id": "c", "ts": "not-a-time", "text": "skipped"}),
            json.dumps({"id": "d", "ts": "2026-10-15T09:00:02+07:00", "text": "  "}),
            json.dumps({"id": "e", "ts": "2026-10-15T09:00:03+07:00", "text": "boom"}),
        ]
    )
    + "\n"
)


def test_requests_are_sent_in_timestamp_order_with_ties_and_duplicates_kept(tmp_path):
    with Server() as server:
        completed, _ = run_replay(tmp_path, MIXED_LOG, server.url)
    assert completed.returncode == 0, completed.stderr
    assert Recorder.received == ["first", "first", "  ", "boom", "third"]


def test_report_reconciles_every_line_and_outcome(tmp_path):
    with Server() as server:
        completed, report = run_replay(tmp_path, MIXED_LOG, server.url)
    assert completed.returncode == 0, completed.stderr
    assert report["total_lines"] == 7
    assert report["readable_lines"] == 6
    assert report["unreadable_lines"] == 1
    assert report["schema_invalid_lines"] == 1
    assert report["sent_requests"] == 5
    assert report["successful_requests"] == 3
    assert report["rejected_by_reason"] == {"blank_text": 1}
    assert report["server_errors"] == 1
    assert report["network_failures"] == 0
    assert report["duplicate_ids"] == 1
    assert report["extra_occurrences_by_id"] == 1
    assert report["latency_ms"]["sample_size"] == 3
    assert report["latency_ms"]["p50"] <= report["latency_ms"]["p95"]


def test_unreachable_service_is_reported_as_network_failure_not_a_rejection(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    completed, report = run_replay(tmp_path, MIXED_LOG, f"http://127.0.0.1:{closed_port}")
    assert completed.returncode == 0, completed.stderr
    assert report["network_failures"] == 5
    assert report["successful_requests"] == 0
    assert report["latency_ms"] == {"p50": None, "p95": None, "sample_size": 0}
    assert report["rejected_by_reason"] == {}


def test_report_is_written_by_the_script_and_source_is_untouched(tmp_path):
    before = MIXED_LOG
    with Server() as server:
        completed, report = run_replay(tmp_path, MIXED_LOG, server.url)
    assert completed.returncode == 0, completed.stderr
    assert report["source"]["sha256"]
    assert (tmp_path / "replay.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    ("values", "p", "expected"),
    [([1, 2, 3, 4], 50, 2), ([1, 2, 3, 4], 95, 4), ([5], 50, 5), ([1, 2, 3], 100, 3)],
)
def test_nearest_rank_percentile(values, p, expected):
    import importlib.util

    spec = importlib.util.spec_from_file_location("replay_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.percentile(values, p) == expected
    assert module.percentile([], 95) is None
