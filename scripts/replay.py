"""Replay the recorded traffic against a running service and write a report.

Order is the recorded timestamp order. Gaps between timestamps are NOT replayed:
this reproduces sequence and payloads, not arrival pacing. The input file is
never modified.

Run: uv run python scripts/replay.py --base-url http://127.0.0.1:8000
"""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx


def percentile(values, p):
    """Nearest rank, no interpolation; empty input has no percentile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def read_records(raw):
    """Split the file into parseable records and counted failures."""
    lines = raw.decode("utf-8").splitlines()
    records, unreadable, schema_invalid = [], [], []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            unreadable.append({"line": number, "error": "blank line"})
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            unreadable.append({"line": number, "error": f"invalid JSON: {exc}"})
            continue
        problem, moment = _check_record(record)
        if problem:
            schema_invalid.append({"line": number, "error": problem})
            continue
        records.append((number, record, moment))
    return lines, records, unreadable, schema_invalid


def _check_record(record):
    """Return (problem, timestamp): one problem string, or the parsed time."""
    if not isinstance(record, dict):
        return "line is not a JSON object", None
    if not isinstance(record.get("id"), str):
        return "id must be a string", None
    if not isinstance(record.get("ts"), str):
        return "ts must be a string", None
    try:
        moment = datetime.fromisoformat(record["ts"])
    except ValueError as exc:
        return f"ts is not ISO-8601: {exc}", None
    if moment.utcoffset() is None:
        return "ts has no timezone", None
    return None, moment


def send(client, base_url, record):
    """Return the outcome of one request; transport failures are their own class."""
    started = time.perf_counter()
    try:
        response = client.post(f"{base_url}/predict", json={"text": record.get("text")})
    except httpx.HTTPError as exc:
        return "network_failure", None, type(exc).__name__
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code == 200:
        return "success", elapsed_ms, None
    if response.status_code == 400:
        try:
            reason = response.json()["error"]["reason"]
        except ValueError, KeyError, TypeError:
            reason = "undefined_reason"
        return "rejected", elapsed_ms, reason
    return "server_error", elapsed_ms, f"HTTP {response.status_code}"


def replay(log_path, base_url, report_path, timeout=30.0):
    raw = log_path.read_bytes()
    lines, records, unreadable, schema_invalid = read_records(raw)
    # Stable sort: equal timestamps keep the order they had in the file.
    ordered = sorted(records, key=lambda item: item[2])

    outcomes = Counter()
    rejections = Counter()
    durations = []
    failures = []
    with httpx.Client(timeout=timeout) as client:
        for number, record, _ in ordered:
            outcome, elapsed_ms, detail = send(client, base_url, record)
            outcomes[outcome] += 1
            if outcome == "success":
                durations.append(elapsed_ms)
            elif outcome == "rejected":
                rejections[detail] += 1
            else:
                failures.append({"line": number, "outcome": outcome, "detail": detail})

    ids = Counter(record["id"] for _, record, _ in records)
    report = {
        "source": {
            "path": str(log_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "base_url": base_url,
        },
        "replay_mode": "timestamp order only; arrival gaps are not replayed",
        "total_lines": len(lines),
        "readable_lines": len(records) + len(schema_invalid),
        "unreadable_lines": len(unreadable),
        "schema_invalid_lines": len(schema_invalid),
        "sent_requests": len(ordered),
        "successful_requests": outcomes["success"],
        "rejected_by_reason": dict(sorted(rejections.items())),
        "network_failures": outcomes["network_failure"],
        "server_errors": outcomes["server_error"],
        "duplicate_ids": sum(1 for count in ids.values() if count > 1),
        "extra_occurrences_by_id": sum(count - 1 for count in ids.values() if count > 1),
        "latency_ms": {
            "p50": percentile(durations, 50),
            "p95": percentile(durations, 95),
            "sample_size": len(durations),
        },
        "failures": failures,
        "unreadable_lines_detail": unreadable,
        "schema_invalid_detail": schema_invalid,
    }
    reconciled = report["sent_requests"] == (
        report["successful_requests"]
        + sum(report["rejected_by_reason"].values())
        + report["network_failures"]
        + report["server_errors"]
    )
    report["reconciled"] = reconciled
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="replay")
    parser.add_argument("--log", default="replay_log.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", default="results/replay_report.json")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    report = replay(Path(args.log), args.base_url.rstrip("/"), Path(args.report), args.timeout)
    latency = report["latency_ms"]
    print(
        f"{report['total_lines']} lines, {report['sent_requests']} sent, "
        f"{report['successful_requests']} ok, {sum(report['rejected_by_reason'].values())} rejected, "
        f"{report['network_failures']} network, {report['server_errors']} server; "
        f"p50={latency['p50']} p95={latency['p95']} ms (n={latency['sample_size']})"
    )
    if not report["reconciled"]:
        print("ERROR: report does not reconcile")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
