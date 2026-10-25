"""Wait for Prometheus to scrape the service; stdlib only, for local runs and CI."""

import argparse
import json
import time
import urllib.parse
import urllib.request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9090")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    query = urllib.parse.urlencode({"query": 'up{job="sentiment-service"}'})
    url = f"{args.base_url.rstrip('/')}/api/v1/query?{query}"
    deadline = time.monotonic() + args.timeout
    last = "no response"
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            with urllib.request.urlopen(url, timeout=min(5, remaining)) as response:
                payload = json.load(response)
            results = payload.get("data", {}).get("result", [])
            if payload.get("status") == "success" and any(
                row.get("metric", {}).get("job") == "sentiment-service"
                and row.get("value", [None, None])[1] == "1"
                for row in results
            ):
                print("prometheus target up")
                return 0
            last = f"target not up: {payload}"
        except (OSError, ValueError) as exc:
            last = str(exc)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(0.2, remaining))
    raise SystemExit(f"Prometheus never scraped sentiment-service: {last}")


if __name__ == "__main__":
    raise SystemExit(main())
