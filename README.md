# sentiment-service

CPU sentiment inference with a pinned DistilBERT bundle, an HTTP replay tool and Prometheus metrics.

## Run

Requires `uv`, `just`, Python 3 for the HTTP tools, and Docker with Compose for containers.
`uv` manages the application's Python 3.14 environment.

```sh
just setup
just check
cp .env.example .env
just model
just serve
```

For containers, use this instead of `just serve`:

```sh
just up          # build and start the service plus Prometheus
```

In another terminal:

```sh
just smoke       # readiness, prediction, rejection and metric names
just monitor     # wait for Prometheus to scrape the service
just down        # stop the Compose stack
```

Open `bruno/sentiment-service/` in Bruno and select `local` for interactive requests.
The default addresses are the service at `http://127.0.0.1:8000` and Prometheus at
`http://127.0.0.1:9090`. Pass a URL to `just smoke` when using another port.

## Flow and code

```text
Bruno / replay / Locust -> Uvicorn -> FastAPI -> tokenizer + CPU model
                                       ^
                                       | /metrics
                                   Prometheus
```

- `settings.py`: environment configuration.
- `artifacts.py`: create the manifest and verify the weights' SHA-256.
- `inference.py`: load local files, check token positions and predict.
- `api.py`: validation, model lifecycle, bounded inference and metrics.
- `__main__.py`: assemble the app and start Uvicorn.
- `scripts/`: external HTTP replay, container smoke and Prometheus checks.
- `load/locustfile.py`: short synthetic requests, one in flight per Locust user.

The model loads in a background thread. Predictions run in a threadpool, with a semaphore limiting
simultaneous predictions. A failed load leaves the process alive but unready. Startup never downloads
weights; the image contains the bundle and loads with `local_files_only=True`.

## API and configuration

`POST /predict` accepts `{"text":"a good film"}` or `{"text":["good","bad"]}` and returns
`{"predictions":[{"label":"POSITIVE","score":0.99}]}` in input order.

Invalid input returns HTTP 400 with `{"error":{"reason":"...","detail":"...","index":0}}`;
`index` is included only for item-level errors. Reasons are `malformed_json`, `not_an_object`,
`missing_text`, `wrong_type`, `empty_batch`, `blank_text`, `batch_too_large`, `too_many_tokens`.
An invalid batch is rejected as a whole. Token limits include special tokens; text is never truncated.

`/health` reports process liveness. `/ready` returns 200 and `model_version` only after loading,
otherwise 503. `/metrics` exposes request counts, a latency histogram and rejections by reason.

| Setting | Default / meaning |
|---|---|
| `MODEL_DIR` | Required by the app; host recipes use `artifacts/sentiment`, image uses `/app/models/sentiment` |
| `HOST`, `PORT` | Host process binding: `0.0.0.0`, `8000` |
| `MAX_BATCH_SIZE` | `32` sentences |
| `MAX_POSITIONS` | `512`; a value above the model's capacity fails loading |
| `MAX_CONCURRENCY` | `4` simultaneous predictions, not the number of Torch threads |
| `HOST_PORT` | Compose's published port, `8000`; internal port stays 8000 |
| `SERVICE_IMAGE` | Build/Compose tag, `sentiment-service:local` |

Defaults and environment keys live in `Settings` (`field.name.upper()`). The Dockerfile sets only
`MODEL_DIR` and Python flags; override the rest with `docker run -e`. `just serve` forwards `.env` to
the host process; the app reads process environment only. Model repository and revision come from
`model_manifest.json` at build time. The image writes its own manifest, with a fresh packaging
timestamp, after download, and runs as UID 10001.

## Tests and CI

`just check` runs Ruff format checking, lint and pytest. `just test -k predict` selects tests.
API tests substitute the model: they verify HTTP behaviour, not real tokenizer or prediction quality.
Replay and smoke exercise the real model; smoke is a basic availability check, not an accuracy gate.

CI runs the same quality gate, builds one image, tests it standalone, then uses it in Compose with
`--no-build`. Smoke and Prometheus checks use stdlib Python; the integration runner does not install
the ML dependencies. Images remain on the runner: no registry publication or deployment is configured.

## Replay and measurements

Place the supplied, unchanged `replay_log.jsonl` at the repository root. It is not distributed here;
its SHA-256 must be `f8bef24d23d57aad8fd44891da447174e1b1b869005a26abdf48f729eed5c589`.

```sh
just replay
```

The script generates `results/replay_report.json`. It sorts by timestamp, preserves ties and duplicate
occurrences, and does not reproduce arrival gaps. Parser errors, HTTP rejections, network failures
and server errors are reported separately. Successful-request p50/p95 use nearest rank.

Recorded snapshot: image `0894d5a10e2a`, **1.56 GB** reported by Podman. Load generator and service shared
an Intel i5-1240P host, 16 logical CPUs and 15.3 GiB RAM. Full configuration is in each `machine.txt`.

| Replay field | Recorded result |
|---|---|
| Total / readable / unreadable lines | 4,575 / 4,565 / 10 |
| Sent / successful requests | 4,565 / 4,385 |
| Rejections | blank_text 120; too_many_tokens 42; wrong_type 18 |
| Duplicate IDs / extra occurrences | 80 / 80 |
| Network failures / server errors | 0 / 0 |
| Successful-request p50 / p95 | 14.08 / 19.76 ms |

| Connections | Requests | Failures | Throughput | p50 | p95 |
|---|---|---|---|---|---|
| 1 | 4,303 | 0 | 72.87 req/s | 14 ms | 16 ms |
| 8 | 5,672 | 0 | 96.03 req/s | 83 ms | 89 ms |

`results/load-1/` and `results/load-8/` contain unedited Locust CSV output and recorded machine metadata.
These are snapshots from one machine, shared by the generator and the service. Latency varies between runs.
To measure again without mixing new CSVs with old metadata:

```sh
just load 1 60s results/load-local/1
just load 8 60s results/load-local/8
just replay http://127.0.0.1:8000 results/load-local/replay_report.json
```

Locust generates the CSVs, not `machine.txt`. Record the machine, image and runtime configuration
for each new measurement; the existing metadata describes only the recorded runs.

## Remaining work

- The previous GitHub run passed quality and image smoke but failed the Compose polling step.
  The revised workflow still needs a successful native Docker Compose run.
- Local `docker` wraps Podman. Sandbox networking requires host networking, so local scrape checks
  are not proof of Compose network wiring. Podman also needs Docker image format to retain HEALTHCHECK.
- The required uncut 2–3 minute video has not been recorded. Bonus features are not claimed.

With another week: add a small real-model regression check, repeat measurements on an isolated host,
and address request-size/queue limits if the deployment workload requires them.
