# Developer entrypoint. The underlying tools stay uv, Ruff, pytest, Locust and Docker;
# this file only gives them one vocabulary so local runs and CI cannot drift.
set dotenv-load

model_dir := env('MODEL_DIR', 'artifacts/sentiment')
service_image := env('SERVICE_IMAGE', 'sentiment-service:local')

# List the recipes.
default:
    @just --list

# Sync the locked dev environment (no model download).
setup:
    uv sync --frozen

# Run the test suite. Extra arguments go to pytest: `just test -k predict`.
test *args:
    uv run pytest {{args}}

# Rewrite formatting and apply safe lint fixes.
format:
    uv run ruff format
    uv run ruff check --fix

# Read-only quality gate; the same command CI runs.
check:
    uv run ruff format --check
    uv run ruff check
    uv run pytest

# Download the pinned bundle; setup and image builds may need the network too.
model:
    #!/bin/sh
    set -eu
    model_id=$(python3 -c "import json; print(json.load(open('model_manifest.json'))['model_name'])")
    revision=$(python3 -c "import json; print(json.load(open('model_manifest.json'))['source_revision'])")
    model_files=$(uv run python -c "from sentiment_service.artifacts import PACK_FILES; print(*PACK_FILES)")
    uv run hf download "$model_id" $model_files \
        --revision "$revision" --local-dir {{quote(model_dir)}}
    uv run python -m sentiment_service.artifacts \
        --model-id "$model_id" --revision "$revision" --destination {{quote(model_dir)}}

# Serve the packaged bundle on the host. MODEL_DIR comes from .env or the shell,
# falling back to the local bundle this repository builds.
serve:
    MODEL_DIR={{quote(model_dir)}} uv run python -m sentiment_service

# Replay the recorded log against a running service.
replay url="http://127.0.0.1:8000" report="results/replay_report.json":
    uv run python scripts/replay.py --base-url {{url}} --report {{report}}

# Headless load test: `just load 8 60s`.
load users="1" duration="60s" out="results/load-local":
    mkdir -p {{out}}
    uv run locust -f load/locustfile.py --headless -u {{users}} -r {{users}} -t {{duration}} \
        --host http://127.0.0.1:8000 --csv {{out}}/locust

# Build once, then use this tag for standalone and Compose checks.
build image=service_image *args:
    docker build {{args}} -t {{quote(image)}} .

# Stdlib only: no serving dependencies or uv required on the smoke runner.
smoke url="http://127.0.0.1:8000":
    python3 scripts/smoke_container.py --base-url {{quote(url)}} --timeout 300

# Wait for an actual scrape, not just a started container.
monitor url="http://127.0.0.1:9090":
    python3 scripts/check_prometheus.py --base-url {{quote(url)}} --timeout 300

# Rebuild by default; CI uses `just up --no-build` with its tested image.
up *args="--build":
    docker compose up -d {{args}}

down:
    docker compose down -v
