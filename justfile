# Developer entrypoint. The underlying tools stay uv, Ruff, pytest, Locust and Docker;
# this file only gives them one vocabulary so local runs and CI cannot drift.
set dotenv-load

model_id := "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model_files := "config.json model.safetensors tokenizer_config.json vocab.txt"
# The delivered manifest is the single source of the pinned revision.
revision := `uv run python -c "import json; print(json.load(open('model_manifest.json'))['source_revision'])"`

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

# Download the bundle pinned by model_manifest.json. The only networked step.
model:
    uv run hf download {{model_id}} {{model_files}} --revision {{revision}} \
        --local-dir artifacts/sentiment
    uv run python -m sentiment_service.artifacts \
        --model-id {{model_id}} --revision {{revision}} --destination artifacts/sentiment

# Serve the packaged bundle on the host. MODEL_DIR comes from .env or the shell,
# falling back to the local bundle this repository builds.
serve:
    MODEL_DIR={{ env_var_or_default('MODEL_DIR', 'artifacts/sentiment') }} uv run python -m sentiment_service

# Replay the recorded log against a running service.
replay url="http://127.0.0.1:8000" report="results/replay_report.json":
    uv run python scripts/replay.py --base-url {{url}} --report {{report}}

# Headless load test: `just load 8 60s`.
load users="1" duration="60s" out="results/load-local":
    mkdir -p {{out}}
    uv run locust -f load/locustfile.py --headless -u {{users}} -r {{users}} -t {{duration}} \
        --host http://127.0.0.1:8000 --csv {{out}}/locust

# Build the image with the pinned revision from the Dockerfile.
build:
    docker build -t sentiment-service:local .

# Check a running container over HTTP.
smoke url="http://127.0.0.1:8000":
    uv run python scripts/smoke_container.py --base-url {{url}} --timeout 300

# Service plus Prometheus.
up:
    docker compose up -d --build

down:
    docker compose down -v
