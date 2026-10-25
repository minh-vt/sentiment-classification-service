"""The image and the delivered manifest must describe the same model and policy."""

import json
import re
from pathlib import Path

from sentiment_service.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def dockerfile_text():
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def dockerfile_revision():
    match = re.search(r"^ARG MODEL_REVISION=(\S+)$", dockerfile_text(), re.MULTILINE)
    assert match, "Dockerfile must pin MODEL_REVISION with a default"
    return match.group(1)


def dockerfile_env():
    """Merge every ENV block; the builder stage has one too."""
    text = dockerfile_text().replace("\\\n", " ")
    env = {}
    for block in re.findall(r"^ENV (.*)$", text, re.MULTILINE):
        env.update(item.split("=", 1) for item in block.split() if "=" in item)
    return env


def test_image_pins_the_revision_the_manifest_declares():
    manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    assert dockerfile_revision() == manifest["source_revision"]


def test_compose_does_not_declare_a_second_revision():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert not re.search(r"\b[0-9a-f]{40}\b", compose)


def test_dockerfile_runtime_defaults_match_the_code_defaults():
    defaults = Settings(model_dir=Path("/unused"))
    env = dockerfile_env()
    assert int(env["MAX_BATCH_SIZE"]) == defaults.max_batch_size
    assert int(env["MAX_POSITIONS"]) == defaults.max_positions
    assert int(env["MAX_CONCURRENCY"]) == defaults.max_concurrency
    assert int(env["PORT"]) == defaults.port
    assert env["HOST"] == defaults.host
