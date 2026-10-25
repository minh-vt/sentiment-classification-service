"""Runtime configuration: required values fail loudly instead of defaulting wrong."""

from dataclasses import fields
from pathlib import Path

import pytest

from sentiment_service.settings import Settings


def test_missing_model_dir_is_refused():
    with pytest.raises(ValueError, match="MODEL_DIR is required"):
        Settings.from_env({})


def test_defaults_match_the_documented_image_contract():
    settings = Settings.from_env({"MODEL_DIR": "/bundle"})
    assert settings.model_dir == Path("/bundle")
    assert (settings.host, settings.port) == ("0.0.0.0", 8000)
    assert (settings.max_batch_size, settings.max_positions) == (32, 512)
    assert settings.max_concurrency == 4


def test_environment_overrides_are_read():
    settings = Settings.from_env(
        {
            "MODEL_DIR": "/bundle",
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "MAX_BATCH_SIZE": "2",
            "MAX_POSITIONS": "128",
            "MAX_CONCURRENCY": "4",
        }
    )
    assert (settings.host, settings.port) == ("127.0.0.1", 9000)
    assert (settings.max_batch_size, settings.max_positions) == (2, 128)
    assert settings.max_concurrency == 4


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"MAX_CONCURRENCY": "0"}, "must be positive"),
        ({"MAX_BATCH_SIZE": "0"}, "must be positive"),
        ({"MAX_POSITIONS": "-1"}, "must be positive"),
        ({"PORT": "0"}, "PORT must be between"),
        ({"PORT": "70000"}, "PORT must be between"),
    ],
)
def test_unusable_values_are_refused(env, message):
    with pytest.raises(ValueError, match=message):
        Settings.from_env({"MODEL_DIR": "/bundle", **env})


def read_env_example():
    """Parse .env.example the way a developer would read it: KEY=VALUE lines."""
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    pairs = (line.strip().split("=", 1) for line in text.splitlines() if "=" in line)
    return {key: value for key, value in pairs if not key.startswith("#")}


def test_env_example_documents_every_setting_the_service_reads():
    documented = set(read_env_example())
    required = {field.name.upper() for field in fields(Settings)}
    assert required <= documented, f"missing from .env.example: {sorted(required - documented)}"


def test_env_example_alone_is_enough_to_configure_the_service():
    example = read_env_example()
    settings = Settings.from_env(example)
    assert settings.model_dir == Path(example["MODEL_DIR"])
    assert settings.port == int(example["PORT"])
    assert settings.max_concurrency == int(example["MAX_CONCURRENCY"])
