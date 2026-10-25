"""Bundle manifests: a manifest must pin its source and match the bytes it describes."""

import hashlib
import json
from pathlib import Path

import pytest

from sentiment_service.artifacts import BundleError, verify_bundle, write_manifest

REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"


def write_bundle(directory, weights=b"fake weights", **overrides):
    (directory / "model.safetensors").write_bytes(weights)
    for name in ("config.json", "tokenizer_config.json", "vocab.txt"):
        (directory / name).write_text("{}", encoding="utf-8")
    manifest = {
        "model_name": "example",
        "source_url": "https://example.invalid/model",
        "source_revision": REVISION,
        "weights_file": "model.safetensors",
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "packaged_at": "2026-10-24T00:00:00Z",
        "version": "1.0.0",
    }
    manifest.update(overrides)
    (directory / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_matching_digest_is_accepted(tmp_path):
    manifest = write_bundle(tmp_path)
    assert verify_bundle(tmp_path)["version"] == manifest["version"]


def test_changed_weights_are_refused(tmp_path):
    write_bundle(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(BundleError, match="digest mismatch"):
        verify_bundle(tmp_path)


def test_missing_manifest_field_is_refused(tmp_path):
    write_bundle(tmp_path, version="")
    with pytest.raises(BundleError, match="version"):
        verify_bundle(tmp_path)


def test_manifest_without_a_pinned_source_revision_is_refused(tmp_path):
    write_bundle(tmp_path, source_revision="main")
    with pytest.raises(BundleError, match="must pin source_revision"):
        verify_bundle(tmp_path)


def test_missing_weights_file_is_refused(tmp_path):
    write_bundle(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(BundleError, match="weights file not found"):
        verify_bundle(tmp_path)


def test_manifest_is_required(tmp_path):
    with pytest.raises(BundleError, match="manifest not found"):
        verify_bundle(Path(tmp_path))


def test_written_manifest_describes_the_files_on_disk(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "vocab.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fake weights")

    manifest = write_manifest(tmp_path, REVISION)

    assert manifest["source_revision"] == REVISION
    assert manifest["version"] == "1.0.0+714eb0f"
    assert manifest["weights_sha256"] == hashlib.sha256(b"fake weights").hexdigest()
    assert verify_bundle(tmp_path)["weights_sha256"] == manifest["weights_sha256"]


@pytest.mark.parametrize("revision", ["", "main", "v1.0.0", "714eb0f", REVISION.upper()])
def test_writing_a_manifest_refuses_anything_but_a_full_commit(tmp_path, revision):
    with pytest.raises(BundleError, match="40-character commit"):
        write_manifest(tmp_path, revision)


def test_writing_a_manifest_reports_missing_files(tmp_path):
    with pytest.raises(BundleError, match="missing bundle files"):
        write_manifest(tmp_path, REVISION)
