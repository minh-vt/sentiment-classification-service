"""Artifact manifest: the record that says which bytes a bundle contains.

Downloading is `hf download`'s job; this module writes and verifies the manifest
for files that step already placed on disk.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_FIELDS = (
    "model_name",
    "source_url",
    "source_revision",
    "weights_file",
    "weights_sha256",
    "packaged_at",
    "version",
)

PACK_FILES = ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt")

DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# A branch or tag can move; only a full commit identifies the weights.
PINNED_REVISION = re.compile(r"\A[0-9a-f]{40}\Z")


class BundleError(ValueError):
    """The local model bundle is missing or does not match its manifest."""


def sha256_file(path, chunk_size=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(model_dir):
    path = Path(model_dir) / "model_manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BundleError(f"manifest not found at {path}") from None
    except json.JSONDecodeError as exc:
        raise BundleError(f"manifest at {path} is not valid JSON: {exc}") from None
    missing = [field for field in REQUIRED_FIELDS if not str(manifest.get(field, "")).strip()]
    if missing:
        raise BundleError(f"manifest at {path} is missing: {', '.join(missing)}")
    if not PINNED_REVISION.match(manifest["source_revision"]):
        raise BundleError(
            f"manifest at {path} must pin source_revision to a full commit, "
            f"got {manifest['source_revision']!r}"
        )
    return manifest


def verify_bundle(model_dir):
    """Return the manifest only when the packaged weights match its digest."""
    model_dir = Path(model_dir)
    manifest = read_manifest(model_dir)
    weights = model_dir / manifest["weights_file"]
    if not weights.is_file():
        raise BundleError(f"weights file not found at {weights}")
    actual = sha256_file(weights)
    if actual != manifest["weights_sha256"]:
        raise BundleError(
            f"weights digest mismatch for {weights.name}: "
            f"manifest {manifest['weights_sha256']}, actual {actual}"
        )
    return manifest


def write_manifest(destination, revision, model_id=DEFAULT_MODEL_ID, version=None):
    """Describe files already present in `destination`; never downloads."""
    if not PINNED_REVISION.match(revision):
        raise BundleError(f"revision must be a full 40-character commit, got {revision!r}")
    destination = Path(destination)
    missing = [name for name in PACK_FILES if not (destination / name).is_file()]
    if missing:
        raise BundleError(f"missing bundle files in {destination}: {', '.join(missing)}")

    manifest = {
        "model_name": model_id,
        "source_url": f"https://huggingface.co/{model_id}",
        "source_revision": revision,
        "weights_file": "model.safetensors",
        "weights_sha256": sha256_file(destination / "model.safetensors"),
        "packaged_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "version": version or f"1.0.0+{revision[:7]}",
    }
    (destination / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="sentiment-pack",
        description="Write model_manifest.json for a bundle that `hf download` placed.",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--version", default=None)
    args = parser.parse_args(argv)
    try:
        manifest = write_manifest(args.destination, args.revision, args.model_id, args.version)
    except BundleError as exc:
        raise SystemExit(f"error: {exc}") from None
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
