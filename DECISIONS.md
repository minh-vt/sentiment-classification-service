# Decisions

## AI assistance

Code, tests and documentation were written with the pi coding agent. The candidate directed the
work and is responsible for the result. Measurements come from actual runs; retained outputs are
in `results/`.

## 2026-10-23 — Preserve the supplied traffic

Replay uses the original log, sorted stably by timestamp. Duplicate occurrences are sent. Arrival
gaps are skipped, so the run reproduces order rather than pacing. Malformed records are reported
separately from requests the service rejects.

## 2026-10-24 — Local bundle and explicit request policy

Weights are baked into the image, with a manifest and startup SHA-256 verification. This avoids
runtime downloads at the cost of a larger image and a rebuild when model identity changes.
`/health` stays available while loading; `/ready` reports the model version after verification.

The API accepts scalar or batch text. Invalid batches fail atomically; a request is either accepted
whole or rejected whole. Inference runs outside the event loop, with a configurable concurrency
limit. The default is a local policy; measure again before assuming it suits other hardware.

### Rejected attempt: editable install in the image

The first container exited with:

```text
/app/.venv/bin/python: No module named sentiment_service
```

The editable install pointed at source that was not copied into the runtime stage. The builder now
uses `--no-editable` and checks that the installed package imports before producing the runtime image.

### Rejected attempt: letting configuration exceed model capacity

With `MAX_POSITIONS=1024`, a 513-position input reached a model supporting 512 positions and returned
HTTP 500. The configured limit is now checked against the loaded model: an excessive value prevents
readiness. Overlong requests at a valid limit return 400 instead of reaching model execution.

## 2026-10-25 — Shared commands and smaller packaging code

`hf download` handles network/cache behaviour; Python only writes and verifies the manifest.
Host and Docker preparation read model identity from the same manifest and use the same runtime
file inventory. Host model preparation and serving share MODEL_DIR. The container has its own path.

`just` is the local/CI entrypoint. Bruno is for interactive requests; stdlib smoke is for CI, avoiding
a Node toolchain or ML installation on the HTTP-check runner. CI builds one image and reuses it for
standalone and Compose checks. It does not publish images.

The original Compose poll piped an empty/invalid response into `json.load`; `bash -e` exited before
retrying. A deadline-based HTTP probe now retries startup failures and requires an actual successful
scrape. Model preparation stays inside its recipe: global backticks had made unrelated tasks call uv.

## 2026-10-25 — One source for runtime configuration

`Settings` previously repeated every environment key and default in `from_env`, so a rename or a
value change had two places to drift. The dataclass fields are now the single source: the key is
`field.name.upper()`, the default is the field default, and a missing default means required.
Error messages no longer embed the `MAX_POSITIONS` key, so renaming the setting cannot leave a
message that names the old key. The Dockerfile and Compose no longer copy runtime policy defaults;
the image sets only `MODEL_DIR`, and overrides go through `docker run -e`. The test that exercised
`just` itself through a PATH shim was removed: it tested developer tooling, not the probe.

## 2026-10-25 — Handoff cleanup

Removed source-string packaging tests, a duplicated test-client lifecycle, experimental sweeps and
assembled verification reports from the handoff. Results now hold the replay report, required load
outputs and their machine metadata. Historical optimization claims were dropped with the artifacts
that backed them. Runtime prediction behaviour is unchanged.

Remaining gaps are stated in README: the revised Compose CI run and the required video are pending.
The API unit tests use a fake model; neither their count nor basic smoke proves model accuracy.
