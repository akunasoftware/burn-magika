# magika-burn

Portable, runtime-agnostic scaffolding for Magika inference in Rust.

## Workspace Layout

- `crates/magika-burn`: main library/API crate
- `crates/magika-burn-tests`: internal dev/test crate for parity harnesses

## Status

This repository currently provides:

- A stable Rust API surface:
  - `detect_bytes(&[u8]) -> Detection`
  - `detect_path(Path) -> Detection`
  - `detect_batch(Vec<&[u8]>) -> Vec<Detection>`
- Deterministic preprocessing and batch plumbing
- Single-load Magika classifier lifecycle with reusable runtime object

It intentionally does **not** include ONNX Runtime.

## Corpus parity testing against Rust `magika` (dev/test only)

A parity harness lives at `crates/magika-burn-tests/tests/magika_parity.rs` and is intentionally `#[ignore]` by default.
It compares this crate against the Rust-native `magika` crate without introducing Python in the workflow.

### Requirements

- Rust toolchain
- Local corpus directory organized by label folders

A practical source for samples is the upstream `google/magika` repository test data.

### Run

```bash
MAGIKA_CORPUS_DIR=/path/to/corpus \
MAGIKA_CORPUS_MAX_FILES=100 \
cargo test -p magika-burn-tests --test magika_parity -- --ignored
```

## Next implementation steps

1. Add a Burn-backed runtime implementing `InferenceRuntime`.
2. Replace preprocessing constants with official Magika config-derived values.
3. Validate outputs against official Magika reference behavior.
4. Add a concurrent batching worker for throughput-oriented serving.
