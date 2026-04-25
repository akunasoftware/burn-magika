[![License: MIT](https://img.shields.io/badge/license-MIT-0f766e?style=for-the-badge)](./LICENSE)
[![Crates.io](https://img.shields.io/crates/v/akuna-infer?style=for-the-badge)](https://crates.io/crates/akuna-infer)
[![Docs.rs](https://img.shields.io/docsrs/akuna-infer?style=for-the-badge)](https://docs.rs/akuna-infer)
[![Last Commit](https://img.shields.io/github/last-commit/akunasoftware/akuna-infer?style=for-the-badge)](https://github.com/akunasoftware/akuna-infer/commits/main)
[![CI](https://img.shields.io/github/actions/workflow/status/akunasoftware/akuna-infer/ci.yml?label=ci&style=for-the-badge)](https://github.com/akunasoftware/akuna-infer/actions/workflows/ci.yml)

# akuna-infer

Intelligent file-type inference using the [Google Magika ML model](https://github.com/google/magika) in native Rust using [Burn](https://github.com/tracel-ai/burn).

Runs natively in Rust, supports hardware acceleration, and has zero runtime dependencies.

## Features

- Magika-compatible preprocessing and output post-processing
- Generic `Session<B>` and `MagikaModel<B>` built on Burn's `Backend` abstraction
- Vendored `standard_v3_3` model from `src/vendor/assets/models/standard_v3_3/model.onnx`
- Tested parity against the Rust `magika` crate on local test fixtures

## Usage

```rust
use akuna_infer::Session;
use burn_wgpu::{Wgpu, WgpuDevice};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();
    let mut session = Session::<Wgpu>::new(&device)?;

    let detected = session.identify_file_sync(Path::new("target/fixtures/text.pdf"))?;
    assert_eq!(detected.info().label, "pdf");
    assert_eq!(detected.info().mime_type, "application/pdf");

    Ok(())
}
```

## Development

This project uses a Nix development shell.

If you use `nix-direnv`, it should activate automatically.

To enter it manually:

```sh
nix develop
```

Run all checks with:

```sh
./scripts/check.sh
```

Run tests only with:

```sh
cargo nextest run
```

## Benchmarks

Run all benchmarks, results are written to `target/criterion/`.

```sh
cargo bench
```

## Scripts

Refresh vendored upstream code and detection model:

- Google Magika model and type structs come directly from https://github.com/google/magika

```sh
./scripts/update_vendor.sh
```
