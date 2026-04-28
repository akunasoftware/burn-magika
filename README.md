# Burn-Magika

Google Magika file-type detection in native Rust using [Burn](https://github.com/tracel-ai/burn).

Runs natively in rust, supports hardware acceleration, with zero runtime dependencies (no onnxruntime).

## Usage

```rust
use burn_magika::Session;
use burn_wgpu::{Wgpu, WgpuDevice};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::default();
    let mut session = Session::<Wgpu>::new(&device)?;

    let detected = session.identify_file_sync(Path::new("fixtures/text.pdf"))?;
    assert_eq!(detected.info().label, "pdf");
    assert_eq!(detected.info().mime_type, "application/pdf");

    Ok(())
}
```

## Features

- Magika-compatible preprocessing and output post-processing
- Generic `Session<B>` and `MagikaModel<B>` built on Burn's `Backend` abstraction
- Vendored `standard_v3_3` model from `src/vendor/assets/models/standard_v3_3/model.onnx`
- Tested parity against the Rust `magika` crate on local test fixtures

## Scripts

Refresh vendored upstream code, detection model, and test fixtures:

- Google Magika model, and type structs come directly from https://github.com/google/magika
- Test fixtures come via script from https://github.com/akunasoftware/test-corpus.

```bash
./scripts/update_vendor.sh
```

## Testing

Run all tests:

```bash
cargo test
```

## Benchmarks

Run all benchmarks:

```bash
cargo bench
```

Results are written to `target/criterion/`.
