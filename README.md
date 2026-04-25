# Magika-Burn

Magika file-type detection in Rust using [Burn](https://github.com/tracel-ai/burn).

At this time, this lib is **fully vibe-coded** and not extensively tested for production!

## Usage

```rust
use burn_cpu::{Cpu, CpuDevice};
use magika_burn::MagikaModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = MagikaModel::<Cpu<f32, i64>>::from_embedded(&CpuDevice::default())?;

    let detection = model.detect_bytes(b"function greet() { return 'hi'; }")?;
    println!("{} {:?}", detection.label, detection.mime_type);

    Ok(())
}
```

## Features

- Magika-compatible preprocessing and output post-processing
- Generic `MagikaModel<B>` built on Burn's native `Backend` abstraction
- Vendored `standard_v3_3` model from `src/vendor/assets/models/standard_v3_3/model.onnx`
- Parity coverage against the Rust `magika` crate on the local fixtures corpus

## Testing

Run all tests:

```bash
cargo test
```

The integration parity test lives in `tests/magika_parity.rs` and compares this crate against the Rust `magika` crate on every file in `fixtures/`.

## Benchmarks

Run Criterion benchmarks:

```bash
cargo bench --bench inference
```

Results are written to `target/criterion/`.

## Scripts

Refresh vendored upstream files and model asset:

```bash
./scripts/update_vendor.sh
```

## License

MIT OR Apache-2.0
