use std::fs;

#[path = "mod.rs"]
mod tests;

use burn::tensor::backend::Backend;
use burn_cpu::{Cpu, CpuDevice};
use burn_magika::MagikaModel;
use burn_wgpu::{Wgpu, WgpuDevice};

#[test]
fn parity_against_rust_magika_on_repo_fixtures_cpu() {
    let classifier = MagikaModel::<Cpu>::from_embedded(&CpuDevice).expect("build cpu classifier");
    assert_parity_against_rust_magika(&classifier);
}

#[test]
fn parity_against_rust_magika_on_repo_fixtures_wgpu() {
    let classifier =
        MagikaModel::<Wgpu>::from_embedded(&WgpuDevice::default()).expect("build wgpu classifier");

    assert_parity_against_rust_magika(&classifier);
}

fn assert_parity_against_rust_magika<B>(classifier: &MagikaModel<B>)
where
    B: Backend<FloatElem = f32>,
{
    let fixture_files = tests::fixture_files();

    let mut rust_magika = magika::Session::new().expect("build rust magika session");
    let expected = fixture_files
        .iter()
        .map(|path| {
            let detection = rust_magika
                .identify_file_sync(path)
                .unwrap_or_else(|err| panic!("rust magika failed for {path:?}: {err}"));
            (
                path.clone(),
                detection.info().label.to_string(),
                detection.info().mime_type.to_string(),
            )
        })
        .collect::<Vec<_>>();

    let fixture_bytes = fixture_files
        .iter()
        .map(|path| fs::read(path).unwrap_or_else(|err| panic!("failed to read {path:?}: {err}")))
        .collect::<Vec<_>>();
    let batch_inputs = fixture_bytes
        .iter()
        .map(|bytes| bytes.as_slice())
        .collect::<Vec<_>>();
    let actual = classifier
        .detect_batch(batch_inputs)
        .expect("classify fixtures");

    let mismatches = expected
        .into_iter()
        .zip(actual)
        .filter_map(|((path, rust_label, rust_mime_type), ours)| {
            if ours.label == rust_label
                && ours.mime_type.as_deref() == Some(rust_mime_type.as_str())
            {
                return None;
            }

            Some((
                path,
                rust_label,
                rust_mime_type,
                ours.label,
                ours.mime_type.unwrap_or_default(),
            ))
        })
        .collect::<Vec<_>>();

    assert!(
        mismatches.is_empty(),
        "found {} parity mismatches across fixtures, first few: {:#?}",
        mismatches.len(),
        mismatches.into_iter().take(10).collect::<Vec<_>>()
    );
}
