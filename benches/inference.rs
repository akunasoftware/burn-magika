use std::{
    fs,
    path::{Path, PathBuf},
};

use burn_cpu::{Cpu, CpuDevice};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use magika_burn::MagikaModel;

fn benchmark_inference(c: &mut Criterion) {
    let fixture_paths =
        collect_fixture_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures"));
    let fixture_bytes = fixture_paths
        .iter()
        .map(|path| fs::read(path).expect("read fixture"))
        .collect::<Vec<_>>();

    let classifier =
        MagikaModel::<Cpu<f32, i64>>::from_embedded(&CpuDevice).expect("build classifier");

    let first = fixture_bytes.first().expect("at least one fixture").clone();
    c.bench_function("detect_bytes_single", |b| {
        b.iter_batched(
            || first.clone(),
            |bytes| {
                classifier
                    .detect_bytes(bytes.as_slice())
                    .expect("single inference")
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("detect_batch_fixtures", |b| {
        b.iter(|| {
            let batch = fixture_bytes
                .iter()
                .map(|bytes| bytes.as_slice())
                .collect::<Vec<_>>();
            classifier.detect_batch(batch).expect("batch inference")
        })
    });
}

fn collect_fixture_files(root: &Path) -> Vec<PathBuf> {
    let mut files = fs::read_dir(root)
        .expect("read fixtures directory")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    files.sort();
    files
}

criterion_group!(benches, benchmark_inference);
criterion_main!(benches);
