use std::{
    fs,
    path::{Path, PathBuf},
};

use magika_burn::{LabelEntry, MagikaClassifierBuilder, ModelConfig};

/// Parity harness for validating this crate against the Rust-native `magika` crate.
///
/// This test is intentionally ignored by default because it requires:
/// - the `magika` crate to resolve in this dev/test environment
/// - a local Magika corpus directory
#[test]
#[ignore = "requires rust magika + local corpus"]
fn parity_against_rust_magika_on_corpus() {
    let corpus_dir = match std::env::var("MAGIKA_CORPUS_DIR") {
        Ok(v) => PathBuf::from(v),
        Err(_) => {
            eprintln!("Skipping: set MAGIKA_CORPUS_DIR to run parity test");
            return;
        }
    };

    if !corpus_dir.exists() {
        eprintln!("Skipping: MAGIKA_CORPUS_DIR does not exist: {corpus_dir:?}");
        return;
    }

    let labels = discover_labels(&corpus_dir);
    if labels.is_empty() {
        eprintln!("Skipping: no label subdirectories found in corpus");
        return;
    }

    let classifier = MagikaClassifierBuilder::new()
        .config(ModelConfig {
            labels,
            metadata: Default::default(),
        })
        .build()
        .expect("build classifier");

    let rust_magika = magika::Magika::new();

    let max_files = std::env::var("MAGIKA_CORPUS_MAX_FILES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);

    let files = collect_sample_files(&corpus_dir, max_files);
    if files.is_empty() {
        eprintln!("Skipping: no files found in corpus");
        return;
    }

    let mut mismatches = Vec::new();
    for file in files {
        let rust_label = match rust_magika.identify_file_sync(&file) {
            Ok(v) => v.info().label.clone(),
            Err(err) => {
                eprintln!("Skipping file with rust magika error: {file:?}: {err}");
                continue;
            }
        };

        let bytes = fs::read(&file).expect("read test file");
        let ours = classifier.detect_bytes(&bytes).expect("classify bytes");

        if ours.label != rust_label {
            mismatches.push((file, rust_label, ours.label));
        }
    }

    assert!(
        mismatches.is_empty(),
        "found {} label mismatches, first few: {:#?}",
        mismatches.len(),
        mismatches.into_iter().take(10).collect::<Vec<_>>()
    );
}

fn discover_labels(root: &Path) -> Vec<LabelEntry> {
    let mut labels = Vec::new();
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(label) = path.file_name().and_then(|n| n.to_str()) {
                    labels.push(LabelEntry {
                        label: label.to_string(),
                        mime_type: None,
                    });
                }
            }
        }
    }
    labels.sort_by(|a, b| a.label.cmp(&b.label));
    labels
}

fn collect_sample_files(root: &Path, max_files: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        if out.len() >= max_files {
            break;
        }

        let read = match fs::read_dir(&dir) {
            Ok(v) => v,
            Err(_) => continue,
        };

        for entry in read.flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.is_file() {
                out.push(path);
                if out.len() >= max_files {
                    break;
                }
            }
        }
    }

    out
}
