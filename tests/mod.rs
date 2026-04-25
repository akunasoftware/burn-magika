use std::{
    fs,
    path::{Path, PathBuf},
};

pub fn fixture_files() -> Vec<PathBuf> {
    let root = fixture_dir();
    let mut files = fs::read_dir(&root)
        .unwrap_or_else(|err| {
            panic!("failed to read fixtures directory {root:?}: {err}")
        })
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();

    files.sort();
    assert!(
        !files.is_empty(),
        "expected at least one fixture file in the shared test corpus at {}",
        root.display()
    );

    files
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target/fixtures")
}
