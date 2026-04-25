#!/usr/bin/env bash

set -euo pipefail

TEST_CORPUS_REPO_GIT="https://github.com/akunasoftware/test-corpus.git"
TEST_CORPUS_FIXTURES_PATH="content/fixtures"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
FIXTURES_DIR="${REPO_DIR}/target/fixtures"
TMP_DIR="$(mktemp -d)"
TEST_CORPUS_REPO_DIR="${TMP_DIR}/test-corpus"
trap 'rm -rf "${TMP_DIR}"' EXIT

if ! git clone --depth 1 "${TEST_CORPUS_REPO_GIT}" "${TEST_CORPUS_REPO_DIR}" >/dev/null 2>&1 \
  || [ ! -d "${TEST_CORPUS_REPO_DIR}/${TEST_CORPUS_FIXTURES_PATH}" ]; then
  printf 'failed to locate shared fixtures at %s or %s\n' \
    "${TEST_CORPUS_REPO_GIT}" "${TEST_CORPUS_FIXTURES_PATH}" >&2
  exit 1
fi

rm -rf "${FIXTURES_DIR}"
mkdir -p "${FIXTURES_DIR}"
cp -R "${TEST_CORPUS_REPO_DIR}/${TEST_CORPUS_FIXTURES_PATH}/." "${FIXTURES_DIR}/"

cargo fmt --all --check
cargo clippy --all-features --all-targets -- -D warnings
cargo check --all-features
cargo nextest run --all-features --no-capture -j 1
cargo test --doc --all-features
cargo deny check --allow=no-license-field
