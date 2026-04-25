#!/usr/bin/env bash

set -euo pipefail

cargo fmt --all --check
cargo clippy --all-features --all-targets -- -D warnings
cargo check --all-features
cargo test
cargo deny check
