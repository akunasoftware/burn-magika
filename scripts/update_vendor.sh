#!/usr/bin/env bash

# Refresh vendored upstream Magika sources and model assets.
set -euo pipefail

# Upstream source and the exact model subtree we vendor.
MAGIKA_REPO_GIT="https://github.com/google/magika.git"
MAGIKA_REF="main"
MAGIKA_MODEL_NAME="standard_v3_3"
# Paths copied from the upstream repo into src/vendor/.
MAGIKA_VENDOR_PATHS=(
  "rust/lib/src/content.rs"
  "rust/lib/src/model.rs"
  "assets/models/${MAGIKA_MODEL_NAME}"
)

# Resolve local paths and make sure the temp clone is always removed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
REPO_DIR="${TMP_DIR}/magika"
VENDOR_DIR="${SCRIPT_DIR}/../src/vendor"
trap 'rm -rf "${TMP_DIR}"' EXIT

# Shallow clone is enough because we only read tracked files.
git clone --depth 1 --branch "${MAGIKA_REF}" --single-branch \
  "${MAGIKA_REPO_GIT}" "${REPO_DIR}" >/dev/null 2>&1

# Keep committed mod.rs, replace everything generated underneath it.
find "${VENDOR_DIR}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# Recreate each vendored upstream path under src/vendor/.
for relative_path in "${MAGIKA_VENDOR_PATHS[@]}"
do
  mkdir -p "$(dirname "${VENDOR_DIR}/${relative_path}")"
  cp -R "${REPO_DIR}/${relative_path}" "${VENDOR_DIR}/${relative_path}"
done

# Emit a short summary so updates are traceable in logs.
printf 'Updated vendor assets from %s at %s using model %s\n' "${MAGIKA_REPO_GIT}" "${MAGIKA_REF}" "${MAGIKA_MODEL_NAME}"
