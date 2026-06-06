#!/usr/bin/env bash
set -euo pipefail

RELEASE_BASE_URL="${RELEASE_BASE_URL:-https://github.com/ZhuochengShang/QARV/releases/download/gsqa2-qwen-embeddings}"
ARCHIVE_NAME="${ARCHIVE_NAME:-gsqa_vector_raster_chroma_embeddings.tar.gz}"
CHECKSUM_NAME="${CHECKSUM_NAME:-gsqa_vector_raster_chroma_embeddings.tar.gz.sha256}"
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEST_DIR="${DEST_DIR:-$BASE_DIR/shared_embeddings}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$DEST_DIR/_downloads}"

mkdir -p "$DOWNLOAD_DIR" "$DEST_DIR"

download_if_missing() {
  local name="$1"
  local url="$RELEASE_BASE_URL/$name"
  local out="$DOWNLOAD_DIR/$name"
  if [[ -s "$out" ]]; then
    echo "[setup] using existing $out"
  else
    echo "[setup] downloading $url"
    curl -L --fail --retry 3 --output "$out" "$url"
  fi
}

download_if_missing "$ARCHIVE_NAME"
download_if_missing "$CHECKSUM_NAME"

echo "[setup] verifying embedding archive"
(
  cd "$DOWNLOAD_DIR"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "$CHECKSUM_NAME"
  else
    shasum -a 256 -c "$CHECKSUM_NAME"
  fi
)

echo "[setup] extracting embeddings"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
tar -xzf "$DOWNLOAD_DIR/$ARCHIVE_NAME" -C "$TMP_DIR"

rm -rf "$DEST_DIR/vector_entities_chroma"
rm -rf "$DEST_DIR/dem_patches_gdal_chroma"
mv "$TMP_DIR/vector_entities_chroma" "$DEST_DIR/"
mv "$TMP_DIR/dem_patches_gdal_chroma" "$DEST_DIR/"

cat <<EOF
[setup] done.

Shared embeddings are ready at:
  $DEST_DIR/vector_entities_chroma
  $DEST_DIR/dem_patches_gdal_chroma

Download Qwen3-32B-AWQ separately, start a local API-compatible endpoint, and
pass the local model path with MODEL_PATH when running experiments.
EOF
