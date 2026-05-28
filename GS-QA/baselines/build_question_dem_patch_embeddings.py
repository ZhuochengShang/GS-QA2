#!/usr/bin/env python3
"""Build DEM GeoTIFF patch embeddings only for question-relevant windows.

Input is the JSONL produced by find_question_dem_patches.py. Each row points to
one DEM GeoTIFF path and one patch window. This avoids embedding every 256x256
window in every DEM tile when the benchmark touches only a subset of patches.

Embedded text includes file-level GDAL metadata from:
    gdalinfo -json -stats dem.tif
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import chromadb
import rasterio
from rasterio.transform import array_bounds
from rasterio.windows import Window
from tqdm import tqdm

from build_dem_patch_embeddings import (
    build_patch_text,
    compact_gdalinfo_metadata,
    default_proj_data,
    finite_values,
    make_embedding_function,
    metadata_for_chroma,
    stats_for_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed question-selected DEM GeoTIFF patch windows.")
    parser.add_argument(
        "--patches",
        required=True,
        help="JSONL file produced by find_question_dem_patches.py.",
    )
    parser.add_argument("--persist-directory", required=True)
    parser.add_argument("--collection-name", default="dem_patches")
    parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "sentence-transformers"],
        default="sentence-transformers",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-valid-fraction", type=float, default=0.25)
    parser.add_argument("--gdalinfo-cmd", default="gdalinfo")
    parser.add_argument(
        "--skip-gdalinfo-stats",
        action="store_true",
        help="Do not pass -stats to gdalinfo. Faster, but band min/max may be missing.",
    )
    parser.add_argument(
        "--proj-data",
        help="Optional PROJ data directory containing proj.db. Defaults to environment/Homebrew paths.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip patch IDs already present in the Chroma collection.",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        help="Optional smoke-test limit.",
    )
    return parser.parse_args()


def load_patch_rows(path: Path) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_id[row["patch_id"]] = row
    return list(rows_by_id.values())


def existing_ids(collection: Any) -> set[str]:
    return set(collection.get(include=[])["ids"])


def add_batch(
    collection: Any,
    embeddings: Any,
    ids: list[str],
    docs: list[str],
    metadatas: list[dict[str, str | int | float | bool]],
) -> None:
    if not docs:
        return
    vectors = embeddings.embed_documents(docs)
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=vectors)


def index_selected_patches(args: argparse.Namespace) -> None:
    patch_rows = load_patch_rows(Path(args.patches))
    if args.max_patches is not None:
        patch_rows = patch_rows[: args.max_patches]

    rows_by_dem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in patch_rows:
        rows_by_dem[row["dem"]].append(row)

    embeddings = make_embedding_function(args)
    client = chromadb.PersistentClient(path=args.persist_directory)
    collection = client.get_or_create_collection(name=args.collection_name)
    already_done = existing_ids(collection) if args.skip_existing else set()

    docs: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []
    ids: list[str] = []
    indexed_count = 0
    skipped_existing = 0

    for dem_text, rows in tqdm(sorted(rows_by_dem.items()), desc="DEM files"):
        dem_path = Path(dem_text)

        with rasterio.open(dem_path) as dem_src:
            raster_file_metadata = compact_gdalinfo_metadata(
                dem_path,
                gdalinfo_cmd=args.gdalinfo_cmd,
                proj_data=args.proj_data or default_proj_data(),
                compute_stats=not args.skip_gdalinfo_stats,
            )

            for row in rows:
                patch_id = row["patch_id"]
                if patch_id in already_done:
                    skipped_existing += 1
                    continue

                window = Window(
                    int(row["col_start"]),
                    int(row["row_start"]),
                    int(row["width"]),
                    int(row["height"]),
                )
                dem = dem_src.read(1, window=window, masked=True)
                elev_values = finite_values(dem)
                valid_pixels = int(elev_values.size)
                total_pixels = int(window.width * window.height)
                valid_fraction = valid_pixels / total_pixels if total_pixels else 0.0
                if valid_fraction < args.min_valid_fraction:
                    continue

                win_transform = dem_src.window_transform(window)
                minx, miny, maxx, maxy = array_bounds(
                    int(window.height),
                    int(window.width),
                    win_transform,
                )

                metadata: dict[str, Any] = {
                    "source_file": str(dem_path),
                    "crs": dem_src.crs.to_string() if dem_src.crs else "",
                    "row_start": int(window.row_off),
                    "row_end": int(window.row_off + window.height),
                    "col_start": int(window.col_off),
                    "col_end": int(window.col_off + window.width),
                    "patch_width": int(window.width),
                    "patch_height": int(window.height),
                    "valid_fraction": float(valid_fraction),
                    "bbox_minx": float(minx),
                    "bbox_miny": float(miny),
                    "bbox_maxx": float(maxx),
                    "bbox_maxy": float(maxy),
                    "question_ids": row.get("question_ids", []),
                    "source_question_files": row.get("source_files", []),
                }
                metadata.update(raster_file_metadata)
                metadata.update(stats_for_array(dem, "elevation"))
                if (
                    metadata["elevation_min"] is not None
                    and metadata["elevation_max"] is not None
                ):
                    metadata["elevation_range"] = (
                        metadata["elevation_max"] - metadata["elevation_min"]
                    )

                text = build_patch_text(metadata)
                ids.append(patch_id)
                docs.append(text)
                metadatas.append(metadata_for_chroma(metadata))
                indexed_count += 1

                if len(docs) >= args.batch_size:
                    add_batch(collection, embeddings, ids, docs, metadatas)
                    ids = []
                    docs = []
                    metadatas = []

    add_batch(collection, embeddings, ids, docs, metadatas)

    print(f"Patch rows requested: {len(patch_rows)}")
    print(f"Indexed patches: {indexed_count}")
    print(f"Skipped existing patches: {skipped_existing}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    index_selected_patches(parse_args())
