#!/usr/bin/env python3
"""Build a Chroma store of vector entities for raster RAG.

The script extracts OSM-like entities from GS-QA2 JSONL question files. It keeps
geometry WKT in Chroma metadata so raster RAG can spatially prefilter DEM patch
records before semantic ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector entity embeddings for raster RAG.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory containing GS-QA2 JSONL files. Repeat for multiple directories.",
    )
    parser.add_argument("--persist-directory", required=True)
    parser.add_argument("--collection-name", default="geo_entities")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reset", action="store_true", help="Delete the existing collection first.")
    return parser.parse_args()


def scalar_metadata(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True)


def geometry_type(wkt: str | None) -> str | None:
    if not wkt:
        return None
    return wkt.split("(", 1)[0].strip().upper() or None


def entity_name(entity: dict[str, Any]) -> str | None:
    for key in (
        "display_name",
        "poi_name",
        "road_name",
        "park_name",
        "lake_name",
        "region_name",
        "name",
        "official_name",
    ):
        value = entity.get(key)
        if value:
            return str(value)
    return None


def entity_geometry(entity: dict[str, Any]) -> str | None:
    for key in ("geometry", "geo_wkt", "wkt"):
        value = entity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def looks_like_entity(value: dict[str, Any]) -> bool:
    return entity_geometry(value) is not None or entity_name(value) is not None


def walk_entities(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if looks_like_entity(value):
            found.append(value)
        for child in value.values():
            found.extend(walk_entities(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_entities(child))
    return found


def build_document(entity: dict[str, Any], source_file: str, source_row: int) -> tuple[str, dict[str, Any]]:
    geom = entity_geometry(entity)
    name = entity_name(entity)
    metadata: dict[str, Any] = {
        "source_file": source_file,
        "source_row": source_row,
    }
    for key in (
        "display_name",
        "poi_name",
        "road_name",
        "park_name",
        "lake_name",
        "region_name",
        "main_category",
        "sub_category",
        "sub_category_label",
        "amenity",
        "tourism",
        "leisure",
        "highway",
        "border_type",
        "addr_city",
        "addr_state",
    ):
        if key in entity and entity[key] is not None:
            metadata[key] = scalar_metadata(entity[key])
    if geom:
        metadata["geometry"] = geom
        metadata["geometry_type"] = geometry_type(geom)
    if name:
        metadata["display_name"] = metadata.get("display_name") or name

    doc_fields = {
        key: entity.get(key)
        for key in (
            "display_name",
            "poi_name",
            "road_name",
            "park_name",
            "lake_name",
            "region_name",
            "main_category",
            "sub_category",
            "sub_category_label",
            "amenity",
            "tourism",
            "leisure",
            "highway",
            "border_type",
            "addr_city",
            "addr_state",
        )
        if entity.get(key) is not None
    }
    doc_fields["geometry_type"] = metadata.get("geometry_type")
    return json.dumps(doc_fields, sort_keys=True), metadata


def iter_question_rows(input_dirs: list[str]) -> list[tuple[Path, int, dict[str, Any]]]:
    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for input_dir in input_dirs:
        for path in sorted(Path(input_dir).glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for row_number, line in enumerate(handle):
                    line = line.strip()
                    if line:
                        rows.append((path, row_number, json.loads(line)))
    return rows


def stable_id(document: str, metadata: dict[str, Any]) -> str:
    payload = json.dumps({"document": document, "metadata": metadata}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    embedder = SentenceTransformerEmbeddings(args.embedding_model, args.embedding_batch_size)
    client = chromadb.PersistentClient(path=args.persist_directory)
    if args.reset:
        try:
            client.delete_collection(args.collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(args.collection_name)

    seen: set[str] = set()
    ids: list[str] = []
    docs: list[str] = []
    metadatas: list[dict[str, Any]] = []

    def flush() -> None:
        if not docs:
            return
        vectors = embedder.embed_documents(docs)
        collection.add(ids=list(ids), documents=list(docs), metadatas=list(metadatas), embeddings=vectors)
        ids.clear()
        docs.clear()
        metadatas.clear()

    for path, row_number, question in iter_question_rows(args.input_dir):
        for entity in walk_entities(question.get("question_entities", {})):
            document, metadata = build_document(entity, path.name, row_number)
            if document == "{}" and "geometry" not in metadata:
                continue
            row_id = stable_id(document, metadata)
            if row_id in seen:
                continue
            seen.add(row_id)
            ids.append(row_id)
            docs.append(document)
            metadatas.append(metadata)
            if len(docs) >= args.batch_size:
                flush()
    flush()
    print(f"Indexed {len(seen)} vector entities into {args.persist_directory} ({args.collection_name}).")


if __name__ == "__main__":
    main()
