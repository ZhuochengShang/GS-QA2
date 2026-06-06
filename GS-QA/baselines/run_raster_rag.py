#!/usr/bin/env python3
"""Run a raster RAG baseline over vector entities and GDAL raster metadata.

This mirrors baselines_rag.ipynb at the raster level:
1. Load natural-language QA records.
2. Retrieve OSM-like vector entity records from Chroma.
3. Retrieve precomputed DEM GeoTIFF metadata records from Chroma.
4. Put retrieved vector and raster metadata into an LLM prompt.
5. Save the generated answer and retrieval context.

This script does not open DEM rasters at query time and does not run SQL.
It relies only on the Chroma metadata index built by build_question_dem_patch_embeddings.py.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import chromadb
import numpy as np


DEFAULT_ANSWER_PROMPT = """Answer the provided user question while satisfying the following requirements:
1. Do not include any parts of the question in the answer; provide the answer directly.
2. Provide only the property the user is asking for like name of an entity, its location, distance, direction, or raster metadata value.
3. Do not provide information the user did not ask for.
4. Any number must be written as words and rounded to the nearest ten when appropriate.
5. Only use metric units.

The following records might be relevant to answering this question, but not necessarily:
"""


GDAL_RAG_RULES = """

The retrieved OSM/vector records may include POIs, roads, parks, lakes, and regions.
The retrieved DEM GeoTIFF records may include:
- patch bbox in EPSG:4326 for spatial filtering
- source DEM GeoTIFF file
- file-level raster metadata from gdalinfo, including CRS/EPSG, GeoTransform, bounds, bands, band nodata, and band min/max
- raster row/column window identifying the patch used for retrieval

Rules:
- Do not invent raster values that are not present in the retrieved records.
- Treat gdalinfo raster metadata as file-level context, not point-level measurement.
- Do not use SQL.
- Do not claim exact point or polygon calculations were performed.
- For numeric answers, use only the retrieved gdalinfo metadata values.
- If the retrieved vector records or patches do not provide enough evidence for the requested location or operation, say so directly.
- Keep the answer concise.
"""


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for "
                "--embedding-provider sentence-transformers. Install with: "
                "pip install sentence-transformers"
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def make_embedding_function(args: argparse.Namespace) -> Any:
    if args.embedding_provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is required for --embedding-provider ollama. "
                "Install with: pip install langchain-ollama"
            ) from exc
        return OllamaEmbeddings(model=args.embedding_model)

    if args.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddings(args.embedding_model, args.embedding_batch_size)

    raise ValueError(f"Unsupported embedding provider: {args.embedding_provider}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raster RAG from DEM GeoTIFF metadata records.")
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Directory containing JSONL questions. Repeat for multiple directories.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--dem-persist-directory", required=True, help="DEM GeoTIFF metadata Chroma directory.")
    parser.add_argument("--dem-collection-name", default="dem_patches")
    parser.add_argument("--entity-persist-directory", required=True, help="Vector entity Chroma directory.")
    parser.add_argument("--entity-collection-name", default="geo_entities")
    parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "sentence-transformers"],
        default="sentence-transformers",
        help="Embedding provider used at query time. Must match the Chroma indexes.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used for query embeddings.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--answer-prompt-file",
        help="Optional prompt template file. Defaults to the built-in RAG answer prompt.",
    )
    parser.add_argument("--samples-per-file", type=int, default=2)
    parser.add_argument("--top-k-dem", type=int, default=10)
    parser.add_argument("--top-k-entities", type=int, default=10)
    parser.add_argument(
        "--spatial-prefilter",
        action="store_true",
        help="Use WKT geometry from question_entities and retrieved entities to prefilter patch bboxes.",
    )
    parser.add_argument(
        "--spatial-candidate-limit",
        type=int,
        default=200,
        help="Maximum spatially intersecting patch records to rank locally.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["none", "openai", "gemini"],
        default="none",
        help="Use 'none' to save prompts/retrievals without generating LLM answers.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model when provider is openai or gemini.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_questions(input_dirs: list[str], samples_per_file: int) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        root = Path(input_dir)
        for path in sorted(root.glob("*.jsonl")):
            kept = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    row["source_file"] = path.name
                    row["source_stem"] = path.stem
                    row["id"] = f"{root.name}:{path.stem}:{kept}"
                    questions.append(row)
                    kept += 1
                    if kept >= samples_per_file:
                        break
    return questions


def compact_entities(entities: dict[str, Any], max_wkt_chars: int = 500) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in entities.items():
        if not isinstance(value, dict):
            continue
        out: dict[str, Any] = {}
        for field in (
            "main_category",
            "sub_category",
            "sub_category_label",
            "display_name",
            "region_name",
            "distance",
            "elevation",
            "value",
        ):
            if field in value:
                out[field] = value[field]
        if "geo_wkt" in value:
            wkt = value["geo_wkt"]
            out["geo_wkt"] = wkt[:max_wkt_chars] + ("..." if len(wkt) > max_wkt_chars else "")
        for nested_key in ("poi", "road", "park", "region"):
            nested = value.get(nested_key)
            if isinstance(nested, dict):
                nested_out = {
                    k: nested[k]
                    for k in (
                        "id",
                        "poi_name",
                        "road_name",
                        "park_name",
                        "region_name",
                        "amenity",
                        "tourism",
                        "leisure",
                        "highway",
                    )
                    if k in nested
                }
                if nested_out:
                    out[nested_key] = nested_out
        compact[key] = out
    return compact


def build_retrieval_query(question: dict[str, Any]) -> str:
    parts = [question.get("question", "")]
    for entity in question.get("question_entities", {}).values():
        if not isinstance(entity, dict):
            continue
        for field in ("display_name", "region_name", "sub_category", "sub_category_label"):
            if entity.get(field):
                parts.append(str(entity[field]))
        for nested_key in ("poi", "road", "park", "region"):
            nested = entity.get(nested_key)
            if isinstance(nested, dict):
                for field in ("poi_name", "road_name", "park_name", "region_name"):
                    if nested.get(field):
                        parts.append(str(nested[field]))
    return "\n".join(parts)


def geometry_from_wkt_text(text: str) -> Any | None:
    try:
        from shapely import wkt
    except ImportError:
        return None
    try:
        return wkt.loads(text)
    except Exception:
        return None


def extract_query_geometries(question: dict[str, Any]) -> list[Any]:
    geoms: list[Any] = []
    for entity in question.get("question_entities", {}).values():
        if not isinstance(entity, dict):
            continue
        if entity.get("geo_wkt"):
            geom = geometry_from_wkt_text(entity["geo_wkt"])
            if geom is not None:
                geoms.append(geom)
        for nested_key in ("poi", "road", "park", "region"):
            nested = entity.get(nested_key)
            if isinstance(nested, dict) and nested.get("geometry"):
                geom = geometry_from_wkt_text(nested["geometry"])
                if geom is not None:
                    geoms.append(geom)
    return geoms


def extract_entity_geometries(entity_records: list[dict[str, Any]]) -> list[Any]:
    geoms: list[Any] = []
    for record in entity_records:
        geom_text = record.get("metadata", {}).get("geometry")
        if isinstance(geom_text, str):
            geom = geometry_from_wkt_text(geom_text)
            if geom is not None:
                geoms.append(geom)
    return geoms


def bbox_intersects(metadata: dict[str, Any], geom: Any) -> bool:
    try:
        from shapely.geometry import box
    except ImportError:
        return False

    keys = ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy")
    if not all(key in metadata for key in keys):
        return False
    patch = box(
        metadata["bbox_minx"],
        metadata["bbox_miny"],
        metadata["bbox_maxx"],
        metadata["bbox_maxy"],
    )
    return patch.intersects(geom)


def cosine_distance(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(va, vb) / denom)


def retrieve_patches(
    collection: Any,
    embedder: SentenceTransformerEmbeddings,
    question: dict[str, Any],
    top_k: int,
    spatial_prefilter: bool,
    spatial_candidate_limit: int,
    extra_geometries: list[Any] | None = None,
) -> list[dict[str, Any]]:
    query = build_retrieval_query(question)
    query_embedding = embedder.embed_query(query)

    if spatial_prefilter:
        query_geoms = extract_query_geometries(question)
        geoms = query_geoms + (extra_geometries or [])  # union both sources
        if geoms:
            candidates: list[dict[str, Any]] = []
            offset = 0
            page_size = 1000
            total_count = collection.count()
            while offset < total_count:
                page = collection.get(
                    include=["documents", "metadatas", "embeddings"],
                    limit=page_size,
                    offset=offset,
                )
                if not page["ids"]:
                    break
                for row_id, doc, md, emb in zip(
                    page["ids"],
                    page["documents"],
                    page["metadatas"],
                    page["embeddings"],
                ):
                    if any(bbox_intersects(md, geom) for geom in geoms):
                        candidates.append(
                            {
                                "id": row_id,
                                "document": doc,
                                "metadata": md,
                                "distance": cosine_distance(query_embedding, emb),
                            }
                        )
                        if len(candidates) >= spatial_candidate_limit:
                            break
                if len(candidates) >= spatial_candidate_limit:
                    break
                offset += page_size
            if candidates:
                candidates.sort(key=lambda row: row["distance"])
                return candidates[:top_k]
            # No spatial candidates (tile not yet indexed) — fall back to semantic search
            # so the LLM gets nearby terrain context rather than an empty prompt.

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    rows: list[dict[str, Any]] = []
    for row_id, doc, md, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        rows.append({"id": row_id, "document": doc, "metadata": md, "distance": dist})
    return rows


def retrieve_entities(
    collection: Any,
    embedder: SentenceTransformerEmbeddings,
    question: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    query = build_retrieval_query(question)
    result = collection.query(
        query_embeddings=[embedder.embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    rows: list[dict[str, Any]] = []
    for row_id, doc, md, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        rows.append({"id": row_id, "document": doc, "metadata": md, "distance": dist})
    return rows


def compact_entity_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata.get(key)
        for key in (
            "entity_type",
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
            "geometry_type",
            "source_file",
            "source_row",
        )
        if key in metadata
    }


def build_prompt(
    question: dict[str, Any],
    entity_records: list[dict[str, Any]],
    patches: list[dict[str, Any]],
) -> str:
    entity_text = []
    for i, entity in enumerate(entity_records, start=1):
        entity_text.append(
            "\n".join(
                [
                    f"Entity {i}: {entity['id']}",
                    f"Distance score: {entity['distance']}",
                    f"Summary: {entity['document']}",
                    "Metadata: " + json.dumps(compact_entity_metadata(entity["metadata"]), sort_keys=True),
                ]
            )
        )

    patch_text = []
    for i, patch in enumerate(patches, start=1):
        md = patch["metadata"]
        patch_text.append(
            "\n".join(
                [
                    f"Record {i}: {patch['id']}",
                    f"Distance score: {patch['distance']}",
                    f"Summary: {patch['document']}",
                    "Metadata: "
                    + json.dumps(
                        {
                            key: md.get(key)
                            for key in (
                                "bbox_minx",
                                "bbox_miny",
                                "bbox_maxx",
                                "bbox_maxy",
                                "raster_driver",
                                "raster_width",
                                "raster_height",
                                "raster_epsg",
                                "raster_origin_x",
                                "raster_origin_y",
                                "raster_pixel_width",
                                "raster_pixel_height",
                                "raster_bbox_minx",
                                "raster_bbox_miny",
                                "raster_bbox_maxx",
                                "raster_bbox_maxy",
                                "raster_band_count",
                                "raster_bands",
                                "source_file",
                                "row_start",
                                "col_start",
                                "row_end",
                                "col_end",
                                "patch_width",
                                "patch_height",
                            )
                            if key in md
                        },
                        sort_keys=True,
                    ),
                ]
            )
        )

    return "\n\n".join(
        [
            "Question:",
            question.get("question", ""),
            "Structured question entities:",
            json.dumps(compact_entities(question.get("question_entities", {})), indent=2),
            "Retrieved OSM/vector records:",
            "\n\n".join(entity_text) if entity_text else "No vector records retrieved.",
            "Retrieved DEM GeoTIFF metadata records:",
            "\n\n".join(patch_text) if patch_text else "No DEM GeoTIFF metadata records retrieved.",
            "Answer:",
        ]
    )


def call_openai(model: str, system_prompt: str, prompt: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    usage = response.usage
    return {
        "text": response.choices[0].message.content or "",
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
    }


def call_gemini(model: str, system_prompt: str, prompt: str) -> dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required when --llm-provider gemini.")

    model_name = model if model.startswith("models/") else f"models/{model}"
    request_body = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
        },
    }
    request = urllib.request.Request(
        url=f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    texts: list[str] = []
    for candidate in payload.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    usage_metadata = payload.get("usageMetadata", {})
    return {
        "text": "\n".join(texts),
        "usage": {
            "prompt_tokens": usage_metadata.get("promptTokenCount"),
            "completion_tokens": usage_metadata.get("candidatesTokenCount"),
            "total_tokens": usage_metadata.get("totalTokenCount"),
        },
        "raw_usage_metadata": usage_metadata,
    }


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in row:
                ids.add(row["id"])
    return ids


def load_system_prompt(path_text: str | None) -> str:
    base_prompt = Path(path_text).read_text(encoding="utf-8") if path_text else DEFAULT_ANSWER_PROMPT
    return base_prompt.rstrip() + "\n\n" + GDAL_RAG_RULES.strip()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.input_dir, args.samples_per_file)
    completed = load_existing_ids(output) if args.resume else set()
    system_prompt = load_system_prompt(args.answer_prompt_file)

    embedder = make_embedding_function(args)
    dem_client = chromadb.PersistentClient(path=args.dem_persist_directory)
    dem_collection = dem_client.get_collection(args.dem_collection_name)
    entity_client = chromadb.PersistentClient(path=args.entity_persist_directory)
    entity_collection = entity_client.get_collection(args.entity_collection_name)

    with output.open("a", encoding="utf-8") as handle:
        total_questions = len(questions)
        for index, question in enumerate(questions, start=1):
            if question["id"] in completed:
                continue
            started = time.time()
            print(f"[{index}/{total_questions}] {question['id']} retrieving entities", flush=True)
            entity_records = retrieve_entities(
                collection=entity_collection,
                embedder=embedder,
                question=question,
                top_k=args.top_k_entities,
            )
            print(
                f"[{index}/{total_questions}] {question['id']} retrieved "
                f"{len(entity_records)} entities; retrieving DEM patches",
                flush=True,
            )
            patches = retrieve_patches(
                collection=dem_collection,
                embedder=embedder,
                question=question,
                top_k=args.top_k_dem,
                spatial_prefilter=args.spatial_prefilter,
                spatial_candidate_limit=args.spatial_candidate_limit,
                extra_geometries=extract_entity_geometries(entity_records),
            )
            print(
                f"[{index}/{total_questions}] {question['id']} retrieved "
                f"{len(patches)} DEM patches; building prompt",
                flush=True,
            )
            prompt = build_prompt(question, entity_records, patches)
            llm_result = {"text": "", "usage": {}}
            if args.llm_provider == "openai":
                print(f"[{index}/{total_questions}] {question['id']} calling OpenAI", flush=True)
                llm_result = call_openai(args.model, system_prompt, prompt)
            elif args.llm_provider == "gemini":
                print(f"[{index}/{total_questions}] {question['id']} calling Gemini", flush=True)
                llm_result = call_gemini(args.model, system_prompt, prompt)
            answer = llm_result["text"]

            row = {
                "id": question["id"],
                "source_file": question.get("source_file"),
                "question": question.get("question"),
                "answer_type": question.get("answer_type"),
                "gold_answers": question.get("answers"),
                "retrieved_entities": entity_records,
                "retrieved_patches": patches,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "rag_answer": answer,
                "usage": llm_result.get("usage", {}),
                "llm_provider": args.llm_provider,
                "model": args.model if args.llm_provider != "none" else None,
                "elapsed_seconds": time.time() - started,
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(
                f"[{index}/{total_questions}] wrote {question['id']} with "
                f"{len(entity_records)} entities, {len(patches)} patches, "
                f"tokens={llm_result.get('usage', {}).get('total_tokens')}"
            )


if __name__ == "__main__":
    main()
