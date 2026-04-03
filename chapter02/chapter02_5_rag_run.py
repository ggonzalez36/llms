#!/usr/bin/env python
"""
Chapter 2.5 (RAG intro with ChromaDB) executable script.

Usage examples:
  python chapter02_5_rag_run.py
  python chapter02_5_rag_run.py --query "Como se representan palabras matematicamente?"
  python chapter02_5_rag_run.py --collection-name mi_practica_ia
  python chapter02_5_rag_run.py --persistent-path .chroma_data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


DEFAULT_DOCUMENTS = (
    "El mecanismo de atencion permite al modelo enfocarse en partes especificas de la frase.",
    "Los embeddings convierten texto en vectores numericos de alta dimension.",
    "Las bases de datos vectoriales son ideales para busquedas semanticas rapidas.",
)

DEFAULT_IDS = ("id1", "id2", "id3")

DEFAULT_QUERY = "Como se representan las palabras matematicamente?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small RAG practice with ChromaDB.")
    parser.add_argument(
        "--collection-name",
        default="mi_practica_ia",
        help="Collection name in ChromaDB.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query text for semantic search.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=1,
        help="How many nearest results to return.",
    )
    parser.add_argument(
        "--documents",
        default="||".join(DEFAULT_DOCUMENTS),
        help="Documents separated by '||'.",
    )
    parser.add_argument(
        "--ids",
        default=",".join(DEFAULT_IDS),
        help="Document ids separated by comma.",
    )
    parser.add_argument(
        "--persistent-path",
        default="",
        help="Optional folder for persistent local Chroma data. Empty means in-memory client.",
    )
    parser.add_argument(
        "--cache-home",
        default="",
        help="Optional writable home folder for Chroma cache. Default: chapter02/.chroma_home",
    )
    return parser.parse_args()


def parse_documents(raw: str) -> list[str]:
    docs = [part.strip() for part in raw.split("||") if part.strip()]
    if not docs:
        raise ValueError("No documents provided. Use --documents with at least one entry.")
    return docs


def parse_ids(raw: str, expected_count: int) -> list[str]:
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    if len(ids) != expected_count:
        raise ValueError(
            f"IDs count ({len(ids)}) must match documents count ({expected_count}). "
            "Use --ids with the same amount of items."
        )
    return ids


def configure_chroma_home(cache_home: str) -> str:
    if not cache_home:
        return ""

    home_path = Path(cache_home)
    home_path.mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(home_path)
    os.environ["HOME"] = str(home_path)
    return str(home_path)


def build_client(persistent_path: str) -> Any:
    import chromadb

    if persistent_path:
        print(f"[INFO] Using persistent Chroma client at: {persistent_path}")
        return chromadb.PersistentClient(path=persistent_path)
    print("[INFO] Using in-memory Chroma client.")
    return chromadb.Client()


def upsert_documents(collection: Any, documents: list[str], ids: list[str]) -> None:
    if hasattr(collection, "upsert"):
        collection.upsert(documents=documents, ids=ids)
        return

    # Backward-compatible fallback for older Chroma versions.
    try:
        collection.add(documents=documents, ids=ids)
    except Exception:
        collection.delete(ids=ids)
        collection.add(documents=documents, ids=ids)


def explain_chroma_error(exc: Exception, cache_home: str) -> str:
    error_text = str(exc)
    if "WinError 10013" in error_text:
        return (
            "Network access was blocked while Chroma tried to download its default embedding model.\n"
            "Run again in a normal terminal session with internet access.\n"
            "Once downloaded, the model is cached and next runs are faster/offline-friendly."
        )
    if "PermissionError" in exc.__class__.__name__ or "Permission denied" in error_text:
        cache_hint = cache_home if cache_home else ".chroma_home"
        return (
            "Permission error while writing Chroma cache files.\n"
            f"Try a writable cache folder, for example: --cache-home {cache_hint}"
        )
    return f"Chroma operation failed: {exc}"


def main() -> None:
    args = parse_args()
    documents = parse_documents(args.documents)
    ids = parse_ids(args.ids, len(documents))
    cache_home = configure_chroma_home(args.cache_home)
    if cache_home:
        print(f"[INFO] Chroma cache home: {cache_home}")
    else:
        print("[INFO] Chroma cache home: default (~/.cache/chroma)")

    try:
        client = build_client(args.persistent_path)
    except ModuleNotFoundError:
        raise SystemExit(
            "chromadb is not installed in this environment.\n"
            "Install it with:\n"
            "  conda run -n holllm python -m pip install chromadb"
        )

    collection = client.get_or_create_collection(name=args.collection_name)
    try:
        upsert_documents(collection, documents, ids)
    except Exception as exc:
        raise SystemExit(explain_chroma_error(exc, cache_home))

    print("\n=== Stored Documents ===")
    for doc_id, doc in zip(ids, documents):
        print(f"- {doc_id}: {doc}")

    try:
        results = collection.query(
            query_texts=[args.query],
            n_results=args.n_results,
        )
    except Exception as exc:
        raise SystemExit(explain_chroma_error(exc, cache_home))

    matched_docs = results.get("documents", [[]])[0]
    matched_ids = results.get("ids", [[]])[0]
    matched_distances = results.get("distances", [[]])[0]

    print("\n=== Query ===")
    print(args.query)
    print("\n=== Top Matches ===")
    if not matched_docs:
        print("[WARN] No results returned.")
        return

    for idx, (doc, doc_id, distance) in enumerate(
        zip(matched_docs, matched_ids, matched_distances),
        start=1,
    ):
        print(f"{idx}. id={doc_id} distance={distance:.6f}")
        print(f"   {doc}")


if __name__ == "__main__":
    main()
