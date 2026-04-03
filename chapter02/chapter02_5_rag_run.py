#!/usr/bin/env python
"""
Chapter 2.5 - Minimal RAG demo with Chroma.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DOCS = [
    "El mecanismo de atencion permite al modelo enfocarse en partes especificas de la frase.",
    "Los embeddings convierten texto en vectores numericos de alta dimension.",
    "Las bases de datos vectoriales son ideales para busquedas semanticas rapidas.",
]
IDS = ["id1", "id2", "id3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 2.5 simple RAG runner.")
    parser.add_argument("--query", default="Como se representan las palabras matematicamente?")
    parser.add_argument("--n-results", type=int, default=1)
    parser.add_argument("--cache-home", default="")
    return parser.parse_args()


def set_cache_home(cache_home: str) -> None:
    if not cache_home:
        return
    cache_path = Path(cache_home).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(cache_path)
    os.environ["USERPROFILE"] = str(cache_path)
    print(f"[INFO] Cache path: {cache_path}")


def main() -> None:
    args = parse_args()
    set_cache_home(args.cache_home)

    try:
        import chromadb
    except ModuleNotFoundError:
        raise SystemExit("chromadb no esta instalado. Ejecuta: conda run -n holllm python -m pip install chromadb")

    client = chromadb.Client()
    collection = client.get_or_create_collection(name="mi_practica_ia")
    collection.upsert(documents=DOCS, ids=IDS)
    results = collection.query(query_texts=[args.query], n_results=args.n_results)

    print("\n=== Pregunta ===")
    print(args.query)
    print("\n=== Resultado(s) ===")

    docs = results["documents"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]
    for i, (doc_id, doc, distance) in enumerate(zip(ids, docs, distances), start=1):
        print(f"{i}. {doc_id} | distance={distance:.6f}")
        print(f"   {doc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"RAG demo failed: {exc}")
