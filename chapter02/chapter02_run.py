#!/usr/bin/env python
"""
Chapter 2 - Tiny concepts demo:
1) tokens
2) embeddings
3) optional tiny generation
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

TEXT = "Transformers convert text into token ids and vectors."
PROMPT = "Write one short sentence about token embeddings."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 2 simple runner.")
    parser.add_argument("--mode", choices=["all", "tokens", "embeddings", "generation"], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    return parser.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def show_tokens(local_files_only: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=local_files_only)
    token_ids = tokenizer(TEXT).input_ids
    print("\n=== Tokens ===")
    print(f"Text: {TEXT}")
    print(f"Token count: {len(token_ids)}")
    for token_id in token_ids[:12]:
        piece = tokenizer.decode([token_id]).replace("\n", "\\n")
        print(f"id={token_id:<6} piece={piece!r}")


def show_embeddings(device: str, local_files_only: bool) -> None:
    model_name = "microsoft/deberta-v3-xsmall"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()

    encoded = tokenizer(TEXT, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state

    print("\n=== Embeddings ===")
    print(f"Model: {model_name}")
    print(f"Shape [batch, tokens, dims]: {tuple(hidden.shape)}")
    print(f"First token, first 8 dims: {[round(x, 4) for x in hidden[0, 0, :8].tolist()]}")


def show_generation(device: str, local_files_only: bool, max_new_tokens: int) -> None:
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)

    generated_ids = output_ids[0][input_ids.shape[-1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\n=== Tiny Generation (optional) ===")
    print(f"Prompt: {PROMPT}")
    print(f"Output: {generated_text if generated_text else '[empty]'}")


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"Device: {device}")

    if args.mode in ("all", "tokens"):
        show_tokens(args.local_files_only)
    if args.mode in ("all", "embeddings"):
        show_embeddings(device, args.local_files_only)
    if args.mode in ("all", "generation"):
        show_generation(device, args.local_files_only, args.max_new_tokens)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Chapter 2 failed: {exc}")
