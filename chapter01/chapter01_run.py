#!/usr/bin/env python
"""
Chapter 1 (Hands-On LLMs) as an executable Python script.

Usage examples:
  python chapter01_run.py
  python chapter01_run.py --device cpu --prompt "Explain transformers simply."
  python chapter01_run.py --model sshleifer/tiny-gpt2 --device cpu --no-chat
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

# Silence a harmless PyTorch Windows warning about unsupported stdio redirects.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def resolve_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chapter 1 generation example.")
    parser.add_argument(
        "--model",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--prompt",
        default="Create a funny joke about chickens.",
        help="Prompt to send to the model",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection. 'auto' picks cuda if available, else cpu.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling (default is deterministic generation)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Only used when --sample is enabled.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold. Only used when --sample is enabled.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty factor to discourage repeated tokens and loops.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Prevent repeating n-grams of this size. Use 0 to disable.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the model",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Force plain text prompt instead of chat messages",
    )
    return parser.parse_args()


def extract_text(generated: Any) -> str:
    if isinstance(generated, str):
        return generated
    if isinstance(generated, list) and generated:
        last = generated[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    return str(generated)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Loading model: {args.model}")
    print(f"Device map: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model.lower() == "sshleifer/tiny-gpt2":
        print("[NOTE] sshleifer/tiny-gpt2 es un modelo de prueba y suele generar texto absurdo.")

    generation_kwargs = {
        "return_full_text": False,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "repetition_penalty": args.repetition_penalty,
    }
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if args.sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    if args.no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **generation_kwargs,
    )

    payload: Any
    if args.no_chat:
        payload = args.prompt
    else:
        payload = [{"role": "user", "content": args.prompt}]

    try:
        output = generator(payload)
    except Exception:
        # Fallback for models that do not accept chat-message payloads.
        output = generator(args.prompt)

    generated = output[0]["generated_text"]
    print("\n=== Model output ===\n")
    print(extract_text(generated).strip())


if __name__ == "__main__":
    main()
