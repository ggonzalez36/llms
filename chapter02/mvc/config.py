from __future__ import annotations

import argparse
from dataclasses import dataclass

DEFAULT_TEXT = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
""".strip()

AVAILABLE_SECTIONS = ("generation", "tokenizers", "contextual", "sentence")
DEFAULT_SECTIONS = ("generation", "tokenizers", "contextual")


@dataclass(frozen=True)
class Chapter2Config:
    sections: tuple[str, ...]
    device: str
    local_files_only: bool
    trust_remote_code: bool
    preview_tokens: int
    max_new_tokens: int
    sample: bool
    temperature: float
    top_p: float
    prompt: str
    generation_model: str
    text: str
    tokenizer_models: tuple[str, ...]
    contextual_text: str
    encoder_tokenizer: str
    encoder_model: str
    sentence_text: str
    sentence_model: str


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_sections(raw: str) -> tuple[str, ...]:
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        return DEFAULT_SECTIONS
    if "all" in parts:
        return AVAILABLE_SECTIONS

    invalid = [item for item in parts if item not in AVAILABLE_SECTIONS]
    if invalid:
        valid = ", ".join(AVAILABLE_SECTIONS)
        raise ValueError(f"Invalid --sections value(s): {invalid}. Valid values: {valid}, all")
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chapter 2 token and embedding demos.")
    parser.add_argument(
        "--sections",
        default="generation,tokenizers,contextual",
        help="Comma-separated sections: generation,tokenizers,contextual,sentence,all",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device selection. 'auto' picks cuda if available, else cpu.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load models/tokenizers only from local Hugging Face cache.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading models/tokenizers.",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=25,
        help="How many tokens to print per section.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of new tokens in generation demo.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling in generation demo.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (used only with --sample).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value (used only with --sample).",
    )
    parser.add_argument(
        "--prompt",
        default="Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>",
        help="Prompt used in generation demo.",
    )
    parser.add_argument(
        "--generation-model",
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Causal language model used for generation demo.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text used for tokenizer comparison.",
    )
    parser.add_argument(
        "--tokenizer-models",
        default="bert-base-uncased,bert-base-cased,gpt2,google/flan-t5-small,microsoft/Phi-3-mini-4k-instruct",
        help="Comma-separated tokenizer model ids for tokenizer comparison.",
    )
    parser.add_argument(
        "--contextual-text",
        default="Hello world",
        help="Text used for contextual embedding demo.",
    )
    parser.add_argument(
        "--encoder-tokenizer",
        default="microsoft/deberta-base",
        help="Tokenizer id for contextual embedding demo.",
    )
    parser.add_argument(
        "--encoder-model",
        default="microsoft/deberta-v3-xsmall",
        help="Encoder model id for contextual embedding demo.",
    )
    parser.add_argument(
        "--sentence-text",
        default="Best movie ever!",
        help="Text used for sentence embedding demo.",
    )
    parser.add_argument(
        "--sentence-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model id for sentence embedding demo.",
    )
    return parser


def load_config() -> Chapter2Config:
    args = build_parser().parse_args()
    sections = parse_sections(args.sections)
    tokenizer_models = parse_csv(args.tokenizer_models)
    if not tokenizer_models:
        raise SystemExit("Invalid --tokenizer-models value: provide at least one model id.")

    return Chapter2Config(
        sections=sections,
        device=args.device,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        preview_tokens=args.preview_tokens,
        max_new_tokens=args.max_new_tokens,
        sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt=args.prompt,
        generation_model=args.generation_model,
        text=args.text,
        tokenizer_models=tokenizer_models,
        contextual_text=args.contextual_text,
        encoder_tokenizer=args.encoder_tokenizer,
        encoder_model=args.encoder_model,
        sentence_text=args.sentence_text,
        sentence_model=args.sentence_model,
    )

