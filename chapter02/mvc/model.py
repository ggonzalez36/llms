from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .config import Chapter2Config


@dataclass(frozen=True)
class TokenPreview:
    token_id: int
    piece: str


@dataclass(frozen=True)
class GenerationResult:
    model_name: str
    prompt_token_count: int
    prompt_preview: tuple[TokenPreview, ...]
    generated_preview: tuple[TokenPreview, ...]
    generated_text: str


@dataclass(frozen=True)
class TokenizerResult:
    model_name: str
    token_count: int
    token_preview: tuple[TokenPreview, ...]
    omitted_count: int
    error: str | None = None


@dataclass(frozen=True)
class ContextualResult:
    tokenizer_name: str
    model_name: str
    input_text: str
    embedding_shape: tuple[int, ...]
    token_preview: tuple[TokenPreview, ...]
    first_dims: tuple[float, ...]


@dataclass(frozen=True)
class SentenceResult:
    model_name: str
    input_text: str
    embedding_shape: tuple[int, ...] | None
    first_dims: tuple[float, ...]
    warning: str | None = None


class Chapter2Model:
    def __init__(self, config: Chapter2Config) -> None:
        self.config = config
        self.device = self.resolve_device(config.device)
        if config.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    @staticmethod
    def resolve_device(user_device: str) -> str:
        if user_device != "auto":
            return user_device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def generation_demo(self) -> GenerationResult:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.generation_model,
            **self.model_load_kwargs(),
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.generation_model,
            torch_dtype="auto",
            **self.model_load_kwargs(),
        )
        model.to(self.device)
        model.eval()

        encoded = tokenizer(self.config.prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, **self.generation_kwargs(tokenizer))

        generated_only_ids = output_ids[0][input_ids.shape[-1] :]
        generated_text = tokenizer.decode(generated_only_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            model_name=self.config.generation_model,
            prompt_token_count=int(input_ids.shape[-1]),
            prompt_preview=self.build_token_preview(input_ids[0], tokenizer),
            generated_preview=self.build_token_preview(generated_only_ids, tokenizer),
            generated_text=generated_text,
        )

    def tokenizer_comparison(self) -> list[TokenizerResult]:
        results: list[TokenizerResult] = []
        for model_name in self.config.tokenizer_models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, **self.model_load_kwargs())
                token_ids = tokenizer(self.config.text).input_ids
                preview = self.build_token_preview(token_ids, tokenizer)
                results.append(
                    TokenizerResult(
                        model_name=model_name,
                        token_count=len(token_ids),
                        token_preview=preview,
                        omitted_count=max(0, len(token_ids) - self.config.preview_tokens),
                    )
                )
            except Exception as exc:
                results.append(
                    TokenizerResult(
                        model_name=model_name,
                        token_count=0,
                        token_preview=(),
                        omitted_count=0,
                        error=str(exc),
                    )
                )
        return results

    def contextual_embedding_demo(self) -> ContextualResult:
        tokenizer = AutoTokenizer.from_pretrained(self.config.encoder_tokenizer, **self.model_load_kwargs())
        model = AutoModel.from_pretrained(self.config.encoder_model, **self.model_load_kwargs())
        model.to(self.device)
        model.eval()

        tokens = tokenizer(self.config.contextual_text, return_tensors="pt")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            output = model(**tokens).last_hidden_state

        first_dims = tuple(round(value, 4) for value in output[0, 0, :8].detach().cpu().tolist())
        token_preview = self.build_token_preview(tokens["input_ids"][0], tokenizer)

        return ContextualResult(
            tokenizer_name=self.config.encoder_tokenizer,
            model_name=self.config.encoder_model,
            input_text=self.config.contextual_text,
            embedding_shape=tuple(output.shape),
            token_preview=token_preview,
            first_dims=first_dims,
        )

    def sentence_embedding_demo(self) -> SentenceResult:
        if importlib.util.find_spec("sentence_transformers") is None:
            return SentenceResult(
                model_name=self.config.sentence_model,
                input_text=self.config.sentence_text,
                embedding_shape=None,
                first_dims=(),
                warning="sentence-transformers is not installed. Install it to run this section.",
            )

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.config.sentence_model, device=self.device)
        vector = model.encode(self.config.sentence_text)

        return SentenceResult(
            model_name=self.config.sentence_model,
            input_text=self.config.sentence_text,
            embedding_shape=tuple(vector.shape),
            first_dims=tuple(round(float(value), 4) for value in vector[:8]),
        )

    def model_load_kwargs(self) -> dict[str, Any]:
        return {
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }

    def generation_kwargs(self, tokenizer: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.sample,
        }
        if tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        if self.config.sample:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p
        return kwargs

    def build_token_preview(self, token_ids: Any, tokenizer: Any) -> tuple[TokenPreview, ...]:
        preview: list[TokenPreview] = []
        for token_id in token_ids[: self.config.preview_tokens]:
            token_int = int(token_id)
            preview.append(TokenPreview(token_id=token_int, piece=tokenizer.decode([token_int])))
        return tuple(preview)

