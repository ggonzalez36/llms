from __future__ import annotations

from typing import Callable

from .config import Chapter2Config
from .model import Chapter2Model
from .presenter import Chapter2Presenter


class Chapter2Controller:
    def __init__(self, config: Chapter2Config) -> None:
        self.config = config
        self.model = Chapter2Model(config)
        self.presenter = Chapter2Presenter()

    def run(self) -> None:
        self.presenter.render_run_context(self.model.device, self.config.sections)

        handlers: dict[str, Callable[[], None]] = {
            "generation": self.run_generation,
            "tokenizers": self.run_tokenizers,
            "contextual": self.run_contextual,
            "sentence": self.run_sentence,
        }

        for section in self.config.sections:
            try:
                handlers[section]()
            except Exception as exc:
                self.presenter.render_section_error(section, exc)

    def run_generation(self) -> None:
        result = self.model.generation_demo()
        self.presenter.render_generation(result)

    def run_tokenizers(self) -> None:
        results = self.model.tokenizer_comparison()
        self.presenter.render_tokenizer_comparison(self.config.text, results, self.config.preview_tokens)

    def run_contextual(self) -> None:
        result = self.model.contextual_embedding_demo()
        self.presenter.render_contextual(result)

    def run_sentence(self) -> None:
        result = self.model.sentence_embedding_demo()
        self.presenter.render_sentence(result)
