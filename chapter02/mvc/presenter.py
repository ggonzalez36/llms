from __future__ import annotations

from .model import ContextualResult, GenerationResult, SentenceResult, TokenPreview, TokenizerResult


class Chapter2Presenter:
    @staticmethod
    def render_run_context(device: str, sections: tuple[str, ...]) -> None:
        print(f"Device: {device}")
        print(f"Sections: {', '.join(sections)}")

    @staticmethod
    def render_section_error(section: str, error: Exception) -> None:
        print(f"[WARN] Section '{section}' failed: {error}")

    def render_generation(self, result: GenerationResult) -> None:
        self.print_header("Generation Demo")
        print(f"Model: {result.model_name}")
        print(f"Prompt token count: {result.prompt_token_count}")
        print("Prompt token ids (preview):")
        self.render_token_preview(result.prompt_preview)

        print("\nGenerated token ids (preview):")
        self.render_token_preview(result.generated_preview)

        print("\nGenerated text:")
        print(result.generated_text if result.generated_text else "[empty output]")

    def render_tokenizer_comparison(self, text: str, results: list[TokenizerResult], preview_tokens: int) -> None:
        self.print_header("Tokenizer Comparison")
        print("Input text:")
        print(text)

        for result in results:
            print(f"\n--- {result.model_name} ---")
            if result.error is not None:
                print(f"[WARN] Could not load tokenizer '{result.model_name}': {result.error}")
                continue
            print(f"Token count: {result.token_count}")
            for idx, token in enumerate(result.token_preview, start=1):
                piece = self.short_piece(token.piece)
                print(f"  {idx:>3}. id={token.token_id:>6}  piece={piece!r}")
            if result.omitted_count > 0:
                print(f"  ... ({result.omitted_count} additional tokens omitted)")
            elif result.token_count == 0:
                print(f"  ... (0 additional tokens omitted, preview size={preview_tokens})")

    def render_contextual(self, result: ContextualResult) -> None:
        self.print_header("Contextual Embedding Demo")
        print(f"Tokenizer: {result.tokenizer_name}")
        print(f"Model: {result.model_name}")
        print(f"Input text: {result.input_text!r}")
        print(f"Embedding tensor shape: {result.embedding_shape}")
        print("Token breakdown:")
        self.render_token_preview(result.token_preview)
        print("First 8 dims of first token embedding:")
        print(list(result.first_dims))

    def render_sentence(self, result: SentenceResult) -> None:
        self.print_header("Sentence Embedding Demo")
        if result.warning is not None:
            print(f"[WARN] {result.warning}")
            return
        print(f"Model: {result.model_name}")
        print(f"Input text: {result.input_text!r}")
        print(f"Embedding shape: {result.embedding_shape}")
        print("First 8 dims:")
        print(list(result.first_dims))

    def render_token_preview(self, preview: tuple[TokenPreview, ...]) -> None:
        for token in preview:
            piece = self.short_piece(token.piece)
            print(f"  {token.token_id:>6}  {piece!r}")

    @staticmethod
    def print_header(title: str) -> None:
        print(f"\n=== {title} ===\n")

    @staticmethod
    def short_piece(text: str, max_len: int = 60) -> str:
        cleaned = text.replace("\n", "\\n")
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3] + "..."
