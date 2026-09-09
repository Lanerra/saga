# tests/test_text_processing_offsets.py
from typing import Any

import pytest

from core.service_context import get_services
from utils import text_processing


class DummySpan:
    def __init__(self, text: str, start: int, end: int) -> None:
        self.text = text
        self.start_char = start
        self.end_char = end


class DummyNLP:
    def __call__(self, text: str) -> "Any":
        class Doc:
            def __init__(self, t: str) -> None:
                self.sents = [DummySpan(t, 0, len(t))]

        return Doc(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("document_vector, expected", [([1.0, 0.0], (0, 3, 0, 3)), ([0.0, 1.0], None)])
async def test_find_quote_offsets_no_model(monkeypatch: pytest.MonkeyPatch, document_vector: list[float], expected: tuple[int, int, int, int] | None) -> None:
    monkeypatch.setattr(text_processing, "_get_spacy_nlp", lambda: None)
    async def embedding(text: str) -> list[float]:
        assert text in {"doc", "quote"}
        return [1.0, 0.0] if text == "quote" else document_vector

    monkeypatch.setattr(get_services().language_model, 'async_get_embedding', embedding)
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("doc", "quote")
    assert result == expected


@pytest.mark.asyncio
async def test_find_quote_offsets_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_processing, "_get_spacy_nlp", lambda: DummyNLP())
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("Hello world", "world")
    assert result == (6, 11, 0, len("Hello world"))


@pytest.mark.asyncio
async def test_find_quote_offsets_fuzzy_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_processing, "_get_spacy_nlp", lambda: DummyNLP())
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("Hello world.", "Hello world!")
    assert result == (0, 11, 0, len("Hello world."))


@pytest.mark.asyncio
async def test_find_quote_offsets_fuzzy_extra_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_processing, "_get_spacy_nlp", lambda: DummyNLP())
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("Hello world.", "Hello world again")
    assert result == (0, 12, 0, len("Hello world."))


@pytest.mark.asyncio
async def test_find_quote_offsets_token_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_processing, "_get_spacy_nlp", lambda: DummyNLP())

    class DummyAlign:
        def __init__(self) -> None:
            self.score = 0.0
            self.dest_start = 0
            self.dest_end = 0

    monkeypatch.setattr(
        text_processing,
        "partial_ratio_alignment",
        lambda *_args, **_kwargs: DummyAlign(),
    )
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy(
        "The quick brown fox jumps over the lazy dog.",
        "Fast brown fox jumps over sleepy dog.",
    )
    assert result == (
        0,
        len("The quick brown fox jumps over the lazy dog."),
        0,
        len("The quick brown fox jumps over the lazy dog."),
    )
