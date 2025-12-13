# tests/test_text_processing_offsets.py
from typing import Any

import pytest

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
async def test_find_quote_offsets_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", None)
    monkeypatch.setattr(text_processing.spacy_manager, "load", lambda: None)
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("doc", "quote")
    assert result is None


@pytest.mark.asyncio
async def test_find_quote_offsets_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", DummyNLP())
    monkeypatch.setattr(text_processing.spacy_manager, "load", lambda: None)
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("Hello world", "world")
    assert result == (6, 11, 0, len("Hello world"))


@pytest.mark.asyncio
async def test_find_quote_offsets_fuzzy_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", DummyNLP())
    monkeypatch.setattr(text_processing.spacy_manager, "load", lambda: None)
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("Hello world.", "Hello world!")
    assert result == (0, 11, 0, len("Hello world."))


@pytest.mark.asyncio
async def test_find_quote_offsets_fuzzy_extra_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", DummyNLP())
    monkeypatch.setattr(text_processing.spacy_manager, "load", lambda: None)
    result = await text_processing.find_quote_and_sentence_offsets_with_spacy("Hello world.", "Hello world again")
    assert result == (0, 12, 0, len("Hello world."))


@pytest.mark.asyncio
async def test_find_quote_offsets_token_similarity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", DummyNLP())
    monkeypatch.setattr(text_processing.spacy_manager, "load", lambda: None)

    # Force partial_ratio_alignment to fail
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
