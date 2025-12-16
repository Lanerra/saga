# tests/test_text_processing_misc.py
import pytest

import config as _config
from utils import text_processing
from utils.common import _is_fill_in as _fill_in


def test_normalize_for_id() -> None:
    assert text_processing._normalize_for_id(" Hello World! ") == "hello_world"
    # Ensure input is string
    assert text_processing._normalize_for_id(str(123)) == "123"
    assert text_processing._normalize_for_id("The Wilderness") == "wilderness"


def test_normalize_trait_name() -> None:
    assert text_processing.normalize_trait_name(" Brave & Bold ") == "brave-bold"


def test_normalize_text_for_matching() -> None:
    assert text_processing._normalize_text_for_matching(" 'Hello...' ") == "hello"
    assert text_processing._normalize_text_for_matching("") == ""


def test_is_fill_in_and_normalization_helpers() -> None:
    assert not _fill_in("test")
    # Use config settings directly rather than a nested import
    assert _fill_in(_config.settings.FILL_IN)


def test_get_text_segments_paragraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", None)
    text = "Para one.\n\nPara two."  # two paragraphs
    segments = text_processing.get_text_segments(text, "paragraph")
    assert len(segments) == 2
    assert segments[0][0] == "Para one."
    assert segments[1][0] == "Para two."


def test_get_text_segments_sentence_without_spacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", None)
    text = "First. Second?"
    segments = text_processing.get_text_segments(text, "sentence")
    assert [s[0] for s in segments] == ["First.", "Second?"]
