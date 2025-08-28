# tests/test_text_processing_misc.py
from utils import text_processing


def test_normalize_for_id():
    assert text_processing._normalize_for_id(" Hello World! ") == "hello_world"
    assert text_processing._normalize_for_id(123) == "123"
    assert text_processing._normalize_for_id("The Wilderness") == "wilderness"


def test_normalize_trait_name():
    assert text_processing.normalize_trait_name(" Brave & Bold ") == "brave bold"


def test_normalize_text_for_matching():
    assert text_processing._normalize_text_for_matching(" 'Hello...' ") == "hello"
    assert text_processing._normalize_text_for_matching("") == ""


def test_is_fill_in_and_normalization_helpers():
    assert not text_processing._is_fill_in("test")
    assert text_processing._is_fill_in(text_processing.config.FILL_IN)


def test_get_text_segments_paragraph(monkeypatch):
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", None)
    text = "Para one.\n\nPara two."  # two paragraphs
    segments = text_processing.get_text_segments(text, "paragraph")
    assert len(segments) == 2
    assert segments[0][0] == "Para one."
    assert segments[1][0] == "Para two."


def test_get_text_segments_sentence_without_spacy(monkeypatch):
    monkeypatch.setattr(text_processing.spacy_manager, "_nlp", None)
    text = "First. Second?"
    segments = text_processing.get_text_segments(text, "sentence")
    assert [s[0] for s in segments] == ["First.", "Second?"]
