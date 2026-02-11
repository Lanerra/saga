# tests/test_text_cleanup_with_spacy.py
"""Tests for spaCy-based text cleanup functionality."""

from unittest.mock import MagicMock, patch

import pytest

from core.spacy_service import SpacyService
from core.text_processing_service import TextProcessingService


class MockToken:
    """Mock spaCy token for testing."""

    def __init__(self, text, lemma="", is_stop=False, is_punct=False, is_space=False):
        self.text = text
        self.lemma_ = lemma or text.lower()
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.is_space = is_space

    def endswith(self, suffix):
        """Mock endswith method."""
        return self.text.endswith(suffix)


class MockSent:
    """Mock spaCy sentence for testing."""

    def __init__(self, text):
        self.text = text


class MockDoc:
    """Mock spaCy document for testing."""

    def __init__(self, text, tokens=None, sentences=None):
        self.text = text
        self._tokens = tokens or []
        self._sents = sentences or []

    def __iter__(self):
        return iter(self._tokens)

    @property
    def sents(self):
        return self._sents


@pytest.fixture
def spacy_service():
    """Create a SpacyService instance for testing."""
    return SpacyService()


@pytest.fixture
def text_processing_service():
    """Create a TextProcessingService instance for testing."""
    return TextProcessingService()


def test_clean_text_conservative_with_spacy(spacy_service):
    """Test conservative text cleaning with spaCy."""
    mock_nlp = MagicMock()
    # Mock tokens: "Hello, world!  This is a test..."
    mock_doc = MockDoc("Hello, world!  This is a test...")
    mock_doc._tokens = [
        MockToken("Hello", lemma="hello"),
        MockToken(",", is_punct=True),
        MockToken("world", lemma="world"),
        MockToken("!", is_punct=True),
        MockToken("  ", is_space=True),
        MockToken("This", lemma="this", is_stop=True),
        MockToken("is", lemma="be", is_stop=True),
        MockToken("a", lemma="a", is_stop=True),
        MockToken("test", lemma="test"),
        MockToken(".", is_punct=True),
    ]
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.clean_text("Hello, world!  This is a test...")

    # Conservative cleaning should preserve content but normalize spaces
    # Note: the ellipsis at the end is treated as punctuation and preserved
    assert result == "Hello, world! This is a test."


def test_clean_text_aggressive_with_spacy(spacy_service):
    """Test aggressive text cleaning with spaCy (removes stop words)."""
    mock_nlp = MagicMock()
    # Mock tokens: "Hello, world!  This is a test..."
    mock_doc = MockDoc("Hello, world!  This is a test...")
    mock_doc._tokens = [
        MockToken("Hello", lemma="hello"),
        MockToken(",", is_punct=True),
        MockToken("world", lemma="world"),
        MockToken("!", is_punct=True),
        MockToken("  ", is_space=True),
        MockToken("This", lemma="this", is_stop=True),
        MockToken("is", lemma="be", is_stop=True),
        MockToken("a", lemma="a", is_stop=True),
        MockToken("test", lemma="test"),
        MockToken(".", is_punct=True),
    ]
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.clean_text("Hello, world!  This is a test...", aggressive=True)

    # Aggressive cleaning should remove stop words and punctuation, lemmatize
    assert result == "hello world test"


def test_clean_text_fallback(spacy_service):
    """Test text cleaning fallback when spaCy not loaded."""
    result = spacy_service.clean_text("  Hello   world!  ")

    # Should use regex-based fallback
    assert result == "Hello world!"


def test_clean_text_empty_input(spacy_service):
    """Test text cleaning with empty input."""
    mock_nlp = MagicMock()
    spacy_service._nlp = mock_nlp

    result = spacy_service.clean_text("")
    assert result == ""


def test_clean_text_invalid_input(spacy_service):
    """Test text cleaning with invalid input."""
    mock_nlp = MagicMock()
    spacy_service._nlp = mock_nlp

    result = spacy_service.clean_text(123)  # type: ignore
    assert result == ""


def test_extract_sentences_with_spacy(spacy_service):
    """Test sentence extraction with spaCy."""
    mock_nlp = MagicMock()
    mock_doc = MockDoc("Hello world. This is a test.")
    mock_doc._sents = [MockSent("Hello world."), MockSent("This is a test.")]
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.extract_sentences("Hello world. This is a test.")

    assert len(result) == 2
    assert "Hello world." in result
    assert "This is a test." in result


def test_extract_sentences_fallback(spacy_service):
    """Test sentence extraction fallback when spaCy not loaded."""
    result = spacy_service.extract_sentences("Hello world. This is a test.")

    # Should use regex-based fallback
    assert len(result) == 2
    assert "Hello world." in result
    assert "This is a test." in result


def test_extract_sentences_empty_input(spacy_service):
    """Test sentence extraction with empty input."""
    mock_nlp = MagicMock()
    spacy_service._nlp = mock_nlp

    result = spacy_service.extract_sentences("")
    assert result == []


def test_extract_sentences_invalid_input(spacy_service):
    """Test sentence extraction with invalid input."""
    mock_nlp = MagicMock()
    spacy_service._nlp = mock_nlp

    result = spacy_service.extract_sentences(123)  # type: ignore
    assert result == []


def test_text_processing_service_clean_text(text_processing_service):
    """Test TextProcessingService clean_text_with_spacy method."""
    with patch.object(text_processing_service.spacy_service, "clean_text") as mock_clean:
        mock_clean.return_value = "cleaned text"

        result = text_processing_service.clean_text_with_spacy("dirty text")

        assert result == "cleaned text"
        mock_clean.assert_called_once_with("dirty text", False)


def test_text_processing_service_extract_sentences(text_processing_service):
    """Test TextProcessingService extract_sentences_with_spacy method."""
    with patch.object(text_processing_service.spacy_service, "extract_sentences") as mock_extract:
        mock_extract.return_value = ["sentence 1", "sentence 2"]

        result = text_processing_service.extract_sentences_with_spacy("text with sentences")

        assert result == ["sentence 1", "sentence 2"]
        mock_extract.assert_called_once_with("text with sentences")


def test_module_level_clean_text():
    """Test module-level clean_text_with_spacy function."""
    fake_service = MagicMock()
    fake_service.clean_text.return_value = "cleaned text"

    with patch("core.spacy_service._singleton", fake_service):
        from core.text_processing_service import clean_text_with_spacy

        result = clean_text_with_spacy("dirty text")

        assert result == "cleaned text"
        fake_service.clean_text.assert_called_once_with("dirty text", False)


def test_module_level_extract_sentences():
    """Test module-level extract_sentences_with_spacy function."""
    fake_service = MagicMock()
    fake_service.extract_sentences.return_value = ["sentence 1", "sentence 2"]

    with patch("core.spacy_service._singleton", fake_service):
        from core.text_processing_service import extract_sentences_with_spacy

        result = extract_sentences_with_spacy("text with sentences")

        assert result == ["sentence 1", "sentence 2"]
        fake_service.extract_sentences.assert_called_once_with("text with sentences")


def test_clean_text_with_special_characters(spacy_service):
    """Test text cleaning with special characters and tabs."""
    mock_nlp = MagicMock()
    mock_doc = MockDoc("Hello\tworld\r\nwith\t\ttabs")
    mock_doc._tokens = [
        MockToken("Hello", lemma="hello"),
        MockToken("\t", is_space=True),
        MockToken("world", lemma="world"),
        MockToken("\r\n", is_space=True),
        MockToken("with", lemma="with", is_stop=True),
        MockToken("\t", is_space=True),
        MockToken("\t", is_space=True),
        MockToken("tabs", lemma="tab"),
    ]
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.clean_text("Hello\tworld\r\nwith\t\ttabs")

    # Should normalize whitespace but preserve content
    assert "\t" not in result
    assert "\r" not in result
    assert "\n" not in result
    assert result == "Hello world with tabs"


def test_clean_text_aggressive_with_punctuation(spacy_service):
    """Test aggressive cleaning removes punctuation."""
    mock_nlp = MagicMock()
    mock_doc = MockDoc("'Hello, world!'")
    mock_doc._tokens = [
        MockToken("'", is_punct=True),
        MockToken("Hello", lemma="hello"),
        MockToken(",", is_punct=True),
        MockToken("world", lemma="world"),
        MockToken("!", is_punct=True),
        MockToken("'", is_punct=True),
    ]
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.clean_text("'Hello, world!'", aggressive=True)

    # Aggressive cleaning should remove all punctuation
    assert result == "hello world"
