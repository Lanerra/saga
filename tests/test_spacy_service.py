# tests/test_spacy_service.py
"""Unit tests for the SpacyService class."""

from unittest.mock import MagicMock

import pytest

from core.spacy_service import SpacyService


class MockToken:
    """Mock spaCy token for testing."""

    def __init__(self, text, lemma="", is_stop=False, is_punct=False, is_space=False):
        self.text = text
        self.lemma_ = lemma or text.lower()
        self.is_stop = is_stop
        self.is_punct = is_punct
        self.is_space = is_space


class MockEntity:
    """Mock spaCy entity for testing."""

    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class MockDoc:
    """Mock spaCy document for testing."""

    def __init__(self, text, entities=None, tokens=None):
        self.text = text
        self.ents = entities or []
        self._tokens = tokens or []

    def __iter__(self):
        return iter(self._tokens)


@pytest.fixture
def spacy_service():
    """Create a SpacyService instance for testing."""
    return SpacyService()


def test_not_loaded_before_first_use(spacy_service):
    """SpacyService does not load the model at construction time."""
    assert spacy_service.is_loaded() is False
    assert spacy_service.get_model_name() is None


def test_load_model_explicit(spacy_service):
    """Explicitly calling load_model loads the model."""
    result = spacy_service.load_model()
    assert result is True
    assert spacy_service.is_loaded() is True


def test_load_model_idempotent(spacy_service):
    """Calling load_model twice returns True and keeps the original model."""
    spacy_service.load_model()
    original_model_name = spacy_service.get_model_name()
    result = spacy_service.load_model("en_core_web_sm")
    assert result is True
    assert spacy_service.get_model_name() == original_model_name


def test_extract_entities_success(spacy_service):
    """Test successful entity extraction."""
    mock_nlp = MagicMock()
    mock_doc = MockDoc("John works at Google", entities=[MockEntity("John", "PERSON"), MockEntity("Google", "ORG")])
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    entities = spacy_service.extract_entities("John works at Google")

    assert len(entities) == 2
    assert ("John", "PERSON") in entities
    assert ("Google", "ORG") in entities


def test_extract_entities_no_model(spacy_service):
    """Test entity extraction when model is not loaded."""
    entities = spacy_service.extract_entities("Some text")

    assert entities == []


def test_verify_entity_presence_exact_match(spacy_service):
    """Test entity verification with exact match."""
    mock_nlp = MagicMock()
    # Mock for text processing
    text_doc = MockDoc("John works at Google", tokens=[MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")])
    # Mock for entity processing
    entity_doc = MockDoc("John", tokens=[MockToken("John")])

    def nlp_side_effect(text):
        if text == "John works at Google":
            return text_doc
        elif text == "John":
            return entity_doc
        return MockDoc(text)

    mock_nlp.side_effect = nlp_side_effect
    spacy_service._nlp = mock_nlp

    result = spacy_service.verify_entity_presence("John works at Google", "John")

    assert result is True


def test_verify_entity_presence_case_insensitive(spacy_service):
    """Test entity verification with case variations."""
    mock_nlp = MagicMock()
    # Mock for text processing
    text_doc = MockDoc("John works at Google", tokens=[MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")])
    # Mock for entity processing
    entity_doc = MockDoc("john", tokens=[MockToken("john")])

    def nlp_side_effect(text):
        if text == "John works at Google":
            return text_doc
        elif text == "john":
            return entity_doc
        return MockDoc(text)

    mock_nlp.side_effect = nlp_side_effect
    spacy_service._nlp = mock_nlp

    result = spacy_service.verify_entity_presence("John works at Google", "john")

    assert result is True


def test_verify_entity_presence_not_found(spacy_service):
    """Test entity verification when entity is not present."""
    mock_nlp = MagicMock()
    # Mock for text processing
    text_doc = MockDoc("John works at Google", tokens=[MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")])
    # Mock for entity processing
    entity_doc = MockDoc("Jane", tokens=[MockToken("Jane")])

    def nlp_side_effect(text):
        if text == "John works at Google":
            return text_doc
        elif text == "Jane":
            return entity_doc
        return MockDoc(text)

    mock_nlp.side_effect = nlp_side_effect
    spacy_service._nlp = mock_nlp

    result = spacy_service.verify_entity_presence("John works at Google", "Jane")

    assert result is False


def test_normalize_entity_name_success(spacy_service):
    """Test entity name normalization."""
    mock_nlp = MagicMock()
    # Mock tokens: "The Dark Tower" -> ["dark", "tower"] (removing stop words)
    mock_doc = MockDoc("The Dark Tower", tokens=[MockToken("The", is_stop=True), MockToken("Dark", lemma="dark"), MockToken("Tower", lemma="tower")])
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.normalize_entity_name("The Dark Tower")

    assert result == "dark tower"


def test_normalize_entity_name_fallback(spacy_service):
    """Test entity name normalization with model loaded."""
    # Model is now loaded in __init__, so this tests the normal path
    result = spacy_service.normalize_entity_name("  The Dark Tower  ")

    # Should return lemmatized form without stop words
    assert result == "dark tower"


def test_normalize_entity_name_empty(spacy_service):
    """Test entity name normalization with empty input."""
    mock_nlp = MagicMock()
    spacy_service._nlp = mock_nlp

    result = spacy_service.normalize_entity_name("")

    assert result == ""


def test_verify_entity_presence_fallback(spacy_service):
    """Test entity verification fallback when model not loaded."""
    result = spacy_service.verify_entity_presence("John works at Google", "John")

    assert result is True  # Should use substring matching fallback


def test_verify_entity_presence_fallback_not_found(spacy_service):
    """Test entity verification fallback when entity not found."""
    result = spacy_service.verify_entity_presence("John works at Google", "Jane")

    assert result is False


def test_extract_entities_empty_text(spacy_service):
    """Test entity extraction with empty text."""
    mock_nlp = MagicMock()
    spacy_service._nlp = mock_nlp

    result = spacy_service.extract_entities("")

    assert result == []


def test_extract_entities_invalid_input(spacy_service):
    """Test entity extraction with invalid input."""
    mock_nlp = MagicMock()
    spacy_service._nlp = mock_nlp

    result = spacy_service.extract_entities(123)  # type: ignore

    assert result == []


def test_normalize_entity_name_with_punctuation(spacy_service):
    """Test entity name normalization with punctuation."""
    mock_nlp = MagicMock()
    mock_doc = MockDoc(
        "'The Dark Tower'", tokens=[MockToken("'", is_punct=True), MockToken("The", is_stop=True), MockToken("Dark", lemma="dark"), MockToken("Tower", lemma="tower"), MockToken("'", is_punct=True)]
    )
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.normalize_entity_name("'The Dark Tower'")

    assert result == "dark tower"


def test_verify_entity_presence_with_threshold(spacy_service):
    """Test entity verification with custom threshold."""
    mock_nlp = MagicMock()
    # Mock for text processing
    text_doc = MockDoc("John works at Google", tokens=[MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")])
    # Mock for entity processing
    john_doc = MockDoc("John", tokens=[MockToken("John")])
    jane_doc = MockDoc("Jane", tokens=[MockToken("Jane")])

    def nlp_side_effect(text):
        if text == "John works at Google":
            return text_doc
        elif text == "John":
            return john_doc
        elif text == "Jane":
            return jane_doc
        return MockDoc(text)

    mock_nlp.side_effect = nlp_side_effect
    spacy_service._nlp = mock_nlp

    # Should pass with lower threshold
    result_low = spacy_service.verify_entity_presence("John works at Google", "John", threshold=0.5)
    assert result_low is True

    # Should fail with higher threshold (no overlap for partial match)
    result_high = spacy_service.verify_entity_presence("John works at Google", "Jane", threshold=0.9)
    assert result_high is False


def test_verify_entity_presence_partial_match(spacy_service):
    """Test entity verification with partial match of significant tokens."""
    mock_nlp = MagicMock()
    # Mock for text processing "Elias went home"
    text_doc = MockDoc("Elias went home", tokens=[MockToken("Elias"), MockToken("went"), MockToken("home")])
    # Mock for entity processing "Elias Thorne"
    entity_doc = MockDoc("Elias Thorne", tokens=[MockToken("Elias"), MockToken("Thorne")])

    def nlp_side_effect(text):
        if text == "Elias went home":
            return text_doc
        elif text == "Elias Thorne":
            return entity_doc
        return MockDoc(text)

    mock_nlp.side_effect = nlp_side_effect
    spacy_service._nlp = mock_nlp

    # Should pass even with high threshold because of "any significant token" logic
    result = spacy_service.verify_entity_presence("Elias went home", "Elias Thorne", threshold=0.9)
    assert result is True


def test_verify_entity_presence_common_title_exclusion(spacy_service):
    """Test that common titles are ignored during verification."""
    mock_nlp = MagicMock()
    # Mock for text processing "Mr. Jones went home"
    text_doc = MockDoc("Mr. Jones went home", tokens=[MockToken("Mr."), MockToken("Jones"), MockToken("went"), MockToken("home")])
    # Mock for entity processing "Mr. Smith"
    entity_doc = MockDoc("Mr. Smith", tokens=[MockToken("Mr."), MockToken("Smith")])

    def nlp_side_effect(text):
        if text == "Mr. Jones went home":
            return text_doc
        elif text == "Mr. Smith":
            return entity_doc
        return MockDoc(text)

    mock_nlp.side_effect = nlp_side_effect
    spacy_service._nlp = mock_nlp

    # "Mr." should be ignored in entity, "Smith" is looked for.
    # "Smith" is not in text.
    result = spacy_service.verify_entity_presence("Mr. Jones went home", "Mr. Smith")
    assert result is False
