# tests/test_spacy_service.py
"""Unit tests for the SpacyService class."""

import pytest
from unittest.mock import MagicMock, patch

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


def test_load_model_success(spacy_service):
    """Test successful model loading."""
    with patch('builtins.__import__', return_value=MagicMock()) as mock_import:
        # Configure the mock to return a spaCy-like module
        mock_spacy = MagicMock()
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp
        mock_import.side_effect = lambda name, *args, **kwargs: mock_spacy if name == 'spacy' else None

        result = spacy_service.load_model("en_core_web_sm")
        
        assert result is True
        assert spacy_service.is_loaded() is True
        assert spacy_service.get_model_name() == "en_core_web_sm"
        mock_spacy.load.assert_called_once_with("en_core_web_sm")


def test_load_model_failure(spacy_service):
    """Test model loading failure."""
    with patch('builtins.__import__', return_value=MagicMock()) as mock_import:
        # Configure the mock to return a spaCy-like module
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("Model not found")
        mock_import.side_effect = lambda name, *args, **kwargs: mock_spacy if name == 'spacy' else None

        result = spacy_service.load_model("nonexistent_model")
        
        assert result is False
        assert spacy_service.is_loaded() is False


def test_load_model_import_error(spacy_service):
    """Test when spaCy is not installed."""
    # Remove the mock to simulate ImportError
    with patch.dict('sys.modules', {'spacy': None}):
        result = spacy_service.load_model("en_core_web_sm")
        assert result is False
        assert spacy_service.is_loaded() is False


def test_extract_entities_success(spacy_service):
    """Test successful entity extraction."""
    mock_nlp = MagicMock()
    mock_doc = MockDoc("John works at Google", entities=[
        MockEntity("John", "PERSON"),
        MockEntity("Google", "ORG")
    ])
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
    text_doc = MockDoc("John works at Google", tokens=[
        MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")
    ])
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
    text_doc = MockDoc("John works at Google", tokens=[
        MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")
    ])
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
    text_doc = MockDoc("John works at Google", tokens=[
        MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")
    ])
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
    mock_doc = MockDoc("The Dark Tower", tokens=[
        MockToken("The", is_stop=True),
        MockToken("Dark", lemma="dark"),
        MockToken("Tower", lemma="tower")
    ])
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.normalize_entity_name("The Dark Tower")
    
    assert result == "dark tower"


def test_normalize_entity_name_fallback(spacy_service):
    """Test entity name normalization fallback when model not loaded."""
    result = spacy_service.normalize_entity_name("  The Dark Tower  ")
    
    # Should return lowercase with stripped whitespace
    assert result == "the dark tower"


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
    mock_doc = MockDoc("'The Dark Tower'", tokens=[
        MockToken("'", is_punct=True),
        MockToken("The", is_stop=True),
        MockToken("Dark", lemma="dark"),
        MockToken("Tower", lemma="tower"),
        MockToken("'", is_punct=True)
    ])
    mock_nlp.return_value = mock_doc
    spacy_service._nlp = mock_nlp

    result = spacy_service.normalize_entity_name("'The Dark Tower'")
    
    assert result == "dark tower"


def test_verify_entity_presence_with_threshold(spacy_service):
    """Test entity verification with custom threshold."""
    mock_nlp = MagicMock()
    # Mock for text processing
    text_doc = MockDoc("John works at Google", tokens=[
        MockToken("John"), MockToken("works"), MockToken("at"), MockToken("Google")
    ])
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