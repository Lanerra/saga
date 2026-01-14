# core/spacy_service.py
"""SpaCy-based NLP service for entity extraction and text processing.

This module provides a centralized SpacyService that:
- Loads and manages the spaCy model
- Extracts entities from text
- Verifies entity presence in text
- Normalizes entity names for deduplication
- Handles lemmatization and fuzzy matching
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    import spacy


class SpacyService:
    """Centralized service for spaCy-based NLP operations."""

    def __init__(self) -> None:
        """Initialize the SpacyService with eager model loading."""
        self._nlp: Any | None = None
        self._model_name: str | None = None
        # Load model eagerly to ensure it's available when needed
        self.load_model()

    def load_model(self, model_name: str | None = None) -> bool:
        """Load the spaCy model if not already loaded.

        Args:
            model_name: Optional model name override. If None, uses config.SPACY_MODEL or defaults to 'en_core_web_sm'.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._nlp is not None:
            return True

        try:
            import spacy
        except ImportError:
            logger.error("spaCy library not installed. Install with: pip install spacy")
            return False

        # Determine model name
        if model_name:
            selected_model = model_name
        else:
            try:
                import config
                selected_model = getattr(config.settings, "SPACY_MODEL", None)
            except Exception:
                selected_model = None

            if not selected_model:
                selected_model = "en_core_web_sm"

        self._model_name = selected_model

        try:
            self._nlp = spacy.load(selected_model)
            logger.info("SpacyService: spaCy model '%s' loaded successfully", selected_model)
            return True
        except OSError as e:
            logger.error(
                "spaCy model '%s' not found. Install with: python -m spacy download %s. "
                "spaCy-dependent features will be disabled.",
                selected_model,
                selected_model,
            )
            return False
        except Exception as e:
            logger.error("Failed to load spaCy model '%s': %s", selected_model, e)
            return False

    def is_loaded(self) -> bool:
        """Check if spaCy model is loaded.

        Returns:
            True if model is loaded and available, False otherwise.
        """
        return self._nlp is not None

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """Extract named entities from text.

        Args:
            text: Input text to analyze.

        Returns:
            List of (entity_text, entity_label) tuples. Empty list if model not loaded or text is empty.
        """
        if not self.is_loaded():
            logger.warning("extract_entities: spaCy model not loaded")
            return []

        if not text or not isinstance(text, str):
            return []

        try:
            doc = self._nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append((ent.text, ent.label_))
            return entities
        except Exception as e:
            logger.error("extract_entities failed: %s", e, exc_info=True)
            return []

    def verify_entity_presence(self, text: str, entity_name: str, threshold: float = 0.7) -> bool:
        """Verify if an entity is present in the text using fuzzy matching.

        This method handles various forms of entity mentions including:
        - Exact matches
        - Case variations
        - Lemmatization (e.g., "guards" matches "guard")
        - Partial matches with context (ANY significant token presence)

        Args:
            text: The source text to search in.
            entity_name: The entity name to verify.
            threshold: Similarity threshold (0.0 to 1.0). Default 0.7.
                       Note: Updated logic checks for ANY significant token presence,
                       effectively behaving as a loose match for multi-word entities.

        Returns:
            True if entity is likely present, False otherwise.
        """
        if not self.is_loaded():
            logger.warning("verify_entity_presence: spaCy model not loaded, using fallback")
            # Fallback to simple substring matching
            return entity_name.lower() in text.lower()

        if not text or not isinstance(text, str) or not entity_name or not isinstance(entity_name, str):
            return False

        try:
            doc = self._nlp(text)
            entity_doc = self._nlp(entity_name)

            # Check for exact match first (case-insensitive)
            if entity_name.lower() in text.lower():
                return True

            # Identify significant tokens in the entity name
            # Exclude stopwords, punctuation, and common titles
            common_titles = {
                "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.",
                "prof", "prof.", "sir", "lady", "lord", "captain", "cpt", "cpt."
            }

            sig_entity_tokens = set()
            for token in entity_doc:
                txt = token.text.lower()
                if not token.is_stop and not token.is_punct and txt not in common_titles:
                    sig_entity_tokens.add(txt)

            # If we filtered everything out (e.g. name was just "The Doctor" and Doctor is stopword/title?),
            # fallback to using tokens that are just not punctuation
            if not sig_entity_tokens:
                sig_entity_tokens = {t.text.lower() for t in entity_doc if not t.is_punct}

            text_tokens = {token.text.lower() for token in doc}

            # Check overlap of significant tokens
            # This logic allows "Elias" to match "Elias Thorne" (valid partial match)
            if sig_entity_tokens & text_tokens:
                return True

            # Check lemmatized forms of significant tokens
            sig_entity_lemmas = set()
            for token in entity_doc:
                 txt = token.text.lower()
                 if not token.is_stop and not token.is_punct and txt not in common_titles:
                     sig_entity_lemmas.add(token.lemma_.lower())

            text_lemmas = {token.lemma_.lower() for token in doc}

            if sig_entity_lemmas & text_lemmas:
                return True

            return False
        except Exception as e:
            logger.error("verify_entity_presence failed: %s", e, exc_info=True)
            # Fallback to simple substring matching on error
            return entity_name.lower() in text.lower()

    def normalize_entity_name(self, name: str) -> str:
        """Normalize an entity name for deduplication.

        This produces a canonical form by:
        1. Converting to lowercase
        2. Lemmatizing (e.g., "guards" -> "guard")
        3. Removing common articles ("the", "a", "an")
        4. Normalizing whitespace and punctuation

        Args:
            name: The entity name to normalize.

        Returns:
            Normalized canonical form of the name.
        """
        if not self.is_loaded():
            logger.warning("normalize_entity_name: spaCy model not loaded, using fallback")
            # Fallback to simple normalization
            import re
            name = name.strip().lower()
            name = re.sub(r"^[\s'\"()]*", "", name)
            name = re.sub(r"[\s'\"()]*$", "", name)
            name = re.sub(r"\s+", " ", name)
            return name

        if not name or not isinstance(name, str):
            return ""

        try:
            doc = self._nlp(name)

            # Get lemmas and filter out stop words and punctuation
            tokens = []
            for token in doc:
                if not token.is_stop and not token.is_punct and not token.is_space:
                    tokens.append(token.lemma_.lower())

            # Join with single spaces
            normalized = " ".join(tokens)
            return normalized
        except Exception as e:
            logger.error("normalize_entity_name failed: %s", e, exc_info=True)
            # Fallback to simple normalization on error
            import re
            name = name.strip().lower()
            name = re.sub(r"^[\s'\"()]*", "", name)
            name = re.sub(r"[\s'\"()]*$", "", name)
            name = re.sub(r"\s+", " ", name)
            return name

    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """Clean text using spaCy for advanced processing.

        This method provides NLP-based text cleaning including:
        - Normalization of whitespace and punctuation
        - Removal of stop words (optional)
        - Lemmatization
        - Sentence boundary detection and normalization

        Args:
            text: Input text to clean.
            aggressive: If True, remove stop words and excessive punctuation.
                      If False, only normalize whitespace and basic punctuation.

        Returns:
            Cleaned text. Returns original text on error or if model not loaded.
        """
        if not self.is_loaded():
            logger.warning("clean_text: spaCy model not loaded, using fallback")
            # Fallback to simple regex-based cleaning
            import re
            cleaned = text.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'[\t\r\f\v]', ' ', cleaned)
            return cleaned

        if not text or not isinstance(text, str):
            return ""

        try:
            doc = self._nlp(text)

            if aggressive:
                # Aggressive cleaning: remove stop words and excessive punctuation
                tokens = []
                for token in doc:
                    # Skip stop words, punctuation, and whitespace
                    if not token.is_stop and not token.is_punct and not token.is_space:
                        tokens.append(token.lemma_.lower())
                cleaned = " ".join(tokens)
            else:
                # Conservative cleaning: normalize whitespace and basic punctuation
                # Reconstruct text from tokens but preserve newlines
                cleaned_parts = []
                for token in doc:
                    if not token.is_space:
                        # Add space before non-punctuation tokens if we have content already
                        # and the previous token didn't end with a newline
                        if cleaned_parts and not token.is_punct:
                            if not cleaned_parts[-1].endswith('\n') and not cleaned_parts[-1].endswith(' '):
                                cleaned_parts.append(' ')
                        cleaned_parts.append(token.text)
                    elif '\n' in token.text:
                         # Preserve newlines (normalize multiple newlines to max 2)
                        newlines = token.text.count('\n')
                        if newlines >= 2:
                            cleaned_parts.append('\n\n')
                        else:
                            cleaned_parts.append('\n')
                    elif cleaned_parts and not cleaned_parts[-1].endswith(' ') and not cleaned_parts[-1].endswith('\n'):
                        # Add single space for other whitespace tokens if needed
                        cleaned_parts.append(' ')
                
                cleaned = "".join(cleaned_parts)
                
                # Normalize common punctuation patterns but preserve newlines
                import re
                cleaned = re.sub(r'[\t\r\f\v]', ' ', cleaned)
                # Collapse multiple spaces but keep newlines
                cleaned = re.sub(r'[ ]+', ' ', cleaned).strip()

            return cleaned
        except Exception as e:
            logger.error("clean_text failed: %s", e, exc_info=True)
            # Fallback to simple cleaning on error
            import re
            cleaned = text.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = re.sub(r'[\t\r\f\v]', ' ', cleaned)
            return cleaned

    def extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text using spaCy's sentence boundary detection.

        Args:
            text: Input text to process.

        Returns:
            List of sentences. Empty list on error or if model not loaded.
        """
        if not self.is_loaded():
            logger.warning("extract_sentences: spaCy model not loaded, using fallback")
            # Fallback to simple regex-based sentence splitting
            import re
            sentences = []
            for match in re.finditer(r'([^\.!?]+(?:[\.!?]|$))', text):
                sent_text = match.group(1).strip()
                if sent_text:
                    sentences.append(sent_text)
            return sentences

        if not text or not isinstance(text, str):
            return []

        try:
            doc = self._nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        except Exception as e:
            logger.error("extract_sentences failed: %s", e, exc_info=True)
            # Fallback to simple sentence splitting
            import re
            sentences = []
            for match in re.finditer(r'([^\.!?]+(?:[\.!?]|$))', text):
                sent_text = match.group(1).strip()
                if sent_text:
                    sentences.append(sent_text)
            return sentences

    def get_model_name(self) -> str | None:
        """Get the name of the loaded model.

        Returns:
            Model name if loaded, None otherwise.
        """
        return self._model_name
