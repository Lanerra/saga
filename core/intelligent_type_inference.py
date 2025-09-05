# core/intelligent_type_inference.py
"""
ML-inspired intelligent type inference system for SAGA knowledge graph.

This module learns patterns from existing data to intelligently infer node types,
replacing static keyword matching with adaptive pattern recognition.
"""

import asyncio
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class IntelligentTypeInference:
    """ML-inspired type inference using pattern learning from existing data."""

    def __init__(self, schema_introspector):
        self.introspector = schema_introspector
        self.learned_patterns = defaultdict(Counter)
        self.confidence_threshold = 0.7
        self.min_pattern_frequency = 3
        self.last_learning_update = None
        self._learning_lock = asyncio.Lock()

    async def learn_from_existing_data(self, sample_size: int = 5000):
        """Learn patterns from existing database entities."""
        async with self._learning_lock:
            try:
                logger.info(
                    f"Learning type patterns from up to {sample_size} existing entities..."
                )

                # Get sample data from introspector
                samples_by_label = await self.introspector.sample_node_properties(
                    sample_size
                )

                if not samples_by_label:
                    logger.warning("No sample data available for learning")
                    return

                # Reset learned patterns
                self.learned_patterns.clear()
                total_learned = 0

                # Extract patterns from each label's samples
                for label, samples in samples_by_label.items():
                    for sample in samples:
                        self._extract_patterns(
                            name=sample.get("name", ""),
                            category=sample.get("category", ""),
                            description=sample.get("description", ""),
                            type_property=sample.get("type_property", ""),
                            actual_label=label,
                        )
                        total_learned += 1

                # Clean up low-frequency patterns
                self._cleanup_patterns()

                self.last_learning_update = datetime.utcnow()

                pattern_count = sum(
                    len(counters) for counters in self.learned_patterns.values()
                )
                logger.info(
                    f"Learned {pattern_count} patterns from {total_learned} entities across {len(samples_by_label)} labels"
                )

            except Exception as e:
                logger.error(f"Failed to learn from existing data: {e}", exc_info=True)

    def _extract_patterns(
        self,
        name: str,
        category: str,
        description: str,
        type_property: str,
        actual_label: str,
    ):
        """Extract meaningful patterns for type inference."""
        if not name or not actual_label:
            return

        name_lower = name.lower().strip()

        # Skip very short or generic names
        if len(name_lower) < 2:
            return

        # Word-based patterns (most reliable)
        words = re.findall(r"\b\w{2,}\b", name_lower)  # At least 2 characters
        for word in words:
            if len(word) >= 3:  # Focus on meaningful words
                self.learned_patterns[f"word:{word}"][actual_label] += (
                    2  # Higher weight
                )

        # Prefix/suffix patterns for shorter names
        if 3 <= len(name_lower) <= 15:  # Reasonable name length
            if len(name_lower) >= 4:
                prefix = name_lower[:3]
                suffix = name_lower[-3:]
                self.learned_patterns[f"prefix:{prefix}"][actual_label] += 1
                self.learned_patterns[f"suffix:{suffix}"][actual_label] += 1

        # Category patterns (high reliability when available)
        if category and category.strip():
            clean_category = category.lower().strip()
            self.learned_patterns[f"category:{clean_category}"][actual_label] += (
                3  # High weight
            )

            # Also extract category words
            cat_words = re.findall(r"\b\w{3,}\b", clean_category)
            for word in cat_words:
                self.learned_patterns[f"cat_word:{word}"][actual_label] += 2

        # Type property patterns (if available)
        if type_property and type_property.strip():
            clean_type = type_property.lower().strip()
            self.learned_patterns[f"type_prop:{clean_type}"][actual_label] += 2

        # Description keywords (lower weight, first few words only)
        if description and description.strip():
            desc_words = re.findall(r"\b\w{4,}\b", description.lower())[
                :5
            ]  # First 5 meaningful words
            for word in desc_words:
                if word not in [
                    "this",
                    "that",
                    "with",
                    "from",
                    "they",
                    "have",
                    "been",
                    "were",
                    "will",
                ]:
                    self.learned_patterns[f"desc_word:{word}"][actual_label] += 1

        # Character-based patterns for unique identifiers
        if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)*$", name):  # CamelCase
            self.learned_patterns["pattern:camelcase"][actual_label] += 1
        elif "_" in name_lower:
            self.learned_patterns["pattern:underscore"][actual_label] += 1
        elif "-" in name_lower:
            self.learned_patterns["pattern:hyphenated"][actual_label] += 1

    def _cleanup_patterns(self):
        """Remove low-frequency patterns to improve accuracy."""
        patterns_to_remove = []

        for pattern_key, label_counts in self.learned_patterns.items():
            # Remove patterns with very low frequency
            total_count = sum(label_counts.values())
            if total_count < self.min_pattern_frequency:
                patterns_to_remove.append(pattern_key)
                continue

            # Remove labels with very low frequency within a pattern
            labels_to_remove = []
            for label, count in label_counts.items():
                if (
                    count < 2 and total_count > 10
                ):  # Remove outliers in high-frequency patterns
                    labels_to_remove.append(label)

            for label in labels_to_remove:
                del label_counts[label]

        # Remove empty patterns
        for pattern_key in patterns_to_remove:
            del self.learned_patterns[pattern_key]

        logger.debug(f"Cleaned up {len(patterns_to_remove)} low-frequency patterns")

    def infer_type(
        self, name: str, category: str = "", description: str = ""
    ) -> tuple[str, float]:
        """Infer node type with confidence score using learned patterns."""
        if not name or not name.strip():
            return "Entity", 0.0

        if not self.learned_patterns:
            logger.debug(
                "No learned patterns available, learning disabled or not initialized"
            )
            return "Entity", 0.0

        name = name.strip()
        name_lower = name.lower()
        category = category.strip() if category else ""
        description = description.strip() if description else ""

        scores = defaultdict(float)
        total_weight = 0.0

        # Category patterns (highest priority)
        if category:
            clean_category = category.lower()
            category_key = f"category:{clean_category}"
            if category_key in self.learned_patterns:
                self._add_pattern_scores(category_key, scores, weight=4.0)
                total_weight += 4.0

            # Category words
            cat_words = re.findall(r"\b\w{3,}\b", clean_category)
            for word in cat_words:
                cat_word_key = f"cat_word:{word}"
                if cat_word_key in self.learned_patterns:
                    self._add_pattern_scores(cat_word_key, scores, weight=2.5)
                    total_weight += 2.5

        # Word patterns from name (high priority)
        words = re.findall(r"\b\w{3,}\b", name_lower)
        for word in words:
            word_key = f"word:{word}"
            if word_key in self.learned_patterns:
                self._add_pattern_scores(word_key, scores, weight=2.0)
                total_weight += 2.0

        # Prefix/suffix patterns (medium priority)
        if 4 <= len(name_lower) <= 15:
            prefix_key = f"prefix:{name_lower[:3]}"
            suffix_key = f"suffix:{name_lower[-3:]}"

            if prefix_key in self.learned_patterns:
                self._add_pattern_scores(prefix_key, scores, weight=1.0)
                total_weight += 1.0

            if suffix_key in self.learned_patterns:
                self._add_pattern_scores(suffix_key, scores, weight=1.0)
                total_weight += 1.0

        # Description word patterns (lower priority)
        if description:
            desc_words = re.findall(r"\b\w{4,}\b", description.lower())[:5]
            for word in desc_words:
                desc_key = f"desc_word:{word}"
                if desc_key in self.learned_patterns:
                    self._add_pattern_scores(desc_key, scores, weight=0.5)
                    total_weight += 0.5

        # Pattern-based features
        if re.match(r"^[A-Z][a-z]+(?:[A-Z][a-z]+)*$", name):
            pattern_key = "pattern:camelcase"
            if pattern_key in self.learned_patterns:
                self._add_pattern_scores(pattern_key, scores, weight=0.5)
                total_weight += 0.5

        if not scores:
            return "Entity", 0.0

        # Get best prediction
        best_type, best_score = max(scores.items(), key=lambda x: x[1])

        # Calculate confidence (normalized by total weight, with diminishing returns)
        confidence = (
            min(best_score / max(total_weight, 1.0), 1.0) if total_weight > 0 else 0.0
        )

        # Apply confidence smoothing to avoid overconfidence
        confidence = confidence * 0.9 if confidence > 0.8 else confidence

        logger.debug(
            f"Inferred type '{best_type}' for '{name}' with confidence {confidence:.3f}"
        )
        return best_type, confidence

    def _add_pattern_scores(
        self, pattern_key: str, scores: dict[str, float], weight: float
    ):
        """Add weighted scores from a pattern to the scores dictionary."""
        if pattern_key not in self.learned_patterns:
            return

        label_counts = self.learned_patterns[pattern_key]
        total_pattern_count = sum(label_counts.values())

        # Add weighted scores for each label
        for label, count in label_counts.items():
            label_probability = count / total_pattern_count
            weighted_score = label_probability * weight
            scores[label] += weighted_score

    def get_pattern_summary(self) -> dict[str, any]:
        """Get summary of learned patterns for debugging."""
        if not self.learned_patterns:
            return {"status": "No patterns learned", "last_update": None}

        pattern_types = defaultdict(int)
        total_patterns = 0

        for pattern_key in self.learned_patterns:
            pattern_type = pattern_key.split(":")[0]
            pattern_types[pattern_type] += 1
            total_patterns += 1

        return {
            "total_patterns": total_patterns,
            "pattern_types": dict(pattern_types),
            "last_update": self.last_learning_update.isoformat()
            if self.last_learning_update
            else None,
            "most_common_patterns": self._get_most_common_patterns(10),
        }

    def _get_most_common_patterns(self, limit: int) -> list[dict[str, any]]:
        """Get most common patterns for debugging."""
        pattern_frequencies = []

        for pattern_key, label_counts in self.learned_patterns.items():
            total_count = sum(label_counts.values())
            most_common_label = (
                label_counts.most_common(1)[0] if label_counts else ("Unknown", 0)
            )

            pattern_frequencies.append(
                {
                    "pattern": pattern_key,
                    "total_frequency": total_count,
                    "most_common_label": most_common_label[0],
                    "label_frequency": most_common_label[1],
                }
            )

        return sorted(
            pattern_frequencies, key=lambda x: x["total_frequency"], reverse=True
        )[:limit]

    async def refresh_patterns_if_needed(self, max_age_hours: int = 24):
        """Refresh learned patterns if they're stale."""
        if not self.last_learning_update:
            await self.learn_from_existing_data()
            return

        age = datetime.utcnow() - self.last_learning_update
        if age.total_seconds() > (max_age_hours * 3600):
            logger.info(
                f"Patterns are {age.total_seconds() / 3600:.1f} hours old, refreshing..."
            )
            await self.learn_from_existing_data()
