# core/triple_processor.py
"""
Triple processing service for knowledge graph validation.

This module handles the complex logic of processing and validating triples,
separating concerns from the main validation logic to improve maintainability.
"""

from typing import Any

import structlog

from core.simple_type_inference import infer_object_type, infer_subject_type

logger = structlog.get_logger(__name__)


class TripleProcessor:
    """
    Service for processing and preparing triples for validation.

    This class handles the complex logic of extracting and preparing entity
    information from triple dictionaries, separating this concern from the
    core validation logic.
    """

    def __init__(self):
        """Initialize the triple processor."""
        self._processing_stats = {
            "total_triples_processed": 0,
            "successful_extractions": 0,
            "extraction_errors": 0,
            "type_inferences": 0,
        }

    def process_triple(self, triple_dict: dict[str, Any]) -> dict[str, Any] | None:
        """
        Process a triple dictionary and extract validated entity information.

        Args:
            triple_dict: Raw triple dictionary from extraction pipeline

        Returns:
            Processed triple information ready for validation, or None if invalid

        Expected triple_dict format:
        {
            "subject": {"name": "...", "type": "...", "category": "..."},
            "predicate": "...",
            "object_entity": {"name": "...", "type": "...", "category": "..."} | None,
            "object_literal": "..." | None,
            "is_literal_object": bool
        }
        """
        self._processing_stats["total_triples_processed"] += 1

        try:
            # Extract and validate subject information
            subject_info = self._extract_subject_info(triple_dict)
            if not subject_info:
                return None

            # Extract predicate
            predicate = self._extract_predicate(triple_dict)
            if not predicate:
                return None

            # Extract and validate object information
            object_info = self._extract_object_info(triple_dict)
            if not object_info:
                return None

            processed_triple = {
                "subject_type": subject_info["type"],
                "subject_name": subject_info["name"],
                "predicate": predicate,
                "object_type": object_info["type"],
                "object_name": object_info.get("name"),
                "is_literal_object": object_info.get("is_literal", False),
                "original_triple": triple_dict,
            }

            self._processing_stats["successful_extractions"] += 1
            return processed_triple

        except Exception as e:
            logger.error(f"Error processing triple {triple_dict}: {e}", exc_info=True)
            self._processing_stats["extraction_errors"] += 1
            return None

    def _extract_subject_info(
        self, triple_dict: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract and validate subject information from triple."""
        subject_info = triple_dict.get("subject")
        if not isinstance(subject_info, dict):
            logger.warning(f"Invalid subject info in triple: {triple_dict}")
            return None

        subject_name = subject_info.get("name")
        if not subject_name or not str(subject_name).strip():
            logger.warning(f"Missing or empty subject name in triple: {triple_dict}")
            return None

        # Use simple type inference to get proper subject type
        subject_type = infer_subject_type(subject_info)
        self._processing_stats["type_inferences"] += 1

        return {
            "name": str(subject_name).strip(),
            "type": subject_type,
            "original_info": subject_info,
        }

    def _extract_predicate(self, triple_dict: dict[str, Any]) -> str | None:
        """Extract and validate predicate from triple."""
        predicate = triple_dict.get("predicate")
        if not predicate or not str(predicate).strip():
            logger.warning(f"Missing or empty predicate in triple: {triple_dict}")
            return None

        return str(predicate).strip()

    def _extract_object_info(
        self, triple_dict: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Extract and validate object information from triple."""
        is_literal_object = triple_dict.get("is_literal_object", False)

        if is_literal_object:
            # Handle literal objects
            literal_value = triple_dict.get("object_literal")
            if literal_value is None:
                logger.warning(
                    f"Missing literal value in literal triple: {triple_dict}"
                )
                return None

            return {
                "type": "ValueNode",
                "name": str(literal_value),
                "is_literal": True,
                "original_value": literal_value,
            }
        else:
            # Handle entity objects
            object_entity_info = triple_dict.get("object_entity")
            if not isinstance(object_entity_info, dict):
                logger.warning(f"Invalid object entity info in triple: {triple_dict}")
                return None

            object_name = object_entity_info.get("name")
            if not object_name or not str(object_name).strip():
                logger.warning(f"Missing or empty object name in triple: {triple_dict}")
                return None

            # Use simple type inference to get proper object type
            object_type = infer_object_type(object_entity_info, is_literal=False)
            self._processing_stats["type_inferences"] += 1

            return {
                "type": object_type,
                "name": str(object_name).strip(),
                "is_literal": False,
                "original_info": object_entity_info,
            }

    def process_batch(self, triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Process a batch of triples efficiently.

        Args:
            triples: List of triple dictionaries

        Returns:
            List of processed triple information (excluding invalid triples)
        """
        processed_triples = []

        for triple_dict in triples:
            processed = self.process_triple(triple_dict)
            if processed:
                processed_triples.append(processed)

        return processed_triples

    def get_processing_statistics(self) -> dict[str, Any]:
        """Get statistics about triple processing operations."""
        total = self._processing_stats["total_triples_processed"]

        if total == 0:
            return self._processing_stats

        return {
            **self._processing_stats,
            "success_rate": self._processing_stats["successful_extractions"]
            / total
            * 100,
            "error_rate": self._processing_stats["extraction_errors"] / total * 100,
            "avg_inferences_per_triple": self._processing_stats["type_inferences"]
            / total
            if total > 0
            else 0,
        }

    def reset_statistics(self):
        """Reset processing statistics (useful for testing)."""
        self._processing_stats = {
            "total_triples_processed": 0,
            "successful_extractions": 0,
            "extraction_errors": 0,
            "type_inferences": 0,
        }
