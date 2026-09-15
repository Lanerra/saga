"""Catalog metadata transport; no graph mutation or validation implementation."""
from typing import Any

import config
from core.schema_readiness import CONSTRAINT_QUERY, CONSTRAINTS, FUNCTION_QUERY, INDEX_QUERY, PROCEDURE_QUERY, REQUIRED_FUNCTIONS, REQUIRED_PROCEDURES


def schema_catalog() -> dict[str, list[dict[str, Any]]]:
    constraints: list[dict[str, Any]] = [{"name": name, "type": "UNIQUENESS", "entityType": "NODE", "labelsOrTypes": [label], "properties": list(properties), "ownedIndex": name} for name, label, properties in CONSTRAINTS]
    indexes: list[dict[str, Any]] = [{"name": name, "type": "RANGE", "entityType": "NODE", "labelsOrTypes": [label], "properties": list(properties), "state": "ONLINE", "owningConstraint": name, "options": {}} for name, label, properties in CONSTRAINTS]
    indexes.append({"name": config.NEO4J_VECTOR_INDEX_NAME, "type": "VECTOR", "entityType": "NODE", "labelsOrTypes": ["Chapter"], "properties": ["embedding_vector"], "state": "ONLINE", "owningConstraint": None, "options": {"indexConfig": {"vector.dimensions": config.NEO4J_VECTOR_DIMENSIONS, "vector.similarity_function": {"cosine": "COSINE", "euclidean": "EUCLIDEAN"}[config.NEO4J_VECTOR_SIMILARITY_FUNCTION]}}})
    return {CONSTRAINT_QUERY: constraints, INDEX_QUERY: indexes, PROCEDURE_QUERY: [{"name": name} for name in sorted(REQUIRED_PROCEDURES)], FUNCTION_QUERY: [{"name": name} for name in sorted(REQUIRED_FUNCTIONS)]}
