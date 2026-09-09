"""Effective prerequisites for the exclusive-project graph's active writers."""
from typing import Any, Protocol

import config
from core.exceptions import DatabaseConnectionError

CONSTRAINT_QUERY = "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties, ownedIndex RETURN *"
INDEX_QUERY = "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state, owningConstraint, options RETURN *"
PROCEDURE_QUERY = "SHOW PROCEDURES EXECUTABLE BY CURRENT USER YIELD name RETURN name"
FUNCTION_QUERY = "SHOW FUNCTIONS EXECUTABLE BY CURRENT USER YIELD name RETURN name"

REQUIRED_PROCEDURES = frozenset({
    "apoc.util.validate", "apoc.merge.node", "apoc.merge.relationship", "db.index.vector.queryNodes",
    "apoc.refactor.mergeNodes", "apoc.refactor.mergeRelationships", "apoc.periodic.iterate", "apoc.create.relationship",
})
REQUIRED_FUNCTIONS = frozenset({
    "apoc.version", "apoc.util.sha256", "apoc.map.merge", "apoc.text.join", "apoc.coll.sort",
    "apoc.coll.toSet", "apoc.map.fromPairs", "apoc.map.removeKeys", "apoc.convert.toJson",
    "apoc.convert.fromJsonList", "apoc.convert.fromJsonMap",
    "apoc.text.levenshteinSimilarity", "apoc.text.replace",
})

CONSTRAINTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("saga_graph_owner_unique", "SagaGraphOwner", ("key",)),
    ("novel_id_unique", "Novel", ("id",)),
    ("chapter_id_unique", "Chapter", ("id",)),
    ("chapter_attempt_id_unique", "ChapterAttempt", ("id",)),
    ("character_id_unique", "Character", ("id",)),
    ("location_id_unique", "Location", ("id",)),
    ("event_id_unique", "Event", ("id",)),
    ("item_id_unique", "Item", ("id",)),
    ("novel_title_unique", "Novel", ("title",)),
    ("chapter_number_unique", "Chapter", ("number",)),
    ("character_name_unique", "Character", ("name",)),
    ("location_name_unique", "Location", ("name",)),
    ("event_name_unique", "Event", ("name",)),
    ("item_name_unique", "Item", ("name",)),
    ("scene_chapter_scene_unique", "Scene", ("chapter_number", "scene_index")),
    ("novelInfo_id_unique", "NovelInfo", ("id",)),
    ("worldContainer_id_unique", "WorldContainer", ("id",)),
    ("valueNode_value_type_unique", "ValueNode", ("value", "type")),
)


class Catalog(Protocol):
    def run(self, query: str, parameters: Any = None, **keywords: Any) -> Any: ...


def identifier(value: str) -> str:
    if not isinstance(value, str) or not value or "`" in value:
        raise ValueError("Invalid schema identifier")
    return f"`{value}`"


def vector_specs() -> list[tuple[str, str, str]]:
    return [
        (config.NEO4J_VECTOR_INDEX_NAME, "Chapter", "embedding_vector"),
        (config.NEO4J_CHARACTER_ENTITY_VECTOR_INDEX_NAME, "Character", config.ENTITY_EMBEDDING_VECTOR_PROPERTY),
        (config.NEO4J_LOCATION_ENTITY_VECTOR_INDEX_NAME, "Location", config.ENTITY_EMBEDDING_VECTOR_PROPERTY),
        (config.NEO4J_ITEM_ENTITY_VECTOR_INDEX_NAME, "Item", config.ENTITY_EMBEDDING_VECTOR_PROPERTY),
        (config.NEO4J_EVENT_ENTITY_VECTOR_INDEX_NAME, "Event", config.ENTITY_EMBEDDING_VECTOR_PROPERTY),
    ]


def vector_statement(name: str, label: str, property_name: str) -> str:
    dimensions = config.NEO4J_VECTOR_DIMENSIONS
    similarity = config.NEO4J_VECTOR_SIMILARITY_FUNCTION
    if type(dimensions) is not int or dimensions <= 0 or similarity not in {"cosine", "euclidean"}:
        raise ValueError("Invalid vector index configuration")
    return (
        f"CREATE VECTOR INDEX {identifier(name)} IF NOT EXISTS FOR (n:{identifier(label)}) ON (n.{identifier(property_name)}) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, `vector.similarity_function`: '{similarity}'}}}}"
    )


def required_statements() -> list[str]:
    constraints = [
        f"CREATE CONSTRAINT {identifier(name)} IF NOT EXISTS FOR (n:{identifier(label)}) "
        f"REQUIRE ({', '.join('n.' + identifier(property_name) for property_name in properties)}) IS UNIQUE"
        for name, label, properties in CONSTRAINTS
    ]
    return constraints + [vector_statement(*vector_specs()[0])]


def verify_capabilities(catalog: Catalog) -> None:
    procedures = {row["name"] for row in catalog.run(PROCEDURE_QUERY)}
    functions = {row["name"] for row in catalog.run(FUNCTION_QUERY)}
    missing = sorted((REQUIRED_PROCEDURES - procedures) | (REQUIRED_FUNCTIONS - functions))
    if missing:
        raise DatabaseConnectionError("Missing executable graph prerequisites", details={"missing": missing})


def verify_schema(catalog: Catalog, *, allow_missing: bool = False) -> None:
    """Reject ineffective or conflicting metadata, never infer success from DDL."""
    constraints = {row["name"]: dict(row) for row in catalog.run(CONSTRAINT_QUERY)}
    indexes = {row["name"]: dict(row) for row in catalog.run(INDEX_QUERY)}
    failures: list[str] = []
    for name, label, properties in CONSTRAINTS:
        expected = {"name": name, "type": "UNIQUENESS", "entityType": "NODE", "labelsOrTypes": [label], "properties": list(properties), "ownedIndex": name}
        actual = constraints.get(name)
        index = indexes.get(name)
        if actual is None and index is None and allow_missing:
            continue
        expected_index = {"name": name, "type": "RANGE", "entityType": "NODE", "labelsOrTypes": [label], "properties": list(properties), "state": "ONLINE", "owningConstraint": name}
        if actual != expected or index is None or any(index.get(key) != value for key, value in expected_index.items()):
            failures.append(name)
    name, label, property_name = vector_specs()[0]
    vector_statement(name, label, property_name)
    vector = indexes.get(name)
    if vector is not None or name in constraints or not allow_missing:
        expected_vector = {"type": "VECTOR", "entityType": "NODE", "labelsOrTypes": [label], "properties": [property_name], "state": "ONLINE", "owningConstraint": None}
        if vector is None or name in constraints or any(vector.get(key) != value for key, value in expected_vector.items()):
            failures.append(name)
        else:
            options = vector.get("options", {}).get("indexConfig", {})
            expected_similarity = {"cosine": "COSINE", "euclidean": "EUCLIDEAN"}[config.NEO4J_VECTOR_SIMILARITY_FUNCTION]
            if type(options.get("vector.dimensions")) is not int or options["vector.dimensions"] != config.NEO4J_VECTOR_DIMENSIONS or options.get("vector.similarity_function") != expected_similarity:
                failures.append(name)
    if failures:
        raise DatabaseConnectionError("Graph schema prerequisites missing or ineffective; explicit schema recovery required", details={"schema": failures})


def verify_write_prerequisites(catalog: Catalog) -> None:
    verify_schema(catalog)
    verify_capabilities(catalog)
