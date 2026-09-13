"""Transaction-bound graph before/after images, never chapter-number deletion authority."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from datetime import tzinfo
from importlib import import_module
from typing import Any, Literal

from neo4j.time import Date, DateTime, Duration, Time
from pydantic import BaseModel, ConfigDict

from core.graph_migration import EDGE_SNAPSHOT_QUERY, NODE_SNAPSHOT_QUERY
from core.schema_readiness import Catalog

DELETE_EDGE = "MATCH ()-[r]->() WHERE elementId(r) = $element_id DELETE r"
DELETE_NODE = "MATCH (n) WHERE elementId(n) = $element_id DELETE n"
RESTORE_NODE = "MATCH (n) WHERE elementId(n) = $element_id SET n = $properties"
RESTORE_EDGE = "MATCH ()-[r]->() WHERE elementId(r) = $element_id SET r = $properties"
CREATE_EDGE = """
MATCH (source), (target)
WHERE elementId(source) = $source AND elementId(target) = $target
CALL apoc.create.relationship(source, $type, $properties, target) YIELD rel
RETURN elementId(rel) AS element_id
"""
DOMAIN_LABELS = frozenset({"Chapter", "Scene", "Character", "Location", "Item", "Event", "ValueNode"})


def _encode_property(value: Any) -> Any:
    if isinstance(value, (DateTime, Date, Time, Duration)):
        zone = getattr(value, "tzinfo", None)
        return {"neo4j_type": type(value).__name__, "value": value.iso_format(), "zone": getattr(zone, "zone", getattr(zone, "key", None))}
    if isinstance(value, bytes):
        return {"neo4j_type": "bytes", "value": value.hex()}
    if isinstance(value, list):
        return [_encode_property(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"Unsupported compensation property type: {type(value).__name__}")


def _decode_property(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_property(item) for item in value]
    if not isinstance(value, dict):
        return value
    kind = value["neo4j_type"]
    if kind == "bytes":
        return bytes.fromhex(value["value"])
    parsers: dict[str, Callable[[str], Any]] = {"DateTime": DateTime.from_iso_format, "Date": Date.from_iso_format, "Time": Time.from_iso_format, "Duration": Duration.from_iso_format}
    restored = parsers[kind](value["value"])
    if kind == "DateTime" and value.get("zone"):
        # Neo4j's nanosecond DateTime requires its pytz timezone implementation.
        # The driver dependency has no installed typing stubs; validate this boundary.
        zone = import_module("pytz").timezone(value["zone"])
        if not isinstance(zone, tzinfo):
            raise ValueError("Invalid compensation timezone")
        restored = restored.as_timezone(zone)
    return restored


class NodeImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    element_id: str
    labels: list[str]
    properties: dict[str, Any]


class EdgeImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    element_id: str
    source: str
    target: str
    type: str
    properties: dict[str, Any]


class GraphImage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    nodes: list[NodeImage]
    edges: list[EdgeImage]


class Compensation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    before: GraphImage
    after: GraphImage


def _encoded_image(image: GraphImage) -> dict[str, Any]:
    encoded = image.model_dump()
    for kind in ("nodes", "edges"):
        for row, original in zip(encoded[kind], getattr(image, kind), strict=True):
            row["properties"] = {key: _encode_property(value) for key, value in original.properties.items()}
    return encoded


def capture_graph(transaction: Catalog) -> dict[str, Any]:
    nodes = [dict(row) for row in transaction.run(NODE_SNAPSHOT_QUERY)]
    for node in nodes:
        node["labels"] = sorted(node["labels"])
    # Attempt receipts are lifecycle authority, not compensated domain projections.
    nodes = [node for node in nodes if "ChapterAttempt" not in node["labels"]]
    edges = [dict(row) for row in transaction.run(EDGE_SNAPSHOT_QUERY)]
    image = {"nodes": nodes, "edges": edges}
    GraphImage.model_validate(image)
    return image


def _index(rows: list[Any]) -> dict[str, Any]:
    indexed = {row.element_id: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate compensation element identity")
    return indexed


def encode_compensation(before: dict[str, Any], after: dict[str, Any]) -> str:
    previous = GraphImage.model_validate(before)
    following = GraphImage.model_validate(after)
    nodes_before, nodes_after = _index(previous.nodes), _index(following.nodes)
    edges_before, edges_after = _index(previous.edges), _index(following.edges)
    changed_nodes = {identity for identity in nodes_before.keys() | nodes_after.keys() if nodes_before.get(identity) != nodes_after.get(identity)}
    changed_edges = {identity for identity in edges_before.keys() | edges_after.keys() if edges_before.get(identity) != edges_after.get(identity)}
    for identity in changed_nodes:
        old, new = nodes_before.get(identity), nodes_after.get(identity)
        if new is None or (old is not None and old.labels != new.labels):
            raise ValueError("Chapter commit cannot journal node deletion or label change")
        if not new.labels or not set(new.labels) <= DOMAIN_LABELS:
            raise ValueError("Chapter commit cannot change protected graph infrastructure")
        if old is not None and "Chapter" in old.labels and old.properties.get("generation_status") == "finalized":
            raise ValueError("Chapter commit cannot change a previously accepted chapter")
    # Retain endpoint identities too: deleted edges cannot be restored onto reused IDs.
    for identity in changed_edges:
        for edge in (edges_before.get(identity), edges_after.get(identity)):
            if edge is not None:
                changed_nodes.update((edge.source, edge.target))
    value = Compensation(
        schema_version=1,
        before=GraphImage(nodes=[nodes_before[key] for key in sorted(changed_nodes) if key in nodes_before], edges=[edges_before[key] for key in sorted(changed_edges) if key in edges_before]),
        after=GraphImage(nodes=[nodes_after[key] for key in sorted(changed_nodes) if key in nodes_after], edges=[edges_after[key] for key in sorted(changed_edges) if key in edges_after]),
    )
    encoded = {"schema_version": 1, "before": _encoded_image(value.before), "after": _encoded_image(value.after)}
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"), allow_nan=False)


def apply_compensation(transaction: Catalog, encoded: str) -> None:
    decoded = json.loads(encoded)
    for phase in ("before", "after"):
        for kind in ("nodes", "edges"):
            for row in decoded[phase][kind]:
                row["properties"] = {key: _decode_property(value) for key, value in row["properties"].items()}
    journal = Compensation.model_validate(decoded)
    current = GraphImage.model_validate(capture_graph(transaction))
    nodes_before, nodes_after, nodes_current = (_index(image.nodes) for image in (journal.before, journal.after, current))
    edges_before, edges_after, edges_current = (_index(image.edges) for image in (journal.before, journal.after, current))
    for expected, actual in ((nodes_after, nodes_current), (edges_after, edges_current)):
        if any(actual.get(identity) != row for identity, row in expected.items()):
            raise ValueError("Compensation conflict: graph differs from committed image")
    if any(identity in edges_current for identity in edges_before.keys() - edges_after.keys()):
        raise ValueError("Compensation conflict: deleted relationship identity reused")
    created_nodes = nodes_after.keys() - nodes_before.keys()
    created_edges = edges_after.keys() - edges_before.keys()
    if any((edge.source in created_nodes or edge.target in created_nodes) and identity not in created_edges for identity, edge in edges_current.items()):
        raise ValueError("Compensation conflict: new node has an unowned relationship")
    if any(identity not in nodes_after for identity in nodes_before):
        raise ValueError("Compensation conflict: node deletion is not restorable")
    if any(node.labels != nodes_after[identity].labels for identity, node in nodes_before.items()):
        raise ValueError("Compensation conflict: node labels differ")
    for identity in edges_before.keys() & edges_after.keys():
        old, new = edges_before[identity], edges_after[identity]
        if (old.source, old.target, old.type) != (new.source, new.target, new.type):
            raise ValueError("Compensation conflict: relationship identity changed")
    for identity in sorted(created_edges):
        transaction.run(DELETE_EDGE, {"element_id": identity}).consume()
    for identity in sorted(created_nodes):
        transaction.run(DELETE_NODE, {"element_id": identity}).consume()
    for identity, node in nodes_before.items():
        if node != nodes_after[identity]:
            transaction.run(RESTORE_NODE, {"element_id": identity, "properties": node.properties}).consume()
    for identity, edge in edges_before.items():
        if identity not in edges_after:
            transaction.run(CREATE_EDGE, {"source": edge.source, "target": edge.target, "type": edge.type, "properties": edge.properties}).consume()
        else:
            transaction.run(RESTORE_EDGE, {"element_id": identity, "properties": edge.properties}).consume()
    restored = GraphImage.model_validate(capture_graph(transaction))
    expected_nodes = {identity: node for identity, node in nodes_current.items() if identity not in created_nodes}
    expected_nodes.update(nodes_before)
    expected_edges = [edge for identity, edge in edges_current.items() if identity not in edges_after] + list(edges_before.values())

    def occurrences(edges: list[EdgeImage]) -> Counter[str]:
        return Counter(json.dumps({"source": edge.source, "target": edge.target, "type": edge.type, "properties": {key: _encode_property(value) for key, value in edge.properties.items()}}, sort_keys=True, allow_nan=False) for edge in edges)

    if _index(restored.nodes) != expected_nodes or occurrences(restored.edges) != occurrences(expected_edges):
        raise ValueError("Compensation readback differs from the exact inverse")
