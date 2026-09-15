"""Explicit snapshot-bound quarantine migration for an offline legacy copy.

Ordinary startup never invokes this adapter. The caller must retain a cold backup,
verify the copy's project identity, provision the required schema separately, and
hold exclusive maintenance access for the entire transaction. Quarantined facts
are not chapter-owned assertions or accepted manuscript/checkpoint authority.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from neo4j import Transaction

from core.graph_ownership import validate_project_id
from core.schema_readiness import CONSTRAINTS, Catalog, verify_write_prerequisites

NODE_SNAPSHOT_QUERY = "MATCH (n) RETURN elementId(n) AS element_id, labels(n) AS labels, properties(n) AS properties ORDER BY element_id"
EDGE_SNAPSHOT_QUERY = (
    "MATCH (source)-[relationship]->(target) RETURN elementId(relationship) AS element_id, "
    "elementId(source) AS source, elementId(target) AS target, type(relationship) AS type, "
    "properties(relationship) AS properties ORDER BY element_id"
)


@dataclass(frozen=True)
class LegacySnapshot:
    payload: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LegacyMigration:
    project_id: str
    snapshot_sha256: str

    def __post_init__(self) -> None:
        validate_project_id(self.project_id)
        if not isinstance(self.snapshot_sha256, str) or re.fullmatch("[0-9a-f]{64}", self.snapshot_sha256) is None:
            raise ValueError("Migration requires an exact retained snapshot SHA-256")


def snapshot_legacy_graph(transaction: Catalog) -> LegacySnapshot:
    nodes = [dict(row) for row in transaction.run(NODE_SNAPSHOT_QUERY)]
    for node in nodes:
        node["labels"] = sorted(node["labels"])
    edges = [dict(row) for row in transaction.run(EDGE_SNAPSHOT_QUERY)]
    return LegacySnapshot(json.dumps({"nodes": nodes, "edges": edges}, sort_keys=True, separators=(",", ":"), allow_nan=False))


def migrate_legacy_snapshot(transaction: Catalog, migration: LegacyMigration) -> str:
    """Quarantine all historical relationships and assign one explicitly verified owner.

    Execute inside one caller-owned transaction; propagate every error for rollback.
    No IDs, chapter memberships, initialization receipts or acceptance are inferred.
    Non-JSON property values and unknown identity shapes need a separate reviewed adapter.
    """
    snapshot = snapshot_legacy_graph(transaction)
    if snapshot.sha256 != migration.snapshot_sha256:
        raise ValueError("Legacy graph snapshot mismatch; no migration permitted")
    graph: dict[str, Any] = json.loads(snapshot.payload)
    nodes = graph["nodes"]
    if not nodes or any("SagaGraphOwner" in node["labels"] for node in nodes):
        raise ValueError("Migration requires a nonempty unowned legacy snapshot")
    labels = {label for _, label, _ in CONSTRAINTS} - {"SagaGraphOwner", "ChapterAttempt"}
    for node in nodes:
        if len(node["labels"]) != 1 or node["labels"][0] not in labels:
            raise ValueError("Unknown historical node identity; explicit adapter required")
        label = node["labels"][0]
        identity = node["properties"].get("id")
        if label not in {"Scene", "ValueNode"} and (not isinstance(identity, str) or not identity.strip()):
            raise ValueError("Legacy node missing canonical identity; no ID inference permitted")
        if label == "Scene" and any(type(node["properties"].get(key)) is not int for key in ("chapter_number", "scene_index")):
            raise ValueError("Legacy scene missing integer composite identity")
        if label == "ValueNode" and any(not isinstance(node["properties"].get(key), str) for key in ("value", "type")):
            raise ValueError("Legacy value missing composite identity")
    if any("legacy_assertion_origin" in edge["properties"] for edge in graph["edges"]):
        raise ValueError("Legacy quarantine metadata collision")
    if not isinstance(transaction, Transaction):
        raise ValueError("Migration requires an explicit Neo4j transaction, never an autocommit session")
    verify_write_prerequisites(transaction)
    transaction.run(
        "MATCH ()-[relationship]->() "
        "SET relationship.legacy_assertion_origin = relationship.assertion_origin, "
        "relationship.assertion_origin = 'legacy_unclassified'"
    ).consume()
    transaction.run(
        "CREATE (:SagaGraphOwner {key: 'exclusive', project_id: $project_id, version: 1, migration_snapshot_sha256: $snapshot_sha256})",
        {"project_id": migration.project_id, "snapshot_sha256": migration.snapshot_sha256},
    ).consume()
    return snapshot.sha256
