"""Exclusive graph ownership and durable, display-name-independent identity."""

import fcntl
import os
from pathlib import Path
from uuid import UUID, uuid4

from neo4j import ManagedTransaction, Session, Transaction

from core.exceptions import DatabaseConnectionError

OWNER_CONSTRAINT = "CREATE CONSTRAINT saga_graph_owner_unique IF NOT EXISTS FOR (owner:SagaGraphOwner) REQUIRE owner.key IS UNIQUE"
OWNER_QUERY = "MATCH (owner:SagaGraphOwner) RETURN owner.key AS key, owner.project_id AS project_id, owner.version AS version"
OWNER_CONSTRAINT_QUERY = (
    "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties, ownedIndex "
    "WHERE name = 'saga_graph_owner_unique' OR ('SagaGraphOwner' IN labelsOrTypes AND 'key' IN properties) "
    "RETURN name, type, entityType, labelsOrTypes, properties, ownedIndex"
)
OWNER_INDEX_QUERY = (
    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state, owningConstraint "
    "WHERE name = 'saga_graph_owner_unique' OR ('SagaGraphOwner' IN labelsOrTypes AND 'key' IN properties) "
    "RETURN name, type, entityType, labelsOrTypes, properties, state, owningConstraint"
)


class GraphOwnershipError(DatabaseConnectionError):
    """The database cannot safely be used by the requested project."""


def validate_project_id(project_id: str) -> str:
    if not isinstance(project_id, str) or str(UUID(project_id)) != project_id:
        raise ValueError("Graph project identity must be a canonical UUID string")
    return project_id


def load_graph_project_id(project_directory: Path) -> str:
    """Keep graph identity with the project across renames and restores.

    Copying this file means restoring the same project, not forking a new story.
    A missing identity never authorizes adoption of a nonempty legacy database.
    """
    identity_path = project_directory / "graph-project-id"
    with (project_directory / "graph-project-id.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if identity_path.exists():
            return validate_project_id(identity_path.read_text(encoding="ascii"))
        identity = str(uuid4())
        temporary = project_directory / f".graph-project-id-{identity}"
        with temporary.open("x", encoding="ascii") as output:
            output.write(identity)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, identity_path)
        directory = os.open(project_directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return identity


def assert_graph_owner(transaction: ManagedTransaction | Transaction, project_id: str) -> None:
    validate_project_id(project_id)
    owners = [dict(record) for record in transaction.run(OWNER_QUERY)]
    if (
        len(owners) != 1
        or type(owners[0].get("key")) is not str
        or type(owners[0].get("project_id")) is not str
        or type(owners[0].get("version")) is not int
        or owners[0] != {"key": "exclusive", "project_id": project_id, "version": 1}
    ):
        raise GraphOwnershipError("Graph project ownership missing, malformed or mismatched; no legacy adoption is permitted")


def ownership_constraint_exists(session: Session) -> bool:
    """Reject conflicting schema; only a wholly absent prerequisite is creatable."""
    constraints = session.run(OWNER_CONSTRAINT_QUERY).data()
    indexes = session.run(OWNER_INDEX_QUERY).data()
    if not constraints and not indexes:
        return False
    expected_constraint = {
        "name": "saga_graph_owner_unique",
        "type": "UNIQUENESS",
        "entityType": "NODE",
        "labelsOrTypes": ["SagaGraphOwner"],
        "properties": ["key"],
        "ownedIndex": "saga_graph_owner_unique",
    }
    expected_index = {
        "name": "saga_graph_owner_unique",
        "type": "RANGE",
        "entityType": "NODE",
        "labelsOrTypes": ["SagaGraphOwner"],
        "properties": ["key"],
        "state": "ONLINE",
        "owningConstraint": "saga_graph_owner_unique",
    }
    if constraints != [expected_constraint] or indexes != [expected_index]:
        raise GraphOwnershipError("Graph ownership constraint or backing index is conflicting or ineffective; schema remains unchanged")
    return True


def claim_empty_graph(transaction: ManagedTransaction, project_id: str, claim_token: str) -> None:
    record = transaction.run(
        "MERGE (owner:SagaGraphOwner {key: 'exclusive'}) "
        "ON CREATE SET owner.project_id = $project_id, owner.version = 1, owner.claim_token = $claim_token "
        "RETURN owner.claim_token AS claim_token",
        project_id=project_id,
        claim_token=claim_token,
    ).single(strict=True)
    assert record is not None
    assert_graph_owner(transaction, project_id)
    if record["claim_token"] == claim_token:
        existing = transaction.run("MATCH (n) WHERE NOT n:SagaGraphOwner RETURN count(n) AS count").single(strict=True)
        assert existing is not None
        if existing["count"] != 0:
            raise GraphOwnershipError("Unowned nonempty legacy graph cannot be assigned to a project; restore remains unchanged")
