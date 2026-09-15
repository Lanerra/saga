"""Fail-closed test doubles and journal-backed compensation controls."""
from copy import deepcopy
from typing import Any, cast

import pytest

from core.graph_migration import EDGE_SNAPSHOT_QUERY, NODE_SNAPSHOT_QUERY
from core.graph_ownership import OWNER_QUERY
from core.langgraph.chapter_lifecycle import ChapterLifecycle
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.revision_node import revise_chapter
from core.langgraph.state import NarrativeState
from core.langgraph.state_helpers import clear_error_state
from data_access.cypher_builders import graph_compensation as compensation
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.fakes.graph_ownership import OWNER_LOCK_QUERY, PROJECT_ID, OwnershipTransaction
from tests.test_langgraph.test_chapter_lifecycle import DriverExample, TransactionExample
from tests.test_langgraph.test_chapter_lifecycle import lifecycle_example as lifecycle_example


@pytest.mark.parametrize("operation", ["execute_read_query", "execute_write_query"])
async def test_unknown_queries_fail_and_explicit_empty_responses_are_valid(operation: str) -> None:
    database = FakeNeo4jManager()
    query = "MATCH (n:Character) RETURN n"
    call = getattr(database, operation)
    with pytest.raises(AssertionError, match="Unconfigured"):
        await call(query, {})
    database.configure_response(r"^MATCH \(n:Character\) RETURN n$", [])
    assert await call(query, {}) == []
    with pytest.raises(AssertionError, match="Unconfigured"):
        await call("MATCH (n:Location) RETURN n", {})


async def test_response_fake_cannot_claim_transaction_success_without_callback() -> None:
    database = FakeNeo4jManager()
    calls: list[str] = []

    def apply(transaction: Any) -> None:
        calls.append("callback")

    with pytest.raises(AssertionError, match="transaction"):
        await database.execute_in_transaction(apply)
    assert calls == []


async def test_unknown_batch_statement_cannot_claim_success() -> None:
    database = FakeNeo4jManager()
    with pytest.raises(AssertionError, match="Unconfigured"):
        await database.execute_cypher_batch([("UNIMPLEMENTED WRITE", {})])
    database.configure_response(r"^KNOWN WRITE$", [])
    await database.execute_cypher_batch([("KNOWN WRITE", {})])
    assert database.batch_statements[-1] == [("KNOWN WRITE", {})]


def test_owner_lock_keeps_metadata_and_unknown_queries_fail() -> None:
    transaction = OwnershipTransaction()
    before = transaction.run(OWNER_QUERY)
    transaction.run(OWNER_LOCK_QUERY, {"project_id": PROJECT_ID}).consume()
    assert transaction.run(OWNER_QUERY) == before
    with pytest.raises(AssertionError, match="does not implement"):
        transaction.run("MATCH (n) RETURN labels(n)")


def test_transaction_snapshots_are_nonempty_detached_and_rollback_is_atomic() -> None:
    driver = DriverExample(PROJECT_ID)
    before = deepcopy((driver.nodes, driver.edges))
    transaction = TransactionExample(driver)
    nodes = transaction.run(NODE_SNAPSHOT_QUERY)
    edges = transaction.run(EDGE_SNAPSHOT_QUERY)
    assert nodes == list(driver.nodes.values()) and nodes
    assert edges == list(driver.edges.values()) and edges
    nodes[0]["properties"]["title"] = "Not a database write"
    assert transaction.nodes == driver.nodes
    transaction.run(compensation.RESTORE_NODE, {"element_id": "chapter-1", "properties": {"number": 1}})
    transaction.run(compensation.DELETE_EDGE, {"element_id": "plan-edge"})
    transaction.run(compensation.DELETE_NODE, {"element_id": "chapter-1"})
    transaction.rollback()
    assert (driver.nodes, driver.edges) == before
    assert driver.writes == [] and driver.commits == 0
    with pytest.raises(AssertionError, match="Unimplemented"):
        TransactionExample(driver).run("MATCH (n) RETURN labels(n)")


def test_transaction_inverse_restores_edge_properties_and_occurrences() -> None:
    driver = DriverExample(PROJECT_ID)
    transaction = TransactionExample(driver)
    before = compensation.capture_graph(transaction)
    transaction.run(compensation.RESTORE_EDGE, {"element_id": "plan-edge", "properties": {"chapter_added": 1}})
    parameters = {"source": "scene-1", "target": "chapter-1", "type": "PART_OF", "properties": {"chapter_added": 1}}
    for _ in range(2):
        transaction.run(compensation.CREATE_EDGE, parameters)
    journal = compensation.encode_compensation(before, compensation.capture_graph(transaction))
    compensation.apply_compensation(transaction, journal)
    assert compensation.capture_graph(transaction) == before
    transaction.commit()
    assert (driver.nodes, driver.edges) == ({row["element_id"]: row for row in before["nodes"]}, {row["element_id"]: row for row in before["edges"]})


async def test_unjournaled_revision_never_writes_or_bypasses_block(
    lifecycle_example: tuple[NarrativeState, DriverExample],
) -> None:
    state, driver = lifecycle_example
    del state["lifecycle_version"]
    state.update({"max_iterations": 3, "needs_revision": True})
    before = deepcopy((driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits))
    result = await revise_chapter(state)
    assert result["has_fatal_error"] is True
    assert result["current_node"] == "revise_blocked"
    assert result["last_error"] == "Revision rollback failed; reconciliation required: Rollback requires a durable attempt journal; legacy data needs explicit reconciliation"
    for controls in ({}, clear_error_state(), {"force_continue": True, "iteration_count": 3}):
        assert await revise_chapter(cast(NarrativeState, {**state, **result, **controls})) == {
            "last_error": result["last_error"], "has_fatal_error": True, "error_node": "revise", "current_node": "revise_blocked",
        }
    assert (driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits) == before


async def test_revision_iteration_limit_never_enters_journal_or_model(
    lifecycle_example: tuple[NarrativeState, DriverExample],
) -> None:
    state, driver = lifecycle_example
    state = {**state, **await commit_to_graph(state)}
    assert state["has_fatal_error"] is False
    lifecycle = ChapterLifecycle(state).stage()
    before = deepcopy((driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits))
    result = await revise_chapter({**state, "max_iterations": 3, "iteration_count": 3, "force_continue": True})
    assert result == {
        "last_error": "Max revision attempts (3) reached", "has_fatal_error": True, "error_node": "revise", "needs_revision": False, "current_node": "revise_failed",
    }
    assert not lifecycle.files.exists(lifecycle.phase_path("compensation_required"))
    assert (driver.nodes, driver.edges, driver.receipts, driver.writes, driver.commits) == before
