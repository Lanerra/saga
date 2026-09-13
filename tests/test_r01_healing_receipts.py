"""Synthetic confidence admission and advisory failure receipts; no model transport."""
from typing import Any, cast

import pytest

from core.exceptions import ValidationError
from core.graph_healing_service import GraphHealingService
from core.service_context import RunServices, inject_services


class Database:
    def __init__(self, failure: str = "") -> None:
        self.failure = failure
        self.writes: list[dict[str, Any]] = []

    async def execute_read_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
        if "cutoff_chapter" in query:
            if self.failure == "cleanup":
                raise RuntimeError("synthetic cleanup unavailable")
            return []
        if "ORDER BY n.created_chapter ASC" in query:
            return [{"element_id": "node", "id": "character_example", "name": "Synthetic Keeper", "type": "Character", "description": "Short", "traits": [], "created_chapter": 1}] if self.failure == "enrich" else []
        if "count(r) AS rel_count" in query:
            return [{"rel_count": 3, "status": "Unknown"}]
        if self.failure == "enrich" and (parameters or {}).get("id_param"):
            raise RuntimeError("synthetic enrichment context unavailable")
        return []

    async def execute_write_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
        self.writes.append(parameters)
        return [{"name": "Synthetic Keeper"}]


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9", None])
@pytest.mark.parametrize("operation", ["enrich", "graduate"])
async def test_confidence_is_typed_finite_and_bounded(confidence: Any, operation: str) -> None:
    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        with pytest.raises(ValidationError, match="confidence"):
            if operation == "enrich":
                await GraphHealingService().apply_enrichment("node", {"confidence": confidence, "inferred_description": "Synthetic description"})
            else:
                await GraphHealingService().graduate_node("node", confidence)
    assert database.writes == []


@pytest.mark.parametrize("confidence, expected", [(0.0, False), (0.59, False), (0.6, True), (1, True)])
async def test_valid_confidence_retains_apply_threshold(confidence: float, expected: bool) -> None:
    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        assert await GraphHealingService().apply_enrichment("node", {"confidence": confidence, "inferred_description": "Synthetic description"}) is expected
    assert len(database.writes) == int(expected)


@pytest.mark.parametrize("failure", ["enrich", "cleanup"])
async def test_advisory_failure_is_reported_without_false_completion(failure: str) -> None:
    database = Database(failure)
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        receipt = await GraphHealingService().heal_graph(2, "unused-offline")
    assert receipt["warnings"]
    assert any(failure in warning for warning in receipt["warnings"])
    assert receipt["status"] == "partial"
    assert database.writes == []
