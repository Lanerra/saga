"""Healing cannot infer deletion ownership from age and provisional status."""
from typing import Any, cast

from core.graph_healing_service import GraphHealingService
from core.service_context import RunServices, inject_services


async def test_unowned_orphan_candidate_is_reported_not_deleted() -> None:
    class Database:
        writes: list[str] = []

        async def execute_read_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
            return [{"element_id": "initialization-event", "name": "Synthetic future event", "type": "Event", "created_chapter": 0}]

        async def execute_write_query(self, query: str, parameters: Any = None) -> list[dict[str, Any]]:
            self.writes.append(query)
            return [{"deleted_count": 1}]

    database = Database()
    with inject_services(RunServices(cast(Any, object()), cast(Any, database))):
        result = await GraphHealingService().cleanup_orphaned_nodes(10)
    assert database.writes == []
    assert result == {"nodes_checked": 1, "nodes_removed": 0, "nodes_requiring_reconciliation": 1}
