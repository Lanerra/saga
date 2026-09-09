"""Explicit legacy quarantine admission, separate from ordinary startup."""
from typing import Any

import pytest

from tests.fakes.graph_ownership import PROJECT_ID, OwnershipRows
from tests.fakes.schema_catalog import schema_catalog


class LegacyCatalog:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = [{"element_id": "node-1", "labels": ["Character"], "properties": {"id": "ada", "name": "Ada"}}]
        self.edges: list[dict[str, Any]] = []
        self.writes: list[str] = []
        self.catalog = schema_catalog()

    def run(self, query: str, parameters: Any = None, **keywords: Any) -> OwnershipRows:
        if query in self.catalog:
            return OwnershipRows(self.catalog[query])
        if query.startswith("MATCH (n) RETURN elementId"):
            return OwnershipRows(self.nodes)
        if query.startswith("MATCH (source)-[relationship]->(target)"):
            return OwnershipRows(self.edges)
        self.writes.append(query)
        return OwnershipRows([])


@pytest.mark.parametrize("defect", ["digest", "identity", "already_owned", "missing_id", "unknown_label"])
def test_legacy_admission_rejects_before_mutation(defect: str) -> None:
    from core.graph_migration import LegacyMigration, migrate_legacy_snapshot, snapshot_legacy_graph

    catalog = LegacyCatalog()
    if defect == "missing_id":
        catalog.nodes[0]["properties"].pop("id")
    elif defect == "unknown_label":
        catalog.nodes[0]["labels"] = ["UnknownHistoricalLabel"]
    elif defect == "already_owned":
        catalog.nodes.append({"element_id": "owner-1", "labels": ["SagaGraphOwner"], "properties": {"key": "exclusive", "project_id": PROJECT_ID, "version": 1}})
    snapshot = snapshot_legacy_graph(catalog)
    with pytest.raises(ValueError):
        plan = LegacyMigration(project_id="not-a-uuid" if defect == "identity" else PROJECT_ID, snapshot_sha256="0" * 64 if defect == "digest" else snapshot.sha256)
        migrate_legacy_snapshot(catalog, plan)
    assert catalog.writes == []


def test_migration_rejects_autocommit_transport() -> None:
    from core.graph_migration import LegacyMigration, migrate_legacy_snapshot, snapshot_legacy_graph

    catalog = LegacyCatalog()
    plan = LegacyMigration(PROJECT_ID, snapshot_legacy_graph(catalog).sha256)
    with pytest.raises(ValueError, match="explicit Neo4j transaction"):
        migrate_legacy_snapshot(catalog, plan)
    assert catalog.writes == []
