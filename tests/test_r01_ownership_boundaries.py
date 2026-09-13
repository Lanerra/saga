"""Ownership file admission and driver lifetime regressions from R07 handoff."""
import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from neo4j.exceptions import ServiceUnavailable

from core.db_manager import GraphDatabase, Neo4jManagerSingleton
from core.graph_ownership import load_graph_project_id
from tests.fakes.graph_ownership import PROJECT_ID, OwnershipDriver, OwnershipRows


def isolated_manager() -> Neo4jManagerSingleton:
    manager = object.__new__(Neo4jManagerSingleton)
    manager._initialized_flag = False
    Neo4jManagerSingleton.__init__(manager)
    return manager


@pytest.mark.parametrize("component", ["graph-project-id", "graph-project-id.lock", "ancestor"])
def test_identity_rejects_symlinks_without_touching_external_files(tmp_path: Path, component: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "identity"
    external.write_text(PROJECT_ID)
    project = tmp_path / "project"
    if component == "ancestor":
        project.symlink_to(outside, target_is_directory=True)
    else:
        project.mkdir()
        (project / component).symlink_to(external)
    with pytest.raises((OSError, ValueError)):
        load_graph_project_id(project)
    assert external.read_text() == PROJECT_ID
    assert sorted(path.name for path in outside.iterdir()) == ["identity"]


class ConnectionRows(OwnershipRows):
    def single(self, strict: bool = False) -> dict[str, Any]:
        assert len(self) == 1
        return self[0]


class ConnectionDriver(OwnershipDriver):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure
        self.closed_count = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = False
        self.closed_while_checking = False
        if failure == "ownership":
            self.transaction.owner = "22222222-2222-4222-8222-222222222222"

    def verify_connectivity(self) -> None:
        self.started.set()
        try:
            if self.failure == "connectivity":
                raise ServiceUnavailable("synthetic unreachable")
            if self.failure == "cancellation":
                assert self.release.wait(3)
        finally:
            self.finished = True

    def run(self, query: str) -> Any:
        if query == "RETURN apoc.version() AS version":
            return ConnectionRows([{"version": "5.26.8"}])
        if "CALL dbms.components()" in query:
            return ConnectionRows([{"component": "Neo4j", "version": "5.26.8", "edition": "community"}])
        return super().run(query)

    def close(self) -> None:
        self.closed_count += 1
        self.closed_while_checking = not self.finished


@pytest.mark.parametrize("failure", ["connectivity", "ownership", "cancellation", ""])
async def test_connection_driver_is_closed_exactly_once_on_failure_or_final_close(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = isolated_manager()
    manager.bind_project(PROJECT_ID)
    driver = ConnectionDriver(failure)
    monkeypatch.setattr(GraphDatabase, "driver", lambda *args, **kwargs: driver)
    if failure == "cancellation":
        pending = asyncio.create_task(manager.connect())
        assert await asyncio.to_thread(driver.started.wait, 3)
        pending.cancel()
        driver.release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
    elif failure:
        with pytest.raises(Exception, match="synthetic unreachable|ownership"):
            await manager.connect()
    else:
        await manager.connect()
        assert manager.driver is driver
        assert driver.closed_count == 0
    if failure:
        assert driver.closed_count == 1
        assert manager.driver is None
    await manager.close()
    assert driver.closed_count == 1
    assert driver.closed_while_checking is False
