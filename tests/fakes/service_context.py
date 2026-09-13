"""Patch a borrowed run interface at execution time, including decorated cases."""
from typing import Any
from unittest.mock import DEFAULT, _patch

from core.service_context import get_services
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager


def configure_empty_entity_names(database: FakeNeo4jManager) -> None:
    """Declare only the known-name read for a fresh synthetic entity graph."""
    database.configure_response(
        r"\A\s*MATCH \(n\)\s+WHERE n:Character OR n:Location OR n:Event OR n:Item\s+RETURN DISTINCT toLower\(n\.name\) AS name\s*\Z",
        [],
    )


def patch_service(target: str, new: Any = DEFAULT, *, new_callable: Any = None, **keywords: Any) -> Any:
    parts = target.split(".")
    if parts[0] not in {"language_model", "database"}:
        raise ValueError(f"Unknown run interface: {target}")

    def owner_at_entry() -> Any:
        owner: Any = get_services()
        for part in parts[:-1]:
            owner = getattr(owner, part)
        return owner

    return _patch(owner_at_entry, parts[-1], new, None, False, None, None, new_callable, keywords)
