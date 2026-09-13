"""Projection completeness from checksum-selected inputs."""
import json
from pathlib import Path

import pytest
import yaml

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.catalog import materialize_entities
from core.langgraph.initialization.commit_init_node import _parse_world_items_extraction
from core.langgraph.initialization.snapshot import select_inputs, select_snapshot
from core.langgraph.initialization.staged_import import InitializationImport
from tests.test_staged_initialization import example_state


@pytest.mark.parametrize("case", ["world", "outline", "character"])
def test_projection_contains_all_selected_data(tmp_path: Path, case: str) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    items = _parse_world_items_extraction('[{"name":"Compass","category":"object","description":" Brass\\n"}]')
    catalog = materialize_entities(select_inputs(state), items, ())
    state["initialization_catalog_ref"] = manager.save_json(catalog.model_dump(mode="json"), "initialization_catalog", catalog.inputs.identity, version=2)
    relationships = dict(schema_version=2, project_id=catalog.inputs.project_id, catalog_identity=catalog.identity, relationships=[], evidence=[])
    state["outline_relationships_ref"] = manager.save_json(relationships, "outline_relationships", "selected", version=2)
    snapshot = select_snapshot(state)
    importer = InitializationImport(str(tmp_path))
    (tmp_path / importer.root).mkdir(parents=True, exist_ok=True)
    projections = json.loads(importer.projection_manifest(snapshot))
    if case == "world":
        projected = yaml.safe_load(projections["world/items.yaml"])["items"]
        assert len(projected) == 1
        assert projected[0] == {key: getattr(items[0], key) for key in ("id", "name", "category", "description")}
    elif case == "outline":
        projected = yaml.safe_load(projections["outline/beats.yaml"])
        assert projected["selected_outlines"] == {name: snapshot.source(name) for name in ("global_outline", "act_outlines", "chapter_outlines")}
    else:
        projected = yaml.safe_load(projections["characters/ada.yaml"])
        assert projected["selected_sheet"] == snapshot.source("character_sheets")["Ada"]
