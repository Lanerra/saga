"""Persisted extraction admission through real, checksum-bound content files."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from core.exceptions import ContentIntegrityError
from core.langgraph import content_manager
from core.langgraph.content_manager import (
    ContentManager,
    get_extracted_entities,
    get_extracted_relationships,
    get_scene_drafts,
    save_extracted_entities,
    save_extracted_relationships,
    save_scenes,
)

Getter = Callable[[Mapping[str, Any], ContentManager], object]
GETTERS = [
    pytest.param(get_extracted_relationships, "extracted_relationships", [], id="relationships"),
    pytest.param(get_extracted_entities, "extracted_entities", {}, id="entities"),
    pytest.param(get_scene_drafts, "scene_drafts", [], id="scenes"),
]


def test_import_uses_assigned_worktree() -> None:
    assert Path(content_manager.__file__).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/content_manager.py"


@pytest.mark.parametrize("payload", [{"unexpected": []}, {}, None, "", 0, False])
def test_relationships_reject_non_list_artifacts(tmp_path: Path, payload: object) -> None:
    manager = ContentManager(str(tmp_path))
    reference = manager.save_json(payload, "extracted_relationships", "chapter_1")
    assert manager.load_json_strict(reference) == payload

    with pytest.raises(ValueError, match="Expected list extracted relationships"):
        get_extracted_relationships({"extracted_relationships_ref": reference, "extracted_relationships": []}, manager)


@pytest.mark.parametrize("payload", [[], ["unexpected"], None, "", 0, False])
def test_entities_reject_non_dict_artifacts(tmp_path: Path, payload: object) -> None:
    manager = ContentManager(str(tmp_path))
    reference = manager.save_json(payload, "extracted_entities", "chapter_1")
    assert manager.load_json_strict(reference) == payload

    with pytest.raises(ValueError, match="Expected dict extracted entities"):
        get_extracted_entities({"extracted_entities_ref": reference, "extracted_entities": {}}, manager)


@pytest.mark.parametrize("payload", [{"unexpected": []}, {}, None, "", 0, False])
def test_scenes_reject_non_list_artifacts(tmp_path: Path, payload: object) -> None:
    manager = ContentManager(str(tmp_path))
    reference = manager.save_json(payload, "scenes", "chapter_1")

    with pytest.raises(ValueError, match="Expected list"):
        get_scene_drafts({"scene_drafts_ref": reference}, manager)


@pytest.mark.parametrize("getter,key,empty", GETTERS)
@pytest.mark.parametrize("reference,error,match", [
    ({}, ValueError, "required key 'path'"),
    ([], TypeError, "expected ContentRef"),
    ("", ValueError, "path"),
    (False, TypeError, "expected ContentRef"),
    (0, TypeError, "expected ContentRef"),
])
def test_malformed_reference_is_not_optional_absence(
    tmp_path: Path, getter: Getter, key: str, empty: object, reference: object, error: type[Exception], match: str,
) -> None:
    manager = ContentManager(str(tmp_path))
    with pytest.raises(error, match=match):
        getter({f"{key}_ref": reference, key: empty}, manager)


@pytest.mark.parametrize("getter,key,empty", GETTERS)
def test_optional_absence_and_valid_empty_artifacts(tmp_path: Path, getter: Getter, key: str, empty: object) -> None:
    manager = ContentManager(str(tmp_path))
    assert getter({}, manager) == empty
    assert getter({f"{key}_ref": None}, manager) == empty
    reference = manager.save_json(empty, key, "chapter_1")
    assert getter({f"{key}_ref": reference, key: ["stale inline value"]}, ContentManager(str(tmp_path))) == empty


@pytest.mark.parametrize("getter,key,payload", [
    (get_extracted_relationships, "extracted_relationships", [{"source_name": "Exact-ID_025"}]),
    (get_extracted_entities, "extracted_entities", {"characters": [{"id": "Exact-ID_025"}]}),
])
def test_absent_reference_preserves_inline_compatibility(tmp_path: Path, getter: Getter, key: str, payload: object) -> None:
    manager = ContentManager(str(tmp_path))
    assert getter({key: payload}, manager) is payload
    assert getter({f"{key}_ref": None, key: payload}, manager) is payload


@pytest.mark.parametrize("getter,key,empty", GETTERS)
def test_referenced_integrity_and_missing_file_failures_propagate(tmp_path: Path, getter: Getter, key: str, empty: object) -> None:
    manager = ContentManager(str(tmp_path))
    reference = manager.save_json(empty, key, "chapter_1")
    bad_checksum = {**reference, "checksum": "0" * 64}
    with pytest.raises(ContentIntegrityError, match="checksum mismatch"):
        getter({f"{key}_ref": bad_checksum, key: empty}, manager)
    with pytest.raises(FileNotFoundError):
        getter({f"{key}_ref": reference, key: empty}, ContentManager(str(tmp_path / "other-project")))


def test_successful_reopen_preserves_payload_and_checkpoint_history(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    entities = {"characters": [{"name": "Astra", "id": "Character_025", "attributes": {"description": "Exact source text."}}], "world_items": []}
    relationships = [{"source_name": "Astra", "target_name": "Archive_029", "relationship_type": "VISITS", "description": "Exact source text.", "chapter": 1, "confidence": 0.75}]
    drafts = ["Astra enters the archive.\n", ""]
    state = {
        "extracted_entities_ref": save_extracted_entities(manager, entities, 1),
        "extracted_relationships_ref": save_extracted_relationships(manager, relationships, 1),
        "scene_drafts_ref": save_scenes(manager, drafts, 1),
    }
    checkpoint = json.dumps(state)
    snapshots = {reference["path"]: ((tmp_path / reference["path"]).read_bytes(), (tmp_path / reference["path"]).stat().st_ino) for reference in state.values()}
    save_extracted_entities(manager, {"characters": [], "world_items": []}, 1, version=2)
    save_extracted_relationships(manager, [], 1, version=2)
    save_scenes(manager, [], 1, version=2)

    reopened = ContentManager(str(tmp_path))
    restored = json.loads(checkpoint)
    assert get_extracted_entities(restored, reopened) == entities
    assert get_extracted_relationships(restored, reopened) == relationships
    assert get_scene_drafts(restored, reopened) == drafts
    assert save_extracted_relationships(reopened, relationships, 1) == state["extracted_relationships_ref"]
    for path, (data, inode) in snapshots.items():
        assert (tmp_path / path).read_bytes() == data
        assert (tmp_path / path).stat().st_ino == inode
    assert json.dumps(state) == checkpoint
