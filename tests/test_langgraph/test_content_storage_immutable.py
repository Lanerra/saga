"""Immutable storage behavior on disposable, real filesystem artifacts."""

import os
import stat
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from core.exceptions import ContentIntegrityError
from core.langgraph.content_manager import ContentManager, ContentRef
from core.service_context import get_services


@pytest.mark.parametrize("kind", ["text", "json", "binary", "list_of_texts"])
def test_identity_conflict_preserves_every_old_reference(tmp_path: Path, kind: str) -> None:
    manager = ContentManager(str(tmp_path))
    payloads: dict[str, tuple[Any, Any]] = {"text": ("first 雨\r\n", "second"), "json": ({"first": 1}, {"second": 2}), "binary": (b"first", b"second"), "list_of_texts": (["first"], ["second"])}
    first, second = payloads[kind]
    save = getattr(manager, "save_" + kind)
    original = save(first, "draft", "chapter_1", 1)
    path = tmp_path / original["path"]
    original_bytes = path.read_bytes()
    identity = path.stat()
    assert save(first, "draft", "chapter_1", 1) == original
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (identity.st_ino, identity.st_mtime_ns)
    with pytest.raises(FileExistsError, match="Immutable content conflict"):
        save(second, "draft", "chapter_1", 1)
    later = save(second, "draft", "chapter_1", 2)
    assert later["path"] != original["path"]
    assert path.read_bytes() == original_bytes
    assert manager._compute_checksum(original_bytes) == original["checksum"]


@pytest.mark.parametrize("operation", ["__ior__", "__init__", "__setitem__", "__delitem__", "clear", "pop", "popitem", "setdefault", "update"])
def test_reference_mutators_reject_changes(tmp_path: Path, operation: str) -> None:
    reference: Any = ContentManager(str(tmp_path)).save_text("first", "draft", "chapter_1")
    expected = dict(reference)
    arguments = {"__ior__": ({"path": "changed"},), "__init__": ({"path": "changed"},), "__setitem__": ("path", "changed"), "__delitem__": ("path",), "clear": (), "pop": ("path",), "popitem": (), "setdefault": ("missing", "changed"), "update": ({"path": "changed"},)}
    with pytest.raises(TypeError, match="ContentRef is immutable"):
        getattr(reference, operation)(*arguments[operation])
    assert dict(reference) == expected


def test_reference_copy_protocols_preserve_checkpoint_values(tmp_path: Path) -> None:
    reference = ContentManager(str(tmp_path)).save_text("first", "draft", "chapter_1")
    assert copy(reference) == reference
    assert deepcopy(reference) == reference


@pytest.mark.parametrize("operation", ["load_text", "load_json", "load_binary", "load_text_strict", "load_json_strict"])
@pytest.mark.parametrize("field,value", [("checksum", None), ("size_bytes", None), ("size_bytes", True), ("size_bytes", 99), ("version", True), ("version", -1), ("content_type", "../escape")])
def test_all_readers_require_integrity_metadata(tmp_path: Path, operation: str, field: str, value: Any) -> None:
    manager = ContentManager(str(tmp_path))
    reference = dict(manager.save_json(["first"], "draft", "chapter_1"))
    if value is None:
        reference.pop(field)
    else:
        reference[field] = value
    with pytest.raises((ContentIntegrityError, ValueError)):
        getattr(manager, operation)(reference)


@pytest.mark.parametrize("operation", ["load_text", "load_json", "load_binary"])
def test_path_only_load_requires_explicit_admission(tmp_path: Path, operation: str) -> None:
    manager = ContentManager(str(tmp_path))
    reference = manager.save_json(["first"], "draft", "chapter_1")
    for path in [reference["path"], Path(reference["path"])]:
        with pytest.raises(ValueError, match="explicit"):
            getattr(manager, operation)(path)


def test_legacy_admission_requires_expected_bytes_and_preserves_source(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    original = manager.save_text("legacy 雨\r\n", "draft", "chapter_1", 0)
    path = tmp_path / original["path"]
    identity = path.stat()
    admitted = manager.admit_legacy_reference(**original)
    assert admitted == original
    assert manager.load_text_strict(admitted) == "legacy 雨\r\n"
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (identity.st_ino, identity.st_mtime_ns)
    invalid: dict[str, Any] = {**original, "checksum": "0" * 64}
    with pytest.raises(ContentIntegrityError, match="checksum mismatch"):
        manager.admit_legacy_reference(**invalid)


async def test_assembly_keeps_partial_scene_checkpoint_readable(tmp_path: Path) -> None:
    from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
    from core.langgraph.state import NarrativeState

    manager = ContentManager(str(tmp_path))
    first = manager.save_list_of_texts(["first"], "scenes", "chapter_1", 1)
    second = manager.save_list_of_texts(["first", "second"], "scenes", "chapter_1", 2)
    state = cast(NarrativeState, {"project_dir": str(tmp_path), "current_chapter": 1, "scene_drafts_ref": second})
    update = await assemble_chapter(state)
    assert manager.load_json_strict(first) == ["first"]
    assert manager.load_json_strict(second) == ["first", "second"]
    assert manager.load_text_strict(cast(ContentRef, update["draft_ref"])) == "first\n\n# ***\n\nsecond"
    assert manager.load_json_strict(cast(ContentRef, update["scene_drafts_ref"])) == ["first", "second"]


@pytest.mark.parametrize("skeleton", [False, True])
async def test_chapter_outline_updates_preserve_prior_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, skeleton: bool) -> None:
    from core.langgraph.initialization.chapter_outline_node import generate_chapter_outline
    from core.langgraph.state import create_initial_state
    

    manager = ContentManager(str(tmp_path))
    state = create_initial_state(project_id="example", project_dir=str(tmp_path), title="Example", genre="Fantasy", theme="Return", setting="Room", total_chapters=2, target_word_count=2000, protagonist_name="Traveler")
    if skeleton:
        state["chapter_outlines_ref"] = manager.save_json({"1": {"version": 0, "act_number": 1}, "2": {"version": 0, "act_number": 1}}, "chapter_outlines", "all", 0)

    async def answer(**keywords: Any) -> tuple[str, dict[str, int]]:
        return '{"scene_description": "A traveler returns", "key_beats": ["arrival"], "plot_point": "Return"}', {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", answer)
    first_update = await generate_chapter_outline(state)
    first = cast(ContentRef, first_update["chapter_outlines_ref"])
    first_bytes = (tmp_path / first["path"]).read_bytes()
    state.update(first_update)
    state["current_chapter"] = 2
    second_update = await generate_chapter_outline(state)
    second = cast(ContentRef, second_update["chapter_outlines_ref"])
    assert (first["version"], second["version"]) == (1, 2)
    assert (tmp_path / first["path"]).read_bytes() == first_bytes
    assert set(manager.load_json_strict(second)) == {"1", "2"}


@pytest.mark.parametrize("operation", ["extract_from_scenes", "consolidate_extraction"])
async def test_relationship_storage_uses_its_own_version_counter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    from core.langgraph.nodes.extraction_nodes import consolidate_extraction
    from core.langgraph.nodes.scene_extraction import extract_from_scenes
    from core.langgraph.state import NarrativeState
    

    manager = ContentManager(str(tmp_path))
    previous = manager.save_json([{"retained": "earlier normalization"}], "extracted_relationships", "chapter_1", 1)
    state = cast(NarrativeState, {"project_dir": str(tmp_path), "current_chapter": 1, "scene_drafts_ref": manager.save_list_of_texts(["Synthetic room."], "scenes", "chapter_1")})

    async def answer(**keywords: Any) -> tuple[dict[str, Any], None]:
        if "response_format" in keywords:
            return {"kg_triples": []}, None
        return {"character_updates": {}, "world_updates": {"Location": {}, "Event": {}}, "kg_triples": []}, None

    monkeypatch.setattr(get_services().language_model, "async_call_llm_json_object", answer)
    update = await {"extract_from_scenes": extract_from_scenes, "consolidate_extraction": consolidate_extraction}[operation](state)
    current = cast(ContentRef, update["extracted_relationships_ref"])
    assert current["version"] == 2
    assert manager.load_json_strict(current) == []
    assert manager.load_json_strict(previous) == [{"retained": "earlier normalization"}]


@pytest.mark.parametrize("target", ["traversal", "absolute", "sibling", "symlink", "fifo"])
def test_legacy_admission_preserves_containment(tmp_path: Path, target: str) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    original = manager.save_text("original", "draft", "chapter_1")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"original")
    paths = {"traversal": "../outside.txt", "absolute": str(outside), "sibling": ".saga/content/../content_sibling/outside.txt", "symlink": ".saga/content/draft/link.txt", "fifo": ".saga/content/draft/fifo.txt"}
    if target == "symlink":
        (manager.project_dir / paths[target]).symlink_to(outside)
    if target == "fifo":
        os.mkfifo(manager.project_dir / paths[target])
    invalid: dict[str, Any] = {**original, "path": paths[target]}
    with pytest.raises((ValueError, OSError)):
        manager.admit_legacy_reference(**invalid)
    assert outside.read_bytes() == b"original"
    assert manager.load_text(original) == "original"


def test_publication_collision_with_symlink_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    original = manager.save_text("original", "draft", "chapter_1")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    link = os.link
    destination = manager.content_dir / "draft/chapter_1_v2.txt"

    def collision(source: Any, target: Any, **keywords: Any) -> None:
        destination.symlink_to(outside)
        link(source, target, **keywords)

    monkeypatch.setattr(os, "link", collision)
    with pytest.raises(OSError):
        manager.save_text("new", "draft", "chapter_1", 2)
    assert destination.is_symlink()
    assert outside.read_text() == "outside"
    assert manager.load_text(original) == "original"
    assert sorted(path.name for path in destination.parent.iterdir()) == ["chapter_1_v1.txt", "chapter_1_v2.txt"]


@pytest.mark.parametrize("boundary", ["file_sync", "directory_sync"])
def test_sync_failure_propagates_and_retry_preserves_prior_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    manager = ContentManager(str(tmp_path))
    original = manager.save_text("original", "draft", "chapter_1")
    target = manager.content_dir / "draft/chapter_1_v2.txt"
    sync = os.fsync
    injected = False

    def fail(descriptor: int) -> None:
        nonlocal injected
        regular = stat.S_ISREG(os.fstat(descriptor).st_mode)
        at_boundary = regular if boundary == "file_sync" else (not regular and target.exists())
        if at_boundary and not injected:
            injected = True
            raise OSError("synthetic sync failure")
        sync(descriptor)

    with monkeypatch.context() as context:
        context.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="synthetic sync failure"):
            manager.save_text("new", "draft", "chapter_1", 2)
    assert injected is True
    assert target.exists() is (boundary == "directory_sync")
    inode = target.stat().st_ino if target.exists() else None
    revised = manager.save_text("new", "draft", "chapter_1", 2)
    if inode is not None:
        assert target.stat().st_ino == inode
    assert manager.load_text(original) == "original"
    assert manager.load_text(revised) == "new"
    assert sorted(path.name for path in target.parent.iterdir()) == ["chapter_1_v1.txt", "chapter_1_v2.txt"]


def test_lock_inode_survives_content_writes_and_reopen(tmp_path: Path) -> None:
    from utils.file_io import ContainedFiles

    files = ContainedFiles(tmp_path, durable=True)
    with files.exclusive_lock("writer.lock"):
        identity = (tmp_path / "writer.lock").stat().st_ino
        with pytest.raises(BlockingIOError):
            with ContainedFiles(tmp_path, durable=True).exclusive_lock("writer.lock"):
                pytest.fail("Second writer entered")
        manager = ContentManager(str(tmp_path))
        reference = manager.save_text("original", "draft", "chapter_1")
        assert manager.save_text("original", "draft", "chapter_1") == reference
    with files.exclusive_lock("writer.lock"):
        assert (tmp_path / "writer.lock").stat().st_ino == identity
        assert manager.load_text(reference) == "original"
