from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager

OPERATIONS = ["load_text", "load_text_strict", "load_json", "load_json_strict", "load_binary", "load_list_of_texts", "exists", "delete"]


def reference(path: str) -> dict[str, Any]:
    payload = b'["outside"]'
    return {"path": path, "content_type": "draft", "version": 1, "size_bytes": len(payload), "checksum": hashlib.sha256(payload).hexdigest()}


def test_imports_resolve_to_assigned_tree() -> None:
    import utils.file_io

    root = Path(__file__).absolute().parents[2]
    assert Path(inspect.getfile(ContentManager)) == root / "core/langgraph/content_manager.py"
    assert Path(inspect.getfile(utils.file_io)) == root / "utils/file_io.py"


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("attack", ["traversal", "absolute", "sibling", "project_artifact"])
def test_references_cannot_cross_content_boundary(tmp_path: Path, operation: str, attack: str) -> None:
    project = tmp_path / "project"
    manager = ContentManager(str(project))
    outside = tmp_path / "outside.json"
    outside.write_bytes(b'["outside"]')
    sibling = project / ".saga/content_sibling/outside.json"
    sibling.parent.mkdir()
    sibling.write_bytes(b'["outside"]')
    artifact = project / "outside.json"
    artifact.write_bytes(b'["outside"]')
    paths = {"traversal": "../outside.json", "absolute": str(outside), "sibling": ".saga/content/../content_sibling/outside.json", "project_artifact": "outside.json"}
    with pytest.raises(ValueError):
        getattr(manager, operation)(reference(paths[attack]))
    assert outside.read_bytes() == sibling.read_bytes() == artifact.read_bytes() == b'["outside"]'


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("path", [None, 1, [], "", " ", ".saga/content", ".saga/content/draft/../file.json", ".saga/content/draft/\x00.json", ".saga/content/draft\\file.json"])
def test_malformed_reference_paths_fail_closed(tmp_path: Path, operation: str, path: Any) -> None:
    manager = ContentManager(str(tmp_path))
    with pytest.raises(ValueError):
        getattr(manager, operation)({"path": path})
    assert list(manager.content_dir.iterdir()) == []


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("link_position", ["leaf", "nested", "bucket", "content", "saga", "project_parent"])
def test_symlinks_cannot_redirect_reference_operations(tmp_path: Path, operation: str, link_position: str) -> None:
    parent = tmp_path / "parent"
    project = parent / "project"
    manager = ContentManager(str(project))
    inside = manager.content_dir / "draft/nested/file.json"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b'["outside"]')
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "file.json"
    sentinel.write_bytes(b'["outside"]')
    target = {"leaf": inside, "nested": inside.parent, "bucket": inside.parent.parent, "content": manager.content_dir, "saga": project / ".saga", "project_parent": parent}[link_position]
    parked = target.with_name(target.name + "-parked")
    target.rename(parked)
    if link_position == "leaf":
        target.symlink_to(sentinel)
    else:
        suffix = inside.relative_to(target)
        external = outside / suffix
        external.parent.mkdir(parents=True, exist_ok=True)
        external.write_bytes(b'["outside"]')
        target.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        getattr(manager, operation)(reference(".saga/content/draft/nested/file.json"))
    assert sentinel.read_bytes() == b'["outside"]'
    if link_position != "leaf":
        assert external.read_bytes() == b'["outside"]'


@pytest.mark.parametrize("bucket", ["../content_sibling", "../escape", "/absolute", "nested/bucket", "..", "", "draft\\escape"])
@pytest.mark.parametrize("operation", ["save_text", "save_json", "save_binary", "get_latest_version"])
def test_invalid_buckets_have_no_filesystem_effect(tmp_path: Path, bucket: str, operation: str) -> None:
    manager = ContentManager(str(tmp_path))
    if bucket == "/absolute":
        bucket = str(tmp_path / "absolute")
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        if operation == "get_latest_version":
            manager.get_latest_version(bucket, "sample")
        else:
            payload: Any = {"value": 1} if operation == "save_json" else "payload"
            getattr(manager, operation)(payload, bucket, "sample")
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("version", ["1/../../escape", -1, True, 1.5])
def test_invalid_versions_do_not_create_buckets(tmp_path: Path, version: Any) -> None:
    manager = ContentManager(str(tmp_path))
    with pytest.raises(ValueError):
        manager.save_text("payload", "draft", "sample", version)
    assert list(manager.content_dir.iterdir()) == []


@pytest.mark.parametrize("link_position", ["leaf", "bucket", "temporary"])
@pytest.mark.parametrize("operation", ["save_text", "save_json", "save_binary"])
def test_writes_never_follow_symlinks(tmp_path: Path, link_position: str, operation: str) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sample_v1.json"
    sentinel.write_bytes(b"sentinel")
    bucket = manager.content_dir / "draft"
    bucket.mkdir()
    extension = {"save_text": "txt", "save_json": "json", "save_binary": "bin"}[operation]
    destination = bucket / f"sample_v1.{extension}"
    payload: Any = {"value": 1} if operation == "save_json" else (b"payload" if operation == "save_binary" else "payload")
    if link_position == "bucket":
        bucket.rmdir()
        bucket.symlink_to(outside, target_is_directory=True)
    else:
        target = destination.with_suffix(".tmp") if link_position == "temporary" else destination
        target.symlink_to(sentinel)
    if link_position == "temporary":
        getattr(manager, operation)(payload, "draft", "sample")
        assert destination.is_file()
    else:
        with pytest.raises((ValueError, OSError)):
            getattr(manager, operation)(payload, "draft", "sample")
    assert sentinel.read_bytes() == b"sentinel"
    assert sorted(path.name for path in outside.iterdir()) == ["sample_v1.json"]


@pytest.mark.parametrize("link_position", ["project_parent", "saga", "content"])
def test_constructor_does_not_follow_symlinked_parents(tmp_path: Path, link_position: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = tmp_path / "parent"
    project = parent / "project"
    target = {"project_parent": parent, "saga": project / ".saga", "content": project / ".saga/content"}[link_position]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        ContentManager(str(project))
    assert list(outside.iterdir()) == []


def test_valid_relative_reference_roundtrips_and_cleanup(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    first = manager.save_text("héllo\r\n", "draft", "chapter_1", 0)
    second = manager.save_text("later", "draft", "chapter_1", 1)
    imported = json.loads(json.dumps(first))
    assert manager.load_text_strict(imported) == "héllo\r\n"
    with pytest.raises(ValueError, match="explicit"):
        manager.load_text(first["path"])
    with pytest.raises(ValueError, match="explicit"):
        manager.load_text(Path(first["path"]))
    assert manager.load_text(manager.admit_legacy_reference(**first)) == "héllo\r\n"
    assert manager.get_latest_version("draft", "chapter_1") == 1
    manager.delete(second)
    assert manager.exists(second) is False
    assert manager.exists(first) is True
    manager.delete(second)
    assert manager.load_text_strict(first) == "héllo\r\n"


@pytest.mark.parametrize("operation", ["load_text", "save_text", "delete"])
def test_leaf_symlink_swap_at_operation_boundary_preserves_outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    ref = manager.save_text("inside", "draft", "sample")
    target = manager.project_dir / ref["path"]
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("outside")
    original_open = os.open
    original_unlink = os.unlink
    original_replace = os.replace
    swapped = False

    def swap() -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            original_unlink(target)
            target.symlink_to(sentinel)

    def opening(path: Any, flags: int, mode: int = 0o777, *, dir_fd: Any = None) -> int:
        if Path(path).name == target.name:
            swap()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def unlinking(path: Any, *, dir_fd: Any = None) -> None:
        if Path(path).name == target.name:
            swap()
        original_unlink(path, dir_fd=dir_fd)

    def replacing(source: Any, destination: Any, *, src_dir_fd: Any = None, dst_dir_fd: Any = None) -> None:
        swap()
        original_replace(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "open", opening)
    monkeypatch.setattr(os, "unlink", unlinking)
    monkeypatch.setattr(os, "replace", replacing)
    if operation == "load_text":
        with pytest.raises(OSError):
            manager.load_text(ref)
    elif operation == "save_text":
        with pytest.raises(OSError):
            manager.save_text("replacement", "draft", "sample")
        assert target.is_symlink()
    else:
        manager.delete(ref)
        assert target.exists() is False
    assert swapped is True
    assert sentinel.read_text() == "outside"


@pytest.mark.parametrize("operation", ["load_text", "save_text", "delete", "get_latest_version"])
def test_directory_swap_before_open_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    ref = manager.save_text("inside", "draft", "sample")
    bucket = manager.content_dir / "draft"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sample_v1.txt"
    sentinel.write_text("outside")
    original_open = os.open
    swapped = False

    def opening(path: Any, flags: int, mode: int = 0o777, *, dir_fd: Any = None) -> int:
        nonlocal swapped
        if Path(path).name == "draft" and not swapped:
            swapped = True
            bucket.rename(bucket.with_name("parked"))
            bucket.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", opening)
    with pytest.raises(OSError):
        if operation == "save_text":
            manager.save_text("replacement", "draft", "sample")
        elif operation == "get_latest_version":
            manager.get_latest_version("draft", "sample")
        else:
            getattr(manager, operation)(ref)
    assert swapped is True
    assert sentinel.read_text() == "outside"
    assert sorted(path.name for path in outside.iterdir()) == ["sample_v1.txt"]


@pytest.mark.parametrize("failure", ["fsync", "link"])
def test_failed_write_cleans_only_its_private_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    ref = manager.save_text("inside", "draft", "sample")
    bucket = manager.content_dir / "draft"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    planted = bucket / "sample_v1.tmp"
    planted.symlink_to(outside)

    def failing(*arguments: Any, **keywords: Any) -> None:
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(os, failure, failing)
    with pytest.raises(OSError, match="^synthetic publication failure$"):
        manager.save_text("replacement", "draft", "sample", 2)
    assert manager.load_text_strict(ref) == "inside"
    assert sorted(path.name for path in bucket.iterdir()) == ["sample_v1.tmp", "sample_v1.txt"]
    assert planted.is_symlink()
    assert outside.read_text() == "outside"


@pytest.mark.parametrize("operation", OPERATIONS)
def test_directory_references_are_not_files(tmp_path: Path, operation: str) -> None:
    manager = ContentManager(str(tmp_path))
    (manager.content_dir / "draft").mkdir()
    with pytest.raises(ValueError):
        getattr(manager, operation)(reference(".saga/content/draft"))
    assert (manager.content_dir / "draft").is_dir()


@pytest.mark.parametrize("operation", ["load_text", "exists", "delete", "save_text"])
def test_special_files_fail_without_blocking(tmp_path: Path, operation: str) -> None:
    manager = ContentManager(str(tmp_path))
    bucket = manager.content_dir / "draft"
    bucket.mkdir()
    os.mkfifo(bucket / "sample_v1.txt")
    with pytest.raises(ValueError):
        if operation == "save_text":
            manager.save_text("replacement", "draft", "sample")
        else:
            getattr(manager, operation)(reference(".saga/content/draft/sample_v1.txt"))


@pytest.mark.parametrize("link_position", ["bucket", "leaf"])
def test_version_discovery_rejects_symlinks(tmp_path: Path, link_position: str) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sample_v9.txt"
    sentinel.write_text("outside")
    bucket = manager.content_dir / "draft"
    if link_position == "bucket":
        bucket.symlink_to(outside, target_is_directory=True)
    else:
        bucket.mkdir()
        (bucket / sentinel.name).symlink_to(sentinel)
    with pytest.raises((ValueError, OSError)):
        manager.get_latest_version("draft", "sample")
    assert sentinel.read_text() == "outside"


def test_version_lookup_uses_literal_sanitized_identifier(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    manager.save_text("inside", "draft", "a[1]..part", 2)
    manager.save_text("other", "draft", "a1_part", 9)
    assert manager.get_latest_version("draft", "a[1]..part") == 2
    assert manager.get_latest_version("missing", "a[1]..part") == 0
    assert (manager.content_dir / "missing").exists() is False


def test_absolute_in_root_references_are_rejected(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    ref = manager.save_text("inside", "draft", "sample")
    with pytest.raises(ValueError):
        manager.load_text(manager.project_dir / ref["path"])
    assert manager.load_text(ref) == "inside"


def test_relative_project_root_keeps_project_relative_references(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    manager = ContentManager("project")
    ref = manager.save_text("inside", "draft", "sample")
    assert ref["path"] == ".saga/content/draft/sample_v1.txt"
    assert manager.load_text_strict(ref) == "inside"


@pytest.mark.parametrize("operation", ["load_text", "save_text", "delete", "failed_write_cleanup"])
def test_opened_parent_remains_anchored_after_symlink_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    ref = manager.save_text("inside", "draft", "sample")
    bucket = manager.content_dir / "draft"
    parked = manager.content_dir / "parked"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sample_v1.txt"
    sentinel.write_text("outside")
    original_open = os.open
    original_link = os.link
    original_unlink = os.unlink
    swapped = False

    def swap() -> None:
        nonlocal swapped
        if not swapped:
            swapped = True
            bucket.rename(parked)
            bucket.symlink_to(outside, target_is_directory=True)

    def opening(path: Any, flags: int, mode: int = 0o777, *, dir_fd: Any = None) -> int:
        if Path(path).name == "sample_v1.txt":
            swap()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    def linking(source: Any, destination: Any, *, src_dir_fd: Any = None, dst_dir_fd: Any = None, follow_symlinks: bool = True) -> None:
        swap()
        if operation == "failed_write_cleanup":
            raise OSError("synthetic publication failure")
        original_link(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, follow_symlinks=follow_symlinks)

    def unlinking(path: Any, *, dir_fd: Any = None) -> None:
        swap()
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", opening)
    monkeypatch.setattr(os, "link", linking)
    monkeypatch.setattr(os, "unlink", unlinking)
    if operation == "load_text":
        assert manager.load_text(ref) == "inside"
    elif operation == "save_text":
        manager.save_text("replacement", "draft", "sample", 2)
        assert (parked / "sample_v2.txt").read_text() == "replacement"
        assert (parked / "sample_v1.txt").read_text() == "inside"
    elif operation == "failed_write_cleanup":
        with pytest.raises(OSError, match="^synthetic publication failure$"):
            manager.save_text("replacement", "draft", "sample", 2)
        assert sorted(path.name for path in parked.iterdir()) == ["sample_v1.txt"]
        assert (parked / "sample_v1.txt").read_text() == "inside"
    else:
        manager.delete(ref)
        assert list(parked.iterdir()) == []
    assert swapped is True
    assert sentinel.read_text() == "outside"
    assert sorted(path.name for path in outside.iterdir()) == ["sample_v1.txt"]


@pytest.mark.parametrize("extension", ["../escape", "/absolute", "", "txt/nested", ".."])
def test_invalid_extensions_do_not_create_buckets(tmp_path: Path, extension: str) -> None:
    manager = ContentManager(str(tmp_path))
    with pytest.raises(ValueError):
        manager._get_content_path("draft", "sample", 1, extension)
    assert list(manager.content_dir.iterdir()) == []


def test_private_writer_validates_destination_independently(tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path / "project"))
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    with pytest.raises(ValueError):
        manager._write_bytes_atomically(outside, b"replacement")
    assert outside.read_text() == "outside"
