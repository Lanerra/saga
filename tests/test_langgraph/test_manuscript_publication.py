"""Durable manuscript publication uses real synthetic files, not fake writers."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager, get_draft_text
from core.langgraph.export import generate_full_export
from core.langgraph.nodes import finalize_node
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from tests.fakes.quality import example_quality_state


@pytest.fixture
def publication_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NarrativeState:
    manager = ContentManager(str(tmp_path))

    async def no_embedding(text: str) -> None:
        return None

    async def acknowledge(**keywords: Any) -> None:
        return None

    monkeypatch.setattr(get_services().language_model, "async_get_embedding", no_embedding)
    monkeypatch.setattr(finalize_node, "save_chapter_data_to_db", acknowledge)
    return {
        "project_dir": str(tmp_path),
        "current_chapter": 1,
        "draft_ref": manager.save_text("Previous synthetic prose.", "draft", "chapter_1", 1),
    }


async def test_acceptance_binds_exact_canonical_bytes(publication_state: NarrativeState, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    outcome = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert outcome["last_error"] is None
    receipt = json.loads((tmp_path / "chapters/chapter_001.accepted.json").read_bytes())
    canonical = (tmp_path / receipt["artifact_path"]).read_bytes()
    assert receipt["markdown_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert receipt["markdown_size"] == len(canonical)
    assert canonical[receipt["body_offset"]:] == b"Previous synthetic prose."
    assert receipt["body_sha256"] == hashlib.sha256(b"Previous synthetic prose.").hexdigest()
    assert "accepted_with_exceptions" in caplog.text
    assert ".quality.json" in caplog.text


async def test_failed_mirror_preserves_previous_accepted_export(publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    manager = ContentManager(str(tmp_path))
    publication_state["draft_ref"] = manager.save_text("Candidate synthetic prose.", "draft", "chapter_1", 2)
    original_replace = os.replace

    def fail_text_mirror(source: Any, destination: Any, **keywords: Any) -> None:
        if Path(destination).name == "chapter_001.txt":
            raise OSError("synthetic mirror interruption")
        original_replace(source, destination, **keywords)

    monkeypatch.setattr(os, "replace", fail_text_mirror)
    outcome = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert outcome["has_fatal_error"] is True
    assert get_draft_text(publication_state, manager) == "Candidate synthetic prose."
    assert generate_full_export(tmp_path).read_bytes() == b"Previous synthetic prose.\n"


async def test_export_preserves_literal_backslashes_and_unicode(publication_state: NarrativeState, tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    body = "A literal \\n remains literal. Café.\nA real newline."
    publication_state["draft_ref"] = manager.save_text(body, "draft", "chapter_1", 2)
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    assert generate_full_export(tmp_path).read_bytes() == (body + "\n").encode()


async def test_manuscript_directory_sync_failure_blocks_graph(publication_state: NarrativeState, monkeypatch: pytest.MonkeyPatch) -> None:
    graph_calls: list[dict[str, Any]] = []

    async def acknowledge(**keywords: Any) -> None:
        graph_calls.append(keywords)

    original_sync = os.fsync

    def fail_directory(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and Path(os.readlink(f"/proc/self/fd/{descriptor}")).name == "chapters":
            raise OSError("synthetic directory sync failure")
        original_sync(descriptor)

    monkeypatch.setattr(finalize_node, "save_chapter_data_to_db", acknowledge)
    monkeypatch.setattr(os, "fsync", fail_directory)
    outcome = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert outcome.get("has_fatal_error") is True
    assert graph_calls == []


async def test_failed_graph_does_not_publish_candidate(publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    manager = ContentManager(str(tmp_path))
    publication_state["draft_ref"] = manager.save_text("Unacknowledged candidate.", "draft", "chapter_1", 2)

    async def fail_graph(**keywords: Any) -> None:
        raise OSError("synthetic graph acknowledgement failure")

    monkeypatch.setattr(finalize_node, "save_chapter_data_to_db", fail_graph)
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["has_fatal_error"] is True
    assert generate_full_export(tmp_path).read_bytes() == b"Previous synthetic prose.\n"


@pytest.mark.parametrize("target", ["canonical", "prepared", "markdown_mirror", "text_mirror", "accepted"])
@pytest.mark.parametrize("operation", ["file_sync", "replace", "directory_sync"])
async def test_publication_fault_matrix_preserves_accepted_prose(
    publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, operation: str,
) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    previous = (tmp_path / "chapters/chapter_001.accepted.json").read_bytes()
    manager = ContentManager(str(tmp_path))
    publication_state["draft_ref"] = manager.save_text("Recoverable candidate.", "draft", "chapter_1", 2)
    original_replace, original_sync = os.replace, os.fsync
    armed = False
    injected = False
    file_syncs = 0
    graph_calls: list[dict[str, Any]] = []

    async def acknowledge(**keywords: Any) -> None:
        graph_calls.append(keywords)

    def replace(source: Any, destination: Any, **keywords: Any) -> None:
        nonlocal armed, injected
        name = Path(destination).name
        matches = {
            "canonical": len(name) == 67 and name.endswith(".md"),
            "prepared": name.endswith(".prepared.json"),
            "markdown_mirror": name == "chapter_001.md",
            "text_mirror": name == "chapter_001.txt",
            "accepted": name == "chapter_001.accepted.json",
        }
        if matches[target] and not injected:
            if operation == "replace":
                injected = True
                raise OSError("synthetic publication fault")
            armed = True
        original_replace(source, destination, **keywords)

    def sync(descriptor: int) -> None:
        nonlocal armed, injected, file_syncs
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            file_syncs += 1
            selected = {"canonical": 1, "prepared": 2, "markdown_mirror": 3, "text_mirror": 4, "accepted": 6}[target]
            if operation == "file_sync" and file_syncs == selected and not injected:
                injected = True
                raise OSError("synthetic publication fault")
        if armed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            armed = False
            injected = True
            raise OSError("synthetic publication fault")
        original_sync(descriptor)

    monkeypatch.setattr(finalize_node, "save_chapter_data_to_db", acknowledge)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "fsync", sync)
    outcome = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert injected is True
    assert outcome["has_fatal_error"] is True
    assert len(graph_calls) == (1 if target == "accepted" else 0)
    assert get_draft_text(publication_state, manager) == "Recoverable candidate."
    assert (tmp_path / "chapters/chapter_001.accepted.json").read_bytes() == previous
    assert generate_full_export(tmp_path).read_bytes() == b"Previous synthetic prose.\n"
    store = ManuscriptStore(tmp_path)
    recovered = store.recover(1)
    markdown, body = store.read(recovered)
    assert (tmp_path / "chapters/chapter_001.md").read_bytes() == markdown
    assert (tmp_path / "chapters/chapter_001.txt").read_bytes() == body
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    assert generate_full_export(tmp_path).read_bytes() == b"Recoverable candidate.\n"


def test_durable_contained_writer_orders_file_replace_and_directory_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from utils.file_io import ContainedFiles

    events: list[tuple[str, str]] = []
    original_sync, original_replace, original_mkdir = os.fsync, os.replace, os.mkdir

    def sync(descriptor: int) -> None:
        name = os.readlink(f"/proc/self/fd/{descriptor}")
        events.append(("directory_sync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file_sync", name))
        original_sync(descriptor)

    def replace(source: Any, destination: Any, **keywords: Any) -> None:
        parent = os.readlink(f"/proc/self/fd/{keywords['dst_dir_fd']}")
        events.append(("replace", str(Path(parent) / destination)))
        original_replace(source, destination, **keywords)

    def mkdir(name: str, *arguments: Any, **keywords: Any) -> None:
        original_mkdir(name, *arguments, **keywords)
        parent = os.readlink(f"/proc/self/fd/{keywords['dir_fd']}")
        events.append(("mkdir", str(Path(parent) / name)))

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "mkdir", mkdir)
    files = ContainedFiles(tmp_path / "new-root", durable=True)
    files.write_bytes("nested/chapter.md", b"Exact bytes.\r\n")
    assert files.read_bytes("nested/chapter.md") == b"Exact bytes.\r\n"
    for index, (operation, name) in enumerate(events):
        if operation == "mkdir":
            assert events[index + 1] == ("directory_sync", str(Path(name).parent))
        if operation == "replace":
            assert events[index - 1][0] == "file_sync"
            assert Path(events[index - 1][1]).parent == Path(name).parent
            assert events[index + 1] == ("directory_sync", str(Path(name).parent))
    assert [operation for operation, _ in events].count("replace") == 1


@pytest.mark.parametrize("target", ["artifact", "receipt", "body_offset", "symlink"])
async def test_export_rejects_corruption_without_replacing_previous_export(
    publication_state: NarrativeState, tmp_path: Path, target: str,
) -> None:
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    export = generate_full_export(tmp_path)
    previous = export.read_bytes()
    receipt_path = tmp_path / "chapters/chapter_001.accepted.json"
    receipt = json.loads(receipt_path.read_bytes())
    artifact = tmp_path / receipt["artifact_path"]
    if target == "artifact":
        artifact.write_bytes(b"Corrupt candidate.")
    elif target == "receipt":
        receipt_path.write_text("{}")
    elif target == "body_offset":
        receipt["body_offset"] += 1
        receipt_path.write_text(json.dumps(receipt))
    else:
        outside = tmp_path / "synthetic-outside.md"
        outside.write_bytes(artifact.read_bytes())
        artifact.unlink()
        artifact.symlink_to(outside)
    with pytest.raises((OSError, ValueError)):
        generate_full_export(tmp_path)
    assert export.read_bytes() == previous


def test_unreceipted_legacy_export_is_an_explicit_unverified_boundary(tmp_path: Path) -> None:
    from core.langgraph.export import generate_legacy_export

    (tmp_path / "chapters").mkdir()
    (tmp_path / "chapters/chapter_001.md").write_text("Historical synthetic prose.")
    assert generate_full_export(tmp_path).exists() is False
    export = generate_legacy_export(tmp_path)
    assert export.name == "novel_legacy_unverified.md"
    assert export.read_bytes() == b"Historical synthetic prose.\n"
    assert (tmp_path / "chapters/chapter_001.accepted.json").exists() is False


async def test_canonical_export_uses_numeric_order_and_exact_body_bytes(publication_state: NarrativeState, tmp_path: Path) -> None:
    manager = ContentManager(str(tmp_path))
    for number, text in [(10, "\nTen.  \r\n"), (2, "Two.\\n")]:
        publication_state["current_chapter"] = number
        publication_state["draft_ref"] = manager.save_text(text, "draft", f"chapter_{number}", 1)
        assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    assert generate_full_export(tmp_path).read_bytes() == b"Two.\\n\n\n\nTen.  \r\n"


def test_imports_resolve_to_this_worktree() -> None:
    from core.langgraph import export, manuscript
    from utils import file_io

    root = Path(__file__).resolve().parents[2]
    assert Path(finalize_node.__file__).resolve() == root / "core/langgraph/nodes/finalize_node.py"
    assert Path(manuscript.__file__).resolve() == root / "core/langgraph/manuscript.py"
    assert Path(export.__file__).resolve() == root / "core/langgraph/export.py"
    assert Path(file_io.__file__).resolve() == root / "utils/file_io.py"


def test_legacy_discovery_rejects_symlink_directory(tmp_path: Path) -> None:
    from core.langgraph.export import generate_legacy_export

    (tmp_path / "outside").mkdir()
    (tmp_path / "chapters").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(OSError):
        generate_legacy_export(tmp_path)


async def test_first_acceptance_sync_failure_leaves_no_accepted_receipt(publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_replace, original_sync = os.replace, os.fsync
    armed = False

    def replace(source: Any, destination: Any, **keywords: Any) -> None:
        nonlocal armed
        original_replace(source, destination, **keywords)
        armed = Path(destination).name == "chapter_001.accepted.json"

    def sync(descriptor: int) -> None:
        nonlocal armed
        if armed:
            armed = False
            raise OSError("synthetic first acceptance failure")
        original_sync(descriptor)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "fsync", sync)
    result = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert result["has_fatal_error"] is True
    assert (tmp_path / "chapters/chapter_001.accepted.json").exists() is False
    assert generate_full_export(tmp_path).exists() is False
    assert get_draft_text(publication_state, ContentManager(str(tmp_path))) == "Previous synthetic prose."


@pytest.mark.parametrize("layout", ["markdown_only", "text_only", "divergent_pair", "matching_draft"])
@pytest.mark.parametrize("graph_failure", [False, True])
async def test_historical_mirrors_block_publication_without_changing_bytes(
    publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, graph_failure: bool,
) -> None:
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    historical = {}
    if layout != "text_only":
        historical["chapter_001.md"] = b"---\nchapter: 1\n---\nHistorical author edit.\r\n"
    if layout != "markdown_only":
        historical["chapter_001.txt"] = b"Different historical text-only edit.\n"
    if layout == "matching_draft":
        historical = {"chapter_001.txt": b"Previous synthetic prose."}
    for name, content in historical.items():
        (chapters / name).write_bytes(content)
    graph_calls: list[dict[str, Any]] = []

    async def acknowledge(**keywords: Any) -> None:
        graph_calls.append(keywords)
        if graph_failure:
            raise OSError("Synthetic graph acknowledgement failure")

    monkeypatch.setattr(finalize_node, "save_chapter_data_to_db", acknowledge)
    outcome = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert {path.name: path.read_bytes() for path in chapters.iterdir() if path.is_file()} == historical
    assert outcome.get("has_fatal_error") is True
    assert graph_calls == []
    assert (chapters / ".manuscripts").exists() is False
    assert get_draft_text(publication_state, ContentManager(str(tmp_path))) == "Previous synthetic prose."


@pytest.mark.parametrize("target", ["canonical", "prepared", "markdown_mirror", "text_mirror", "accepted"])
@pytest.mark.parametrize("moment", ["before_replace", "after_replace", "after_directory_sync"])
async def test_first_publication_interruption_reopens_and_retries(
    publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, moment: str,
) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    original_replace, original_sync = os.replace, os.fsync
    armed = False
    injected = False

    def replace(source: Any, destination: Any, **keywords: Any) -> None:
        nonlocal armed, injected
        name = Path(destination).name
        selected = {
            "canonical": len(name) == 67 and name.endswith(".md"),
            "prepared": name.endswith(".prepared.json"),
            "markdown_mirror": name == "chapter_001.md",
            "text_mirror": name == "chapter_001.txt",
            "accepted": name == "chapter_001.accepted.json",
        }[target] and not injected
        if selected and moment == "before_replace":
            injected = True
            raise OSError("Synthetic first-publication interruption")
        original_replace(source, destination, **keywords)
        if selected and moment == "after_replace":
            injected = True
            raise OSError("Synthetic first-publication interruption")
        armed = selected

    def sync(descriptor: int) -> None:
        nonlocal injected
        original_sync(descriptor)
        if armed and not injected and stat.S_ISDIR(os.fstat(descriptor).st_mode) and moment == "after_directory_sync":
            injected = True
            raise OSError("Synthetic first-publication interruption")

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", replace)
        fault.setattr(os, "fsync", sync)
        outcome = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert injected is True
    assert outcome["has_fatal_error"] is True
    assert generate_full_export(tmp_path).exists() is False
    assert get_draft_text(publication_state, ContentManager(str(tmp_path))) == "Previous synthetic prose."
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    store = ManuscriptStore(tmp_path)
    receipt = store.accepted(1)
    markdown, body = store.read(receipt)
    assert (tmp_path / "chapters/chapter_001.md").read_bytes() == markdown
    assert (tmp_path / "chapters/chapter_001.txt").read_bytes() == body == b"Previous synthetic prose."
    assert generate_full_export(tmp_path).read_bytes() == body + b"\n"


@pytest.mark.parametrize("target", ["artifact", "receipt", "receipt_name", "chapter", "body_offset", "artifact_symlink", "receipt_symlink", "markdown_mirror", "text_mirror"])
def test_prepared_corruption_blocks_retry_without_replacing_mirrors(tmp_path: Path, target: str) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    receipt = ManuscriptStore(tmp_path).prepare(1, "Prepared synthetic prose.")
    artifact = tmp_path / receipt.artifact_path
    prepared = tmp_path / receipt.prepared_path
    if target == "artifact":
        artifact.write_bytes(b"Corrupted artifact.")
    elif target == "receipt":
        prepared.write_bytes(b"{}")
    elif target == "receipt_name":
        prepared.rename(prepared.with_name("wrong.prepared.json"))
    elif target in {"chapter", "body_offset"}:
        payload = receipt.model_dump()
        if target == "chapter":
            payload["chapter_number"] = 2
            payload["artifact_path"] = payload["artifact_path"].replace("chapter_001", "chapter_002")
        else:
            payload["body_offset"] += 1
        prepared.write_text(json.dumps(payload))
    elif target.endswith("symlink"):
        path = artifact if target == "artifact_symlink" else prepared
        retained = tmp_path / "synthetic-link-target"
        path.rename(retained)
        path.symlink_to(retained)
    else:
        suffix = ".md" if target == "markdown_mirror" else ".txt"
        (tmp_path / ("chapters/chapter_001" + suffix)).write_bytes(b"Unowned author edit.")
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises((OSError, ValueError)):
        ManuscriptStore(tmp_path).prepare(1, "Replacement synthetic prose.")
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("operation", ["prepare", "recover"])
@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_accepted_receipt_does_not_authorize_unretained_mirror_edits(tmp_path: Path, operation: str, suffix: str) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Accepted synthetic prose.")
    store.accept(receipt)
    (tmp_path / ("chapters/chapter_001" + suffix)).write_bytes(b"Unretained author edit.\r\n")
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Unowned manuscript mirror"):
        if operation == "prepare":
            store.prepare(1, "Candidate synthetic prose.")
        else:
            store.recover(1)
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("layout", ["no_mirrors", "markdown_only", "text_only", "pair", "mixed_prepared_pair"])
async def test_verified_prepared_bytes_admit_retry_without_auto_acceptance(
    publication_state: NarrativeState, tmp_path: Path, layout: str,
) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    store = ManuscriptStore(tmp_path)
    first = store.prepare(1, "First retained prepared prose.")
    second = store.prepare(1, "Second retained prepared prose.")
    markdown, body = store.read(first)
    if layout in {"no_mirrors", "text_only"}:
        (tmp_path / "chapters/chapter_001.md").unlink()
    if layout in {"no_mirrors", "markdown_only"}:
        (tmp_path / "chapters/chapter_001.txt").unlink()
    if layout == "mixed_prepared_pair":
        (tmp_path / "chapters/chapter_001.md").write_bytes(markdown)
    assert generate_full_export(tmp_path).exists() is False
    with pytest.raises(FileNotFoundError):
        ManuscriptStore(tmp_path).recover(1)
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    assert generate_full_export(tmp_path).read_bytes() == b"Previous synthetic prose.\n"
    assert store.read(first) == (markdown, body)
    assert store.read(second)[1] == b"Second retained prepared prose."


async def test_first_graph_failure_retains_prepared_bytes_and_allows_explicit_retry(
    publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    graph_calls: list[dict[str, Any]] = []

    async def fail_acknowledgement(**keywords: Any) -> None:
        graph_calls.append(keywords)
        raise OSError("Synthetic first graph failure")

    with monkeypatch.context() as fault:
        fault.setattr(finalize_node, "save_chapter_data_to_db", fail_acknowledgement)
        outcome = await finalize_node.finalize_chapter(example_quality_state(publication_state))
    assert outcome["has_fatal_error"] is True
    assert len(graph_calls) == 1
    assert generate_full_export(tmp_path).exists() is False
    assert (tmp_path / "chapters/chapter_001.accepted.json").exists() is False
    retained = {str(path.relative_to(tmp_path)): path.read_bytes() for path in (tmp_path / "chapters/.manuscripts").rglob("*") if path.is_file()}
    assert len(retained) == 3
    quality = [json.loads(content) for path, content in retained.items() if path.endswith(".quality.json")]
    assert len(quality) == 1
    assert quality[0]["quality"]["status"] == "accepted_with_exceptions"
    assert get_draft_text(publication_state, ContentManager(str(tmp_path))) == "Previous synthetic prose."
    assert (await finalize_node.finalize_chapter(example_quality_state(publication_state)))["last_error"] is None
    assert {path: (tmp_path / path).read_bytes() for path in retained} == retained
    assert ManuscriptStore(tmp_path).read(ManuscriptStore(tmp_path).accepted(1))[1] == b"Previous synthetic prose."


async def test_legacy_quality_receipt_cannot_be_reinterpreted(publication_state: NarrativeState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from datetime import UTC, datetime

    from core.langgraph import manuscript

    class FixedClock:
        @staticmethod
        def now(zone: Any) -> datetime:
            return datetime(2026, 1, 1, tzinfo=UTC)

    # Same prepared manuscript identity; normal new publication timestamps select
    # a distinct immutable object and may legitimately carry a different policy.
    monkeypatch.setattr(manuscript, "datetime", FixedClock)
    state = example_quality_state(publication_state)
    assert (await finalize_node.finalize_chapter(state))["last_error"] is None
    sidecar, = list((tmp_path / "chapters/.manuscripts").rglob("*.quality.json"))
    before = sidecar.read_bytes()
    state["quality_policy"] = {**state["quality_policy"], "identity": "strict"}
    outcome = await finalize_node.finalize_chapter(state)
    assert outcome["has_fatal_error"] is True
    assert sidecar.read_bytes() == before


async def test_legacy_advisory_receipt_is_durable(publication_state: NarrativeState, tmp_path: Path) -> None:
    from core.langgraph.quality_policy import retain_maintenance

    state = example_quality_state(publication_state)
    assert (await finalize_node.finalize_chapter(state))["last_error"] is None
    retain_maintenance(state, "healing", {"errors": ["Synthetic maintenance failure"]})
    receipt, = [json.loads(path.read_bytes()) for path in (tmp_path / "chapters/.manuscripts").rglob("healing-*.json")]
    assert receipt["status"] == "accepted_with_exceptions"
    assert receipt["outcome"]["errors"] == ["Synthetic maintenance failure"]


@pytest.mark.parametrize("fail_write", [False, True])
@pytest.mark.parametrize("resume_mirrors", [False, True])
def test_accepted_replay_verifies_without_republishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_write: bool, resume_mirrors: bool,
) -> None:
    from core.langgraph.manuscript import ManuscriptStore
    from utils.file_io import ContainedFiles

    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Retained synthetic prose. 雨\r\n")
    store.accept(receipt)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    writes: list[str] = []
    original_write = ContainedFiles.write_bytes

    def observe(files: ContainedFiles, path: str, content: bytes) -> None:
        writes.append(path)
        if fail_write:
            raise OSError(errno.ENOSPC, "Synthetic unnecessary publication failure")
        original_write(files, path, content)

    monkeypatch.setattr(ContainedFiles, "write_bytes", observe)
    reopened = ManuscriptStore(tmp_path)
    if resume_mirrors:
        reopened.resume_prepared(receipt)
    reopened.accept(receipt)
    assert reopened.accepted(1) == receipt
    assert writes == []
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("target", ["canonical", "prepared", "prepared_identity", "accepted", "canonical_symlink", "prepared_symlink"])
def test_accepted_replay_rejects_corrupt_identity_before_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    from core.langgraph.manuscript import ManuscriptStore
    from utils.file_io import ContainedFiles

    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Retained synthetic prose.")
    store.accept(receipt)
    if target == "canonical":
        (tmp_path / receipt.artifact_path).write_bytes(b"Corrupt canonical bytes.")
    elif target == "prepared_identity":
        payload = receipt.model_dump()
        payload["body_offset"] += 1
        (tmp_path / receipt.prepared_path).write_text(json.dumps(payload))
    elif target.endswith("symlink"):
        path = tmp_path / (receipt.artifact_path if target == "canonical_symlink" else receipt.prepared_path)
        retained = tmp_path / "synthetic-link-target"
        path.rename(retained)
        path.symlink_to(retained)
    else:
        (tmp_path / (receipt.prepared_path if target == "prepared" else receipt.accepted_path)).write_bytes(b"{}")
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    writes: list[str] = []
    original_write = ContainedFiles.write_bytes

    def observe(files: ContainedFiles, path: str, content: bytes) -> None:
        writes.append(path)
        original_write(files, path, content)

    monkeypatch.setattr(ContainedFiles, "write_bytes", observe)
    with pytest.raises((OSError, ValueError)):
        ManuscriptStore(tmp_path).accept(receipt)
    assert writes == []
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("operation", ["resume_prepared", "recover"])
@pytest.mark.parametrize("layout", ["healthy", "missing_markdown", "missing_text", "mixed_retained"])
def test_accepted_mirror_recovery_writes_only_missing_or_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, layout: str,
) -> None:
    from core.langgraph.manuscript import ManuscriptStore
    from utils.file_io import ContainedFiles

    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Accepted synthetic prose.")
    store.accept(receipt)
    markdown, body = store.read(receipt)
    prefix = "chapters/chapter_001"
    expected_writes = []
    if layout == "mixed_retained":
        store.prepare(1, "Other retained prepared prose.")
        (tmp_path / (prefix + ".md")).write_bytes(markdown)
        expected_writes = [prefix + ".txt"]
    elif layout.startswith("missing_"):
        suffix = ".md" if layout == "missing_markdown" else ".txt"
        (tmp_path / (prefix + suffix)).unlink()
        expected_writes = [prefix + suffix]
    selection = (tmp_path / receipt.accepted_path).read_bytes()
    writes: list[str] = []
    original_write = ContainedFiles.write_bytes

    def observe(files: ContainedFiles, path: str, content: bytes) -> None:
        writes.append(path)
        original_write(files, path, content)

    monkeypatch.setattr(ContainedFiles, "write_bytes", observe)
    reopened = ManuscriptStore(tmp_path)
    if operation == "resume_prepared":
        reopened.resume_prepared(receipt)
    else:
        assert reopened.recover(1) == receipt
    reopened.accept(receipt)
    assert writes == expected_writes
    assert (tmp_path / receipt.accepted_path).read_bytes() == selection
    assert (tmp_path / (prefix + ".md")).read_bytes() == markdown
    assert (tmp_path / (prefix + ".txt")).read_bytes() == body


@pytest.mark.parametrize("suffix", [".md", ".txt"])
def test_accepted_resume_preserves_unretained_author_edits(tmp_path: Path, suffix: str) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Accepted synthetic prose.")
    store.accept(receipt)
    (tmp_path / ("chapters/chapter_001" + suffix)).write_bytes(b"Unretained author edit.\r\n")
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Unowned manuscript mirror"):
        ManuscriptStore(tmp_path).resume_prepared(receipt)
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("operation", ["accept", "resume_prepared", "recover"])
def test_visible_publication_replay_completes_directory_durability_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Synthetic durable prose.")
    if operation != "accept":
        store.accept(receipt)
        (tmp_path / "chapters/chapter_001.txt").unlink()
    original_replace, original_sync = os.replace, os.fsync
    target = Path(receipt.accepted_path).name if operation == "accept" else "chapter_001.txt"
    armed = False

    def interrupt(source: Any, destination: Any, **keywords: Any) -> None:
        nonlocal armed
        original_replace(source, destination, **keywords)
        armed = Path(destination).name == target

    def interrupt_sync(descriptor: int) -> None:
        if armed and stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise KeyboardInterrupt("Synthetic interruption between rename and directory sync")
        original_sync(descriptor)

    with monkeypatch.context() as fault:
        fault.setattr(os, "replace", interrupt)
        fault.setattr(os, "fsync", interrupt_sync)
        with pytest.raises(KeyboardInterrupt):
            if operation == "accept":
                store.accept(receipt)
            elif operation == "resume_prepared":
                store.resume_prepared(receipt)
            else:
                store.recover(1)
    synced_directories: list[str] = []
    replacements: list[str] = []

    def sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            synced_directories.append(os.readlink(f"/proc/self/fd/{descriptor}"))
        original_sync(descriptor)

    def observe(source: Any, destination: Any, **keywords: Any) -> None:
        replacements.append(str(destination))
        original_replace(source, destination, **keywords)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", observe)
    reopened = ManuscriptStore(tmp_path)
    if operation == "accept":
        reopened.accept(receipt)
    elif operation == "resume_prepared":
        reopened.resume_prepared(receipt)
    else:
        assert reopened.recover(1) == receipt
    assert synced_directories.count(str(tmp_path / "chapters")) == 1
    assert replacements == []
    assert reopened.accepted(1) == receipt


@pytest.mark.parametrize("operation", ["accept", "resume_prepared", "recover"])
def test_replay_directory_sync_failure_propagates_without_changing_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    from core.langgraph.manuscript import ManuscriptStore

    store = ManuscriptStore(tmp_path)
    receipt = store.prepare(1, "Accepted synthetic prose.")
    store.accept(receipt)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    original_sync = os.fsync

    def fail_sync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and os.readlink(f"/proc/self/fd/{descriptor}") == str(tmp_path / "chapters"):
            raise OSError(errno.ENOSPC, "Synthetic replay durability failure")
        original_sync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="Synthetic replay durability failure"):
        if operation == "accept":
            store.accept(receipt)
        elif operation == "resume_prepared":
            store.resume_prepared(receipt)
        else:
            store.recover(1)
    assert {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
