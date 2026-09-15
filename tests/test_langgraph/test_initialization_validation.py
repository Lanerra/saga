# tests/test_langgraph/test_initialization_validation.py
from pathlib import Path

import pytest

from core.langgraph.initialization.validation import (
    validate_initialization_artifacts,
)
from orchestration.langgraph_orchestrator import LangGraphOrchestrator


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_validate_initialization_artifacts_complete(tmp_path: Path) -> None:
    # Arrange: create all required initialization artifacts
    _touch(tmp_path / "saga.yaml")
    _touch(tmp_path / "outline" / "structure.yaml")
    _touch(tmp_path / "outline" / "beats.yaml")
    _touch(tmp_path / "characters" / "hero.yaml")
    _touch(tmp_path / "world" / "items.yaml")
    _touch(tmp_path / "world" / "rules.yaml")
    _touch(tmp_path / "world" / "history.yaml")

    # Act
    ok, missing = validate_initialization_artifacts(tmp_path)

    # Assert
    assert ok is True
    assert missing == []


def test_validate_initialization_artifacts_reports_missing(tmp_path: Path) -> None:
    # Arrange: create only a subset of required files
    _touch(tmp_path / "saga.yaml")
    _touch(tmp_path / "outline" / "structure.yaml")
    # Intentionally omit beats.yaml, characters/*, and world/*.yaml

    # Act
    ok, missing = validate_initialization_artifacts(tmp_path)

    # Assert
    assert ok is False
    # Ensure each expected artifact is represented in missing messages
    # These assertions rely on the stable phrasing defined in the helper
    assert "Missing saga.yaml" not in missing  # saga.yaml exists
    assert "Missing outline/structure.yaml" not in missing  # exists
    assert "Missing outline/beats.yaml" in missing
    assert "Missing any characters/*.yaml files" in missing
    assert "Missing world/items.yaml" in missing
    assert "Missing world/rules.yaml" in missing
    assert "Missing world/history.yaml" in missing


@pytest.mark.asyncio
async def test_load_or_create_state_logs_missing_initialization_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify real empty project storage is reported as incomplete and requires init.
    """
    # Isolate chapter progress and character profile providers.
    from data_access import chapter_queries as chapter_queries_module

    async def _fake_load_chapter_progress_from_db() -> chapter_queries_module.ChapterProgress:
        return chapter_queries_module.ChapterProgress(0, ())

    monkeypatch.setattr(
        chapter_queries_module,
        "load_chapter_progress_from_db",
        _fake_load_chapter_progress_from_db,
    )

    from data_access import character_queries as character_queries_module

    async def _fake_get_character_profiles() -> list[dict[str, str]]:
        # Simulate no characters; initialization_complete should be False
        return []

    monkeypatch.setattr(
        character_queries_module,
        "get_character_profiles",
        _fake_get_character_profiles,
    )

    orchestrator = LangGraphOrchestrator(project_dir=tmp_path)

    caplog.set_level("WARNING")

    # Act
    state = await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)

    # Empty project artifacts require initialization, independently of graph characters.
    assert state["initialization_complete"] is False

    # Assert: warning about incomplete initialization artifacts was logged
    warnings = [record for record in caplog.records if "Initialization artifacts incomplete" in record.getMessage()]
    assert warnings, f"Expected warning about incomplete initialization artifacts. Captured: {[r.getMessage() for r in caplog.records]}"


@pytest.mark.asyncio
async def test_load_or_create_state_no_warning_when_artifacts_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Verify that no incomplete-artifacts warning is logged when all artifacts exist.
    """
    # Use tmp_path as BASE_OUTPUT_DIR
    import config as config_module

    monkeypatch.setitem(vars(config_module), "settings", config_module.settings.model_copy(update={"BASE_OUTPUT_DIR": str(tmp_path)}))

    # Create all required artifacts in tmp_path
    _touch(tmp_path / "saga.yaml")
    _touch(tmp_path / "outline" / "structure.yaml")
    _touch(tmp_path / "outline" / "beats.yaml")
    _touch(tmp_path / "characters" / "hero.yaml")
    _touch(tmp_path / "world" / "items.yaml")
    _touch(tmp_path / "world" / "rules.yaml")
    _touch(tmp_path / "world" / "history.yaml")

    from data_access import chapter_queries as chapter_queries_module

    async def _fake_load_chapter_progress_from_db() -> chapter_queries_module.ChapterProgress:
        return chapter_queries_module.ChapterProgress(0, ())

    monkeypatch.setattr(
        chapter_queries_module,
        "load_chapter_progress_from_db",
        _fake_load_chapter_progress_from_db,
    )

    from data_access import character_queries as character_queries_module

    async def _fake_get_character_profiles() -> list[dict[str, str]]:
        # Simulate characters exist so initialization_complete will be True
        return [{"name": "Hero"}]

    monkeypatch.setattr(
        character_queries_module,
        "get_character_profiles",
        _fake_get_character_profiles,
    )

    orchestrator = LangGraphOrchestrator()

    caplog.set_level("WARNING")

    with pytest.raises(ValueError, match="Missing initialization receipt"):
        await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)

    # Assert: no warning about incomplete artifacts
    warnings = [record for record in caplog.records if "Initialization artifacts incomplete" in record.getMessage()]
    assert not warnings
