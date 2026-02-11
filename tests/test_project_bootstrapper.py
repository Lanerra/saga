# tests/test_project_bootstrapper.py
from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from core.llm_interface_refactored import RefactoredLLMService
from core.project_bootstrapper import ProjectBootstrapper
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager

VALID_BOOTSTRAP_DATA = {
    "title": "My Novel",
    "genre": "Fantasy",
    "theme": "Adventure",
    "setting": "Medieval",
    "protagonist_name": "Hero",
    "narrative_style": "Third Person Limited",
    "total_chapters": 12,
}


@pytest.fixture()
def fake_language_model_service() -> MagicMock:
    service = MagicMock(spec=RefactoredLLMService)
    service.async_call_llm = AsyncMock()
    return service


@pytest.fixture()
def bootstrapper(fake_language_model_service: MagicMock) -> ProjectBootstrapper:
    return ProjectBootstrapper(language_model_service=fake_language_model_service)


@pytest.fixture()
def _patch_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "LARGE_MODEL", "test-model")
    monkeypatch.setattr(config, "TEMPERATURE_INITIAL_SETUP", 0.7)
    monkeypatch.setattr(config, "DEFAULT_NARRATIVE_STYLE", "Third Person Limited")
    monkeypatch.setattr(config, "TOTAL_CHAPTERS", 12)


@pytest.fixture(autouse=True)
def _use_temp_projects_root(tmp_path: Path) -> Generator[None, None, None]:
    original = ProjectManager.projects_root
    ProjectManager.projects_root = tmp_path / "projects"
    yield
    ProjectManager.projects_root = original


class TestGenerateMetadataInputValidation:
    async def test_empty_string_raises_value_error(
        self, bootstrapper: ProjectBootstrapper
    ) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            await bootstrapper.generate_metadata("")

    async def test_whitespace_only_raises_value_error(
        self, bootstrapper: ProjectBootstrapper
    ) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            await bootstrapper.generate_metadata("   \n\t  ")


@pytest.mark.usefixtures("_patch_config")
class TestGenerateMetadataSuccess:
    @patch("core.project_bootstrapper.render_prompt", return_value="rendered prompt")
    async def test_returns_narrative_project_config(
        self,
        _fake_render_prompt: MagicMock,
        bootstrapper: ProjectBootstrapper,
        fake_language_model_service: MagicMock,
    ) -> None:
        valid_response = json.dumps(VALID_BOOTSTRAP_DATA)
        fake_language_model_service.async_call_llm.return_value = (
            valid_response,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        result = await bootstrapper.generate_metadata("Write me a fantasy novel")

        assert isinstance(result, NarrativeProjectConfig)
        assert result.title == "My Novel"
        assert result.genre == "Fantasy"
        assert result.theme == "Adventure"
        assert result.setting == "Medieval"
        assert result.protagonist_name == "Hero"
        assert result.narrative_style == "Third Person Limited"
        assert result.total_chapters == 12

    @patch("core.project_bootstrapper.render_prompt", return_value="rendered prompt")
    async def test_sets_created_from_and_original_prompt(
        self,
        _fake_render_prompt: MagicMock,
        bootstrapper: ProjectBootstrapper,
        fake_language_model_service: MagicMock,
    ) -> None:
        valid_response = json.dumps(VALID_BOOTSTRAP_DATA)
        fake_language_model_service.async_call_llm.return_value = (
            valid_response,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        result = await bootstrapper.generate_metadata("  Write me a fantasy novel  ")

        assert result.created_from == "bootstrap"
        assert result.original_prompt == "Write me a fantasy novel"

    @patch("core.project_bootstrapper.render_prompt", return_value="rendered prompt")
    async def test_calls_language_model_service_with_correct_arguments(
        self,
        _fake_render_prompt: MagicMock,
        bootstrapper: ProjectBootstrapper,
        fake_language_model_service: MagicMock,
    ) -> None:
        valid_response = json.dumps(VALID_BOOTSTRAP_DATA)
        fake_language_model_service.async_call_llm.return_value = (
            valid_response,
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        await bootstrapper.generate_metadata("Write me a fantasy novel")

        fake_language_model_service.async_call_llm.assert_awaited_once_with(
            model_name="test-model",
            prompt="rendered prompt",
            temperature=0.7,
        )


@pytest.mark.usefixtures("_patch_config")
class TestGenerateMetadataInvalidResponse:
    @patch("core.project_bootstrapper.render_prompt", return_value="rendered prompt")
    async def test_invalid_json_raises_value_error(
        self,
        _fake_render_prompt: MagicMock,
        bootstrapper: ProjectBootstrapper,
        fake_language_model_service: MagicMock,
    ) -> None:
        fake_language_model_service.async_call_llm.return_value = (
            "this is not json at all",
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        with pytest.raises(ValueError, match="did not contain valid JSON"):
            await bootstrapper.generate_metadata("Write me a fantasy novel")

    @patch("core.project_bootstrapper.render_prompt", return_value="rendered prompt")
    async def test_missing_required_key_raises_value_error(
        self,
        _fake_render_prompt: MagicMock,
        bootstrapper: ProjectBootstrapper,
        fake_language_model_service: MagicMock,
    ) -> None:
        incomplete_data = {
            "title": "My Novel",
            "genre": "Fantasy",
            "theme": "Adventure",
        }
        fake_language_model_service.async_call_llm.return_value = (
            json.dumps(incomplete_data),
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        with pytest.raises(ValueError, match="must contain exactly keys"):
            await bootstrapper.generate_metadata("Write me a fantasy novel")

    @patch("core.project_bootstrapper.render_prompt", return_value="rendered prompt")
    async def test_narrative_style_mismatch_raises_value_error(
        self,
        _fake_render_prompt: MagicMock,
        bootstrapper: ProjectBootstrapper,
        fake_language_model_service: MagicMock,
    ) -> None:
        mismatched_data = {
            **VALID_BOOTSTRAP_DATA,
            "narrative_style": "First Person",
        }
        fake_language_model_service.async_call_llm.return_value = (
            json.dumps(mismatched_data),
            {"prompt_tokens": 100, "completion_tokens": 50},
        )

        with pytest.raises(ValueError, match="must match DEFAULT_NARRATIVE_STYLE"):
            await bootstrapper.generate_metadata("Write me a fantasy novel")


class TestSaveConfig:
    def test_delegates_to_project_manager_and_returns_directory(
        self,
        bootstrapper: ProjectBootstrapper,
    ) -> None:
        project_config = NarrativeProjectConfig(
            title="My Novel",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval",
            protagonist_name="Hero",
            narrative_style="Third Person Limited",
            total_chapters=12,
            created_from="bootstrap",
            original_prompt="Write me a fantasy novel",
        )

        result = bootstrapper.save_config(project_config, review=False)

        assert isinstance(result, Path)
        assert (result / "config.json").exists()
        saved_data = json.loads((result / "config.json").read_text(encoding="utf-8"))
        assert saved_data["title"] == "My Novel"
        assert saved_data["created_from"] == "bootstrap"

    def test_review_true_creates_candidate_config(
        self,
        bootstrapper: ProjectBootstrapper,
    ) -> None:
        project_config = NarrativeProjectConfig(
            title="My Novel",
            genre="Fantasy",
            theme="Adventure",
            setting="Medieval",
            protagonist_name="Hero",
            narrative_style="Third Person Limited",
            total_chapters=12,
        )

        result = bootstrapper.save_config(project_config, review=True)

        assert (result / "config.candidate.json").exists()
        assert not (result / "config.json").exists()
