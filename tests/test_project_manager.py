# tests/test_project_manager.py
from __future__ import annotations

import json
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager


@pytest.fixture(autouse=True)
def _use_temp_projects_root(tmp_path: Path) -> Generator[None, None, None]:
    original = ProjectManager.projects_root
    ProjectManager.projects_root = tmp_path / "projects"
    yield
    ProjectManager.projects_root = original


@pytest.fixture()
def example_config() -> NarrativeProjectConfig:
    return NarrativeProjectConfig(
        title="My Fantasy Novel",
        genre="Fantasy",
        theme="Good vs Evil",
        setting="Medieval World",
        protagonist_name="Hero",
        narrative_style="Third Person",
        total_chapters=12,
    )


class TestSanitizeProjectTitle:
    def test_lowercase_and_spaces_to_underscores(self) -> None:
        result = ProjectManager.sanitize_project_title("My Great Novel")
        assert result == "my_great_novel"

    def test_special_characters_removed(self) -> None:
        result = ProjectManager.sanitize_project_title("Hello! World@ #2024")
        assert result == "hello_world_2024"

    def test_hyphens_converted_to_underscores(self) -> None:
        result = ProjectManager.sanitize_project_title("my-great-novel")
        assert result == "my_great_novel"

    def test_consecutive_spaces_collapsed(self) -> None:
        result = ProjectManager.sanitize_project_title("word   another   word")
        assert result == "word_another_word"

    def test_leading_trailing_underscores_stripped(self) -> None:
        result = ProjectManager.sanitize_project_title("  hello  ")
        assert result == "hello"

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            ProjectManager.sanitize_project_title("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            ProjectManager.sanitize_project_title("   ")

    def test_all_special_characters_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one valid character"):
            ProjectManager.sanitize_project_title("!@#$%^&*()")

    def test_truncated_to_50_characters(self) -> None:
        long_title = "a" * 100
        result = ProjectManager.sanitize_project_title(long_title)
        assert len(result) == 50
        assert result == "a" * 50


class TestSaveConfig:
    def test_review_false_creates_config_json(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=False)

        config_path = project_directory / "config.json"
        assert config_path.exists()
        assert not (project_directory / "config.candidate.json").exists()

        saved_data = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved_data["title"] == "My Fantasy Novel"
        assert saved_data["total_chapters"] == 12

    def test_review_false_creates_checkpoints_directory(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=False)

        checkpoints_directory = project_directory / "checkpoints"
        assert checkpoints_directory.exists()
        assert checkpoints_directory.is_dir()

    def test_review_true_creates_candidate_json(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=True)

        candidate_path = project_directory / "config.candidate.json"
        assert candidate_path.exists()
        assert not (project_directory / "config.json").exists()

        saved_data = json.loads(candidate_path.read_text(encoding="utf-8"))
        assert saved_data["title"] == "My Fantasy Novel"

    def test_saved_config_roundtrips(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=False)

        loaded = ProjectManager.load_config(project_directory)
        assert loaded == example_config


class TestLoadConfig:
    def test_returns_valid_config(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=False)

        loaded = ProjectManager.load_config(project_directory)

        assert loaded.title == "My Fantasy Novel"
        assert loaded.genre == "Fantasy"
        assert loaded.theme == "Good vs Evil"
        assert loaded.setting == "Medieval World"
        assert loaded.protagonist_name == "Hero"
        assert loaded.narrative_style == "Third Person"
        assert loaded.total_chapters == 12

    def test_missing_config_raises_file_not_found(self, tmp_path: Path) -> None:
        empty_directory = tmp_path / "empty_project"
        empty_directory.mkdir()

        with pytest.raises(FileNotFoundError, match="Missing config.json"):
            ProjectManager.load_config(empty_directory)


class TestLoadCandidateConfig:
    def test_returns_valid_candidate_config(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=True)

        loaded = ProjectManager.load_candidate_config(project_directory)

        assert loaded == example_config

    def test_missing_candidate_raises_file_not_found(self, tmp_path: Path) -> None:
        empty_directory = tmp_path / "empty_project"
        empty_directory.mkdir()

        with pytest.raises(FileNotFoundError, match="Missing config.candidate.json"):
            ProjectManager.load_candidate_config(empty_directory)


class TestPromoteCandidate:
    def test_renames_candidate_to_final(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=True)

        ProjectManager.promote_candidate(project_directory)

        assert (project_directory / "config.json").exists()
        assert not (project_directory / "config.candidate.json").exists()

        loaded = ProjectManager.load_config(project_directory)
        assert loaded == example_config

    def test_raises_when_no_candidate_exists(self, tmp_path: Path) -> None:
        empty_directory = tmp_path / "empty_project"
        empty_directory.mkdir()

        with pytest.raises(FileNotFoundError, match="No candidate config found"):
            ProjectManager.promote_candidate(empty_directory)

    def test_raises_when_final_already_exists(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=True)
        (project_directory / "config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(FileExistsError, match="Final config already exists"):
            ProjectManager.promote_candidate(project_directory)


class TestFindCandidateProject:
    def test_returns_most_recent_candidate(self) -> None:
        projects_root = ProjectManager.ensure_projects_root()

        older_project = projects_root / "older_project"
        older_project.mkdir()
        older_candidate = older_project / "config.candidate.json"
        older_candidate.write_text("{}", encoding="utf-8")

        time.sleep(0.05)

        newer_project = projects_root / "newer_project"
        newer_project.mkdir()
        newer_candidate = newer_project / "config.candidate.json"
        newer_candidate.write_text("{}", encoding="utf-8")

        result = ProjectManager.find_candidate_project()
        assert result == newer_project

    def test_returns_none_when_no_candidates(self) -> None:
        ProjectManager.ensure_projects_root()

        result = ProjectManager.find_candidate_project()
        assert result is None

    def test_returns_none_when_root_does_not_exist(self) -> None:
        result = ProjectManager.find_candidate_project()
        assert result is None

    def test_ignores_non_directory_entries(self) -> None:
        projects_root = ProjectManager.ensure_projects_root()

        stray_file = projects_root / "not_a_directory.txt"
        stray_file.write_text("stray", encoding="utf-8")

        result = ProjectManager.find_candidate_project()
        assert result is None


class TestFindResumeProject:
    def test_returns_incomplete_project(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=False)

        result = ProjectManager.find_resume_project()
        assert result == project_directory

    def test_returns_none_when_all_chapters_complete(self, example_config: NarrativeProjectConfig) -> None:
        project_directory = ProjectManager.save_config(example_config, review=False)
        chapters_directory = project_directory / "chapters"
        chapters_directory.mkdir()
        for index in range(1, example_config.total_chapters + 1):
            (chapters_directory / f"chapter_{index}.md").write_text(f"Chapter {index}", encoding="utf-8")

        result = ProjectManager.find_resume_project()
        assert result is None

    def test_returns_none_when_root_does_not_exist(self) -> None:
        result = ProjectManager.find_resume_project()
        assert result is None

    def test_skips_projects_without_config(self) -> None:
        projects_root = ProjectManager.ensure_projects_root()
        (projects_root / "no_config_project").mkdir()

        result = ProjectManager.find_resume_project()
        assert result is None


class TestCountCompletedChapters:
    def test_counts_chapter_markdown_files(self, tmp_path: Path) -> None:
        project_directory = tmp_path / "counting_project"
        chapters_directory = project_directory / "chapters"
        chapters_directory.mkdir(parents=True)

        (chapters_directory / "chapter_1.md").write_text("One", encoding="utf-8")
        (chapters_directory / "chapter_2.md").write_text("Two", encoding="utf-8")
        (chapters_directory / "chapter_3.md").write_text("Three", encoding="utf-8")

        result = ProjectManager.count_completed_chapters(project_directory)
        assert result == 3

    def test_returns_zero_for_missing_chapters_directory(self, tmp_path: Path) -> None:
        project_directory = tmp_path / "no_chapters_project"
        project_directory.mkdir()

        result = ProjectManager.count_completed_chapters(project_directory)
        assert result == 0

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        project_directory = tmp_path / "mixed_project"
        chapters_directory = project_directory / "chapters"
        chapters_directory.mkdir(parents=True)

        (chapters_directory / "chapter_1.md").write_text("One", encoding="utf-8")
        (chapters_directory / "notes.md").write_text("Notes", encoding="utf-8")
        (chapters_directory / "outline.txt").write_text("Outline", encoding="utf-8")

        result = ProjectManager.count_completed_chapters(project_directory)
        assert result == 1
