from __future__ import annotations

import json
import re
from pathlib import Path

import config
from core.project_config import NarrativeProjectConfig


class ProjectManager:
    projects_root = Path("projects")

    @classmethod
    def ensure_projects_root(class_type) -> Path:
        class_type.projects_root.mkdir(parents=True, exist_ok=True)
        return class_type.projects_root

    @classmethod
    def sanitize_project_title(class_type, title: str) -> str:
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Project title must be a non-empty string")

        safe_title = re.sub(r"[^\w\s-]", "", title.lower())
        safe_title = re.sub(r"[-\s]+", "_", safe_title).strip("_")
        if not safe_title:
            raise ValueError("Project title must contain at least one valid character")
        return safe_title[:50]

    @classmethod
    def project_directory_from_title(class_type, title: str) -> Path:
        safe_title = class_type.sanitize_project_title(title)
        return class_type.ensure_projects_root() / safe_title

    @classmethod
    def save_config(class_type, project_config: NarrativeProjectConfig, *, review: bool) -> Path:
        project_directory = class_type.project_directory_from_title(project_config.title)
        project_directory.mkdir(parents=True, exist_ok=True)
        (project_directory / "checkpoints").mkdir(exist_ok=True)

        file_name = "config.candidate.json" if review else "config.json"
        config_path = project_directory / file_name
        config_payload = project_config.model_dump()
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        return project_directory

    @classmethod
    def load_config(class_type, project_directory: Path) -> NarrativeProjectConfig:
        config_path = project_directory / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {project_directory}")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        return NarrativeProjectConfig.model_validate(config_data)

    @classmethod
    def load_candidate_config(class_type, project_directory: Path) -> NarrativeProjectConfig:
        config_path = project_directory / "config.candidate.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.candidate.json in {project_directory}")
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        return NarrativeProjectConfig.model_validate(config_data)

    @classmethod
    def promote_candidate(class_type, project_directory: Path) -> None:
        candidate_path = project_directory / "config.candidate.json"
        final_path = project_directory / "config.json"

        if not candidate_path.exists():
            raise FileNotFoundError(f"No candidate config found in {project_directory}")
        if final_path.exists():
            raise FileExistsError(f"Final config already exists in {project_directory}")

        candidate_path.rename(final_path)

    @classmethod
    def find_candidate_project(class_type) -> Path | None:
        if not class_type.projects_root.exists():
            return None

        candidates: list[Path] = []
        for project_directory in class_type.projects_root.iterdir():
            if not project_directory.is_dir():
                continue
            candidate_path = project_directory / "config.candidate.json"
            if candidate_path.exists():
                candidates.append(project_directory)

        if not candidates:
            return None

        candidates.sort(key=lambda path: (path / "config.candidate.json").stat().st_mtime, reverse=True)
        return candidates[0]

    @classmethod
    def find_resume_project(class_type) -> Path | None:
        if not class_type.projects_root.exists():
            return None

        project_directories = [path for path in class_type.projects_root.iterdir() if path.is_dir()]
        project_directories.sort(key=lambda path: path.stat().st_mtime, reverse=True)

        for project_directory in project_directories:
            config_path = project_directory / "config.json"
            if not config_path.exists():
                continue

            project_config = class_type.load_config(project_directory)
            completed_chapters = class_type.count_completed_chapters(project_directory)
            if completed_chapters < project_config.total_chapters:
                return project_directory

        return None

    @classmethod
    def count_completed_chapters(class_type, project_directory: Path) -> int:
        chapters_directory = project_directory / "chapters"
        if not chapters_directory.exists():
            return 0

        return len(list(chapters_directory.glob("chapter_*.md")))

    @classmethod
    def create_default_project(class_type) -> Path:
        project_config = NarrativeProjectConfig(
            title=config.DEFAULT_PLOT_OUTLINE_TITLE,
            genre=config.CONFIGURED_GENRE,
            theme=config.CONFIGURED_THEME,
            setting=config.CONFIGURED_SETTING_DESCRIPTION,
            protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
            narrative_style=config.DEFAULT_NARRATIVE_STYLE,
            total_chapters=config.TOTAL_CHAPTERS or 12,
            created_from="settings",
            original_prompt="",
        )

        project_directory = class_type.project_directory_from_title(project_config.title)
        config_path = project_directory / "config.json"
        if config_path.exists():
            return project_directory

        return class_type.save_config(project_config, review=False)
