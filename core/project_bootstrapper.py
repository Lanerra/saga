from __future__ import annotations

from pathlib import Path

import structlog

import config
from core.llm_interface_refactored import RefactoredLLMService
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from prompts.prompt_renderer import render_prompt
from utils.common import ensure_exact_keys, try_load_json_from_response

logger = structlog.get_logger(__name__)


class ProjectBootstrapper:
    def __init__(self, language_model_service: RefactoredLLMService) -> None:
        self.language_model_service = language_model_service

    async def generate_metadata(self, user_prompt: str) -> NarrativeProjectConfig:
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("User prompt must be a non-empty string")

        prompt = render_prompt(
            "initialization/bootstrap_project.j2",
            {
                "user_input": user_prompt.strip(),
                "default_narrative_style": config.DEFAULT_NARRATIVE_STYLE,
                "target_plot_points": config.TARGET_PLOT_POINTS_INITIAL_GENERATION,
                "total_chapters": config.TOTAL_CHAPTERS,
            },
        )

        response_text, _usage = await self.language_model_service.async_call_llm(
            model_name=config.LARGE_MODEL,
            prompt=prompt,
            temperature=config.TEMPERATURE_INITIAL_SETUP,
        )

        parsed, _candidates, parse_errors = try_load_json_from_response(response_text, expected_root=dict)
        if parsed is None:
            raise ValueError(f"Bootstrap response did not contain valid JSON: {parse_errors}")

        required_keys = {
            "title",
            "genre",
            "theme",
            "setting",
            "protagonist_name",
            "narrative_style",
            "total_chapters",
        }
        ensure_exact_keys(value=parsed, required_keys=required_keys, context="Bootstrap response")

        project_config = NarrativeProjectConfig.model_validate(parsed)
        if project_config.narrative_style != config.DEFAULT_NARRATIVE_STYLE:
            raise ValueError("Bootstrap narrative style must match DEFAULT_NARRATIVE_STYLE")

        return project_config.model_copy(
            update={
                "created_from": "bootstrap",
                "original_prompt": user_prompt.strip(),
            }
        )

    def save_config(self, project_config: NarrativeProjectConfig, *, review: bool) -> Path:
        project_directory = ProjectManager.save_config(project_config, review=review)
        logger.info(
            "Project configuration saved",
            project_dir=str(project_directory),
            title=project_config.title,
            review=review,
        )
        return project_directory
