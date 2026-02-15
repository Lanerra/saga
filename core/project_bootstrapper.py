# core/project_bootstrapper.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

import config
from core.llm_interface_refactored import RefactoredLLMService
from core.project_config import NarrativeProjectConfig
from core.project_manager import ProjectManager
from prompts.prompt_renderer import render_prompt
from utils.common import ensure_exact_keys, try_load_json_from_response

logger = structlog.get_logger(__name__)

# NOT YET WIRED INTO CLI — available for future interactive bootstrap flow
GENRE_TEMPLATES: dict[str, dict[str, str]] = {
    "epic_fantasy": {
        "genre": "Epic Fantasy",
        "theme": "The struggle between duty and personal desire in a world where ancient powers are reawakening",
        "setting": "A sprawling continent of rival kingdoms, ancient forests, and forgotten ruins. Magic is rare but potent, tied to bloodlines and sacred oaths. Political alliances shift as an old darkness stirs beneath the mountains.",
    },
    "hard_science_fiction": {
        "genre": "Hard Science Fiction",
        "theme": "The ethical boundaries of technological progress when survival demands compromise",
        "setting": "A network of space stations and asteroid mining colonies in the Belt, 2340 CE. Humanity has expanded beyond Earth but remains fragmented by corporate territories and resource scarcity. Communication delays make each outpost effectively autonomous.",
    },
    "cozy_mystery": {
        "genre": "Cozy Mystery",
        "theme": "How well do we truly know the people in our communities, and what secrets hide behind familiar faces",
        "setting": "A picturesque coastal village in Devon, England. Cobblestone streets, a centuries-old pub, and a weekly farmers market where gossip flows freely. Beneath the charm, old family rivalries and hidden debts create fertile ground for mischief.",
    },
    "literary_fiction": {
        "genre": "Literary Fiction",
        "theme": "The way memory reshapes identity, and whether we can ever truly escape the stories our families tell about us",
        "setting": "A university town in the American Midwest, spanning three decades from the 1990s to the present. Academic corridors, aging farmhouses, and the slow erosion of a once-thriving downtown mirror the protagonist's inner landscape.",
    },
    "gothic_horror": {
        "genre": "Gothic Horror",
        "theme": "The inherited sins of the past and the impossibility of escaping the darkness encoded in place and bloodline",
        "setting": "A decaying manor house on the Scottish moors, isolated by miles of peat bog and perpetual mist. The estate has been in the family for centuries, each generation adding wings and sealing off rooms. The locals avoid the grounds after dark.",
    },
    "historical_romance": {
        "genre": "Historical Romance - Regency England",
        "theme": "Whether love can flourish when constrained by rigid social expectations and the weight of reputation",
        "setting": "London during the 1815 Season, in the wake of Waterloo. Glittering ballrooms, Hyde Park promenades, and drawing-room politics. Society rewards propriety and punishes scandal, yet beneath the surface, desire and ambition collide.",
    },
    "cyberpunk": {
        "genre": "Cyberpunk",
        "theme": "The erosion of human autonomy in a world where identity itself has become a commodity",
        "setting": "Neo-Shenzhen, 2089. A megalopolis of neon-drenched towers and flooded lower districts. Megacorporations control infrastructure, memory augmentation is commonplace, and the line between digital and physical existence blurs for anyone who can afford the implants.",
    },
    "grimdark_fantasy": {
        "genre": "Grimdark Fantasy",
        "theme": "The cost of power and whether moral compromise is inevitable in a world that punishes idealism",
        "setting": "A war-ravaged continent where a generation-long conflict between three empires has left nations bankrupt and fields salted. Mercenary companies hold more power than kings. Magic exacts a physical toll on its users, and the gods, if they exist, are silent.",
    },
}

# NOT YET WIRED INTO CLI — available for future interactive bootstrap flow
STORY_STRUCTURES: dict[str, dict[str, Any]] = {
    "three_act": {
        "name": "Three-Act Structure",
        "description": "Classic setup-confrontation-resolution arc",
        "recommended_chapter_range": (8, 15),
        "act_proportions": {"setup": 0.25, "confrontation": 0.50, "resolution": 0.25},
    },
    "heroes_journey": {
        "name": "Hero's Journey",
        "description": "Mythic departure-initiation-return cycle",
        "recommended_chapter_range": (12, 24),
        "act_proportions": {"departure": 0.30, "initiation": 0.45, "return": 0.25},
    },
    "five_act": {
        "name": "Five-Act Structure",
        "description": "Shakespearean exposition-rising-climax-falling-denouement",
        "recommended_chapter_range": (15, 30),
        "act_proportions": {
            "exposition": 0.15,
            "rising_action": 0.25,
            "climax": 0.20,
            "falling_action": 0.25,
            "denouement": 0.15,
        },
    },
    "episodic": {
        "name": "Episodic Structure",
        "description": "Self-contained episodes linked by character and theme rather than a single arc",
        "recommended_chapter_range": (10, 20),
        "act_proportions": {"introduction": 0.10, "episodes": 0.80, "resolution": 0.10},
    },
}


class ProjectBootstrapper:
    def __init__(self, language_model_service: RefactoredLLMService) -> None:
        self.language_model_service = language_model_service

    # NOT YET WIRED INTO CLI — available for future interactive bootstrap flow
    @staticmethod
    def get_genre_templates() -> dict[str, dict[str, str]]:
        return GENRE_TEMPLATES

    # NOT YET WIRED INTO CLI — available for future interactive bootstrap flow
    @staticmethod
    def get_story_structures() -> dict[str, dict[str, Any]]:
        return STORY_STRUCTURES

    async def generate_metadata(self, user_prompt: str) -> NarrativeProjectConfig:
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("User prompt must be a non-empty string")

        prompt = render_prompt(
            "initialization/bootstrap_project.j2",
            {
                "user_input": user_prompt.strip(),
                "default_narrative_style": config.DEFAULT_NARRATIVE_STYLE,
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

    # NOT YET WIRED INTO CLI — available for future interactive bootstrap flow
    async def generate_metadata_from_template(
        self,
        template_name: str,
        user_prompt: str,
    ) -> NarrativeProjectConfig:
        if template_name not in GENRE_TEMPLATES:
            raise ValueError(f"Unknown genre template '{template_name}'. " f"Available templates: {sorted(GENRE_TEMPLATES.keys())}")
        template = GENRE_TEMPLATES[template_name]
        enriched_prompt = f"{user_prompt.strip()}\n\n" f"Genre guidance: {template['genre']}\n" f"Thematic direction: {template['theme']}\n" f"Setting inspiration: {template['setting']}"
        return await self.generate_metadata(enriched_prompt)

    # NOT YET WIRED INTO CLI — available for future interactive bootstrap flow
    @staticmethod
    def suggest_story_structure(genre: str) -> dict[str, Any]:
        genre_lower = genre.lower()
        if any(keyword in genre_lower for keyword in ("epic", "fantasy", "saga")):
            return STORY_STRUCTURES["heroes_journey"]
        if any(keyword in genre_lower for keyword in ("mystery", "thriller", "detective")):
            return STORY_STRUCTURES["three_act"]
        if any(keyword in genre_lower for keyword in ("literary", "slice of life")):
            return STORY_STRUCTURES["episodic"]
        if any(keyword in genre_lower for keyword in ("drama", "tragedy", "gothic", "horror")):
            return STORY_STRUCTURES["five_act"]
        return STORY_STRUCTURES["three_act"]

    # NOT YET WIRED INTO CLI — available for future interactive bootstrap flow
    # Uses prompts/initialization/world_building_questions.j2
    async def generate_world_building_questions(
        self,
        project_config: NarrativeProjectConfig,
    ) -> list[str]:
        prompt = render_prompt(
            "initialization/world_building_questions.j2",
            {
                "title": project_config.title,
                "genre": project_config.genre,
                "theme": project_config.theme,
                "setting": project_config.setting,
                "protagonist_name": project_config.protagonist_name,
            },
        )

        response_text, _usage = await self.language_model_service.async_call_llm(
            model_name=config.LARGE_MODEL,
            prompt=prompt,
            temperature=config.TEMPERATURE_INITIAL_SETUP,
        )

        parsed, _candidates, parse_errors = try_load_json_from_response(response_text, expected_root=list)
        if parsed is None:
            raise ValueError(f"World-building response did not contain valid JSON: {parse_errors}")

        if not all(isinstance(question, str) for question in parsed):
            raise TypeError("World-building response must be a JSON array of strings")

        return parsed

    def save_config(self, project_config: NarrativeProjectConfig, *, review: bool) -> Path:
        project_directory = ProjectManager.save_config(project_config, review=review)
        logger.info(
            "Project configuration saved",
            project_dir=str(project_directory),
            title=project_config.title,
            review=review,
        )
        return project_directory
