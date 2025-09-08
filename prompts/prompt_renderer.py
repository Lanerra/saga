# prompts/prompt_renderer.py
"""Utilities for rendering LLM prompts using Jinja2 templates."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from jinja2.exceptions import TemplateNotFound
import logging

import config

PROMPTS_PATH = Path(__file__).parent
_env = Environment(loader=FileSystemLoader(PROMPTS_PATH), autoescape=False)

logger = logging.getLogger(__name__)


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 template from the prompts directory."""
    try:
        template = _env.get_template(template_name)
        # Always include config in template context
        template_context = {"config": config, **context}
        return template.render(**template_context)
    except TemplateNotFound:
        logger.warning(f"Template '{template_name}' not found in {PROMPTS_PATH}")
        return f"[ERROR: Template '{template_name}' missing. Using fallback prompt.]"
