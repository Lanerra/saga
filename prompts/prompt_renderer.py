# prompts/prompt_renderer.py
"""Render prompt templates and load per-agent system prompts.

This module provides a small, opinionated wrapper around Jinja2 for rendering
templates under the `prompts/` directory.

Rendering behavior and contracts:

- Templates are loaded relative to `PROMPTS_PATH`.
- Undefined variables are treated as errors via Jinja2's `StrictUndefined`.
  Missing template variables will raise at render time rather than rendering a
  placeholder.
- Auto-escaping is disabled. If templates interpolate user-provided content, the
  caller is responsible for any escaping/sanitization appropriate for the target
  model and downstream consumers.
- The `config` module is always injected into the template context as `config`.

System prompt loading:

- System prompts are read from `prompts/<agent_name>/system.md`.
- Reads are cached in-process via `functools.lru_cache`. Cache invalidation is
  manual (e.g., via `get_system_prompt.cache_clear()`).
- Failures to read the system prompt file are treated as best-effort and return
  an empty string.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config

PROMPTS_PATH = Path(__file__).parent
_env = Environment(
    loader=FileSystemLoader(PROMPTS_PATH),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, context: dict[str, Any]) -> str:
    """Render a Jinja2 prompt template with a strict variable contract.

    Args:
        template_name: Template path relative to `PROMPTS_PATH`, for example
            `narrative_agent/draft_scene.j2`.
        context: Mapping of template variables to values. The required variables
            depend on the selected template. The `config` module is always added
            to the context as `config` and may be referenced from templates.

    Returns:
        Rendered prompt text.

    Raises:
        jinja2.TemplateNotFound: If `template_name` does not exist under
            `PROMPTS_PATH`.
        jinja2.UndefinedError: If the template references a variable that is not
            provided in `context` (or otherwise available), because templates are
            rendered with `StrictUndefined`.

    Notes:
        Auto-escaping is disabled. This function does not sanitize or escape
        user-provided content that may be included in `context`.
    """
    template = _env.get_template(template_name)
    # Always include config in template context
    template_context = {"config": config, **context}
    return template.render(**template_context)


@lru_cache(maxsize=16)
def get_system_prompt(agent_name: str) -> str:
    """Load a per-agent system prompt, returning an empty string when unavailable.

    Args:
        agent_name: Agent directory name under `PROMPTS_PATH` that contains
            `system.md`.

    Returns:
        The stripped contents of `prompts/<agent_name>/system.md` if the file
        exists and can be read. Returns an empty string when the file is missing
        or cannot be read.

    Notes:
        Results are cached in-process (LRU, `maxsize=16`). Call
        `get_system_prompt.cache_clear()` to invalidate the cache if prompt files
        change during the process lifetime.
    """
    system_path = PROMPTS_PATH / agent_name / "system.md"
    try:
        if system_path.exists():
            return system_path.read_text(encoding="utf-8").strip()
    except Exception:
        # Fail open: return empty string if read fails
        return ""
    return ""
