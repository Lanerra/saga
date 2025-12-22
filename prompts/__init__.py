"""Provide prompt rendering and prompt-context helpers.

This package contains:

- Template rendering utilities in `prompts.prompt_renderer`.
- Helper functions that build plain-text context snippets for templates in
  `prompts.prompt_data_getters`.

Notes:
    Templates live under the `prompts/` directory alongside these modules (for
    example `prompts/<agent_name>/*.j2` and `prompts/<agent_name>/system.md`).
"""
