# initialization/bootstrappers/common.py
import json
from typing import Any

import structlog

import config
from core.llm_interface import llm_service
from prompt_renderer import render_prompt

logger = structlog.get_logger(__name__)


async def bootstrap_field(
    field_name: str,
    context_data: dict[str, Any],
    prompt_template_path: str,
    is_list: bool = False,
    list_count: int = 1,
) -> tuple[Any, dict[str, int] | None]:
    """Call LLM to fill a single field or list of fields."""
    logger.info("Bootstrapping field: '%s'...", field_name)
    prompt = render_prompt(
        prompt_template_path,
        {"context": context_data, "field_name": field_name, "list_count": list_count},
    )

    response_text, usage_data = await llm_service.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt,
        temperature=config.Temperatures.INITIAL_SETUP,
        stream_to_disk=False,
        auto_clean_response=True,
    )

    if not response_text.strip():
        logger.warning(
            "LLM returned empty response for bootstrapping field '%s'.", field_name
        )
        return ([] if is_list else ""), usage_data

    try:
        parsed_json = json.loads(response_text)
        if isinstance(parsed_json, dict):
            value = parsed_json.get(field_name)

            if is_list:
                if isinstance(value, list):
                    return value, usage_data
                if isinstance(value, str):
                    logger.info(
                        "LLM returned a string for list field '%s'. Parsing string into list.",
                        field_name,
                    )
                    items = [
                        item.strip().lstrip("-* ").strip()
                        for item in value.replace("\n", ",").split(",")
                        if item.strip()
                    ]
                    return items, usage_data
            elif isinstance(value, str):
                return value.strip(), usage_data

            logger.warning(
                "LLM JSON for '%s' had unexpected type: %s", field_name, type(value)
            )
        else:
            logger.warning(
                "LLM response for '%s' was not a JSON object. Response: %s",
                field_name,
                response_text[:100],
            )
    except json.JSONDecodeError:
        if is_list:
            return (
                [line.strip() for line in response_text.splitlines() if line.strip()],
                usage_data,
            )
        return response_text.strip(), usage_data

    return ([] if is_list else ""), usage_data
