# initialization/bootstrappers/common.py
import json
from typing import Any

import structlog

import config
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import render_prompt

logger = structlog.get_logger(__name__)


async def bootstrap_field(
    field_name: str,
    context_data: dict[str, Any],
    prompt_template_path: str,
    is_list: bool = False,
    list_count: int = 1,
) -> tuple[Any, dict[str, int] | None]:
    """Call LLM to fill a single field or list of fields."""
    logger.info(f"Bootstrapping field: '{field_name}'...")
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
        from ..error_handling import handle_bootstrap_error, ErrorSeverity
        handle_bootstrap_error(
            ValueError("Empty LLM response"),
            f"LLM bootstrap field generation: {field_name}",
            ErrorSeverity.WARNING,
            {"field_name": field_name, "is_list": is_list}
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

            from ..error_handling import handle_bootstrap_error, ErrorSeverity
            handle_bootstrap_error(
                TypeError(f"Unexpected type {type(value)} for field {field_name}"),
                f"LLM field type validation: {field_name}",
                ErrorSeverity.WARNING,
                {"field_name": field_name, "expected_type": "list" if is_list else "str", "actual_type": type(value).__name__}
            )
        else:
            from ..error_handling import handle_bootstrap_error, ErrorSeverity
            handle_bootstrap_error(
                ValueError("LLM response was not a JSON object"),
                f"LLM JSON parsing: {field_name}",
                ErrorSeverity.WARNING,
                {"field_name": field_name, "response_text": response_text[:100]}
            )
    except json.JSONDecodeError:
        if is_list:
            return (
                [line.strip() for line in response_text.splitlines() if line.strip()],
                usage_data,
            )
        return response_text.strip(), usage_data

    return ([] if is_list else ""), usage_data
