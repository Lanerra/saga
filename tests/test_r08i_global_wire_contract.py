"""Wire schema matches the already-strict selected global-outline admission."""
import json
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.initialization.global_outline_node import GlobalOutlineSchema, generate_global_outline
from core.service_context import get_services
from tests.test_staged_initialization import example_state


async def test_selected_global_outline_requires_arcs_and_nested_prompt_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["global_outline_ref"] is not None
    response = manager.load_json_strict(state["global_outline_ref"])
    response = {key: value for key, value in response.items() if key in GlobalOutlineSchema.model_fields}
    calls = []

    async def provider(**options: Any) -> tuple[str, dict[str, Any]]:
        calls.append(options)
        schema = options["response_format"]["json_schema"]["schema"]
        assert "character_arcs" in schema["required"]
        assert "key_events" in schema["$defs"]["ActOutline"]["required"]
        assert "key_moments" in schema["$defs"]["CharacterArc"]["required"]
        assert schema["properties"]["character_arcs"]["minItems"] == 1
        assert schema["properties"]["character_arcs"]["maxItems"] == 1
        assert schema["$defs"]["CharacterArc"]["properties"]["character_name"]["enum"] == ["Ada"]
        assert options["response_format"]["json_schema"]["strict"] is False
        return json.dumps(response), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    result = await generate_global_outline(state)
    assert not result.get("last_error"), result
    assert len(calls) == 1
    assert result["initialization_step"] == "global_outline_complete"


async def test_missing_selected_arcs_still_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["global_outline_ref"] is not None
    response = manager.load_json_strict(state["global_outline_ref"])
    response = {key: value for key, value in response.items() if key in GlobalOutlineSchema.model_fields}
    response.pop("character_arcs")

    async def provider(**options: Any) -> tuple[str, dict[str, Any]]:
        return json.dumps(response), {}

    monkeypatch.setattr(get_services().language_model, "async_call_llm", provider)
    result = await generate_global_outline(state)
    assert result["initialization_step"] == "global_outline_failed"
    assert "exactly one arc" in str(result["last_error"])
    assert result.get("global_outline_ref") is None
