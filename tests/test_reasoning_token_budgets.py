"""Completion budgets include reasoning; retained context budgets do not."""
from unittest.mock import AsyncMock

import pytest

import config
from config.settings import SagaSettings
from core.http_client_service import prepare_completion_payload
from core.langgraph.nodes.context_scene_retrieval import _summarize_scene_text
from core.parsers.act_outline_parser import ActOutlineParser
from core.service_context import get_services


@pytest.mark.parametrize("name", [
    "MAX_GENERATION_TOKENS", "MAX_PLANNING_TOKENS", "MAX_SUMMARY_TOKENS",
    "MAX_KG_TRIPLE_TOKENS", "MAX_PREPOP_KG_TOKENS",
])
def test_reasoning_completion_defaults(name: str) -> None:
    assert SagaSettings.model_fields[name].default == 65536


def test_context_default_reserves_prompt_space() -> None:
    assert SagaSettings.model_fields["MAX_CONTEXT_TOKENS"].default == 131072
    payload = prepare_completion_payload(
        get_services().configuration, "synthetic", [{"role": "user", "content": "Write a scene."}], 0.7, 65536,
    )
    assert payload["max_tokens"] == 65536


async def test_act_selectors_use_configured_completion_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_KG_TRIPLE_TOKENS", 65536)
    completion = AsyncMock(side_effect=[('[{"name":"Ada","role":"witness"}]', {}), ('{"location":"Hall"}', {})])
    monkeypatch.setattr(get_services().language_model, "async_call_llm", completion)
    parser = ActOutlineParser()
    assert await parser._extract_character_names("Arrival", "Ada arrives", "Rain", "Shelter", ["Ada"]) == [("Ada", "witness")]
    assert await parser._extract_event_location("Arrival", "Ada arrives", "Rain", "Shelter", [{"name": "Hall", "description": "Shelter"}]) == "Hall"
    assert [call.kwargs["max_tokens"] for call in completion.await_args_list] == [65536, 65536]


async def test_summary_retention_does_not_cap_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_SUMMARY_TOKENS", 65536)
    completion = AsyncMock(return_value=("Ada arrives.", {}))
    monkeypatch.setattr(get_services().language_model, "async_call_llm", completion)
    assert await _summarize_scene_text("Ada arrives in the rain.", "Arrival", "synthetic", 7) == "Ada arrives."
    assert completion.await_args is not None
    assert completion.await_args.kwargs["max_tokens"] == 65536
