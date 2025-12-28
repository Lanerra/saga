from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.exceptions import ValidationError
from core.graph_healing_service import GraphHealingService

_ENRICHMENT_JSON_CONTRACT_ERROR_MESSAGE = (
    "Graph healing enrichment JSON contract violated: could not parse a JSON object from the model response."
)


@pytest.mark.asyncio
async def test_enrich_node_from_context_invalid_json_then_valid_retries_and_succeeds() -> None:
    service = GraphHealingService()

    node = {
        "element_id": "neo4j-internal-element-id-123",
        "id": "stable-app-id-ABC",
        "name": "Provisional Entity",
        "type": "Character",
        "description": "Unknown",
        "traits": [],
        "created_chapter": 1,
    }

    mentions = [
        {
            "chapter_number": 1,
            "summary": "Provisional Entity was brave and led the group through the storm.",
        }
    ]

    invalid_response = "I think the answer is:\n```json\n{}\n```"
    valid_response = (
        '{"inferred_description":"A brave leader.","inferred_traits":["Brave"],'
        '"inferred_role":"Protagonist","confidence":0.9}'
    )

    with (
        patch(
            "data_access.kg_queries.get_chapter_context_for_entity",
            new=AsyncMock(return_value=mentions),
        ),
        patch(
            "core.graph_healing_service.llm_service.async_call_llm",
            new=AsyncMock(side_effect=[(invalid_response, {}), (valid_response, {})]),
        ) as mock_llm,
    ):
        enriched = await service.enrich_node_from_context(node, model="irrelevant-model")

    assert enriched == {
        "inferred_description": "A brave leader.",
        "inferred_traits": ["Brave"],
        "inferred_role": "Protagonist",
        "confidence": 0.9,
    }

    assert mock_llm.await_count == 2
    second_call_kwargs = mock_llm.await_args_list[1].kwargs
    assert "Corrective instruction:" in second_call_kwargs["prompt"]


@pytest.mark.asyncio
async def test_enrich_node_from_context_all_attempts_invalid_json_raises_stable_error() -> None:
    service = GraphHealingService()

    node = {
        "element_id": "neo4j-internal-element-id-123",
        "id": "stable-app-id-ABC",
        "name": "Provisional Entity",
        "type": "Character",
        "description": "Unknown",
        "traits": [],
        "created_chapter": 1,
    }

    mentions = [
        {
            "chapter_number": 1,
            "summary": "Provisional Entity appeared briefly.",
        }
    ]

    invalid_response_one = "not json"
    invalid_response_two = "also not json"

    with (
        patch(
            "data_access.kg_queries.get_chapter_context_for_entity",
            new=AsyncMock(return_value=mentions),
        ),
        patch(
            "core.graph_healing_service.llm_service.async_call_llm",
            new=AsyncMock(side_effect=[(invalid_response_one, {}), (invalid_response_two, {})]),
        ) as mock_llm,
    ):
        with pytest.raises(ValidationError) as exc:
            await service.enrich_node_from_context(node, model="irrelevant-model")

    assert str(exc.value) == _ENRICHMENT_JSON_CONTRACT_ERROR_MESSAGE
    assert mock_llm.await_count == 2


@pytest.mark.asyncio
async def test_enrich_node_from_context_valid_json_first_attempt_succeeds() -> None:
    service = GraphHealingService()

    node = {
        "element_id": "neo4j-internal-element-id-123",
        "id": "stable-app-id-ABC",
        "name": "Provisional Entity",
        "type": "Character",
        "description": "Unknown",
        "traits": [],
        "created_chapter": 1,
    }

    mentions = [
        {
            "chapter_number": 1,
            "summary": "Provisional Entity appeared briefly.",
        }
    ]

    valid_response = (
        '{"inferred_description":"A minor figure.","inferred_traits":[],'
        '"inferred_role":"Ally","confidence":0.6}'
    )

    with (
        patch(
            "data_access.kg_queries.get_chapter_context_for_entity",
            new=AsyncMock(return_value=mentions),
        ),
        patch(
            "core.graph_healing_service.llm_service.async_call_llm",
            new=AsyncMock(return_value=(valid_response, {})),
        ) as mock_llm,
    ):
        enriched = await service.enrich_node_from_context(node, model="irrelevant-model")

    assert enriched == {
        "inferred_description": "A minor figure.",
        "inferred_traits": [],
        "inferred_role": "Ally",
        "confidence": 0.6,
    }

    assert mock_llm.await_count == 1