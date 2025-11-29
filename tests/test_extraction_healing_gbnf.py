import json
from unittest.mock import AsyncMock, patch

import pytest

from core.graph_healing_service import graph_healing_service
from core.langgraph.nodes.extraction_nodes import (
    extract_characters,
    extract_events,
    extract_locations,
    extract_relationships,
)


@pytest.fixture
def mock_llm_service():
    with patch("core.langgraph.nodes.extraction_nodes.llm_service") as mock:
        mock.async_call_llm = AsyncMock()
        yield mock


@pytest.fixture
def mock_healing_llm_service():
    with patch("core.graph_healing_service.llm_service") as mock:
        mock.async_call_llm = AsyncMock()
        yield mock


@pytest.fixture
def mock_content_manager():
    with patch("core.langgraph.nodes.extraction_nodes.ContentManager") as mock:
        instance = mock.return_value
        instance.read_chapter_text.return_value = "Chapter text content"
        yield mock


@pytest.fixture
def mock_get_draft_text():
    with patch("core.langgraph.nodes.extraction_nodes.get_draft_text") as mock:
        mock.return_value = "Chapter text content"
        yield mock


@pytest.fixture
def narrative_state():
    return {
        "project_dir": "test_project",
        "current_chapter": 1,
        "title": "Test Novel",
        "genre": "Fantasy",
        "protagonist_name": "Hero",
        "medium_model": "test-model",
    }


@pytest.mark.asyncio
async def test_extract_characters_gbnf(
    mock_llm_service, mock_content_manager, mock_get_draft_text, narrative_state
):
    # Mock LLM response
    mock_response = json.dumps(
        {
            "character_updates": {
                "Hero": {
                    "description": "The protagonist",
                    "traits": ["Brave", "Strong"],
                    "status": "Active",
                    "relationships": {},
                }
            }
        }
    )
    mock_llm_service.async_call_llm.return_value = (mock_response, {})

    # Execute
    result = await extract_characters(narrative_state)

    # Verify LLM call used grammar
    call_kwargs = mock_llm_service.async_call_llm.call_args[1]
    assert "grammar" in call_kwargs
    assert "root ::= character_extraction" in call_kwargs["grammar"]

    # Verify result
    assert "character_updates" in result
    assert len(result["character_updates"]) == 1
    assert result["character_updates"][0].name == "Hero"


@pytest.mark.asyncio
async def test_extract_locations_gbnf(
    mock_llm_service, mock_content_manager, mock_get_draft_text, narrative_state
):
    # Mock LLM response
    mock_response = json.dumps(
        {
            "world_updates": {
                "Location": {
                    "Castle": {
                        "description": "A big castle",
                        "goals": [],
                        "rules": [],
                        "key_elements": [],
                    }
                }
            }
        }
    )
    mock_llm_service.async_call_llm.return_value = (mock_response, {})

    # Execute
    result = await extract_locations(narrative_state)

    # Verify LLM call used grammar
    call_kwargs = mock_llm_service.async_call_llm.call_args[1]
    assert "grammar" in call_kwargs
    assert "root ::= world_extraction" in call_kwargs["grammar"]

    # Verify result
    assert "location_updates" in result
    assert len(result["location_updates"]) == 1
    assert result["location_updates"][0].name == "Castle"


@pytest.mark.asyncio
async def test_extract_events_gbnf(
    mock_llm_service, mock_content_manager, mock_get_draft_text, narrative_state
):
    # Mock LLM response
    mock_response = json.dumps(
        {
            "world_updates": {
                "Event": {
                    "Battle": {"description": "A fierce battle", "key_elements": []}
                }
            }
        }
    )
    mock_llm_service.async_call_llm.return_value = (mock_response, {})

    # Execute
    result = await extract_events(narrative_state)

    # Verify LLM call used grammar
    call_kwargs = mock_llm_service.async_call_llm.call_args[1]
    assert "grammar" in call_kwargs
    assert "root ::= world_extraction" in call_kwargs["grammar"]

    # Verify result
    assert "event_updates" in result
    assert len(result["event_updates"]) == 1
    assert result["event_updates"][0].name == "Battle"


@pytest.mark.asyncio
async def test_extract_relationships_gbnf(
    mock_llm_service, mock_content_manager, mock_get_draft_text, narrative_state
):
    # Mock LLM response
    mock_response = json.dumps(
        {
            "kg_triples": [
                {
                    "subject": "Hero",
                    "predicate": "LIVES_IN",
                    "object_entity": "Castle",
                    "description": "Hero lives in the castle",
                }
            ]
        }
    )
    mock_llm_service.async_call_llm.return_value = (mock_response, {})

    # Execute
    result = await extract_relationships(narrative_state)

    # Verify LLM call used grammar
    call_kwargs = mock_llm_service.async_call_llm.call_args[1]
    assert "grammar" in call_kwargs
    assert "root ::= relationship_extraction" in call_kwargs["grammar"]

    # Verify result
    assert "relationship_updates" in result
    assert len(result["relationship_updates"]) == 1
    assert result["relationship_updates"][0].source_name == "Hero"
    assert result["relationship_updates"][0].target_name == "Castle"


@pytest.mark.asyncio
async def test_enrich_node_gbnf(mock_healing_llm_service):
    # Mock data
    node = {
        "element_id": "1",
        "name": "Hero",
        "type": "Character",
        "description": "Unknown",
        "traits": [],
    }

    # Mock KG query for context - patching the source module since it's imported inside function
    with patch(
        "data_access.kg_queries.get_chapter_context_for_entity", new_callable=AsyncMock
    ) as mock_ctx:
        mock_ctx.return_value = [{"chapter": 1, "summary": "Hero did something."}]

        # Mock LLM response (tuple)
        mock_response_str = json.dumps(
            {
                "inferred_description": "A verified hero",
                "inferred_traits": ["Brave"],
                "inferred_role": "Protagonist",
                "confidence": 0.9,
            }
        )
        mock_healing_llm_service.async_call_llm.return_value = (mock_response_str, {})

        # Execute
        enriched = await graph_healing_service.enrich_node_from_context(
            node, "test-model"
        )

        # Verify
        assert enriched["inferred_description"] == "A verified hero"
        assert enriched["confidence"] == 0.9

        # Verify grammar usage
        call_kwargs = mock_healing_llm_service.async_call_llm.call_args[1]
        assert "grammar" in call_kwargs
        # The grammar loader reads the file, checking content length to ensure it's loaded
        assert len(call_kwargs["grammar"]) > 0
