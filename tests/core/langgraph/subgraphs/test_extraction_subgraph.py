from unittest.mock import patch

import pytest

from core.langgraph.state import create_initial_state
from core.langgraph.subgraphs.extraction import create_extraction_subgraph


@pytest.mark.asyncio
async def test_extraction_subgraph():
    # Setup
    workflow = create_extraction_subgraph()
    state = create_initial_state(
        project_id="test_project",
        title="Test Novel",
        genre="Fantasy",
        theme="Heroism",
        setting="A magical world",
        target_word_count=50000,
        total_chapters=10,
        project_dir="/tmp/test_project",
        protagonist_name="Test Hero",
    )
    state["draft_text"] = (
        "Elara walked into the Sunken Library. She found the Starfall Map."
    )
    state["current_chapter"] = 1

    # Mock LLM responses (note: LLM still returns character_updates, world_updates in JSON,
    # but extraction nodes transform these to extracted_entities/extracted_relationships)
    mock_char_response = '{"character_updates": {"Elara": {"description": "A brave hero", "traits": ["brave"], "status": "active"}}}'
    mock_loc_response = '{"world_updates": {"Location": {"Sunken Library": {"description": "A dusty old library"}}}}'
    mock_event_response = '{"world_updates": {"Event": {"Discovery": {"description": "Elara finds the map"}}}}'
    mock_rel_response = '{"kg_triples": [{"subject": "Elara", "predicate": "LOCATED_IN", "object_entity": "Sunken Library", "description": "Elara is in the library"}]}'

    with patch("core.llm_interface_refactored.llm_service.async_call_llm") as mock_llm:
        # Configure mock to return different responses based on prompt content
        async def side_effect(*args, **kwargs):
            prompt = kwargs.get("prompt", "")
            if "specialized character extraction agent" in prompt:
                return mock_char_response, None
            elif "specialized location extraction agent" in prompt:
                return mock_loc_response, None
            elif "specialized event extraction agent" in prompt:
                return mock_event_response, None
            elif "specialized relationship extraction agent" in prompt:
                return mock_rel_response, None
            return "{}", None

        mock_llm.side_effect = side_effect

        # Run workflow
        result = await workflow.ainvoke(state)

        # Verify results
        assert "extracted_entities" in result
        entities = result["extracted_entities"]

        # Check characters
        assert len(entities["characters"]) == 1
        assert entities["characters"][0].name == "Elara"

        # Check world items (locations + events)
        assert len(entities["world_items"]) == 2
        names = [item.name for item in entities["world_items"]]
        assert "Sunken Library" in names
        assert "Discovery" in names

        # Check relationships
        assert len(result["extracted_relationships"]) == 1
        assert result["extracted_relationships"][0].source_name == "Elara"
        assert result["extracted_relationships"][0].target_name == "Sunken Library"
