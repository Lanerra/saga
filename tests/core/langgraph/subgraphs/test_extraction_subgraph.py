from unittest.mock import patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.state import create_initial_state
from core.langgraph.subgraphs.extraction import create_extraction_subgraph


@pytest.mark.asyncio
async def test_extraction_subgraph():
    # Setup
    workflow = create_extraction_subgraph()
    project_dir = "/tmp/test_project"
    state = create_initial_state(
        project_id="test_project",
        title="Test Novel",
        genre="Fantasy",
        theme="Heroism",
        setting="A magical world",
        target_word_count=50000,
        total_chapters=10,
        project_dir=project_dir,
        protagonist_name="Test Hero",
    )
    draft_text = "Elara walked into the Sunken Library. She found the Starfall Map."
    content_manager = ContentManager(project_dir)
    draft_ref = content_manager.save_text(draft_text, "draft", "chapter_1", 1)
    state["draft_ref"] = draft_ref
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


@pytest.mark.asyncio
async def test_extraction_nodes_do_not_log_prompt_or_response_fragments(tmp_path):
    """
    Regression test for LANGGRAPH-021 / remediation 9.1 #3.

    Extraction nodes must not log any prompt/response fragments (e.g., `prompt_head`,
    `response_head`) or raw LLM output (`raw_text`) because those can contain narrative
    manuscript content.
    """
    from core.langgraph.nodes.extraction_nodes import extract_locations

    project_dir = str(tmp_path / "test_project")
    state = create_initial_state(
        project_id="test_project",
        title="Test Novel",
        genre="Fantasy",
        theme="Heroism",
        setting="A magical world",
        target_word_count=50000,
        total_chapters=10,
        project_dir=project_dir,
        protagonist_name="Test Hero",
    )

    content_manager = ContentManager(project_dir)
    draft_text = "Elara walked into the Sunken Library. She found the Starfall Map."
    state["draft_ref"] = content_manager.save_text(draft_text, "draft", "chapter_1", 1)
    state["current_chapter"] = 1

    # Force a JSON decode error to exercise the logging path.
    with (
        patch("core.llm_interface_refactored.llm_service.async_call_llm", return_value=("{not json", None)),
        patch("core.langgraph.nodes.extraction_nodes.logger") as mock_logger,
    ):
        result = await extract_locations(state)
        assert result == {}

        # Ensure the JSON parse error log does not include raw text.
        error_calls = mock_logger.error.call_args_list
        assert error_calls, "Expected an error log on JSON parsing failure"

        # Look for the specific JSON parse error event.
        found_parse_error = False
        for call in error_calls:
            args, kwargs = call
            if args and args[0] == "extract_locations: failed to parse JSON":
                found_parse_error = True
                assert "raw_text" not in kwargs
                assert "response_head" not in kwargs
                assert "prompt_head" not in kwargs
                assert "response_sha1" in kwargs
                assert "response_len" in kwargs
        assert found_parse_error, "Expected 'extract_locations: failed to parse JSON' log event"

        # Ensure *no* log call contains content fragments.
        forbidden_keys = {"prompt_head", "response_head", "raw_text"}
        for method_name in ("debug", "info", "warning", "error"):
            method = getattr(mock_logger, method_name)
            for call in method.call_args_list:
                _args, kwargs = call
                assert forbidden_keys.isdisjoint(set(kwargs.keys()))

@pytest.mark.asyncio
async def test_extraction_clears_previous_state():
    """
    Test that extraction properly clears previous state to prevent accumulation.

    This verifies the fix for the issue where tens of thousands of entities
    were being accumulated across chapters.

    Uses SEQUENTIAL extraction instead of parallel with reducers.
    """
    workflow = create_extraction_subgraph()
    project_dir = "/tmp/test_project"
    state = create_initial_state(
        project_id="test_project",
        title="Test Novel",
        genre="Fantasy",
        theme="Heroism",
        setting="A magical world",
        target_word_count=50000,
        total_chapters=10,
        project_dir=project_dir,
        protagonist_name="Test Hero",
    )

    # Simulate having leftover extraction state from a previous chapter
    from core.langgraph.state import ExtractedEntity, ExtractedRelationship

    state["extracted_entities"] = {
        "characters": [
            ExtractedEntity(
                name="OldCharacter",
                type="Character",
                description="From previous chapter",
                first_appearance_chapter=1,
                attributes={},
            )
        ],
        "world_items": [
            ExtractedEntity(
                name="OldLocation",
                type="Location",
                description="From previous chapter",
                first_appearance_chapter=1,
                attributes={},
            )
        ],
    }
    state["extracted_relationships"] = [
        ExtractedRelationship(
            source_name="OldCharacter",
            target_name="OldLocation",
            relationship_type="LOCATED_IN",
            description="Old relationship",
            chapter=1,
            confidence=0.8,
        )
    ]

    # Set up for new chapter extraction
    draft_text = "New character arrives at new location."
    content_manager = ContentManager(project_dir)
    draft_ref = content_manager.save_text(draft_text, "draft", "chapter_2", 1)
    state["draft_ref"] = draft_ref
    state["current_chapter"] = 2

    # Mock LLM to return new entities
    mock_char_response = '{"character_updates": {"NewCharacter": {"description": "A new hero", "traits": ["brave"], "status": "active"}}}'
    mock_loc_response = '{"world_updates": {"Location": {"NewLocation": {"description": "A new place"}}}}'
    mock_event_response = '{"world_updates": {}}'
    mock_rel_response = '{"kg_triples": [{"subject": "NewCharacter", "predicate": "LOCATED_IN", "object_entity": "NewLocation", "description": "New relationship"}]}'

    with patch("core.llm_interface_refactored.llm_service.async_call_llm") as mock_llm:

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

        # Run workflow (now sequential - extract_characters clears state first)
        result = await workflow.ainvoke(state)

        # Verify that ONLY the new entities are present (old ones were cleared)
        entities = result["extracted_entities"]

        # Should have exactly 1 character (NewCharacter), not 2
        # extract_characters clears the state as the FIRST node
        assert len(entities.get("characters", [])) == 1
        assert entities["characters"][0].name == "NewCharacter"

        # Should have exactly 1 world item (NewLocation), not 2
        assert len(entities.get("world_items", [])) == 1
        assert entities["world_items"][0].name == "NewLocation"

        # Should have exactly 1 relationship, not 2
        assert len(result["extracted_relationships"]) == 1
        assert result["extracted_relationships"][0].source_name == "NewCharacter"
        assert result["extracted_relationships"][0].target_name == "NewLocation"
