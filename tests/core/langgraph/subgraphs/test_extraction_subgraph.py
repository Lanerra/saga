from unittest.mock import patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.state import create_initial_state
from core.langgraph.subgraphs.extraction import create_extraction_subgraph


@pytest.mark.asyncio
async def test_extraction_subgraph():
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

    mock_char_payload = {
        "character_updates": {
            "Elara": {
                "description": "A brave hero",
                "traits": ["brave"],
                "status": "active",
                "relationships": {},
            }
        }
    }
    mock_loc_payload = {
        "world_updates": {
            "Location": {
                "Sunken Library": {
                    "description": "A dusty old library",
                    "category": "Structure",
                    "goals": [],
                    "rules": [],
                    "key_elements": ["dust", "books"],
                }
            }
        }
    }
    mock_event_payload = {
        "world_updates": {
            "Event": {
                "Discovery": {
                    "description": "Elara finds the map",
                    "category": "Scene",
                    "goals": [],
                    "rules": [],
                    "key_elements": ["Starfall Map"],
                }
            }
        }
    }
    mock_rel_payload = {
        "kg_triples": [
            {
                "subject": "Elara",
                "predicate": "LOCATED_IN",
                "object_entity": "Sunken Library",
                "description": "Elara is in the library",
            }
        ]
    }

    with patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object") as mock_llm_json_object:

        async def side_effect(*args, **kwargs):
            prompt = kwargs.get("prompt", "")
            if "specialized character extraction agent" in prompt:
                return mock_char_payload, None
            if "specialized location extraction agent" in prompt:
                return mock_loc_payload, None
            if "specialized event extraction agent" in prompt:
                return mock_event_payload, None
            if "specialized relationship extraction agent" in prompt:
                return mock_rel_payload, None
            return {}, None

        mock_llm_json_object.side_effect = side_effect

        result = await workflow.ainvoke(state)

    assert "extracted_entities" in result
    entities = result["extracted_entities"]

    assert len(entities["characters"]) == 1
    assert entities["characters"][0].name == "Elara"

    assert len(entities["world_items"]) == 2
    names = [item.name for item in entities["world_items"]]
    assert "Sunken Library" in names
    assert "Discovery" in names

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

    with (
        patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object", side_effect=ValueError("LLM returned invalid JSON")),
        patch("core.langgraph.nodes.extraction_nodes.logger") as mock_logger,
    ):
        result = await extract_locations(state)

    assert result.get("has_fatal_error") is True
    assert result.get("error_node") == "extract_locations"
    assert result.get("last_error") is not None

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

    draft_text = "New character arrives at new location."
    content_manager = ContentManager(project_dir)
    draft_ref = content_manager.save_text(draft_text, "draft", "chapter_2", 1)
    state["draft_ref"] = draft_ref
    state["current_chapter"] = 2

    mock_char_payload = {
        "character_updates": {
            "NewCharacter": {
                "description": "A new hero",
                "traits": ["brave"],
                "status": "active",
                "relationships": {},
            }
        }
    }
    mock_loc_payload = {
        "world_updates": {
            "Location": {
                "NewLocation": {
                    "description": "A new place",
                    "category": "Region",
                    "goals": [],
                    "rules": [],
                    "key_elements": [],
                }
            }
        }
    }
    mock_event_payload = {"world_updates": {"Event": {}}}
    mock_rel_payload = {
        "kg_triples": [
            {
                "subject": "NewCharacter",
                "predicate": "LOCATED_IN",
                "object_entity": "NewLocation",
                "description": "New relationship",
            }
        ]
    }

    with patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object") as mock_llm_json_object:

        async def side_effect(*args, **kwargs):
            prompt = kwargs.get("prompt", "")
            if "specialized character extraction agent" in prompt:
                return mock_char_payload, None
            if "specialized location extraction agent" in prompt:
                return mock_loc_payload, None
            if "specialized event extraction agent" in prompt:
                return mock_event_payload, None
            if "specialized relationship extraction agent" in prompt:
                return mock_rel_payload, None
            return {}, None

        mock_llm_json_object.side_effect = side_effect

        result = await workflow.ainvoke(state)

    entities = result["extracted_entities"]

    assert len(entities.get("characters", [])) == 1
    assert entities["characters"][0].name == "NewCharacter"

    assert len(entities.get("world_items", [])) == 1
    assert entities["world_items"][0].name == "NewLocation"

    assert len(result["extracted_relationships"]) == 1
    assert result["extracted_relationships"][0].source_name == "NewCharacter"
    assert result["extracted_relationships"][0].target_name == "NewLocation"


@pytest.mark.asyncio
async def test_extraction_subgraph_propagates_llm_failure_as_fatal(tmp_path):
    """
    CORE-007: Ensure extraction failures propagate in a controlled way.

    If a core LLM call fails (typed exception), extraction should set:
    - has_fatal_error=True
    - last_error populated
    - error_node set to the failing node
    """
    from core.exceptions import LLMServiceError

    workflow = create_extraction_subgraph()

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

    with patch(
        "core.llm_interface_refactored.llm_service.async_call_llm_json_object",
        side_effect=LLMServiceError("LLM completion failed", details={"primary_model": "test-model"}),
    ):
        result = await workflow.ainvoke(state)

    assert result.get("has_fatal_error") is True
    assert result.get("error_node") == "extract_characters"
    assert result.get("last_error") is not None


@pytest.mark.asyncio
async def test_extract_characters_rejects_invalid_root_keyset(tmp_path):
    from core.langgraph.nodes.extraction_nodes import extract_characters

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
    draft_text = "Elara walked into the Sunken Library."
    state["draft_ref"] = content_manager.save_text(draft_text, "draft", "chapter_1", 1)
    state["current_chapter"] = 1

    payload = {
        "character_updates": {
            "Elara": {
                "description": "A brave hero",
                "traits": ["brave"],
                "status": "active",
                "relationships": {},
            }
        },
        "extra_root_key": "nope",
    }

    with patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object", return_value=(payload, None)):
        result = await extract_characters(state)

    assert result.get("has_fatal_error") is True
    assert result.get("error_node") == "extract_characters"
    assert "schema" in str(result.get("last_error", "")).lower()


@pytest.mark.asyncio
async def test_extract_locations_rejects_invalid_bucket_name(tmp_path):
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
    draft_text = "Elara walked into the Sunken Library."
    state["draft_ref"] = content_manager.save_text(draft_text, "draft", "chapter_1", 1)
    state["current_chapter"] = 1

    payload = {
        "world_updates": {
            "Locations": {
                "Sunken Library": {
                    "description": "A dusty old library",
                    "category": "Structure",
                    "goals": [],
                    "rules": [],
                    "key_elements": ["dust"],
                }
            }
        }
    }

    with patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object", return_value=(payload, None)):
        result = await extract_locations(state)

    assert result.get("has_fatal_error") is True
    assert result.get("error_node") == "extract_locations"
    assert "schema" in str(result.get("last_error", "")).lower()


@pytest.mark.asyncio
async def test_extract_events_rejects_missing_required_keys(tmp_path):
    from core.langgraph.nodes.extraction_nodes import extract_events

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
    draft_text = "Elara finds the map."
    state["draft_ref"] = content_manager.save_text(draft_text, "draft", "chapter_1", 1)
    state["current_chapter"] = 1

    payload = {
        "world_updates": {
            "Event": {
                "Discovery": {
                    "description": "Elara finds the map",
                    "category": "Scene",
                    "key_elements": ["Starfall Map"],
                }
            }
        }
    }

    with patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object", return_value=(payload, None)):
        result = await extract_events(state)

    assert result.get("has_fatal_error") is True
    assert result.get("error_node") == "extract_events"
    assert "schema" in str(result.get("last_error", "")).lower()


@pytest.mark.asyncio
async def test_extract_locations_rejects_extra_keys_in_entity_object(tmp_path):
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
    draft_text = "Elara walked into the Sunken Library."
    state["draft_ref"] = content_manager.save_text(draft_text, "draft", "chapter_1", 1)
    state["current_chapter"] = 1

    payload = {
        "world_updates": {
            "Location": {
                "Sunken Library": {
                    "description": "A dusty old library",
                    "category": "Structure",
                    "goals": [],
                    "rules": [],
                    "key_elements": ["dust"],
                    "extra": "nope",
                }
            }
        }
    }

    with patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object", return_value=(payload, None)):
        result = await extract_locations(state)

    assert result.get("has_fatal_error") is True
    assert result.get("error_node") == "extract_locations"
    assert "schema" in str(result.get("last_error", "")).lower()


@pytest.mark.asyncio
async def test_extract_relationships_log_is_privacy_safe_when_kg_triples_not_list(tmp_path):
    from core.langgraph.nodes.extraction_nodes import extract_relationships

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
    draft_text = "Elara walked into the Sunken Library."
    state["draft_ref"] = content_manager.save_text(draft_text, "draft", "chapter_1", 1)
    state["current_chapter"] = 1

    payload = {"kg_triples": {"not": "a list"}}

    with (
        patch("core.llm_interface_refactored.llm_service.async_call_llm_json_object", return_value=(payload, None)),
        patch("core.langgraph.nodes.extraction_nodes.logger") as mock_logger,
    ):
        result = await extract_relationships(state)

    assert result.get("extracted_relationships") == []

    warning_calls = mock_logger.warning.call_args_list
    assert warning_calls, "Expected a warning log"
    for call in warning_calls:
        _args, kwargs = call
        assert "raw_data" not in kwargs
        assert "raw_text" not in kwargs
        assert "response_head" not in kwargs
        assert "prompt_head" not in kwargs
