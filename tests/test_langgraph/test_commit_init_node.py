# tests/test_langgraph/test_commit_init_node.py
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.initialization.commit_init_node import (
    _extract_structured_character_data,
    _extract_world_items_from_outline,
    _parse_character_extraction_response,
    _parse_character_sheets_to_profiles,
    _parse_world_items_extraction,
    commit_initialization_to_graph,
)
from core.langgraph.state import NarrativeState, create_initial_state
from tests.fakes.service_context import patch_service

pytestmark = pytest.mark.usefixtures("offline_commit_providers")


@pytest.fixture
def base_state() -> NarrativeState:
    """Create a base state for testing."""
    return create_initial_state(
        project_id="test-project",
        title="Test Novel",
        genre="Fantasy",
        theme="Adventure",
        setting="Medieval world",
        target_word_count=80000,
        total_chapters=20,
        project_dir="/tmp/test-project",
        protagonist_name="Hero",
        extraction_model="test-model",
        revision_model="test-model",
    )


@pytest.fixture
def mock_content_manager() -> Iterator[MagicMock]:
    """Create a mock ContentManager."""
    with patch("core.langgraph.initialization.commit_init_node.ContentManager") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_get_functions() -> Iterator[dict[str, MagicMock]]:
    """Mock the content getter functions."""
    with patch("core.langgraph.initialization.commit_init_node.get_character_sheets") as mock_chars, patch("core.langgraph.initialization.commit_init_node.get_global_outline") as mock_global:
        mock_chars.return_value = {
            "Hero": {
                "description": "A brave warrior",
                "traits": ["brave", "loyal", "determined"],
                "status": "Active",
                "motivations": "Save the kingdom",
                "background": "Born in a small village",
                "skills": ["swordfighting", "leadership"],
                "relationships": {"Mentor": {"description": "Wise guide"}},
                "is_protagonist": True,
                "internal_conflict": "Doubt in own abilities",
            },
            "Villain": {
                "description": "Evil sorcerer",
                "traits": ["cunning", "ruthless"],
                "status": "Active",
                "motivations": "Conquer the world",
                "background": "Former court mage",
                "skills": ["dark magic"],
                "relationships": {},
                "is_protagonist": False,
                "internal_conflict": "",
            },
        }

        mock_global.return_value = {
            "raw_text": """The story takes place in the Kingdom of Eldoria.
            Key locations include the Royal Castle and the Dark Forest.
            The magical Sword of Light plays a crucial role.""",
            "act_count": 3,
        }

        yield {
            "chars": mock_chars,
            "global": mock_global,
        }


@pytest.fixture
def mock_neo4j_manager() -> Iterator[MagicMock]:
    """Mock the Neo4j manager."""
    with patch_service('database') as mock:
        mock.execute_cypher_batch = AsyncMock()
        yield mock


@pytest.fixture
def mock_llm_service() -> Iterator[MagicMock]:
    """Create a mock LLM service."""
    with patch_service('language_model') as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                '{"traits":["brave","loyal","strong"],"status":"Active","motivations":"Protect the innocent.","background":"Trained as a knight from childhood."}',
                {"prompt_tokens": 100, "completion_tokens": 50},
            )
        )
        yield mock


@pytest.mark.asyncio
async def test_commit_initialization_to_graph_success(tmp_path: Path) -> None:
    from core.langgraph.initialization.staged_import import InitializationImport
    from tests.test_staged_initialization import example_state, with_catalog

    state = with_catalog(example_state(tmp_path))
    with patch_service('language_model') as provider, patch("config.ENABLE_ENTITY_EMBEDDING_PERSISTENCE", False), patch("data_access.cache_coordinator.clear_character_read_caches") as clear_characters, patch("data_access.cache_coordinator.clear_world_read_caches") as clear_world:
        provider.async_call_llm = AsyncMock(return_value=("[]", {}))
        result = await commit_initialization_to_graph(state)
    assert result["initialization_step"] == "initialization_prepared"
    assert result["current_node"] == "commit_initialization"
    assert result["last_error"] is None
    assert InitializationImport(str(tmp_path)).load().identity == result["initialization_id"]
    clear_characters.assert_not_called()
    clear_world.assert_not_called()


@pytest.mark.asyncio
async def test_commit_initialization_no_writes_does_not_invalidate_caches(tmp_path: Path) -> None:
    with patch("data_access.cache_coordinator.clear_character_read_caches") as clear_characters, patch("data_access.cache_coordinator.clear_world_read_caches") as clear_world:
        result = await commit_initialization_to_graph({"project_dir": str(tmp_path)})
    assert result["initialization_step"] == "commit_failed"
    assert result["current_node"] == "commit_initialization"
    assert result["has_fatal_error"] is True
    clear_characters.assert_not_called()
    clear_world.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact", ["character_sheets", "global_outline"])
async def test_missing_required_initialization_source(tmp_path: Path, artifact: str) -> None:
    from tests.test_staged_initialization import example_state

    state = example_state(tmp_path)
    if artifact == "character_sheets":
        state.pop("character_sheets_ref")
    else:
        assert artifact == "global_outline"
        state.pop("global_outline_ref")
    with patch_service('language_model') as provider:
        provider.async_call_llm = AsyncMock(side_effect=AssertionError("Admission must precede provider calls"))
        result = await commit_initialization_to_graph(state)
    assert result["initialization_step"] == "commit_failed"
    assert result["current_node"] == "commit_initialization"
    assert result["has_fatal_error"] is True
    assert result["last_error"] == f"Initialization admission failed: Initialization requires selected {artifact}_ref"
    provider.async_call_llm.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("Synthetic provider failure"), OSError("Synthetic interrupted producer")])
async def test_producer_failure_never_runs_graph_batch(tmp_path: Path, failure: Exception) -> None:
    from core.service_context import get_services
    from tests.test_staged_initialization import example_state, with_catalog

    state = with_catalog(example_state(tmp_path))
    with patch_service('language_model') as provider, patch.object(get_services().database, "execute_in_transaction", new_callable=AsyncMock) as writes:
        provider.async_call_llm = AsyncMock(side_effect=failure)
        result = await commit_initialization_to_graph(state)
    assert result["initialization_step"] == "commit_failed"
    assert result["current_node"] == "commit_initialization"
    assert result["last_error"] == f"Initialization admission failed: {failure}"
    assert result["has_fatal_error"] is True
    assert result["error_node"] == "commit_initialization"
    writes.assert_not_awaited()
    assert not (tmp_path / ".saga/initialization/selected").exists()


@pytest.mark.asyncio
async def test_parse_character_sheets_to_profiles_with_traits() -> None:
    """Verify parsing character sheets with pre-parsed traits."""
    character_sheets = {
        "Hero": {
            "description": "A brave warrior",
            "traits": ["brave", "loyal", "determined"],
            "status": "Active",
            "motivations": "Save the kingdom",
            "background": "Born in a small village",
            "skills": ["swordfighting", "leadership"],
            "relationships": {"Mentor": {"description": "Wise guide"}},
            "is_protagonist": True,
            "internal_conflict": "Doubt in own abilities",
        },
    }

    profiles = await _parse_character_sheets_to_profiles(character_sheets)

    assert len(profiles) == 1
    assert profiles[0].name == "Hero"
    assert profiles[0].personality_description == "A brave warrior"
    assert "brave" in profiles[0].traits
    assert profiles[0].status == "Active"
    assert profiles[0].created_chapter == 0
    assert profiles[0].is_provisional is False


@pytest.mark.asyncio
async def test_parse_character_sheets_to_profiles_filters_invalid_traits() -> None:
    """Verify filtering of invalid traits."""
    character_sheets = {
        "Hero": {
            "description": "A brave warrior",
            "traits": ["brave", "very very long trait that exceeds limits", "loyal"],
            "status": "Active",
            "motivations": "",
            "background": "",
            "skills": [],
            "relationships": {},
            "is_protagonist": False,
            "internal_conflict": "",
        },
    }

    with patch("core.langgraph.initialization.commit_init_node.validate_and_filter_traits") as mock_validate:
        mock_validate.return_value = ["brave", "loyal"]

        profiles = await _parse_character_sheets_to_profiles(character_sheets)

        assert len(profiles) == 1
        assert len(profiles[0].traits) == 2


@pytest.mark.asyncio
async def test_parse_character_sheets_to_profiles_no_traits_uses_llm(
    mock_llm_service: MagicMock,
) -> None:
    """Verify LLM extraction when no pre-parsed traits exist."""
    character_sheets = {
        "Hero": {
            "description": "A brave warrior who fights for justice",
            "traits": [],
            "status": "Unknown",
            "motivations": "",
            "background": "",
            "skills": [],
            "relationships": {},
            "is_protagonist": False,
            "internal_conflict": "",
        },
    }

    with patch("core.langgraph.initialization.commit_init_node.validate_and_filter_traits") as mock_validate:
        mock_validate.side_effect = lambda traits: traits if traits else []

        profiles = await _parse_character_sheets_to_profiles(character_sheets)

        assert len(profiles) == 1
        assert mock_llm_service.async_call_llm.called


@pytest.mark.asyncio
async def test_extract_structured_character_data_success(mock_llm_service: MagicMock) -> None:
    """Verify successful extraction of structured character data."""
    result = await _extract_structured_character_data("Hero", "A brave knight who protects the realm")

    assert "traits" in result
    assert "status" in result
    assert "motivations" in result
    assert "background" in result
    assert result["status"] == "Active"


@pytest.mark.asyncio
async def test_extract_structured_character_data_exception(mock_llm_service: MagicMock) -> None:
    """LLM failures should propagate (init is hard-fail on contract violations)."""
    mock_llm_service.async_call_llm = AsyncMock(side_effect=Exception("LLM error"))

    with pytest.raises(Exception, match="LLM error"):
        await _extract_structured_character_data("Hero", "A brave knight")


def test_parse_character_extraction_response() -> None:
    """Verify parsing of character extraction response (strict JSON)."""
    response = '{"traits":["brave","loyal","determined","strong"],"status":"Active","motivations":"Protect the innocent and uphold justice.","background":"Trained as a knight from childhood."}'

    result = _parse_character_extraction_response(response)

    assert len(result["traits"]) == 4
    assert "brave" in result["traits"]
    assert result["status"] == "Active"
    assert "Protect the innocent" in result["motivations"]
    assert "Trained as a knight" in result["background"]


def test_parse_character_extraction_response_rejects_more_than_seven_traits() -> None:
    """Trait list must be 3-7 items (no silent truncation)."""
    response = '{"traits":["t1","t2","t3","t4","t5","t6","t7","t8"],"status":"Active","motivations":"Test","background":"Test"}'

    with pytest.raises(ValueError, match="3-7"):
        _parse_character_extraction_response(response)


def test_parse_character_extraction_response_empty() -> None:
    """Empty output is invalid JSON."""
    response = ""

    with pytest.raises(json.JSONDecodeError):
        _parse_character_extraction_response(response)


@pytest.mark.asyncio
async def test_extract_world_items_from_outline_success(mock_llm_service: MagicMock) -> None:
    """Verify successful extraction of world items."""
    mock_llm_service.async_call_llm = AsyncMock(
        return_value=(
            '[{"name":"Royal Castle","category":"location","description":"The seat of power in the kingdom."},'
            '{"name":"Sword of Light","category":"object","description":"A magical blade that defeats darkness."},'
            '{"name":"Dark Forest","category":"location","description":"A dangerous woodland filled with monsters."}]',
            {"prompt_tokens": 100, "completion_tokens": 50},
        )
    )

    global_outline = {"raw_text": "A story about a kingdom threatened by darkness."}

    result = await _extract_world_items_from_outline(global_outline, "Medieval fantasy kingdom")

    assert len(result) == 3
    assert any(item.name == "Royal Castle" for item in result)


@pytest.mark.asyncio
async def test_extract_world_items_from_outline_empty() -> None:
    """Verify handling when outline text is empty."""
    global_outline = {"raw_text": ""}

    result = await _extract_world_items_from_outline(global_outline, "Test setting")

    assert result == []


@pytest.mark.asyncio
async def test_extract_world_items_from_outline_exception(mock_llm_service: MagicMock) -> None:
    """LLM failures should propagate (init is hard-fail on contract violations)."""
    mock_llm_service.async_call_llm = AsyncMock(side_effect=Exception("LLM error"))

    global_outline = {"raw_text": "Test outline"}

    with pytest.raises(Exception, match="LLM error"):
        await _extract_world_items_from_outline(global_outline, "Test setting")


def test_parse_world_items_extraction() -> None:
    """Verify parsing of world items from LLM response (strict JSON)."""
    response = (
        '[{"name":"Royal Castle","category":"location","description":"The seat of power"},'
        '{"name":"Magic Sword","category":"object","description":"A legendary blade"},'
        '{"name":"Dark Forest","category":"location","description":"A dangerous place"}]'
    )

    with patch("utils.text_processing.generate_entity_id") as mock_id:
        mock_id.side_effect = lambda name, cat: f"{cat}_{name}"

        result = _parse_world_items_extraction(response)

        assert len(result) == 3
        assert result[0].name == "Royal Castle"
        assert result[0].category == "location"
        assert result[0].created_chapter == 0
        assert result[0].is_provisional is False


def test_parse_world_items_extraction_rejects_invalid_category() -> None:
    """Category must be constrained to allowed enum values."""
    response = '[{"name":"Valid Item","category":"invalid","description":"Description"}]'

    with pytest.raises(ValueError, match="category"):
        _parse_world_items_extraction(response)


def test_parse_world_items_extraction_empty() -> None:
    """Empty output is invalid JSON."""
    response = ""

    with pytest.raises(json.JSONDecodeError):
        _parse_world_items_extraction(response)


def test_parse_world_items_extraction_allows_empty_list() -> None:
    """An empty JSON list is valid and should produce no items."""
    response = "[]"

    result = _parse_world_items_extraction(response)

    assert result == []


@pytest.mark.asyncio
async def test_parse_character_sheets_multiple_characters() -> None:
    """Verify parsing multiple character sheets."""
    character_sheets = {
        f"Character{i}": {
            "description": f"Character {i} description",
            "traits": ["trait1", "trait2"],
            "status": "Active",
            "motivations": f"Motivation {i}",
            "background": f"Background {i}",
            "skills": [f"skill{i}"],
            "relationships": {},
            "is_protagonist": i == 0,
            "internal_conflict": "",
        }
        for i in range(5)
    }

    profiles = await _parse_character_sheets_to_profiles(character_sheets)

    assert len(profiles) == 5
    assert sum(1 for p in profiles if p.updates.get("is_protagonist")) == 1
