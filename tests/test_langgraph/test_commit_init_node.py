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
from core.langgraph.state import create_initial_state


@pytest.fixture
def base_state():
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
        generation_model="test-model",
        extraction_model="test-model",
        revision_model="test-model",
    )


@pytest.fixture
def mock_content_manager():
    """Create a mock ContentManager."""
    with patch("core.langgraph.initialization.commit_init_node.ContentManager") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_get_functions():
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
def mock_neo4j_manager():
    """Mock the Neo4j manager."""
    with patch("core.langgraph.initialization.commit_init_node.neo4j_manager") as mock:
        mock.execute_cypher_batch = AsyncMock()
        yield mock


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    with patch("core.langgraph.initialization.commit_init_node.llm_service") as mock:
        mock.async_call_llm = AsyncMock(
            return_value=(
                """TRAITS: brave, loyal, strong
STATUS: Active
MOTIVATIONS: Protect the innocent
BACKGROUND: Trained as a knight from childhood""",
                {"prompt_tokens": 100, "completion_tokens": 50},
            )
        )
        yield mock


@pytest.mark.asyncio
async def test_commit_initialization_to_graph_success(
    base_state,
    mock_content_manager,
    mock_get_functions,
    mock_neo4j_manager,
):
    """Verify successful commit of initialization data."""
    state = {**base_state}

    result = await commit_initialization_to_graph(state)

    assert result["initialization_step"] == "committed_to_graph"
    assert result["current_node"] == "commit_initialization"
    assert result["last_error"] is None
    assert "active_characters" in result
    assert len(result["active_characters"]) <= 5
    assert mock_neo4j_manager.execute_cypher_batch.called


@pytest.mark.asyncio
async def test_commit_initialization_no_characters(
    base_state,
    mock_content_manager,
    mock_get_functions,
    mock_neo4j_manager,
):
    """Verify handling when no character sheets exist."""
    mock_get_functions["chars"].return_value = {}

    state = {**base_state}

    result = await commit_initialization_to_graph(state)

    assert result["initialization_step"] == "committed_to_graph"
    assert result["current_node"] == "commit_initialization"


@pytest.mark.asyncio
async def test_commit_initialization_no_global_outline(
    base_state,
    mock_content_manager,
    mock_get_functions,
    mock_neo4j_manager,
):
    """Verify handling when no global outline exists."""
    mock_get_functions["global"].return_value = None

    state = {**base_state}

    result = await commit_initialization_to_graph(state)

    assert result["initialization_step"] == "committed_to_graph"
    assert result["current_node"] == "commit_initialization"


@pytest.mark.asyncio
async def test_commit_initialization_batch_execution_failure(
    base_state,
    mock_content_manager,
    mock_get_functions,
    mock_neo4j_manager,
):
    """Verify handling when batch execution fails."""
    mock_neo4j_manager.execute_cypher_batch = AsyncMock(side_effect=Exception("Batch execution failed"))

    state = {**base_state}

    result = await commit_initialization_to_graph(state)

    assert result["initialization_step"] == "commit_failed"
    assert "Failed to commit initialization data" in result["last_error"]


@pytest.mark.asyncio
async def test_commit_initialization_exception(
    base_state,
    mock_content_manager,
    mock_get_functions,
):
    """Verify exception handling during batch execution."""
    with patch("core.langgraph.initialization.commit_init_node.neo4j_manager") as mock_manager:
        mock_manager.execute_cypher_batch = AsyncMock(side_effect=Exception("Database error"))

        state = {**base_state}

        result = await commit_initialization_to_graph(state)

        assert result["initialization_step"] == "commit_failed"
        assert result["current_node"] == "commit_initialization"
        assert "Failed to commit initialization data" in result["last_error"]


@pytest.mark.asyncio
async def test_parse_character_sheets_to_profiles_with_traits():
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
    assert profiles[0].description == "A brave warrior"
    assert "brave" in profiles[0].traits
    assert profiles[0].status == "Active"
    assert profiles[0].created_chapter == 0
    assert profiles[0].is_provisional is False


@pytest.mark.asyncio
async def test_parse_character_sheets_to_profiles_filters_invalid_traits():
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
    mock_llm_service,
):
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
async def test_extract_structured_character_data_success(mock_llm_service):
    """Verify successful extraction of structured character data."""
    result = await _extract_structured_character_data("Hero", "A brave knight who protects the realm")

    assert "traits" in result
    assert "status" in result
    assert "motivations" in result
    assert "background" in result
    assert result["status"] == "Active"


@pytest.mark.asyncio
async def test_extract_structured_character_data_exception(mock_llm_service):
    """Verify fallback when extraction fails."""
    mock_llm_service.async_call_llm = AsyncMock(side_effect=Exception("LLM error"))

    result = await _extract_structured_character_data("Hero", "A brave knight")

    assert result["traits"] == []
    assert result["status"] == "Active"
    assert result["motivations"] == ""
    assert result["background"] == ""


def test_parse_character_extraction_response():
    """Verify parsing of character extraction response."""
    response = """TRAITS: brave, loyal, determined, strong
STATUS: Active
MOTIVATIONS: Protect the innocent and uphold justice
BACKGROUND: Trained as a knight from childhood"""

    result = _parse_character_extraction_response(response)

    assert len(result["traits"]) == 4
    assert "brave" in result["traits"]
    assert result["status"] == "Active"
    assert "Protect the innocent" in result["motivations"]
    assert "Trained as a knight" in result["background"]


def test_parse_character_extraction_response_limits_traits():
    """Verify trait count is limited to 7."""
    response = """TRAITS: trait1, trait2, trait3, trait4, trait5, trait6, trait7, trait8, trait9
STATUS: Active
MOTIVATIONS: Test
BACKGROUND: Test"""

    result = _parse_character_extraction_response(response)

    assert len(result["traits"]) == 7


def test_parse_character_extraction_response_empty():
    """Verify handling of empty response."""
    response = ""

    result = _parse_character_extraction_response(response)

    assert result["traits"] == []
    assert result["status"] == "Active"
    assert result["motivations"] == ""
    assert result["background"] == ""


@pytest.mark.asyncio
async def test_extract_world_items_from_outline_success(mock_llm_service):
    """Verify successful extraction of world items."""
    mock_llm_service.async_call_llm = AsyncMock(
        return_value=(
            """[Location] Royal Castle: The seat of power in the kingdom
[Object] Sword of Light: A magical blade that defeats darkness
[Location] Dark Forest: A dangerous woodland filled with monsters""",
            {"prompt_tokens": 100, "completion_tokens": 50},
        )
    )

    global_outline = {"raw_text": "A story about a kingdom threatened by darkness."}

    result = await _extract_world_items_from_outline(global_outline, "Medieval fantasy kingdom")

    assert len(result) >= 1
    assert any(item.name == "Royal Castle" for item in result)


@pytest.mark.asyncio
async def test_extract_world_items_from_outline_empty():
    """Verify handling when outline text is empty."""
    global_outline = {"raw_text": ""}

    result = await _extract_world_items_from_outline(global_outline, "Test setting")

    assert result == []


@pytest.mark.asyncio
async def test_extract_world_items_from_outline_exception(mock_llm_service):
    """Verify exception handling during extraction."""
    mock_llm_service.async_call_llm = AsyncMock(side_effect=Exception("LLM error"))

    global_outline = {"raw_text": "Test outline"}

    result = await _extract_world_items_from_outline(global_outline, "Test setting")

    assert result == []


def test_parse_world_items_extraction():
    """Verify parsing of world items from LLM response."""
    response = """[Location] Royal Castle: The seat of power
[Object] Magic Sword: A legendary blade
[Location] Dark Forest: A dangerous place"""

    with patch("processing.entity_deduplication.generate_entity_id") as mock_id:
        mock_id.side_effect = lambda name, cat, chapter: f"{cat}_{name}_{chapter}"

        result = _parse_world_items_extraction(response)

        assert len(result) == 3
        assert result[0].name == "Royal Castle"
        assert result[0].category == "location"
        assert result[0].created_chapter == 0
        assert result[0].is_provisional is False


def test_parse_world_items_extraction_invalid_lines():
    """Verify handling of invalid lines in response."""
    response = """[Location] Valid Item: Description
Invalid line without brackets
[Invalid] Missing colon
[Object] No description here
[Location] Valid Item 2: Another description"""

    with patch("processing.entity_deduplication.generate_entity_id") as mock_id:
        mock_id.side_effect = lambda name, cat, chapter: f"{cat}_{name}_{chapter}"

        result = _parse_world_items_extraction(response)

        assert len(result) >= 1
        assert all(item.name and item.description for item in result)


def test_parse_world_items_extraction_empty():
    """Verify handling of empty response."""
    response = ""

    result = _parse_world_items_extraction(response)

    assert result == []


def test_parse_world_items_extraction_whitespace_lines():
    """Verify handling of response with whitespace."""
    response = """

[Location] Castle: A fortress


[Object] Sword: A weapon

"""

    with patch("processing.entity_deduplication.generate_entity_id") as mock_id:
        mock_id.side_effect = lambda name, cat, chapter: f"{cat}_{name}_{chapter}"

        result = _parse_world_items_extraction(response)

        assert len(result) == 2


@pytest.mark.asyncio
async def test_parse_character_sheets_multiple_characters():
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


@pytest.mark.asyncio
async def test_commit_initialization_limits_active_characters(
    base_state,
    mock_content_manager,
    mock_get_functions,
    mock_knowledge_graph_service,
):
    """Verify active_characters is limited to top 5."""
    character_sheets = {
        f"Character{i}": {
            "description": f"Description {i}",
            "traits": ["trait"],
            "status": "Active",
            "motivations": "",
            "background": "",
            "skills": [],
            "relationships": {},
            "is_protagonist": False,
            "internal_conflict": "",
        }
        for i in range(10)
    }

    mock_get_functions["chars"].return_value = character_sheets

    state = {**base_state}

    result = await commit_initialization_to_graph(state)

    assert len(result["active_characters"]) <= 5
