import pytest
from unittest.mock import AsyncMock, patch
from agents.revision_agent import RevisionAgent


@pytest.mark.asyncio
async def test_revision_agent_initialization():
    """Test that RevisionAgent initializes correctly."""
    agent = RevisionAgent()
    assert agent.model_name == "Qwen3-14B-Q4"  # Default from config
    assert agent.threshold == 0.85  # REVISION_EVALUATION_THRESHOLD


@pytest.mark.asyncio
async def test_validate_revision_no_issues():
    """Test validate_revision when no issues are found."""
    agent = RevisionAgent()
    
    # Mock world state
    world_state = {
        "plot_outline": {
            "title": "Test Novel",
            "protagonist_name": "Test Protagonist",
            "genre": "Fantasy",
            "theme": "Adventure",
            "character_arc": "Hero's Journey",
            "logline": "A test story",
            "plot_points": ["Point 1", "Point 2"]
        },
        "chapter_number": 1,
        "previous_chapters_context": ""
    }
    
    # Mock the internal methods to return no problems
    with patch.object(agent, '_check_continuity', return_value=[]), \
         patch.object(agent, '_evaluate_quality', return_value=(False, [])):
        
        is_valid, issues = await agent.validate_revision(
            "This is a test chapter text.",
            "This is the previous chapter text.",
            world_state
        )
        
        assert is_valid is True
        assert len(issues) == 1
        assert "Revision validation passed" in issues[0]


@pytest.mark.asyncio
async def test_validate_revision_with_issues():
    """Test validate_revision when issues are found."""
    agent = RevisionAgent()
    
    # Mock world state
    world_state = {
        "plot_outline": {
            "title": "Test Novel",
            "protagonist_name": "Test Protagonist",
            "genre": "Fantasy",
            "theme": "Adventure",
            "character_arc": "Hero's Journey",
            "logline": "A test story",
            "plot_points": ["Point 1", "Point 2"]
        },
        "chapter_number": 2,
        "previous_chapters_context": "Previous chapter context"
    }
    
    # Mock continuity problems
    continuity_problems = [
        {
            "issue_category": "consistency",
            "problem_description": "Character name inconsistency",
            "quote_from_original_text": "John said",
            "quote_char_start": 0,
            "quote_char_end": 9,
            "sentence_char_start": 0,
            "sentence_char_end": 15,
            "suggested_fix_focus": "Fix character name"
        }
    ]
    
    # Mock quality issues
    quality_issues = ["Chapter is too short", "Plot needs improvement"]
    
    with patch.object(agent, '_check_continuity', return_value=continuity_problems), \
         patch.object(agent, '_evaluate_quality', return_value=(True, quality_issues)):
        
        is_valid, issues = await agent.validate_revision(
            "This is a test chapter text.",
            "This is the previous chapter text.",
            world_state
        )
        
        assert is_valid is False
        assert len(issues) >= 3  # At least continuity + quality issues


@pytest.mark.asyncio
async def test_check_continuity_empty_text():
    """Test _check_continuity with empty chapter text."""
    agent = RevisionAgent()
    
    world_state = {
        "plot_outline": {"title": "Test"},
        "chapter_number": 1
    }
    
    problems = await agent._check_continuity("", world_state)
    assert len(problems) == 0


@pytest.mark.asyncio
async def test_evaluate_quality_empty_text():
    """Test _evaluate_quality with empty chapter text."""
    agent = RevisionAgent()
    
    world_state = {
        "plot_outline": {"title": "Test"},
        "chapter_number": 1
    }
    
    has_issues, issues = await agent._evaluate_quality("", world_state)
    assert has_issues is True
    assert "Draft is empty" in issues


@pytest.mark.asyncio
async def test_validate_patch_no_problems():
    """Test _validate_patch with no problems."""
    agent = RevisionAgent()
    
    problems = []
    result = await agent._validate_patch("Test text", problems)
    assert result is True


@pytest.mark.asyncio
async def test_validate_patch_with_problems():
    """Test _validate_patch with problems (placeholder implementation)."""
    agent = RevisionAgent()
    
    problems = [
        {
            "issue_category": "consistency",
            "problem_description": "Test problem",
            "quote_from_original_text": "Test quote",
            "suggested_fix_focus": "Test fix"
        }
    ]
    
    result = await agent._validate_patch("Test text", problems)
    # Currently returns True as placeholder
    assert result is True


@pytest.mark.asyncio
async def test_revision_agent_with_custom_model():
    """Test RevisionAgent initialization with custom model."""
    custom_model = "custom-model"
    agent = RevisionAgent(model_name=custom_model)
    assert agent.model_name == custom_model