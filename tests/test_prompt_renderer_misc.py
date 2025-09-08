# tests/test_prompt_renderer_misc.py
from jinja2 import DictLoader, Environment

import prompts.prompt_renderer


def test_render_prompt_with_custom_env(monkeypatch):
    env = Environment(
        loader=DictLoader({"greet.j2": "Hello {{ name }}"}), autoescape=False
    )
    monkeypatch.setattr(prompts.prompt_renderer, "_env", env)
    result = prompts.prompt_renderer.render_prompt("greet.j2", {"name": "Bob"})
    assert result == "Hello Bob"

import pytest
from unittest.mock import AsyncMock, patch
import asyncio

from prompts.prompt_data_getters import (
    clear_context_cache,
    _cached_character_info,
    _cached_world_elements,
    get_filtered_character_profiles_for_prompt_plain_text,
    get_filtered_world_data_for_prompt_plain_text,
)

from models import WorldItem

@pytest.mark.asyncio
async def test_cached_character_info_uses_cache():
    mock_result = {"name": "Test Char"}
    mock_get = AsyncMock(return_value=mock_result)
    
    with patch("prompts.prompt_data_getters.character_queries.get_character_info_for_snippet_from_db", mock_get):
        # First call - populates cache
        result1 = await _cached_character_info("test_char", 1)
        assert result1 == mock_result
        assert mock_get.call_count == 1
        
        # Second call - uses cache
        result2 = await _cached_character_info("test_char", 1)
        assert result2 == mock_result
        assert mock_get.call_count == 1

@pytest.mark.asyncio
async def test_clear_context_cache_invalidates_cache():
    mock_result = {"name": "Test Char"}
    mock_get = AsyncMock(return_value=mock_result)
    
    with patch("prompts.prompt_data_getters.character_queries.get_character_info_for_snippet_from_db", mock_get):
        # First call
        result1 = await _cached_character_info("test_char", 1)
        assert mock_get.call_count == 1
        
        # Clear cache
        clear_context_cache()
        
        # Second call after clear
        result2 = await _cached_character_info("test_char", 1)
        assert mock_get.call_count == 2
        assert result2 == mock_result

@pytest.mark.asyncio
async def test_cached_world_elements_uses_cache():
    mock_result = [WorldItem(name="Test World")]
    mock_get = AsyncMock(return_value=mock_result)
    
    with patch("prompts.prompt_data_getters.world_queries.get_all_world_items", mock_get):
        # First call
        result1 = await _cached_world_elements()
        assert result1 == mock_result
        assert mock_get.call_count == 1
        
        # Second call
        result2 = await _cached_world_elements()
        assert result2 == mock_result
        assert mock_get.call_count == 1

@pytest.mark.asyncio
async def test_clear_context_cache_invalidates_world_cache():
    mock_result = [WorldItem(name="Test World")]
    mock_get = AsyncMock(return_value=mock_result)
    
    with patch("prompts.prompt_data_getters.world_queries.get_all_world_items", mock_get):
        # First call
        result1 = await _cached_world_elements()
        assert mock_get.call_count == 1
        
        # Clear cache
        clear_context_cache()
        
        # Second call after clear
        result2 = await _cached_world_elements()
        assert mock_get.call_count == 2
        assert result2 == mock_result

@pytest.mark.asyncio
async def test_get_filtered_character_profiles_uses_cache():
    mock_profiles = {"test_char": {"data": "test"}}
    mock_get = AsyncMock(return_value=mock_profiles)
    
    with patch("prompts.prompt_data_getters._get_character_profiles_dict_with_notes", mock_get):
        char_names = ["test_char"]
        
        # First call
        result1 = await get_filtered_character_profiles_for_prompt_plain_text(char_names, 1)
        assert mock_get.call_count == 1
        
        # Second call
        result2 = await get_filtered_character_profiles_for_prompt_plain_text(char_names, 1)
        assert mock_get.call_count == 1

@pytest.mark.asyncio
async def test_clear_cache_affects_character_profiles():
    mock_profiles = {"test_char": {"data": "test"}}
    mock_get = AsyncMock(return_value=mock_profiles)
    
    with patch("prompts.prompt_data_getters._get_character_profiles_dict_with_notes", mock_get):
        char_names = ["test_char"]
        
        # First call
        await get_filtered_character_profiles_for_prompt_plain_text(char_names, 1)
        assert mock_get.call_count == 1
        
        # Clear cache
        clear_context_cache()
        
        # Second call
        await get_filtered_character_profiles_for_prompt_plain_text(char_names, 1)
        assert mock_get.call_count == 2

@pytest.mark.asyncio
async def test_get_filtered_world_data_uses_cache():
    mock_data = {"category": ["id"]}
    mock_get = AsyncMock(return_value=mock_data)
    
    with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes", mock_get):
        world_ids = {"category": ["id"]}
        
        # First call
        result1 = await get_filtered_world_data_for_prompt_plain_text(world_ids, 1)
        assert mock_get.call_count == 1
        
        # Second call
        result2 = await get_filtered_world_data_for_prompt_plain_text(world_ids, 1)
        assert mock_get.call_count == 1

@pytest.mark.asyncio
async def test_clear_cache_affects_world_data():
    mock_data = {"category": ["id"]}
    mock_get = AsyncMock(return_value=mock_data)
    
    with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes", mock_get):
        world_ids = {"category": ["id"]}
        
        # First call
        await get_filtered_world_data_for_prompt_plain_text(world_ids, 1)
        assert mock_get.call_count == 1
        
        # Clear cache
        clear_context_cache()
        
        # Second call
        await get_filtered_world_data_for_prompt_plain_text(world_ids, 1)
        assert mock_get.call_count == 2
