# tests/test_prompt_data_getters.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from models import CharacterProfile, WorldItem
from prompts import prompt_data_getters


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the context cache before each test."""
    prompt_data_getters.clear_context_cache()
    yield
    prompt_data_getters.clear_context_cache()


class TestCacheManagement:
    """Cache management functions."""

    def test_clears_cache(self):
        prompt_data_getters._context_cache["test_key"] = "test_value"
        prompt_data_getters._current_cache_chapter = 5

        prompt_data_getters.clear_context_cache()

        assert len(prompt_data_getters._context_cache) == 0
        assert prompt_data_getters._current_cache_chapter is None

    def test_ensures_cache_scoped_to_chapter_with_none(self):
        prompt_data_getters._current_cache_chapter = 5
        prompt_data_getters._context_cache["test"] = "value"

        prompt_data_getters._ensure_cache_is_scoped_to_chapter(None)

        assert prompt_data_getters._current_cache_chapter == 5
        assert "test" in prompt_data_getters._context_cache

    def test_ensures_cache_scoped_to_chapter_same_chapter(self):
        prompt_data_getters._current_cache_chapter = 5
        prompt_data_getters._context_cache["test"] = "value"

        prompt_data_getters._ensure_cache_is_scoped_to_chapter(5)

        assert prompt_data_getters._current_cache_chapter == 5
        assert "test" in prompt_data_getters._context_cache

    def test_ensures_cache_scoped_to_chapter_different_chapter(self):
        prompt_data_getters._current_cache_chapter = 5
        prompt_data_getters._context_cache["test"] = "value"

        prompt_data_getters._ensure_cache_is_scoped_to_chapter(6)

        assert prompt_data_getters._current_cache_chapter == 6
        assert len(prompt_data_getters._context_cache) == 0

    def test_ensures_cache_scoped_to_chapter_first_call(self):
        prompt_data_getters._current_cache_chapter = None

        prompt_data_getters._ensure_cache_is_scoped_to_chapter(1)

        assert prompt_data_getters._current_cache_chapter == 1


class TestCharacterOrdering:
    """Character ordering function."""

    def test_orders_protagonist_first(self):
        characters = {"Alice", "Bob", "Charlie"}
        result = prompt_data_getters._deterministic_character_order(characters, "Alice")

        assert result[0] == "Alice"

    def test_orders_alphabetically_after_protagonist(self):
        characters = {"Alice", "Charlie", "Bob"}
        result = prompt_data_getters._deterministic_character_order(characters, "Alice")

        assert result == ["Alice", "Bob", "Charlie"]

    def test_handles_no_protagonist(self):
        characters = {"Charlie", "Alice", "Bob"}
        result = prompt_data_getters._deterministic_character_order(characters, None)

        assert result == ["Alice", "Bob", "Charlie"]

    def test_handles_missing_protagonist(self):
        characters = {"Alice", "Bob", "Charlie"}
        result = prompt_data_getters._deterministic_character_order(characters, "David")

        assert result == ["Alice", "Bob", "Charlie"]

    def test_handles_empty_set(self):
        result = prompt_data_getters._deterministic_character_order(set(), "Alice")
        assert result == []

    def test_normalizes_names_for_comparison(self):
        characters = {"Alice", "ALICE", "alice"}
        result = prompt_data_getters._deterministic_character_order(characters, "alice")

        assert len(result) == 3

    def test_handles_case_insensitive_protagonist(self):
        characters = {"alice", "Bob", "Charlie"}
        result = prompt_data_getters._deterministic_character_order(characters, "ALICE")

        assert result[0] == "alice"


class TestFormattingFunction:
    """Dictionary formatting function."""

    def test_formats_simple_dict(self):
        data = {"description": "A test description", "name": "Test"}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        assert any("Description: A test description" in line for line in result)

    def test_formats_nested_dict(self):
        data = {"info": {"description": "Nested", "value": 42}}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "Info:" in lines_text
        assert "Description: Nested" in lines_text

    def test_formats_list_values(self):
        data = {"traits": ["brave", "loyal", "honest"]}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "Traits:" in lines_text
        assert "- brave" in lines_text

    def test_formats_empty_list(self):
        data = {"traits": []}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "(empty list or N/A)" in lines_text

    def test_skips_source_quality_keys(self):
        data = {
            "description": "Test",
            "source_quality_chapter_1": "provisional",
            "updated_in_chapter_2": "data",
        }

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "source_quality" not in lines_text.lower()
        assert "updated_in_chapter" not in lines_text.lower()

    def test_skips_provisional_hint(self):
        data = {"description": "Test", "is_provisional_hint": True}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "provisional_hint" not in lines_text.lower()

    def test_handles_bool_values(self):
        data = {"active": True, "hidden": False}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "Active: True" in lines_text
        assert "Hidden: False" in lines_text

    def test_handles_none_values(self):
        data = {"name": "Test", "value": None}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "value" not in lines_text.lower()

    def test_handles_empty_strings(self):
        data = {"name": "Test", "value": ""}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "Value:" not in lines_text

    def test_formats_list_with_dicts(self):
        data = {"items": ["item1", {"name": "item2", "value": 42}]}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        assert "Items:" in lines_text
        assert "- item1" in lines_text
        assert "- Item:" in lines_text

    def test_uses_name_override(self):
        data = {"description": "Test"}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data, name_override="Custom Name")

        assert result[0] == "Custom Name:"

    def test_prioritizes_keys(self):
        data = {
            "zebra": "last",
            "description": "first",
            "apple": "middle",
            "traits": ["test"],
        }

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        lines_text = "\n".join(result)
        desc_index = lines_text.find("Description:")
        traits_index = lines_text.find("Traits:")
        apple_index = lines_text.find("Apple:")

        assert desc_index < traits_index
        assert traits_index < apple_index

    def test_handles_unsortable_lists(self):
        data = {"items": [{"a": 1}, None, "string", 42]}

        result = prompt_data_getters._format_dict_for_plain_text_prompt(data)

        assert len(result) > 0


class TestProvisionalNotesAndFiltering:
    """Provisional notes and development filtering."""

    def test_filters_developments_after_chapter(self):
        item_data = {
            "name": "Test",
            "development_in_chapter_5": "later development",
            "development_in_chapter_2": "early development",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=3, is_character=True)

        assert "development_in_chapter_2" in result
        assert "development_in_chapter_5" not in result

    def test_filters_elaborations_for_world_items(self):
        item_data = {
            "name": "Test",
            "elaboration_in_chapter_5": "later",
            "elaboration_in_chapter_2": "early",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=3, is_character=False)

        assert "elaboration_in_chapter_2" in result
        assert "elaboration_in_chapter_5" not in result

    def test_adds_provisional_notes_when_quality_is_provisional(self):
        item_data = {
            "name": "Test",
            "source_quality_chapter_1": "provisional_from_unrevised_draft",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=2, is_character=True)

        assert "prompt_notes" in result
        assert any("provisional" in note.lower() for note in result["prompt_notes"])
        assert result.get("is_provisional_hint") is True

    def test_skips_provisional_notes_after_chapter_limit(self):
        item_data = {
            "name": "Test",
            "source_quality_chapter_10": "provisional_from_unrevised_draft",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=5, is_character=True)

        assert "prompt_notes" not in result
        assert result.get("is_provisional_hint") is not True

    def test_handles_added_in_chapter_filtering(self):
        item_data = {"name": "Test", "added_in_chapter_10": "new data"}

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=5, is_character=True)

        assert "added_in_chapter_10" not in result

    def test_preserves_non_chapter_keys(self):
        item_data = {
            "name": "Test",
            "description": "A test item",
            "development_in_chapter_10": "later",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=5, is_character=True)

        assert result["name"] == "Test"
        assert result["description"] == "A test item"

    def test_handles_prepopulation_chapter(self):
        with patch.object(config, "KG_PREPOPULATION_CHAPTER_NUM", 0):
            item_data = {
                "name": "Test",
                "development_in_chapter_1": "chapter 1",
            }

            result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=0, is_character=True)

            assert "development_in_chapter_1" not in result

    def test_handles_malformed_chapter_keys(self):
        item_data = {
            "name": "Test",
            "development_in_chapter_invalid": "bad key",
            "development_in_chapter_": "no number",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=5, is_character=True)

        assert result["name"] == "Test"

    def test_handles_none_chapter_limit(self):
        item_data = {
            "name": "Test",
            "development_in_chapter_5": "data",
            "source_quality_chapter_3": "provisional_from_unrevised_draft",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=None, is_character=True)

        assert "development_in_chapter_5" in result
        assert "prompt_notes" in result

    def test_deduplicates_provisional_notes(self):
        item_data = {
            "name": "Test",
            "source_quality_chapter_1": "provisional_from_unrevised_draft",
            "source_quality_chapter_2": "provisional_from_unrevised_draft",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=5, is_character=True)

        assert len(result["prompt_notes"]) == 2

    def test_sorts_provisional_notes(self):
        item_data = {
            "name": "Test",
            "source_quality_chapter_5": "provisional_from_unrevised_draft",
            "source_quality_chapter_1": "provisional_from_unrevised_draft",
        }

        result = prompt_data_getters._add_provisional_notes_and_filter_developments(item_data, up_to_chapter_inclusive=10, is_character=True)

        assert result["prompt_notes"] == sorted(result["prompt_notes"])


class TestCachedCharacterInfo:
    """Cached character info function."""

    @pytest.mark.asyncio
    async def test_caches_character_queries(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db") as mock_query:
            mock_query.return_value = {"name": "Alice", "summary": "Test"}

            result1 = await prompt_data_getters._cached_character_info("Alice", 5)
            result2 = await prompt_data_getters._cached_character_info("Alice", 5)

            assert result1 == result2
            mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_differentiates_by_chapter_limit(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db") as mock_query:
            mock_query.return_value = {"name": "Alice", "summary": "Test"}

            await prompt_data_getters._cached_character_info("Alice", 5)
            await prompt_data_getters._cached_character_info("Alice", 6)

            assert mock_query.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_none_chapter_limit(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db") as mock_query:
            mock_query.return_value = {"name": "Alice", "summary": "Test"}

            result = await prompt_data_getters._cached_character_info("Alice", None)

            assert result is not None
            mock_query.assert_called_once_with("Alice", 1000)


class TestCachedWorldItem:
    """Cached world item function."""

    @pytest.mark.asyncio
    async def test_caches_world_item_queries(self):
        mock_item = WorldItem(
            id="item_001",
            category="location",
            name="Castle",
            description="A grand castle",
            created_chapter=1,
        )

        with patch("data_access.world_queries.get_world_item_by_id") as mock_query:
            mock_query.return_value = mock_item

            result1 = await prompt_data_getters._cached_world_item_by_id("item_001")
            result2 = await prompt_data_getters._cached_world_item_by_id("item_001")

            assert result1 == result2
            mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_differentiates_by_item_id(self):
        with patch("data_access.world_queries.get_world_item_by_id") as mock_query:
            mock_query.return_value = WorldItem(id="test", category="test", name="Test", created_chapter=1)

            await prompt_data_getters._cached_world_item_by_id("item_001")
            await prompt_data_getters._cached_world_item_by_id("item_002")

            assert mock_query.call_count == 2


class TestGetCharacterProfilesDictWithNotes:
    """Character profiles dict with notes function."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_list(self):
        result = await prompt_data_getters._get_character_profiles_dict_with_notes([], up_to_chapter_inclusive=5)

        assert result == {}

    @pytest.mark.asyncio
    async def test_fetches_and_processes_character_profiles(self):
        mock_profile = CharacterProfile(
            name="Alice",
            personality_description="Test character",
            traits=["brave"],
            status="alive",
            created_chapter=1,
        )

        with patch("data_access.character_queries.get_character_profile_by_name") as mock_get:
            mock_get.return_value = mock_profile

            result = await prompt_data_getters._get_character_profiles_dict_with_notes(["Alice"], up_to_chapter_inclusive=5)

            assert "Alice" in result
            assert result["Alice"]["personality_description"] == "Test character"

    @pytest.mark.asyncio
    async def test_handles_missing_character(self):
        with patch("data_access.character_queries.get_character_profile_by_name") as mock_get:
            mock_get.return_value = None

            result = await prompt_data_getters._get_character_profiles_dict_with_notes(["Alice"], up_to_chapter_inclusive=5)

            assert result == {}

    @pytest.mark.asyncio
    async def test_handles_query_exception(self):
        with patch("data_access.character_queries.get_character_profile_by_name") as mock_get:
            mock_get.side_effect = Exception("Database error")

            result = await prompt_data_getters._get_character_profiles_dict_with_notes(["Alice"], up_to_chapter_inclusive=5)

            assert result == {}

    @pytest.mark.asyncio
    async def test_processes_multiple_characters(self):
        mock_alice = CharacterProfile(name="Alice", personality_description="Alice desc", created_chapter=1)
        mock_bob = CharacterProfile(name="Bob", personality_description="Bob desc", created_chapter=1)

        with patch("data_access.character_queries.get_character_profile_by_name") as mock_get:
            mock_get.side_effect = [mock_alice, mock_bob]

            result = await prompt_data_getters._get_character_profiles_dict_with_notes(["Alice", "Bob"], up_to_chapter_inclusive=5)

            assert "Alice" in result
            assert "Bob" in result


class TestGetFilteredCharacterProfilesPlainText:
    """Main character profiles plain text function."""

    @pytest.mark.asyncio
    async def test_returns_message_when_no_profiles(self):
        with patch("prompts.prompt_data_getters._get_character_profiles_dict_with_notes") as mock_get:
            mock_get.return_value = {}

            result = await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text([])

            assert result == "No character profiles available."

    @pytest.mark.asyncio
    async def test_formats_character_profiles_as_plain_text(self):
        mock_profiles = {"Alice": {"description": "A brave warrior", "traits": ["brave", "loyal"]}}

        with patch("prompts.prompt_data_getters._get_character_profiles_dict_with_notes") as mock_get:
            mock_get.return_value = mock_profiles

            result = await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text(["Alice"])

            assert "Key Character Profiles:" in result
            assert "Alice" in result
            assert "brave warrior" in result

    @pytest.mark.asyncio
    async def test_sorts_characters_alphabetically(self):
        mock_profiles = {
            "Charlie": {"description": "Charlie desc"},
            "Alice": {"description": "Alice desc"},
            "Bob": {"description": "Bob desc"},
        }

        with patch("prompts.prompt_data_getters._get_character_profiles_dict_with_notes") as mock_get:
            mock_get.return_value = mock_profiles

            result = await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text(["Charlie", "Alice", "Bob"])

            alice_index = result.find("Alice")
            bob_index = result.find("Bob")
            charlie_index = result.find("Charlie")

            assert alice_index < bob_index < charlie_index

    @pytest.mark.asyncio
    async def uses_prepopulation_chapter_for_chapter_0(self):
        with patch.object(config, "KG_PREPOPULATION_CHAPTER_NUM", 0):
            with patch("prompts.prompt_data_getters._get_character_profiles_dict_with_notes") as mock_get:
                mock_get.return_value = {}

                await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text(["Alice"], up_to_chapter_inclusive=0)

                mock_get.assert_called_once()
                assert mock_get.call_args[0][1] == 0

    @pytest.mark.asyncio
    async def test_skips_empty_profiles(self):
        mock_profiles = {
            "Alice": {"description": "Valid"},
            "Bob": {},
            "Charlie": None,
        }

        with patch("prompts.prompt_data_getters._get_character_profiles_dict_with_notes") as mock_get:
            mock_get.return_value = mock_profiles

            result = await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text(["Alice", "Bob", "Charlie"])

            assert "Alice" in result
            assert "Bob" not in result or result.count("Bob") == 0


class TestGetWorldDataDictWithNotes:
    """World data dict with notes function."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_empty_dict(self):
        result = await prompt_data_getters._get_world_data_dict_with_notes({}, up_to_chapter_inclusive=5)

        assert result == {}

    @pytest.mark.asyncio
    async def test_fetches_and_processes_world_items(self):
        mock_item = WorldItem(
            id="item_001",
            category="location",
            name="Castle",
            description="A grand castle",
            created_chapter=1,
        )

        with patch("prompts.prompt_data_getters._cached_world_item_by_id") as mock_get:
            mock_get.return_value = mock_item

            result = await prompt_data_getters._get_world_data_dict_with_notes({"locations": ["item_001"]}, up_to_chapter_inclusive=5)

            assert "locations" in result
            assert "Castle" in result["locations"]

    @pytest.mark.asyncio
    async def test_handles_missing_world_item(self):
        with patch("prompts.prompt_data_getters._cached_world_item_by_id") as mock_get:
            mock_get.return_value = None

            result = await prompt_data_getters._get_world_data_dict_with_notes({"locations": ["item_001"]}, up_to_chapter_inclusive=5)

            assert result["locations"] == {}

    @pytest.mark.asyncio
    async def test_handles_query_exception(self):
        with patch("prompts.prompt_data_getters._cached_world_item_by_id") as mock_get:
            mock_get.side_effect = Exception("Database error")

            result = await prompt_data_getters._get_world_data_dict_with_notes({"locations": ["item_001"]}, up_to_chapter_inclusive=5)

            assert result["locations"] == {}

    @pytest.mark.asyncio
    async def test_processes_multiple_categories(self):
        mock_location = WorldItem(
            id="loc_001",
            category="location",
            name="Castle",
            description="A castle",
            created_chapter=1,
        )
        mock_artifact = WorldItem(
            id="art_001",
            category="artifact",
            name="Sword",
            description="A sword",
            created_chapter=1,
        )

        with patch("prompts.prompt_data_getters._cached_world_item_by_id") as mock_get:
            mock_get.side_effect = [mock_location, mock_artifact]

            result = await prompt_data_getters._get_world_data_dict_with_notes(
                {"locations": ["loc_001"], "artifacts": ["art_001"]},
                up_to_chapter_inclusive=5,
            )

            assert "locations" in result
            assert "artifacts" in result
            assert "Castle" in result["locations"]
            assert "Sword" in result["artifacts"]


class TestGetFilteredWorldDataPlainText:
    """Main world data plain text function."""

    @pytest.mark.asyncio
    async def test_returns_message_when_no_data(self):
        with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes") as mock_get:
            mock_get.return_value = {}

            result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text({})

            assert result == "No world-building data available."

    @pytest.mark.asyncio
    async def test_formats_overview_section(self):
        mock_data = {
            "_overview_": {"description": "A fantasy world with magic"},
            "locations": {"Castle": {"description": "A grand castle"}},
        }

        with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes") as mock_get:
            mock_get.return_value = mock_data

            result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text({"_overview_": ["item"], "locations": ["loc"]})

            assert "World-Building Overview:" in result
            assert "fantasy world" in result

    @pytest.mark.asyncio
    async def test_skips_overview_if_no_description(self):
        mock_data = {
            "_overview_": {},
            "locations": {"Castle": {"description": "A castle"}},
        }

        with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes") as mock_get:
            mock_get.return_value = mock_data

            result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text({"_overview_": ["item"], "locations": ["loc"]})

            assert "World-Building Overview:" not in result

    @pytest.mark.asyncio
    async def test_excludes_special_categories(self):
        mock_data = {
            "is_default": {"item": {}},
            "source": {"item": {}},
            "user_supplied_data": {"item": {}},
            "locations": {"Castle": {"description": "A castle"}},
        }

        with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes") as mock_get:
            mock_get.return_value = mock_data

            result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text(
                {
                    "is_default": ["x"],
                    "source": ["x"],
                    "user_supplied_data": ["x"],
                    "locations": ["loc"],
                }
            )

            assert "is_default" not in result.lower()
            assert "source" not in result or "source" in result.lower()
            assert "user_supplied" not in result.lower()
            assert "Locations:" in result

    @pytest.mark.asyncio
    async def test_sorts_categories_alphabetically(self):
        mock_data = {
            "zebra": {"Item1": {"description": "Test"}},
            "alpha": {"Item2": {"description": "Test"}},
            "beta": {"Item3": {"description": "Test"}},
        }

        with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes") as mock_get:
            mock_get.return_value = mock_data

            result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text({"zebra": ["z"], "alpha": ["a"], "beta": ["b"]})

            alpha_index = result.find("Alpha:")
            beta_index = result.find("Beta:")
            zebra_index = result.find("Zebra:")

            assert alpha_index < beta_index < zebra_index

    @pytest.mark.asyncio
    async def test_returns_fallback_message_when_all_filtered(self):
        mock_data = {"_overview_": {}}

        with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes") as mock_get:
            mock_get.return_value = mock_data

            result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text({"_overview_": ["item"]})

            assert "No significant world-building data available after filtering" in result

    @pytest.mark.asyncio
    async def test_strips_trailing_empty_lines(self):
        mock_data = {"locations": {"Castle": {"description": "A castle"}}}

        with patch("prompts.prompt_data_getters._get_world_data_dict_with_notes") as mock_get:
            mock_get.return_value = mock_data

            result = await prompt_data_getters.get_filtered_world_data_for_prompt_plain_text({"locations": ["loc"]})

            assert not result.endswith("\n\n")


class TestDiscoverCharactersOfInterest:
    """Character discovery function."""

    @pytest.mark.asyncio
    async def test_includes_protagonist(self):
        result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="Alice", chapter_plan=None, chapter_number=1)

        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_excludes_fill_in_protagonist(self):
        with patch("utils._is_fill_in") as mock_fill:
            mock_fill.return_value = True

            result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="TBD", chapter_plan=None, chapter_number=1)

            assert "TBD" not in result

    @pytest.mark.asyncio
    async def test_extracts_characters_from_chapter_plan(self):
        chapter_plan = [
            {"characters_involved": ["Alice", "Bob"]},
            {"characters_involved": ["Charlie"]},
        ]

        result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="Alice", chapter_plan=chapter_plan, chapter_number=1)

        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" in result

    @pytest.mark.asyncio
    async def test_handles_empty_chapter_plan(self):
        result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="Alice", chapter_plan=[], chapter_number=1)

        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_handles_none_chapter_plan(self):
        result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="Alice", chapter_plan=None, chapter_number=1)

        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_excludes_fill_in_characters_from_plan(self):
        with patch("utils._is_fill_in") as mock_fill:
            mock_fill.side_effect = lambda x: x == "TBD"

            chapter_plan = [{"characters_involved": ["Alice", "TBD"]}]

            result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="Hero", chapter_plan=chapter_plan, chapter_number=1)

            assert "Alice" in result
            assert "TBD" not in result

    @pytest.mark.asyncio
    async def test_handles_malformed_scene_details(self):
        chapter_plan = [
            "not a dict",
            {"no_characters_key": ["test"]},
            {"characters_involved": "not a list"},
            {"characters_involved": [None, "", "  ", "Alice"]},
        ]

        result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="Hero", chapter_plan=chapter_plan, chapter_number=1)

        assert "Hero" in result
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_character_names(self):
        chapter_plan = [{"characters_involved": ["  Alice  ", "Bob"]}]

        result = await prompt_data_getters._discover_characters_of_interest(protagonist_name="Hero", chapter_plan=chapter_plan, chapter_number=1)

        assert "Alice" in result or "  Alice  " in result


class TestApplyProtagonistProximityFiltering:
    """Protagonist proximity filtering function."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_characters(self):
        result = await prompt_data_getters._apply_protagonist_proximity_filtering(characters_of_interest=set(), protagonist_name="Alice")

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_returns_protagonist_only_when_no_others(self):
        result = await prompt_data_getters._apply_protagonist_proximity_filtering(characters_of_interest={"Alice"}, protagonist_name="Alice")

        assert result == {"Alice"}

    @pytest.mark.asyncio
    async def test_keeps_protagonist_without_protagonist_name(self):
        with patch("data_access.kg_queries.get_shortest_path_length_between_entities") as mock_path:
            mock_path.return_value = 2

            result = await prompt_data_getters._apply_protagonist_proximity_filtering(characters_of_interest={"Alice", "Bob"}, protagonist_name="")

            assert result == {"Alice", "Bob"}

    @pytest.mark.asyncio
    async def test_filters_distant_characters_small_set(self):
        with patch("data_access.kg_queries.get_shortest_path_length_between_entities") as mock_path:

            async def get_path(protag, char):
                if char == "Bob":
                    return 2
                elif char == "Charlie":
                    return 5
                return None

            mock_path.side_effect = get_path

            result = await prompt_data_getters._apply_protagonist_proximity_filtering(
                characters_of_interest={"Alice", "Bob", "Charlie"},
                protagonist_name="Alice",
            )

            assert "Alice" in result
            assert "Bob" in result
            assert "Charlie" not in result

    @pytest.mark.asyncio
    async def test_uses_parallel_queries_for_large_set(self):
        with patch("data_access.kg_queries.get_shortest_path_length_between_entities") as mock_path:
            mock_path.return_value = 2

            result = await prompt_data_getters._apply_protagonist_proximity_filtering(
                characters_of_interest={"Alice", "Bob", "Charlie", "David"},
                protagonist_name="Alice",
            )

            assert "Alice" in result
            assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_handles_none_path_length(self):
        with patch("data_access.kg_queries.get_shortest_path_length_between_entities") as mock_path:
            mock_path.return_value = None

            result = await prompt_data_getters._apply_protagonist_proximity_filtering(characters_of_interest={"Alice", "Bob"}, protagonist_name="Alice")

            assert "Alice" in result
            assert "Bob" not in result

    @pytest.mark.asyncio
    async def test_handles_query_exceptions(self):
        with patch("data_access.kg_queries.get_shortest_path_length_between_entities") as mock_path:

            async def raise_error(*args, **kwargs):
                raise Exception("Query failed")

            mock_path.side_effect = raise_error

            result = await prompt_data_getters._apply_protagonist_proximity_filtering(
                characters_of_interest={"Alice", "Bob", "Charlie", "David"},
                protagonist_name="Alice",
            )

            assert "Alice" in result
            assert "Bob" not in result
            assert "Charlie" not in result
            assert "David" not in result

    @pytest.mark.asyncio
    async def test_includes_close_characters(self):
        with patch("data_access.kg_queries.get_shortest_path_length_between_entities") as mock_path:
            mock_path.return_value = 1

            result = await prompt_data_getters._apply_protagonist_proximity_filtering(characters_of_interest={"Alice", "Bob"}, protagonist_name="Alice")

            assert "Alice" in result
            assert "Bob" in result

    @pytest.mark.asyncio
    async def test_threshold_is_three_hops(self):
        with patch("data_access.kg_queries.get_shortest_path_length_between_entities") as mock_path:

            async def get_path(protag, char):
                if char == "Bob":
                    return 3
                elif char == "Charlie":
                    return 4
                return None

            mock_path.side_effect = get_path

            result = await prompt_data_getters._apply_protagonist_proximity_filtering(
                characters_of_interest={"Alice", "Bob", "Charlie"},
                protagonist_name="Alice",
            )

            assert "Alice" in result
            assert "Bob" in result
            assert "Charlie" not in result


class TestGatherNovelInfoFacts:
    """Novel info facts gathering function."""

    @pytest.mark.asyncio
    async def test_stops_when_max_facts_reached(self):
        facts = ["fact1", "fact2"]

        await prompt_data_getters._gather_novel_info_facts(facts_list=facts, max_total_facts=2)

        assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_gathers_theme_from_kg(self):
        with patch("data_access.kg_queries.get_novel_info_property_from_db") as mock_query:
            mock_query.return_value = "Redemption"

            facts = []
            await prompt_data_getters._gather_novel_info_facts(facts_list=facts, max_total_facts=10)

            assert len(facts) > 0
            assert any("theme" in fact.lower() for fact in facts)

    @pytest.mark.asyncio
    async def test_gathers_central_conflict_from_kg(self):
        with patch("data_access.kg_queries.get_novel_info_property_from_db") as mock_query:
            mock_query.side_effect = [None, "Good vs Evil"]

            facts = []
            await prompt_data_getters._gather_novel_info_facts(facts_list=facts, max_total_facts=10)

            assert any("conflict" in fact.lower() for fact in facts)

    @pytest.mark.asyncio
    async def test_handles_query_exceptions(self):
        with patch("data_access.kg_queries.get_novel_info_property_from_db") as mock_query:
            mock_query.side_effect = Exception("Query failed")

            facts = []
            await prompt_data_getters._gather_novel_info_facts(facts_list=facts, max_total_facts=10)

            assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_skips_empty_values(self):
        with patch("data_access.kg_queries.get_novel_info_property_from_db") as mock_query:
            mock_query.return_value = None

            facts = []
            await prompt_data_getters._gather_novel_info_facts(facts_list=facts, max_total_facts=10)

            assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_avoids_duplicate_facts(self):
        with patch("data_access.kg_queries.get_novel_info_property_from_db") as mock_query:
            mock_query.side_effect = ["Redemption", "Main conflict"]

            facts = ["- The novel's central theme is: Redemption."]
            await prompt_data_getters._gather_novel_info_facts(facts_list=facts, max_total_facts=10)

            assert "- The novel's central theme is: Redemption." in facts
            assert len([f for f in facts if "Redemption" in f]) == 1


class TestGatherCharacterFacts:
    """Character facts gathering function."""

    @pytest.mark.asyncio
    async def test_returns_early_when_no_characters(self):
        facts = []
        await prompt_data_getters._gather_character_facts(
            characters_of_interest=set(),
            kg_chapter_limit=5,
            facts_list=facts,
            max_facts_per_char=2,
            max_total_facts=10,
            protagonist_name="Alice",
        )

        assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_limits_to_three_characters(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock1:
                with patch("data_access.kg_queries.query_kg_from_db") as mock2:
                    mock_info.return_value = None
                    mock1.return_value = None
                    mock2.return_value = []

                    characters = {"Alice", "Bob", "Charlie", "David"}
                    facts = []

                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest=characters,
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=2,
                        max_total_facts=20,
                        protagonist_name="Alice",
                    )

                    call_count = mock_info.call_count + mock1.call_count + mock2.call_count
                    assert call_count == 9

    @pytest.mark.asyncio
    async def test_gathers_character_status(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_location:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.return_value = {"current_status": "alive"}
                    mock_location.return_value = None
                    mock_rel.return_value = []

                    facts = []
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=3,
                        max_total_facts=10,
                        protagonist_name="Alice",
                    )

                    assert any("status" in fact.lower() for fact in facts)
                    assert any("alive" in fact for fact in facts)

    @pytest.mark.asyncio
    async def test_gathers_character_location(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_location:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.return_value = None
                    mock_location.return_value = "Castle"
                    mock_rel.return_value = []

                    facts = []
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=3,
                        max_total_facts=10,
                        protagonist_name="Alice",
                    )

                    assert any("located in" in fact.lower() for fact in facts)
                    assert any("Castle" in fact for fact in facts)

    @pytest.mark.asyncio
    async def test_gathers_character_relationships(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.return_value = None
                    mock_value.return_value = None
                    mock_rel.return_value = [
                        {"predicate": "ally_of", "object": "Bob"},
                        {"predicate": "enemy_of", "object": "Charlie"},
                    ]

                    facts = []
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=3,
                        max_total_facts=10,
                        protagonist_name="Alice",
                    )

                    assert any("relationship" in fact.lower() for fact in facts)
                    assert any("Bob" in fact for fact in facts)

    @pytest.mark.asyncio
    async def test_respects_max_facts_per_char(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.return_value = {"current_status": "alive"}
                    mock_value.return_value = "Castle"
                    mock_rel.return_value = [
                        {"predicate": "ally_of", "object": "Bob"},
                        {"predicate": "enemy_of", "object": "Charlie"},
                    ]

                    facts = []
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=2,
                        max_total_facts=10,
                        protagonist_name="Alice",
                    )

                    assert len(facts) == 2

    @pytest.mark.asyncio
    async def test_respects_max_total_facts(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.return_value = {"current_status": "alive"}
                    mock_value.return_value = None
                    mock_rel.return_value = []

                    facts = ["fact1", "fact2", "fact3"]
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice", "Bob"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=3,
                        max_total_facts=3,
                        protagonist_name="Alice",
                    )

                    assert len(facts) == 3

    @pytest.mark.asyncio
    async def test_filters_relationship_types(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.return_value = None
                    mock_value.return_value = None
                    mock_rel.return_value = [
                        {"predicate": "uninteresting_rel", "object": "Bob"},
                        {"predicate": "ally_of", "object": "Charlie"},
                    ]

                    facts = []
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=3,
                        max_total_facts=10,
                        protagonist_name="Alice",
                    )

                    assert any("Charlie" in fact for fact in facts)
                    assert not any("Bob" in fact for fact in facts)

    @pytest.mark.asyncio
    async def test_handles_query_exceptions(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.side_effect = Exception("Query failed")
                    mock_value.side_effect = Exception("Query failed")
                    mock_rel.side_effect = Exception("Query failed")

                    facts = []
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=3,
                        max_total_facts=10,
                        protagonist_name="Alice",
                    )

                    assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_avoids_duplicate_facts(self):
        with patch("data_access.character_queries.get_character_info_for_snippet_from_db", new_callable=AsyncMock) as mock_info:
            with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                    mock_info.return_value = {"current_status": "alive"}
                    mock_value.return_value = None
                    mock_rel.return_value = []

                    facts = ["- Alice's status is: alive."]
                    await prompt_data_getters._gather_character_facts(
                        characters_of_interest={"Alice"},
                        kg_chapter_limit=5,
                        facts_list=facts,
                        max_facts_per_char=3,
                        max_total_facts=10,
                        protagonist_name="Alice",
                    )

                    assert len(facts) == 1


class TestGetReliableKGFactsForDraftingPrompt:
    """Main KG facts gathering function."""

    @pytest.mark.asyncio
    async def test_returns_message_for_chapter_zero(self):
        result = await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=0)

        assert "No KG facts applicable" in result

    @pytest.mark.asyncio
    async def test_returns_snapshot_kg_facts_when_available(self):
        mock_snapshot = MagicMock()
        mock_snapshot.kg_facts_block = "Cached KG facts"

        result = await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=1, snapshot=mock_snapshot)

        assert result == "Cached KG facts"

    @pytest.mark.asyncio
    async def test_uses_default_protagonist_when_none_provided(self):
        with patch.object(config, "DEFAULT_PROTAGONIST_NAME", "DefaultHero"):
            with patch("prompts.prompt_data_getters._discover_characters_of_interest") as mock_discover:
                with patch("prompts.prompt_data_getters._apply_protagonist_proximity_filtering") as mock_filter:
                    with patch("prompts.prompt_data_getters._gather_novel_info_facts"):
                        with patch("prompts.prompt_data_getters._gather_character_facts"):
                            mock_discover.return_value = set()
                            mock_filter.return_value = set()

                            await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=1, protagonist_name=None)

                            mock_discover.assert_called_once()
                            assert mock_discover.call_args[0][0] == "DefaultHero"

    @pytest.mark.asyncio
    async def test_returns_no_facts_message_when_empty(self):
        with patch("prompts.prompt_data_getters._discover_characters_of_interest") as mock_discover:
            with patch("prompts.prompt_data_getters._apply_protagonist_proximity_filtering") as mock_filter:
                with patch("prompts.prompt_data_getters._gather_novel_info_facts"):
                    with patch("prompts.prompt_data_getters._gather_character_facts"):
                        mock_discover.return_value = set()
                        mock_filter.return_value = set()

                        result = await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=1)

                        assert "No specific reliable KG facts" in result

    @pytest.mark.asyncio
    async def uses_prepopulation_chapter_for_chapter_1(self):
        with patch.object(config, "KG_PREPOPULATION_CHAPTER_NUM", 0):
            with patch("prompts.prompt_data_getters._discover_characters_of_interest") as mock_discover:
                with patch("prompts.prompt_data_getters._apply_protagonist_proximity_filtering") as mock_filter:
                    with patch("prompts.prompt_data_getters._gather_novel_info_facts"):
                        with patch("prompts.prompt_data_getters._gather_character_facts") as mock_char:
                            mock_discover.return_value = {"Alice"}
                            mock_filter.return_value = {"Alice"}

                            await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=1)

                            mock_char.assert_called_once()
                            assert mock_char.call_args[0][1] == 0

    @pytest.mark.asyncio
    async def test_limits_facts_to_max_total(self):
        with patch("prompts.prompt_data_getters._discover_characters_of_interest") as mock_discover:
            with patch("prompts.prompt_data_getters._apply_protagonist_proximity_filtering") as mock_filter:
                with patch("data_access.kg_queries.get_novel_info_property_from_db") as mock_novel:
                    with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                        with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                            mock_discover.return_value = {"Alice"}
                            mock_filter.return_value = {"Alice"}
                            mock_novel.return_value = "Theme"
                            mock_value.return_value = "alive"
                            mock_rel.return_value = []

                            result = await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=2, max_total_facts=2)

                            lines = result.split("\n")
                            facts = [l for l in lines if l.startswith("-")]
                            assert len(facts) <= 2

    @pytest.mark.asyncio
    async def test_deduplicates_and_sorts_facts(self):
        with patch("prompts.prompt_data_getters._discover_characters_of_interest") as mock_discover:
            with patch("prompts.prompt_data_getters._apply_protagonist_proximity_filtering") as mock_filter:
                with patch("data_access.kg_queries.get_novel_info_property_from_db") as mock_novel:
                    with patch("data_access.kg_queries.get_most_recent_value_from_db") as mock_value:
                        with patch("data_access.kg_queries.query_kg_from_db") as mock_rel:
                            mock_discover.return_value = {"Alice"}
                            mock_filter.return_value = {"Alice"}
                            mock_novel.side_effect = ["Theme B", "Theme A"]
                            mock_value.return_value = None
                            mock_rel.return_value = []

                            result = await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=2)

                            lines = [l for l in result.split("\n") if l.startswith("-")]
                            if len(lines) >= 2:
                                assert lines == sorted(lines)

    @pytest.mark.asyncio
    async def test_handles_snapshot_without_kg_facts(self):
        mock_snapshot = MagicMock()
        del mock_snapshot.kg_facts_block

        with patch("prompts.prompt_data_getters._discover_characters_of_interest") as mock_discover:
            with patch("prompts.prompt_data_getters._apply_protagonist_proximity_filtering") as mock_filter:
                with patch("prompts.prompt_data_getters._gather_novel_info_facts"):
                    with patch("prompts.prompt_data_getters._gather_character_facts"):
                        mock_discover.return_value = set()
                        mock_filter.return_value = set()

                        result = await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=1, snapshot=mock_snapshot)

                        assert "No specific reliable KG facts" in result


class TestGetCharacterStateSnippet:
    """Character state snippet function."""

    @pytest.mark.asyncio
    async def test_returns_empty_string_for_no_profiles(self):
        result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[])

        assert result == ""

    @pytest.mark.asyncio
    async def test_prioritizes_protagonist(self):
        alice = CharacterProfile(name="Alice", personality_description="Alice desc", created_chapter=1)
        bob = CharacterProfile(name="Bob", personality_description="Bob desc", created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[bob, alice], protagonist_name="Alice")

            lines = result.split("\n")
            alice_index = next((i for i, l in enumerate(lines) if "Alice" in l and "**" in l), -1)
            bob_index = next((i for i, l in enumerate(lines) if "Bob" in l and "**" in l), -1)

            if alice_index >= 0 and bob_index >= 0:
                assert alice_index < bob_index

    @pytest.mark.asyncio
    async def test_limits_characters_to_config_max(self):
        profiles = [CharacterProfile(name=f"Char{i}", created_chapter=1) for i in range(10)]

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=profiles)

            character_count = result.count("**Char")
            assert character_count <= 10

    @pytest.mark.asyncio
    async def test_includes_character_description(self):
        alice = CharacterProfile(name="Alice", personality_description="A brave warrior", created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "brave warrior" in result

    @pytest.mark.asyncio
    async def test_includes_character_traits(self):
        alice = CharacterProfile(name="Alice", traits=["brave", "loyal", "honest"], created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "brave" in result
            assert "Traits:" in result

    @pytest.mark.asyncio
    async def test_includes_character_status_when_not_unknown(self):
        alice = CharacterProfile(name="Alice", status="alive", created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "Status: alive" in result

    @pytest.mark.asyncio
    async def test_skips_unknown_status(self):
        alice = CharacterProfile(name="Alice", status="Unknown", created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "Status:" not in result or "Unknown" not in result

    @pytest.mark.asyncio
    async def test_includes_updates_personality(self):
        alice = CharacterProfile(
            name="Alice",
            updates={"personality": "Introverted and thoughtful"},
            created_chapter=1,
        )

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "Personality:" in result
            assert "Introverted" in result

    @pytest.mark.asyncio
    async def test_includes_updates_background(self):
        alice = CharacterProfile(
            name="Alice",
            updates={"background": "Grew up in the mountains"},
            created_chapter=1,
        )

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "Background:" in result
            assert "mountains" in result

    @pytest.mark.asyncio
    async def includes_neo4j_current_state(self):
        alice = CharacterProfile(name="Alice", created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {"summary": "Currently searching for the artifact"}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "Current State:" in result
            assert "artifact" in result

    @pytest.mark.asyncio
    async def includes_neo4j_relationships(self):
        alice = CharacterProfile(name="Alice", created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {"key_relationships": ["friend of Bob", "mentor to Charlie"]}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "Key Relationships:" in result
            assert "Bob" in result

    @pytest.mark.asyncio
    async def test_limits_traits_to_three(self):
        alice = CharacterProfile(
            name="Alice",
            traits=["brave", "loyal", "honest", "wise", "strong"],
            created_chapter=1,
        )

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            traits_line = next((line for line in result.split("\n") if "Traits:" in line), "")
            trait_count = traits_line.count(",") + 1
            assert trait_count <= 3

    @pytest.mark.asyncio
    async def test_limits_relationships_to_three(self):
        alice = CharacterProfile(name="Alice", created_chapter=1)

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {"key_relationships": ["rel1", "rel2", "rel3", "rel4", "rel5"]}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            rel_line = next(
                (line for line in result.split("\n") if "Key Relationships:" in line),
                "",
            )
            rel_count = rel_line.count(",") + 1
            assert rel_count <= 3

    @pytest.mark.asyncio
    async def prefers_profile_personality_over_neo4j(self):
        alice = CharacterProfile(
            name="Alice",
            updates={"personality": "Profile personality"},
            created_chapter=1,
        )

        with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
            mock_info.return_value = {"personality": "Neo4j personality"}

            result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[alice])

            assert "Profile personality" in result
            assert result.count("Personality:") == 1

    @pytest.mark.asyncio
    async def test_uses_default_protagonist_when_none_provided(self):
        with patch.object(config, "DEFAULT_PROTAGONIST_NAME", "DefaultHero"):
            alice = CharacterProfile(name="DefaultHero", personality_description="Hero", created_chapter=1)
            bob = CharacterProfile(name="Bob", personality_description="Bob", created_chapter=1)

            with patch("prompts.prompt_data_getters._cached_character_info") as mock_info:
                mock_info.return_value = {}

                result = await prompt_data_getters.get_character_state_snippet_for_prompt(character_profiles=[bob, alice], protagonist_name=None)

                lines = result.split("\n")
                hero_index = next((i for i, l in enumerate(lines) if "DefaultHero" in l), -1)
                bob_index = next((i for i, l in enumerate(lines) if "Bob" in l), -1)

                if hero_index >= 0 and bob_index >= 0:
                    assert hero_index < bob_index


class TestGetWorldStateSnippet:
    """World state snippet function."""

    @pytest.mark.asyncio
    async def test_returns_empty_string_for_no_items(self):
        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[])

        assert result == ""

    @pytest.mark.asyncio
    async def test_groups_items_by_category(self):
        item1 = WorldItem(
            id="loc1",
            category="location",
            name="Castle",
            description="A castle",
            created_chapter=1,
        )
        item2 = WorldItem(
            id="art1",
            category="artifact",
            name="Sword",
            description="A sword",
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item1, item2])

        assert "**location:**" in result
        assert "**artifact:**" in result

    @pytest.mark.asyncio
    async def test_uses_miscellaneous_for_none_category(self):
        item = WorldItem(
            id="item1",
            category="Miscellaneous",
            name="Unknown",
            description="An item",
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        assert "**Miscellaneous:**" in result

    @pytest.mark.asyncio
    async def test_limits_items_per_category(self):
        items = [
            WorldItem(
                id=f"loc{i}",
                category="location",
                name=f"Location{i}",
                description=f"Desc{i}",
                created_chapter=1,
            )
            for i in range(10)
        ]

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=items)

        location_count = result.count("**Location")
        assert location_count <= 10

    @pytest.mark.asyncio
    async def test_includes_item_description(self):
        item = WorldItem(
            id="loc1",
            category="location",
            name="Castle",
            description="A grand medieval castle",
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        assert "Description: A grand medieval castle" in result

    @pytest.mark.asyncio
    async def test_includes_item_goals(self):
        item = WorldItem(
            id="art1",
            category="artifact",
            name="Sword",
            goals=["Defeat evil", "Protect the realm"],
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        assert "Goals:" in result
        assert "Defeat evil" in result

    @pytest.mark.asyncio
    async def test_includes_item_rules(self):
        item = WorldItem(
            id="loc1",
            category="location",
            name="Castle",
            rules=["No magic allowed", "Respect the king"],
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        assert "Rules:" in result
        assert "No magic allowed" in result

    @pytest.mark.asyncio
    async def test_includes_item_key_elements(self):
        item = WorldItem(
            id="loc1",
            category="location",
            name="Castle",
            key_elements=["throne room", "dungeon", "armory"],
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        assert "Key Elements:" in result
        assert "throne room" in result

    @pytest.mark.asyncio
    async def test_limits_goals_to_two(self):
        item = WorldItem(
            id="art1",
            category="artifact",
            name="Sword",
            goals=["Goal1", "Goal2", "Goal3", "Goal4"],
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        goals_line = next((line for line in result.split("\n") if "Goals:" in line), "")
        goal_count = goals_line.count(",") + 1
        assert goal_count <= 2

    @pytest.mark.asyncio
    async def test_limits_rules_to_two(self):
        item = WorldItem(
            id="loc1",
            category="location",
            name="Castle",
            rules=["Rule1", "Rule2", "Rule3", "Rule4"],
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        rules_line = next((line for line in result.split("\n") if "Rules:" in line), "")
        rule_count = rules_line.count(",") + 1
        assert rule_count <= 2

    @pytest.mark.asyncio
    async def test_limits_key_elements_to_three(self):
        item = WorldItem(
            id="loc1",
            category="location",
            name="Castle",
            key_elements=["Elem1", "Elem2", "Elem3", "Elem4", "Elem5"],
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        elements_line = next((line for line in result.split("\n") if "Key Elements:" in line), "")
        element_count = elements_line.count(",") + 1
        assert element_count <= 3

    @pytest.mark.asyncio
    async def test_prioritizes_items_with_descriptions(self):
        item1 = WorldItem(
            id="loc1",
            category="location",
            name="ZZZ No Desc",
            description="",
            created_chapter=1,
        )
        item2 = WorldItem(
            id="loc2",
            category="location",
            name="AAA With Desc",
            description="Has description",
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item1, item2])

        assert "AAA With Desc" in result or "ZZZ No Desc" in result

    @pytest.mark.asyncio
    async def test_handles_items_with_no_additional_fields(self):
        item = WorldItem(
            id="loc1",
            category="location",
            name="Castle",
            description="",
            goals=[],
            rules=[],
            key_elements=[],
            created_chapter=1,
        )

        result = await prompt_data_getters.get_world_state_snippet_for_prompt(world_building=[item])

        assert "**Castle**" in result
