from unittest.mock import AsyncMock, patch

import pytest

from data_access.scene_queries import (
    get_act_events,
    get_character_items,
    get_character_relationships_for_scene,
    get_scene_events,
    get_scene_items,
)


@pytest.mark.asyncio
class TestSceneQueries:
    async def test_get_scene_events_returns_events(self):
        expected_records = [
            {
                "name": "Test Event",
                "description": "Test description",
                "conflict": "Test conflict",
                "outcome": "Test outcome",
                "pov_character": "Test Character",
                "characters_involved": ["Character A", "Character B"],
            }
        ]

        with patch("data_access.scene_queries.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_read_query = AsyncMock(return_value=expected_records)

            events = await get_scene_events(chapter_number=1, scene_index=0)

            assert len(events) == 1
            assert events[0]["name"] == "Test Event"
            assert events[0]["description"] == "Test description"
            assert events[0]["conflict"] == "Test conflict"
            assert events[0]["outcome"] == "Test outcome"
            assert events[0]["characters_involved"] == ["Character A", "Character B"]

    async def test_get_scene_events_empty_result(self):
        with patch("data_access.scene_queries.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_read_query = AsyncMock(return_value=[])

            events = await get_scene_events(chapter_number=1, scene_index=0)

            assert events == []

    async def test_get_character_relationships_for_scene(self):
        expected_records = [
            {
                "source": "Alice",
                "relationship_type": "FRIENDS_WITH",
                "target": "Bob",
                "description": "Close friends since childhood",
                "chapter_added": 0,
            }
        ]

        with patch("data_access.scene_queries.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_read_query = AsyncMock(return_value=expected_records)

            relationships = await get_character_relationships_for_scene(
                character_names=["Alice", "Bob"],
                chapter_limit=1,
            )

            assert len(relationships) == 1
            assert relationships[0]["source"] == "Alice"
            assert relationships[0]["target"] == "Bob"

    async def test_get_character_items(self):
        expected_records = [
            {
                "character_name": "Alice",
                "item_name": "Magic Sword",
                "item_description": "A legendary blade",
                "item_category": "Weapon",
                "acquired_chapter": 0,
            }
        ]

        with patch("data_access.scene_queries.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_read_query = AsyncMock(return_value=expected_records)

            items = await get_character_items(
                character_names=["Alice"],
                chapter_limit=1,
            )

            assert len(items) == 1
            assert items[0]["character_name"] == "Alice"
            assert items[0]["item_name"] == "Magic Sword"

    async def test_get_scene_items(self):
        expected_records = [
            {
                "item_name": "Ancient Map",
                "item_description": "Shows hidden passages",
                "item_category": "Tool",
            }
        ]

        with patch("data_access.scene_queries.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_read_query = AsyncMock(return_value=expected_records)

            items = await get_scene_items(chapter_number=1, scene_index=0)

            assert len(items) == 1
            assert items[0]["item_name"] == "Ancient Map"

    async def test_get_act_events(self):
        expected_records = [
            {
                "major_points": [
                    {
                        "name": "Inciting Incident",
                        "description": "The hero's journey begins",
                        "sequence_order": 1,
                    }
                ],
                "act_events": [
                    {
                        "name": "Meeting the Mentor",
                        "description": "Hero meets wise guide",
                        "sequence_in_act": 1,
                        "cause": "Hero seeks guidance",
                        "effect": "Hero gains confidence",
                        "characters_involved": ["Hero", "Mentor"],
                        "location": "Ancient Temple",
                        "part_of": "Inciting Incident",
                    }
                ],
            }
        ]

        with patch("data_access.scene_queries.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_read_query = AsyncMock(return_value=expected_records)

            events_data = await get_act_events(act_number=1)

            assert "major_plot_points" in events_data
            assert "act_key_events" in events_data
            assert len(events_data["major_plot_points"]) == 1
            assert len(events_data["act_key_events"]) == 1
            assert events_data["major_plot_points"][0]["name"] == "Inciting Incident"
            assert events_data["act_key_events"][0]["name"] == "Meeting the Mentor"

    async def test_get_act_events_empty_result(self):
        with patch("data_access.scene_queries.neo4j_manager") as fake_neo4j:
            fake_neo4j.execute_read_query = AsyncMock(return_value=[])

            events_data = await get_act_events(act_number=1)

            assert events_data == {
                "major_plot_points": [],
                "act_key_events": [],
            }
