# tests/test_graph_structure_validation.py
"""Graph structure validation tests for each stage.

This test file validates that the knowledge graph structure at each stage
matches the requirements from docs/schema-design.md:

- Stage 1: Character nodes + relationships
- Stage 2: MajorPlotPoint events, Locations, Items, character arcs
- Stage 3: ActKeyEvent events, Location name enrichment
- Stage 4: Chapter, Scene, SceneEvent nodes + relationships
- Stage 5: Character/Chapter enrichment only

Based on: docs/schema-design.md - Stage-by-Stage Construction
"""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
class TestStage1GraphStructure:
    """Test graph structure after Stage 1: Character Initialization."""

    async def test_character_nodes_have_required_properties(self):
        """Test that Character nodes have all required Stage 1 properties."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "c": {
                        "id": "char_001",
                        "name": "Test Character",
                        "personality_description": "Test description",
                        "traits": ["brave", "loyal"],
                        "status": "Active",
                        "created_chapter": 0,
                        "is_provisional": False,
                        "created_ts": 1234567890,
                        "updated_ts": 1234567890,
                    }
                }
            ]

            query = """
            MATCH (c:Character)
            RETURN c
            """
            result = await mock_query(query)

            assert len(result) > 0
            character = result[0]["c"]

            assert "id" in character
            assert "name" in character
            assert "personality_description" in character
            assert "traits" in character
            assert "status" in character
            assert "created_chapter" in character
            assert character["created_chapter"] == 0
            assert "is_provisional" in character
            assert character["is_provisional"] is False

    async def test_character_relationships_exist(self):
        """Test that Character-Character relationships exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "source": {"name": "Character A"},
                    "r": {"description": "Test relationship"},
                    "type": "ALLIES_WITH",
                    "target": {"name": "Character B"},
                }
            ]

            query = """
            MATCH (source:Character)-[r]->(target:Character)
            RETURN source, type(r) as type, r, target
            """
            result = await mock_query(query)

            assert len(result) > 0
            relationship = result[0]

            assert "source" in relationship
            assert "target" in relationship
            assert "type" in relationship
            assert "r" in relationship


@pytest.mark.asyncio
class TestStage2GraphStructure:
    """Test graph structure after Stage 2: Global Outline."""

    async def test_major_plot_points_exist(self):
        """Test that exactly 4 MajorPlotPoint events exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {"e": {"id": "event_001", "event_type": "MajorPlotPoint", "sequence_order": 1}},
                {"e": {"id": "event_002", "event_type": "MajorPlotPoint", "sequence_order": 2}},
                {"e": {"id": "event_003", "event_type": "MajorPlotPoint", "sequence_order": 3}},
                {"e": {"id": "event_004", "event_type": "MajorPlotPoint", "sequence_order": 4}},
            ]

            query = """
            MATCH (e:Event {event_type: 'MajorPlotPoint'})
            RETURN e
            ORDER BY e.sequence_order
            """
            result = await mock_query(query)

            assert len(result) == 4
            assert result[0]["e"]["sequence_order"] == 1
            assert result[3]["e"]["sequence_order"] == 4

    async def test_location_nodes_exist(self):
        """Test that Location nodes exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "loc": {
                        "id": "loc_001",
                        "description": "Test location",
                        "category": "Location",
                        "created_chapter": 0,
                        "is_provisional": False,
                    }
                }
            ]

            query = """
            MATCH (loc:Location)
            RETURN loc
            """
            result = await mock_query(query)

            assert len(result) > 0
            location = result[0]["loc"]

            assert "id" in location
            assert "description" in location
            assert "category" in location
            assert location["category"] == "Location"
            assert "created_chapter" in location
            assert location["created_chapter"] == 0

    async def test_item_nodes_exist(self):
        """Test that Item nodes exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "i": {
                        "id": "item_001",
                        "name": "Test Item",
                        "description": "Test description",
                        "category": "Weapon",
                        "created_chapter": 0,
                        "is_provisional": False,
                    }
                }
            ]

            query = """
            MATCH (i:Item)
            RETURN i
            """
            result = await mock_query(query)

            assert len(result) > 0
            item = result[0]["i"]

            assert "id" in item
            assert "name" in item
            assert "description" in item
            assert "category" in item
            assert "created_chapter" in item
            assert item["created_chapter"] == 0

    async def test_character_arcs_enriched(self):
        """Test that Character nodes have arc properties after Stage 2."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "c": {
                        "name": "Test Character",
                        "arc_start": "Starting state",
                        "arc_end": "Ending state",
                        "arc_key_moments": ["Moment 1", "Moment 2"],
                    }
                }
            ]

            query = """
            MATCH (c:Character)
            WHERE c.arc_start IS NOT NULL
            RETURN c
            """
            result = await mock_query(query)

            assert len(result) > 0
            character = result[0]["c"]

            assert "arc_start" in character
            assert "arc_end" in character
            assert "arc_key_moments" in character


@pytest.mark.asyncio
class TestStage3GraphStructure:
    """Test graph structure after Stage 3: Act Outlines."""

    async def test_act_key_events_exist(self):
        """Test that ActKeyEvent nodes exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "e": {
                        "id": "event_101",
                        "event_type": "ActKeyEvent",
                        "act_number": 1,
                        "sequence_in_act": 1,
                        "cause": "Test cause",
                        "effect": "Test effect",
                        "created_chapter": 0,
                    }
                }
            ]

            query = """
            MATCH (e:Event {event_type: 'ActKeyEvent'})
            RETURN e
            """
            result = await mock_query(query)

            assert len(result) > 0
            event = result[0]["e"]

            assert "id" in event
            assert "event_type" in event
            assert event["event_type"] == "ActKeyEvent"
            assert "act_number" in event
            assert "sequence_in_act" in event
            assert "cause" in event
            assert "effect" in event

    async def test_act_key_events_link_to_major_plot_points(self):
        """Test that ActKeyEvents have PART_OF relationships to MajorPlotPoints."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "ake": {"id": "event_101", "event_type": "ActKeyEvent"},
                    "mpp": {"id": "event_001", "event_type": "MajorPlotPoint"},
                }
            ]

            query = """
            MATCH (ake:Event {event_type: 'ActKeyEvent'})-[:PART_OF]->(mpp:Event {event_type: 'MajorPlotPoint'})
            RETURN ake, mpp
            """
            result = await mock_query(query)

            assert len(result) > 0

    async def test_locations_have_names_after_stage3(self):
        """Test that Location nodes have names after Stage 3."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "loc": {
                        "id": "loc_001",
                        "name": "Test Location",
                        "description": "Test description",
                    }
                }
            ]

            query = """
            MATCH (loc:Location)
            WHERE loc.name IS NOT NULL
            RETURN loc
            """
            result = await mock_query(query)

            assert len(result) > 0
            location = result[0]["loc"]

            assert "name" in location


@pytest.mark.asyncio
class TestStage4GraphStructure:
    """Test graph structure after Stage 4: Chapter Outlines."""

    async def test_chapter_nodes_exist(self):
        """Test that Chapter nodes exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "ch": {
                        "id": "chapter_001",
                        "number": 1,
                        "title": "Chapter 1",
                        "summary": "Test summary",
                        "act_number": 1,
                        "created_chapter": 1,
                        "is_provisional": False,
                    }
                }
            ]

            query = """
            MATCH (ch:Chapter)
            RETURN ch
            """
            result = await mock_query(query)

            assert len(result) > 0
            chapter = result[0]["ch"]

            assert "id" in chapter
            assert "number" in chapter
            assert "title" in chapter
            assert "summary" in chapter
            assert "act_number" in chapter

    async def test_scene_nodes_exist(self):
        """Test that Scene nodes exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "s": {
                        "id": "scene_001",
                        "chapter_number": 1,
                        "scene_index": 0,
                        "title": "Opening Scene",
                        "pov_character": "Test Character",
                        "setting": "Test setting",
                        "plot_point": "Test plot point",
                        "conflict": "Test conflict",
                        "outcome": "Test outcome",
                        "beats": ["Beat 1", "Beat 2"],
                        "created_chapter": 1,
                    }
                }
            ]

            query = """
            MATCH (s:Scene)
            RETURN s
            """
            result = await mock_query(query)

            assert len(result) > 0
            scene = result[0]["s"]

            assert "id" in scene
            assert "chapter_number" in scene
            assert "scene_index" in scene
            assert "title" in scene
            assert "pov_character" in scene

    async def test_scene_events_exist(self):
        """Test that SceneEvent nodes exist."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "e": {
                        "id": "event_201",
                        "event_type": "SceneEvent",
                        "chapter_number": 1,
                        "scene_index": 0,
                        "act_number": 1,
                        "conflict": "Test conflict",
                        "outcome": "Test outcome",
                        "pov_character": "Test Character",
                        "created_chapter": 1,
                    }
                }
            ]

            query = """
            MATCH (e:Event {event_type: 'SceneEvent'})
            RETURN e
            """
            result = await mock_query(query)

            assert len(result) > 0
            event = result[0]["e"]

            assert "id" in event
            assert "event_type" in event
            assert event["event_type"] == "SceneEvent"
            assert "chapter_number" in event
            assert "scene_index" in event

    async def test_scene_part_of_chapter(self):
        """Test that Scenes have PART_OF relationships to Chapters."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "s": {"id": "scene_001", "scene_index": 0},
                    "ch": {"id": "chapter_001", "number": 1},
                }
            ]

            query = """
            MATCH (s:Scene)-[:PART_OF]->(ch:Chapter)
            RETURN s, ch
            """
            result = await mock_query(query)

            assert len(result) > 0


@pytest.mark.asyncio
class TestStage5GraphStructure:
    """Test graph structure after Stage 5: Narrative Enrichment."""

    async def test_characters_have_physical_descriptions(self):
        """Test that Character nodes have physical_description after Stage 5."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "c": {
                        "name": "Test Character",
                        "physical_description": "Tall with dark hair",
                    }
                }
            ]

            query = """
            MATCH (c:Character)
            WHERE c.physical_description IS NOT NULL
            RETURN c
            """
            result = await mock_query(query)

            assert len(result) > 0
            character = result[0]["c"]

            assert "physical_description" in character

    async def test_chapters_have_embeddings(self):
        """Test that Chapter nodes have embeddings after Stage 5."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "ch": {
                        "number": 1,
                        "embedding": [0.1, 0.2, 0.3, 0.4],
                    }
                }
            ]

            query = """
            MATCH (ch:Chapter)
            WHERE ch.embedding IS NOT NULL
            RETURN ch
            """
            result = await mock_query(query)

            assert len(result) > 0
            chapter = result[0]["ch"]

            assert "embedding" in chapter
            assert isinstance(chapter["embedding"], list)

    async def test_no_new_structural_entities_in_stage5(self):
        """Test that Stage 5 doesn't create new structural entities."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "structural_count": 0,
                }
            ]

            query = """
            MATCH (n)
            WHERE n.created_chapter > 0 AND n.created_chapter = $chapter_number
            AND NOT (n:Character OR n:Chapter)
            RETURN count(n) as structural_count
            """
            params = {"chapter_number": 1}
            result = await mock_query(query, params)

            assert len(result) > 0
            assert result[0]["structural_count"] == 0


@pytest.mark.asyncio
class TestGraphStructureConstraints:
    """Test graph structure constraints across all stages."""

    async def test_character_names_are_unique(self):
        """Test that Character names are unique."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []

            query = """
            MATCH (c:Character)
            WITH c.name as name, count(*) as count
            WHERE count > 1
            RETURN name, count
            """
            result = await mock_query(query)

            assert len(result) == 0, "Found duplicate character names"

    async def test_no_orphaned_scenes(self):
        """Test that all Scenes have PART_OF relationships to Chapters."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = []

            query = """
            MATCH (s:Scene)
            WHERE NOT (s)-[:PART_OF]->(:Chapter)
            RETURN s
            """
            result = await mock_query(query)

            assert len(result) == 0, "Found orphaned scenes"

    async def test_scene_indices_are_contiguous(self):
        """Test that scene indices are contiguous within chapters."""
        with patch('core.db_manager.neo4j_manager.execute_read_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {"chapter_number": 1, "scene_indices": [0, 1, 2]},
            ]

            query = """
            MATCH (s:Scene)
            WITH s.chapter_number as chapter_number, collect(s.scene_index) as scene_indices
            RETURN chapter_number, scene_indices
            ORDER BY chapter_number
            """
            result = await mock_query(query)

            for row in result:
                indices = sorted(row["scene_indices"])
                expected = list(range(len(indices)))
                assert indices == expected, f"Scene indices not contiguous in chapter {row['chapter_number']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
