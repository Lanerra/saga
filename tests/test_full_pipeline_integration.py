# tests/test_full_pipeline_integration.py
"""Full pipeline integration test for Stages 1-5.

This test verifies the complete knowledge graph construction pipeline:
- Stage 1: Character Initialization (Character sheets → Character nodes + relationships)
- Stage 2: Global Outline (MajorPlotPoint events, Locations, Items, character arcs)
- Stage 3: Act Outlines (ActKeyEvent events, Location name enrichment)
- Stage 4: Chapter Outlines (Chapter, Scene, SceneEvent nodes + relationships)
- Stage 5: Narrative Enrichment (Character/Chapter enrichment only)

Based on: docs/schema-design.md - Stage-by-Stage Construction
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from core.parsers.act_outline_parser import ActOutlineParser
from core.parsers.chapter_outline_parser import ChapterOutlineParser
from core.parsers.character_sheet_parser import CharacterSheetParser
from core.parsers.global_outline_parser import GlobalOutlineParser
from core.parsers.narrative_enrichment_parser import NarrativeEnrichmentParser
from models.kg_models import Chapter, CharacterProfile


@pytest.fixture
def stage1_character_sheets():
    """Sample character sheets for Stage 1."""
    return {
        "characters": [
            {
                "name": "Eleanor Whitaker",
                "personality_description": "Haunted protector of refugee camp, driven by loss",
                "traits": ["protective", "haunted", "determined"],
                "status": "Active",
                "relationships": [
                    {
                        "type": "LOVES",
                        "target": "Sarah Whitaker",
                        "description": "Eleanor's missing daughter",
                    }
                ],
            },
            {
                "name": "Sarah Whitaker",
                "personality_description": "Eleanor's young daughter, captured by creature",
                "traits": ["innocent", "vulnerable"],
                "status": "Missing",
                "relationships": [],
            },
        ]
    }


@pytest.fixture
def stage2_global_outline():
    """Sample global outline for Stage 2."""
    return {
        "act_count": 3,
        "acts": [
            {
                "act_number": 1,
                "title": "Whispers in the Mist",
                "summary": "First attacks",
                "key_events": ["Discovery of missing child"],
                "chapters_start": 1,
                "chapters_end": 3,
            }
        ],
        "inciting_incident": "Sarah goes missing",
        "midpoint": "Creature nest discovered",
        "climax": "Final confrontation",
        "resolution": "Camp survives",
        "character_arcs": [
            {
                "character_name": "Eleanor Whitaker",
                "starting_state": "Haunted protector",
                "ending_state": "Reluctant hunter",
                "key_moments": ["Losing Sarah", "Destroying creature"],
            }
        ],
        "locations": [
            {
                "name": "Blackwater Creek",
                "description": "Misty swamp where creature resides",
            }
        ],
        "items": [
            {
                "name": "Bloodstained doll",
                "description": "Sarah's doll",
                "category": "Keepsake",
            }
        ],
        "thematic_progression": "War to supernatural terror",
        "pacing_notes": "Slow dread escalating",
        "total_chapters": 3,
        "structure_type": "3-act",
        "generated_at": "initialization",
    }


@pytest.fixture
def stage3_act_outline():
    """Sample act outline for Stage 3."""
    return {
        "act_number": 1,
        "title": "Whispers in the Mist",
        "summary": "First attacks fracture peace",
        "key_events": [
            {
                "name": "Missing Child Discovery",
                "description": "Eleanor discovers Sarah missing from tent",
                "sequence_in_act": 1,
                "cause": "Creature infiltrates camp",
                "effect": "Eleanor begins search",
                "character_involvements": [{"character_name": "Eleanor Whitaker", "role": "protagonist"}],
                "location": {"name": "Refugee Camp", "description": "Settlement"},
                "items": [{"item_name": "Bloodstained doll", "role": "clue"}],
            }
        ],
        "chapter_count": 3,
        "pacing_notes": "Slow build",
    }


@pytest.fixture
def stage4_chapter_outline():
    """Sample chapter outline for Stage 4."""
    return {
        "chapter_number": 1,
        "act_number": 1,
        "title": "The Vanishing",
        "summary": "Sarah disappears from the camp",
        "scenes": [
            {
                "scene_index": 0,
                "title": "Empty Tent",
                "pov_character": "Eleanor Whitaker",
                "setting": "Eleanor's tent in the refugee camp",
                "plot_point": "Eleanor discovers Sarah is missing",
                "conflict": "Eleanor's growing panic",
                "outcome": "Eleanor alerts the camp",
                "beats": ["Waking up", "Finding empty bed", "Discovering doll"],
                "events": [
                    {
                        "name": "Discovery",
                        "description": "Eleanor finds Sarah gone",
                        "conflict": "Growing dread",
                        "outcome": "Raises alarm",
                        "pov_character": "Eleanor Whitaker",
                    }
                ],
                "location": {
                    "name": "Eleanor's Tent",
                    "description": "Small canvas tent in refugee camp",
                },
            }
        ],
    }


@pytest.fixture
def stage5_narrative_text():
    """Sample narrative text for Stage 5."""
    return """
    Chapter 1: The Vanishing

    Eleanor Whitaker was a tall woman with long dark hair streaked with gray,
    her face lined with worry and fatigue. She wore a tattered blue dress
    that had once been fine. Her eyes were sharp and watchful, missing nothing
    in the predawn gloom of the refugee camp.
    """


@pytest.mark.asyncio
class TestFullPipelineIntegration:
    """Integration tests for the full Stage 1-5 pipeline."""

    async def test_stage1_character_initialization(self, stage1_character_sheets):
        """Test Stage 1: Character Initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stage1_character_sheets, f)
            temp_file = f.name

        try:
            parser = CharacterSheetParser(character_sheets_path=temp_file)

            with patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write:
                mock_write.return_value = []

                result = await parser.parse_and_persist()

                assert result is True
                assert mock_write.called

                all_queries = [call[0][0] for call in mock_write.call_args_list]
                character_query = any("CREATE (c:Character" in q for q in all_queries)
                relationship_query = any("MERGE (source)-[r:LOVES]->(target)" in q for q in all_queries)

                assert character_query, "Character creation query not found"
                assert relationship_query, "Relationship creation query not found"

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    async def test_stage2_global_outline(self, stage2_global_outline):
        """Test Stage 2: Global Outline."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stage2_global_outline, f)
            temp_file = f.name

        try:
            parser = GlobalOutlineParser(global_outline_path=temp_file)

            with patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write:
                mock_write.return_value = []

                result = await parser.parse_and_persist()

                assert result is True
                assert mock_write.called

                all_queries = [call[0][0] for call in mock_write.call_args_list]

                major_plot_point = any("event_type: 'MajorPlotPoint'" in q for q in all_queries)
                location_creation = any("CREATE (loc:Location" in q for q in all_queries)
                item_creation = any("CREATE (i:Item" in q for q in all_queries)

                assert major_plot_point, "MajorPlotPoint creation not found"
                assert location_creation, "Location creation not found"
                assert item_creation, "Item creation not found"

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    async def test_stage3_act_outline(self, stage3_act_outline):
        """Test Stage 3: Act Outline."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stage3_act_outline, f)
            temp_file = f.name

        try:
            parser = ActOutlineParser(act_outline_path=temp_file, act_number=1)

            with (
                patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write,
                patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_read,
            ):
                mock_write.return_value = []
                mock_read.return_value = []

                result = await parser.parse_and_persist()

                assert result is True
                assert mock_write.called

                all_queries = [call[0][0] for call in mock_write.call_args_list]

                act_key_event = any("event_type: 'ActKeyEvent'" in q for q in all_queries)
                involves_relationship = any("INVOLVES" in q for q in all_queries)

                assert act_key_event, "ActKeyEvent creation not found"
                assert involves_relationship, "INVOLVES relationship not found"

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    async def test_stage4_chapter_outline(self, stage4_chapter_outline):
        """Test Stage 4: Chapter Outline."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stage4_chapter_outline, f)
            temp_file = f.name

        try:
            parser = ChapterOutlineParser(chapter_outline_path=temp_file, chapter_number=1)

            with (
                patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write,
                patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_read,
                patch("data_access.character_queries.get_all_character_names", new_callable=AsyncMock) as mock_chars,
            ):
                mock_write.return_value = []
                mock_read.return_value = []
                mock_chars.return_value = ["Eleanor Whitaker"]

                result = await parser.parse_and_persist()

                assert result is True
                assert mock_write.called

                all_queries = [call[0][0] for call in mock_write.call_args_list]

                chapter_creation = any("CREATE (c:Chapter" in q for q in all_queries)
                scene_creation = any("CREATE (s:Scene" in q for q in all_queries)
                scene_event = any("event_type: 'SceneEvent'" in q for q in all_queries)

                assert chapter_creation, "Chapter creation not found"
                assert scene_creation, "Scene creation not found"
                assert scene_event, "SceneEvent creation not found"

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    async def test_stage5_narrative_enrichment(self, stage5_narrative_text):
        """Test Stage 5: Narrative Enrichment."""
        parser = NarrativeEnrichmentParser(
            narrative_text=stage5_narrative_text,
            chapter_number=1,
        )

        with (
            patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write,
            patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_read,
            patch("data_access.character_queries.get_character_profiles", new_callable=AsyncMock) as mock_chars,
            patch("data_access.chapter_queries.get_chapter_data_from_db", new_callable=AsyncMock) as mock_chapter,
        ):
            mock_write.return_value = []
            mock_read.return_value = []
            mock_chars.return_value = [
                CharacterProfile(
                    id="char_001",
                    name="Eleanor Whitaker",
                    personality_description="Haunted protector",
                    traits=["protective"],
                    status="Active",
                    created_chapter=0,
                    is_provisional=False,
                    created_ts=1234567890,
                    updated_ts=1234567890,
                )
            ]
            mock_chapter.return_value = Chapter(
                id="chapter_001",
                number=1,
                title="The Vanishing",
                summary="Sarah disappears",
                act_number=1,
                created_chapter=1,
                is_provisional=False,
                created_ts=1234567890,
                updated_ts=1234567890,
            )

            result = await parser.parse_and_persist()

            assert result is True

    async def test_full_pipeline_sequential(
        self,
        stage1_character_sheets,
        stage2_global_outline,
        stage3_act_outline,
        stage4_chapter_outline,
        stage5_narrative_text,
    ):
        """Test full pipeline Stages 1-5 sequentially."""
        stage1_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        stage2_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        stage3_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        stage4_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

        try:
            json.dump(stage1_character_sheets, stage1_file)
            stage1_file.flush()
            json.dump(stage2_global_outline, stage2_file)
            stage2_file.flush()
            json.dump(stage3_act_outline, stage3_file)
            stage3_file.flush()
            json.dump(stage4_chapter_outline, stage4_file)
            stage4_file.flush()

            with (
                patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write,
                patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_read,
                patch("data_access.character_queries.get_character_profiles", new_callable=AsyncMock) as mock_chars,
                patch("data_access.character_queries.get_all_character_names", new_callable=AsyncMock) as mock_char_names,
                patch("data_access.chapter_queries.get_chapter_data_from_db", new_callable=AsyncMock) as mock_chapter,
            ):
                mock_write.return_value = []
                mock_read.return_value = []
                mock_chars.return_value = [
                    CharacterProfile(
                        id="char_001",
                        name="Eleanor Whitaker",
                        personality_description="Haunted protector",
                        traits=["protective"],
                        status="Active",
                        created_chapter=0,
                        is_provisional=False,
                        created_ts=1234567890,
                        updated_ts=1234567890,
                    )
                ]
                mock_char_names.return_value = ["Eleanor Whitaker", "Sarah Whitaker"]
                mock_chapter.return_value = Chapter(
                    id="chapter_001",
                    number=1,
                    title="The Vanishing",
                    summary="Sarah disappears",
                    act_number=1,
                    created_chapter=1,
                    is_provisional=False,
                    created_ts=1234567890,
                    updated_ts=1234567890,
                )

                parser1 = CharacterSheetParser(character_sheets_path=stage1_file.name)
                result1 = await parser1.parse_and_persist()
                assert result1 is True, "Stage 1 failed"

                parser2 = GlobalOutlineParser(global_outline_path=stage2_file.name)
                result2 = await parser2.parse_and_persist()
                assert result2 is True, "Stage 2 failed"

                parser3 = ActOutlineParser(act_outline_path=stage3_file.name, act_number=1)
                result3 = await parser3.parse_and_persist()
                assert result3 is True, "Stage 3 failed"

                parser4 = ChapterOutlineParser(chapter_outline_path=stage4_file.name, chapter_number=1)
                result4 = await parser4.parse_and_persist()
                assert result4 is True, "Stage 4 failed"

                parser5 = NarrativeEnrichmentParser(
                    narrative_text=stage5_narrative_text,
                    chapter_number=1,
                )
                result5 = await parser5.parse_and_persist()
                assert result5 is True, "Stage 5 failed"

        finally:
            for file in [stage1_file, stage2_file, stage3_file, stage4_file]:
                file.close()
                if os.path.exists(file.name):
                    os.unlink(file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
