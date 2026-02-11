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
        "Eleanor Whitaker": {
            "name": "Eleanor Whitaker",
            "description": "Haunted protector of refugee camp, driven by loss",
            "traits": ["protective", "haunted", "determined"],
            "status": "Active",
            "relationships": {
                "Sarah Whitaker": {
                    "type": "LOVES",
                    "description": "Eleanor's missing daughter",
                }
            },
        },
        "Sarah Whitaker": {
            "name": "Sarah Whitaker",
            "description": "Eleanor's young daughter, captured by creature",
            "traits": ["innocent", "vulnerable"],
            "status": "Missing",
            "relationships": {},
        },
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
        "acts": [
            {
                "act_number": 1,
                "title": "Whispers in the Mist",
                "summary": "First attacks fracture peace",
                "sections": {
                    "key_events": [
                        {
                            "event": "Missing Child Discovery",
                            "description": "Eleanor discovers Sarah missing from tent",
                            "sequence": 1,
                            "cause": "Creature infiltrates camp",
                            "effect": "Eleanor begins search",
                        }
                    ],
                },
            }
        ],
    }


@pytest.fixture
def stage4_chapter_outline():
    """Sample chapter outline for Stage 4."""
    return {
        "chapter_1": {
            "chapter_number": 1,
            "act_number": 1,
            "title": "The Vanishing",
            "summary": "Sarah disappears from the camp",
            "scene_description": "Eleanor's tent in the refugee camp",
            "plot_point": "Eleanor discovers Sarah is missing",
            "key_beats": [
                "Eleanor Whitaker wakes up and finds empty bed",
                "Eleanor Whitaker discovers doll",
            ],
        }
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

            with (
                patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write,
                patch("core.db_manager.neo4j_manager.execute_cypher_batch", new_callable=AsyncMock) as mock_batch,
            ):
                mock_write.return_value = []
                mock_batch.return_value = None

                success, message = await parser.parse_and_persist()

                assert success is True

                all_queries = []
                for call in mock_write.call_args_list:
                    all_queries.append(call[0][0])
                for call in mock_batch.call_args_list:
                    for statement in call[0][0]:
                        all_queries.append(statement[0])

                character_query = any("MERGE" in q and "Character" in q for q in all_queries)
                relationship_query = any("LOVES" in q for q in all_queries)

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

                success, message = await parser.parse_and_persist()

                assert success is True
                assert mock_write.called

                all_queries = [call[0][0] for call in mock_write.call_args_list]
                all_params = [call[0][1] if len(call[0]) > 1 else {} for call in mock_write.call_args_list]

                major_plot_point = any("Event" in q and p.get("event_type") == "MajorPlotPoint" for q, p in zip(all_queries, all_params, strict=False))
                location_creation = any("Location" in q and "MERGE" in q for q in all_queries)
                item_creation = any("Item" in q and "MERGE" in q for q in all_queries)

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
            parser = ActOutlineParser(act_outline_path=temp_file)

            def read_side_effect(query, params=None):
                if "Character" in query:
                    return [{"name": "Eleanor Whitaker"}]
                return []

            llm_character_response = json.dumps([{"name": "Eleanor Whitaker", "role": "protagonist"}])

            with (
                patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write,
                patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_read,
                patch("core.parsers.act_outline_parser.llm_service.async_call_llm", new_callable=AsyncMock) as mock_llm,
            ):
                mock_write.return_value = []
                mock_read.side_effect = read_side_effect
                mock_llm.return_value = (llm_character_response, {})

                success, message = await parser.parse_and_persist()

                assert success is True
                assert mock_write.called

                all_queries = [call[0][0] for call in mock_write.call_args_list]
                all_params = [call[0][1] if len(call[0]) > 1 else {} for call in mock_write.call_args_list]

                act_key_event = any("Event" in q and p.get("event_type") == "ActKeyEvent" for q, p in zip(all_queries, all_params, strict=False))
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

                success, message = await parser.parse_and_persist()

                assert success is True
                assert mock_write.called

                all_queries = [call[0][0] for call in mock_write.call_args_list]

                chapter_creation = any("MERGE" in q and "Chapter" in q for q in all_queries)
                scene_creation = any("MERGE" in q and "Scene" in q for q in all_queries)

                assert chapter_creation, "Chapter creation not found"
                assert scene_creation, "Scene creation not found"

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

            success, message = await parser.parse_and_persist()

            assert success is True

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

            def read_side_effect(query, params=None):
                if "Character" in query and "name" in query:
                    return [{"name": "Eleanor Whitaker"}, {"name": "Sarah Whitaker"}]
                return []

            llm_character_response = json.dumps([{"name": "Eleanor Whitaker", "role": "protagonist"}])

            with (
                patch("core.db_manager.neo4j_manager.execute_write_query", new_callable=AsyncMock) as mock_write,
                patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_read,
                patch("core.db_manager.neo4j_manager.execute_cypher_batch", new_callable=AsyncMock) as mock_batch,
                patch("data_access.character_queries.get_character_profiles", new_callable=AsyncMock) as mock_chars,
                patch("data_access.character_queries.get_all_character_names", new_callable=AsyncMock) as mock_char_names,
                patch("data_access.chapter_queries.get_chapter_data_from_db", new_callable=AsyncMock) as mock_chapter,
                patch("core.parsers.act_outline_parser.llm_service.async_call_llm", new_callable=AsyncMock) as mock_llm,
            ):
                mock_write.return_value = []
                mock_read.side_effect = read_side_effect
                mock_batch.return_value = None
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
                mock_llm.return_value = (llm_character_response, {})

                parser1 = CharacterSheetParser(character_sheets_path=stage1_file.name)
                success1, message1 = await parser1.parse_and_persist()
                assert success1 is True, f"Stage 1 failed: {message1}"

                parser2 = GlobalOutlineParser(global_outline_path=stage2_file.name)
                success2, message2 = await parser2.parse_and_persist()
                assert success2 is True, f"Stage 2 failed: {message2}"

                parser3 = ActOutlineParser(act_outline_path=stage3_file.name)
                success3, message3 = await parser3.parse_and_persist()
                assert success3 is True, f"Stage 3 failed: {message3}"

                parser4 = ChapterOutlineParser(chapter_outline_path=stage4_file.name, chapter_number=1)
                success4, message4 = await parser4.parse_and_persist()
                assert success4 is True, f"Stage 4 failed: {message4}"

                parser5 = NarrativeEnrichmentParser(
                    narrative_text=stage5_narrative_text,
                    chapter_number=1,
                )
                success5, message5 = await parser5.parse_and_persist()
                assert success5 is True, f"Stage 5 failed: {message5}"

        finally:
            for file in [stage1_file, stage2_file, stage3_file, stage4_file]:
                file.close()
                if os.path.exists(file.name):
                    os.unlink(file.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
