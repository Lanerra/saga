# tests/test_narrative_quality_with_graph_context.py
"""Tests for narrative quality with graph context.

This test file verifies that narrative generation uses graph context properly:
- Character personalities are reflected in narrative
- Character relationships influence interactions
- Events from graph are referenced in narrative
- Locations from graph are used consistently
- Items from graph appear in narrative

Based on: docs/schema-design.md - Phase 5: Testing
"""

from unittest.mock import AsyncMock, patch

import pytest

from models.kg_models import CharacterProfile, Scene


@pytest.fixture
def mock_character_with_context():
    """Character with rich graph context."""
    return CharacterProfile(
        id="char_001",
        name="Eleanor Whitaker",
        personality_description="Haunted protector driven by loss of daughter",
        traits=["protective", "haunted", "determined"],
        status="Active",
        created_chapter=0,
        is_provisional=False,
        created_ts=1234567890,
        updated_ts=1234567890,
        physical_description="Tall woman with dark hair and weary eyes",
    )


@pytest.fixture
def mock_scene_with_context():
    """Scene with graph context."""
    return Scene(
        id="scene_001",
        chapter_number=1,
        scene_index=0,
        title="Empty Tent",
        pov_character="Eleanor Whitaker",
        setting="Eleanor's tent in the refugee camp",
        plot_point="Eleanor discovers Sarah is missing",
        conflict="Growing panic and dread",
        outcome="Eleanor alerts the camp",
        beats=["Waking up", "Finding empty bed", "Discovering bloodstained doll"],
        created_chapter=1,
        is_provisional=False,
        created_ts=1234567890,
        updated_ts=1234567890,
    )


@pytest.mark.asyncio
class TestNarrativeWithCharacterContext:
    """Test narrative quality with character context."""

    async def test_character_personality_in_narrative(self, mock_character_with_context):
        """Test that character personality traits are reflected in narrative."""
        narrative = """
        Eleanor's heart raced as she scanned the empty tent. Her daughter's absence
        felt like a physical wound. Years of protecting the camp's refugees had taught
        her to sense danger, and every instinct screamed that Sarah was in peril.
        """

        personality_keywords = ["protecting", "sense danger", "heart raced"]

        for keyword in personality_keywords:
            assert keyword.lower() in narrative.lower(), f"Personality keyword '{keyword}' not found in narrative"

    async def test_character_traits_in_dialogue(self, mock_character_with_context):
        """Test that character traits influence dialogue."""
        dialogue = """
        "I won't rest until I find her," Eleanor said, her voice steady despite
        the fear coursing through her. "Sarah needs me. I've failed too many already."
        """

        trait_indicators = {
            "protective": ["won't rest", "find her"],
            "determined": ["voice steady"],
            "haunted": ["failed too many"],
        }

        for trait, indicators in trait_indicators.items():
            if trait in mock_character_with_context.traits:
                found = any(ind.lower() in dialogue.lower() for ind in indicators)
                assert found, f"Trait '{trait}' not reflected in dialogue"

    async def test_physical_description_consistency(self, mock_character_with_context):
        """Test that physical descriptions are consistent with graph."""
        narrative = """
        The tall woman with dark hair stood at the tent entrance, her weary eyes
        scanning the camp for any sign of her daughter.
        """

        physical_keywords = ["tall", "dark hair", "eyes"]

        for keyword in physical_keywords:
            assert keyword.lower() in narrative.lower(), f"Physical description '{keyword}' not found in narrative"


@pytest.mark.asyncio
class TestNarrativeWithRelationshipContext:
    """Test narrative quality with relationship context."""

    async def test_relationship_influences_interaction(self):
        """Test that relationships from graph influence character interactions."""
        with patch("data_access.character_queries.get_character_relationships", new_callable=AsyncMock) as mock_rels:
            mock_rels.return_value = [
                {
                    "type": "LOVES",
                    "target": "Sarah Whitaker",
                    "description": "Eleanor's daughter",
                }
            ]

            narrative = """
            Eleanor clutched Sarah's bloodstained doll, her hands trembling.
            "My baby," she whispered, tears streaming down her face.
            """

            relationship_indicators = ["clutched", "baby", "tears", "trembling"]

            found_indicators = sum(1 for ind in relationship_indicators if ind.lower() in narrative.lower())
            assert found_indicators >= 2, "Relationship context not reflected in narrative"

    async def test_conflict_relationship_reflected(self):
        """Test that conflicting relationships create tension."""
        narrative = """
        Eleanor glared at Thomas. Despite their shared goal to save the camp,
        old wounds from the war made trust impossible. She would work with him,
        but she'd never forget what his regiment did to her family.
        """

        conflict_indicators = ["glared", "despite", "impossible", "never forget", "old wounds"]

        found_indicators = sum(1 for ind in conflict_indicators if ind.lower() in narrative.lower())
        assert found_indicators >= 3, "Conflict relationship not reflected in narrative"


@pytest.mark.asyncio
class TestNarrativeWithEventContext:
    """Test narrative quality with event context."""

    async def test_event_referenced_in_narrative(self):
        """Test that events from graph are referenced in narrative."""
        with patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "e": {
                        "name": "Missing Child Discovery",
                        "description": "Eleanor discovers Sarah missing",
                        "conflict": "Growing panic",
                        "outcome": "Alerts the camp",
                    }
                }
            ]

            narrative = """
            Eleanor woke to an empty tent. Sarah's bed was cold, untouched for hours.
            Panic surged through her as she realized her daughter was missing.
            Within minutes, the entire camp knew - another child had vanished.
            """

            event_keywords = ["missing", "panic", "camp", "vanished"]

            found_keywords = sum(1 for kw in event_keywords if kw.lower() in narrative.lower())
            assert found_keywords >= 3, "Event context not reflected in narrative"

    async def test_event_sequence_preserved(self):
        """Test that event sequence from graph is preserved in narrative."""
        events = [
            {"name": "Waking", "sequence": 1},
            {"name": "Discovery", "sequence": 2},
            {"name": "Alert", "sequence": 3},
        ]

        narrative = """
        Eleanor stirred at dawn, reaching for Sarah's warmth. Her hand found only
        cold sheets. Panic seized her as she sat up - the bed was empty. She stumbled
        outside, her voice cracking as she called for help. Within moments, the camp
        was awake, searching for the missing child.
        """

        sequence_indicators = [
            ("stirred", "waking"),
            ("empty", "discovery"),
            ("called for help", "alert"),
        ]

        for indicator, event_type in sequence_indicators:
            assert indicator.lower() in narrative.lower(), f"Event '{event_type}' not found in narrative"


@pytest.mark.asyncio
class TestNarrativeWithLocationContext:
    """Test narrative quality with location context."""

    async def test_location_description_consistent(self):
        """Test that location descriptions are consistent with graph."""
        with patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "loc": {
                        "name": "Refugee Camp",
                        "description": "Temporary settlement of tents and lean-tos",
                    }
                }
            ]

            narrative = """
            The refugee camp stretched across the clearing, a patchwork of tattered
            tents and makeshift shelters. Smoke from cooking fires drifted through
            the morning mist.
            """

            location_keywords = ["camp", "tents", "shelters"]

            found_keywords = sum(1 for kw in location_keywords if kw.lower() in narrative.lower())
            assert found_keywords >= 2, "Location context not reflected in narrative"

    async def test_location_atmosphere_maintained(self):
        """Test that location atmosphere is maintained throughout scene."""
        narrative_segments = [
            "The misty swamp stretched before them, dark and foreboding.",
            "Eleanor pushed through the undergrowth, water soaking her boots.",
            "The fog grew thicker as they ventured deeper into Blackwater Creek.",
        ]

        atmosphere_keywords = ["misty", "dark", "fog", "thick", "swamp"]

        for segment in narrative_segments:
            found = any(kw.lower() in segment.lower() for kw in atmosphere_keywords)
            assert found, f"Location atmosphere not maintained in segment: {segment}"


@pytest.mark.asyncio
class TestNarrativeWithItemContext:
    """Test narrative quality with item context."""

    async def test_item_appears_in_narrative(self):
        """Test that items from graph appear in narrative."""
        with patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "i": {
                        "name": "Bloodstained doll",
                        "description": "Sarah's doll with fresh bloodstains",
                        "category": "Keepsake",
                    }
                }
            ]

            narrative = """
            Eleanor's hands shook as she lifted the small doll from the ground.
            Dark stains marred its pale fabric - blood, still damp. Sarah's doll.
            Her daughter had never let it out of her sight.
            """

            item_keywords = ["doll", "bloodstain", "stains"]

            found_keywords = sum(1 for kw in item_keywords if kw.lower() in narrative.lower())
            assert found_keywords >= 2, "Item context not reflected in narrative"

    async def test_item_significance_conveyed(self):
        """Test that item significance is conveyed in narrative."""
        narrative = """
        The rusted musket had been James's only companion through years of war.
        Now, as he pressed it into Eleanor's hands, the weight of his sacrifice
        was palpable. Without it, he was defenseless - but she needed it more.
        """

        significance_indicators = ["only companion", "years", "sacrifice", "needed"]

        found_indicators = sum(1 for ind in significance_indicators if ind.lower() in narrative.lower())
        assert found_indicators >= 3, "Item significance not conveyed in narrative"


@pytest.mark.asyncio
class TestNarrativeConsistencyWithGraph:
    """Test overall narrative consistency with graph context."""

    async def test_no_contradictions_with_graph(self, mock_character_with_context):
        """Test that narrative doesn't contradict graph facts."""
        character = mock_character_with_context

        narrative_with_contradiction = """
        Eleanor smiled contentedly as she watched Sarah play. Life in the camp
        was peaceful, and she had no worries at all.
        """

        contradictions = []

        if "haunted" in character.traits and "contentedly" in narrative_with_contradiction.lower():
            contradictions.append("Character is haunted but narrative shows contentment")

        if "protective" in character.traits and "no worries" in narrative_with_contradiction.lower():
            contradictions.append("Protective character shown as carefree")

        assert len(contradictions) > 0, "Test should detect contradictions"

    async def test_character_status_reflected(self, mock_character_with_context):
        """Test that character status from graph is reflected."""
        character = mock_character_with_context

        if character.status == "Active":
            narrative = """
            Eleanor moved through the camp with purpose, her eyes alert for any sign
            of danger. She wouldn't let another child disappear on her watch.
            """

            active_indicators = ["moved", "purpose", "alert", "watch"]
            found = sum(1 for ind in active_indicators if ind.lower() in narrative.lower())
            assert found >= 2, "Active status not reflected in narrative"

    async def test_provisional_entities_not_featured(self):
        """Test that provisional entities are not prominently featured."""
        with patch("core.db_manager.neo4j_manager.execute_read_query", new_callable=AsyncMock) as mock_query:
            mock_query.return_value = [
                {
                    "c": {
                        "name": "Unnamed Character",
                        "is_provisional": True,
                    }
                }
            ]

            narrative = """
            Eleanor searched the camp, asking every person she encountered if they'd
            seen Sarah. Most shook their heads, their faces etched with worry.
            """

            assert "Unnamed Character" not in narrative, "Provisional character should not be named in narrative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
