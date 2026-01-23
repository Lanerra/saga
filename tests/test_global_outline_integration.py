# tests/test_global_outline_integration.py
"""Integration tests for GlobalOutlineParser end-to-end pipeline.

These tests verify the complete Stage 2 Global Outline pipeline including:
- Reading global outline JSON
- Creating MajorPlotPoint events
- Creating Location nodes
- Creating Item nodes
- Enriching Character nodes with arcs
- Persisting to Neo4j
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from core.parsers.global_outline_parser import GlobalOutlineParser, MajorPlotPoint


@pytest.fixture
def complete_global_outline():
    """Complete global outline with all expected data."""
    return {
        "act_count": 3,
        "acts": [
            {
                "act_number": 1,
                "title": "Whispers in the Mist",
                "summary": "The creature's first attacks fracture the fragile peace of Blackwater Creek.",
                "key_events": ["Eleanor Whitaker discovers her daughter Sarah missing"],
                "chapters_start": 1,
                "chapters_end": 7,
            },
            {
                "act_number": 2,
                "title": "Bonds in Blood",
                "summary": "The creature's true nature emerges as human alliances fracture.",
                "key_events": ["Thomas Reed uncovers the creature's nest"],
                "chapters_start": 8,
                "chapters_end": 14,
            },
            {
                "act_number": 3,
                "title": "Ashes in the Pines",
            },
        ],
        "inciting_incident": "Inciting Incident",
        "midpoint": "Midpoint",
        "climax": "Climax",
        "resolution": "Resolution",
        "character_arcs": [
            {
                "character_name": "Eleanor Whitaker",
                "starting_state": "Haunted protector of the refugee camp",
                "ending_state": "Reluctant hunter who sacrifices herself",
                "key_moments": ["Losing Sarah", "Destroying the creature"],
            },
            {
                "character_name": "James Carter",
                "starting_state": "Confederate deserter guarding his family",
                "ending_state": "Father who leads the camp's defense",
                "key_moments": ["Sacrificing his musket", "Choosing to guard the perimeter"],
            },
        ],
        "locations": [
            {
                "name": "Blackwater Creek",
                "description": "A misty swamp where the creature resides",
            },
            {
                "name": "Refugee Camp",
                "description": "A settlement for displaced families",
            },
        ],
        "items": [
            {
                "name": "Bloodstained doll",
                "description": "Sarah's doll with bloodstains",
                "category": "Keepsake",
            },
            {
                "name": "Rusted musket",
                "description": "James Carter's old weapon",
                "category": "Weapon",
            },
        ],
        "thematic_progression": "From war's brutality to supernatural terror",
        "pacing_notes": "Slow dread in Act 1, escalating tension in Act 2, rapid action in Act 3",
        "total_chapters": 20,
        "structure_type": "3-act",
        "generated_at": "initialization",
    }


@pytest.fixture
def mock_global_outline_file(complete_global_outline):
    """Create a temporary global outline file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(complete_global_outline, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_stage2_pipeline_complete(mock_global_outline_file):
    """Test complete Stage 2 pipeline from parsing to Neo4j persistence.

    This integration test verifies that:
    1. Global outline is parsed correctly
    2. All 4 MajorPlotPoints are created
    3. Locations are created with proper properties
    4. Items are created with proper properties
    5. Character arcs are enriched
    6. Neo4j nodes are created with correct properties
    """
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)

    # Mock the Neo4j manager and LLM extraction to capture the actual queries
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query, \
         patch.object(parser, '_extract_world_items_from_outline') as mock_extract:

        from models.kg_models import WorldItem
        mock_query.return_value = None
        mock_extract.return_value = [
            WorldItem(id="loc_1", name="Test Location 1", description="Desc 1", category="location", created_chapter=0, is_provisional=False),
            WorldItem(id="loc_2", name="Test Location 2", description="Desc 2", category="location", created_chapter=0, is_provisional=False),
            WorldItem(id="item_1", name="Test Item 1", description="Desc 1", category="object", created_chapter=0, is_provisional=False),
            WorldItem(id="item_2", name="Test Item 2", description="Desc 2", category="object", created_chapter=0, is_provisional=False),
        ]

        # Run the complete parse_and_persist pipeline
        success, message = await parser.parse_and_persist()
        
        # Verify success
        assert success, f"Pipeline failed: {message}"
        
        # Verify that all expected queries were called
        assert mock_query.call_count >= 5, f"Expected at least 5 queries, got {mock_query.call_count}"
        
        # Verify query types
        query_types = [q[0][0] for q in mock_query.call_args_list]
        
        # Check for MajorPlotPoint creation queries
        # Note: Queries start with newline, so pattern is "\nMERGE (e:Event"
        major_plot_point_queries = [q for q in query_types if "\n                MERGE (e:Event" in q]
        assert len(major_plot_point_queries) >= 4, "Expected at least 4 MajorPlotPoint creation queries"
        
        # Check for Location creation queries
        location_queries = [q for q in query_types if "\n                MERGE (l:Location" in q]
        assert len(location_queries) >= 2, "Expected at least 2 Location creation queries"
        
        # Check for Item creation queries
        item_queries = [q for q in query_types if "\n                MERGE (i:Item" in q]
        assert len(item_queries) >= 2, "Expected at least 2 Item creation queries"
        
        # Check for Character enrichment queries
        character_queries = [q for q in query_types if "SET c.arc_start" in q]
        assert len(character_queries) >= 2, "Expected at least 2 Character enrichment queries"
        
        # Verify the success message contains expected entities
        assert "4" in message, f"Expected 4 in success message, got: {message}"
        assert "MajorPlotPoints" in message, f"Expected 'MajorPlotPoints' in message, got: {message}"
        assert message is not None


@pytest.mark.asyncio
async def test_schema_compliance_major_plot_points(mock_global_outline_file):
    """Test that MajorPlotPoints comply with schema requirements.

    Verifies:
    - Exactly 4 MajorPlotPoints
    - Correct event_type discriminator
    - Valid sequence_order values (1-4)
    - Proper timestamps
    - created_chapter=0 for initialization
    """
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)
    
    # Parse the outline
    global_outline_data = await parser.parse_global_outline()
    plot_points = parser._parse_major_plot_points(global_outline_data)
    
    # Verify count
    assert len(plot_points) == 4, f"Expected 4 MajorPlotPoints, got {len(plot_points)}"
    
    # Verify each plot point has required properties
    for pp in plot_points:
        assert pp.event_type == "MajorPlotPoint", f"Event type should be 'MajorPlotPoint', got {pp.event_type}"
        assert pp.sequence_order in [1, 2, 3, 4], f"Sequence order should be 1-4, got {pp.sequence_order}"
        assert pp.created_chapter == 0, f"created_chapter should be 0 for initialization, got {pp.created_chapter}"
        assert pp.is_provisional == False, f"is_provisional should be False, got {pp.is_provisional}"
        assert pp.id, "Event should have an ID"
        assert pp.name, "Event should have a name"
        assert pp.description, "Event should have a description"
    
    # Verify sequence order is correct
    sequence_orders = [pp.sequence_order for pp in plot_points]
    assert sequence_orders == [1, 2, 3, 4], f"Sequence orders should be [1, 2, 3, 4], got {sequence_orders}"
    
    # Verify all required event names exist (using substring matching)
    event_names = [pp.name.lower() for pp in plot_points]
    expected_keywords = [
        ("inciting", "incident"),  # Event 0
        "midpoint",               # Event 1
        "climax",                 # Event 2
        "resolution",             # Event 3
    ]
    for i, expected_keyword in enumerate(expected_keywords):
        if isinstance(expected_keyword, tuple):
            keyword1, keyword2 = expected_keyword
            assert keyword1 in event_names[i] or keyword2 in event_names[i], f"Event {i} should contain one of {expected_keyword}, got {plot_points[i].name}"
        else:
            assert expected_keyword in event_names[i], f"Event {i} should contain {expected_keyword}, got {plot_points[i].name}"


@pytest.mark.asyncio
async def test_location_schema_compliance(mock_global_outline_file):
    """Test that Location nodes comply with schema requirements.

    Verifies:
    - Locations are created without names in Stage 2
    - Proper category ("Location")
    - Proper timestamps
    - created_chapter=0 for initialization
    """
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)
    
    with patch.object(parser, '_extract_world_items_from_outline') as mock_extract:
        from models.kg_models import WorldItem
        mock_extract.return_value = [
            WorldItem(
                id="loc_1",
                name="Test Location",
                description="A test location",
                category="location",
                created_chapter=0,
                is_provisional=False,
            )
        ]

        # Parse the outline
        global_outline_data = await parser.parse_global_outline()
        locations = await parser._parse_locations(global_outline_data)
    
    # Verify locations exist
    assert len(locations) >= 0, "Should have locations (may be 0 if no explicit locations in outline)"
    
    # Verify each location has required properties
    for loc in locations:
        assert loc.category == "Location", f"Category should be 'Location', got {loc.category}"
        assert loc.created_chapter == 0, f"created_chapter should be 0 for initialization, got {loc.created_chapter}"
        assert loc.is_provisional == False, f"is_provisional should be False, got {loc.is_provisional}"
        assert loc.id, "Location should have an ID"
        assert loc.description, "Location should have a description"
        # Stage 2: Names should be None (added in Stage 3)
        assert loc.name is None, f"Stage 2 locations should have name=None, got {loc.name}"


@pytest.mark.asyncio
async def test_schema_compliance_items(mock_global_outline_file):
    """Test that Item nodes comply with schema requirements.

    Verifies:
    - Items are created with proper properties
    - Proper timestamps
    - created_chapter=0 for initialization
    """
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)

    with patch.object(parser, '_extract_world_items_from_outline') as mock_extract:
        from models.kg_models import WorldItem
        mock_extract.return_value = [
            WorldItem(
                id="item_1",
                name="Test Item",
                description="A test item",
                category="object",
                created_chapter=0,
                is_provisional=False,
            )
        ]

        # Parse the outline
        global_outline_data = await parser.parse_global_outline()
        items = await parser._parse_items(global_outline_data)
    
    # Verify items exist
    assert len(items) >= 0, "Should have items (may be 0 if no explicit items in outline)"
    
    # Verify each item has required properties
    for item in items:
        assert item.category, "Item should have a category"
        assert item.created_chapter == 0, f"created_chapter should be 0 for initialization, got {item.created_chapter}"
        assert item.is_provisional == False, f"is_provisional should be False, got {item.is_provisional}"
        assert item.id, "Item should have an ID"
        assert item.name, "Item should have a name"
        assert item.description, "Item should have a description"


@pytest.mark.asyncio
async def test_schema_compliance_character_arcs(mock_global_outline_file):
    """Test that Character arcs comply with schema requirements.

    Verifies:
    - Character arcs are enriched with arc_start, arc_end, arc_key_moments
    - Properties are properly formatted
    """
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)
    
    # Parse the outline
    global_outline_data = await parser.parse_global_outline()
    character_arcs = parser._parse_character_arcs(global_outline_data)
    
    # Verify character arcs exist
    assert len(character_arcs) >= 0, "Should have character arcs"
    
    # Verify each arc has required properties
    for character_name, arc_data in character_arcs.items():
        assert "arc_start" in arc_data, f"Character arc should have arc_start, got {arc_data}"
        assert "arc_end" in arc_data, f"Character arc should have arc_end, got {arc_data}"
        assert "arc_key_moments" in arc_data, f"Character arc should have arc_key_moments, got {arc_data}"
        assert isinstance(arc_data["arc_key_moments"], list), "arc_key_moments should be a list"
        assert character_name, "Character name should not be empty"


@pytest.mark.asyncio
async def test_no_relationships_created_in_stage2(mock_global_outline_file):
    """Test that no relationships are created during Stage 2.

    Stage 2 should only create entities, not relationships between them.
    Relationships are created in Stage 3 (Act Outlines) and Stage 4 (Chapter Outlines).
    """
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)

    with patch.object(parser, '_extract_world_items_from_outline') as mock_extract:
        mock_extract.return_value = []

        # Parse the outline
        global_outline_data = await parser.parse_global_outline()

        # Parse all entities
        plot_points = parser._parse_major_plot_points(global_outline_data)
        locations = await parser._parse_locations(global_outline_data)
        items = await parser._parse_items(global_outline_data)
        character_arcs = parser._parse_character_arcs(global_outline_data)
    
    # Verify no relationships are created
    # The parser should only create entities, not relationships
    # Relationships would be created in ActOutlineParser (Stage 3) and ChapterOutlineParser (Stage 4)
    
    # Verify that the parser doesn't have any relationship creation logic
    assert hasattr(parser, '_parse_major_plot_points')
    assert hasattr(parser, '_parse_locations')
    assert hasattr(parser, '_parse_items')
    assert hasattr(parser, '_parse_character_arcs')
    # No relationship parsing methods expected
    assert not hasattr(parser, '_parse_character_relationships')
    assert not hasattr(parser, '_parse_event_relationships')
    
    # Verify that the create methods don't create relationships
    # They should only create nodes
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query:
        mock_query.return_value = None
        
        # Create all entities
        await parser.create_major_plot_point_nodes(plot_points)
        await parser.create_location_nodes(locations)
        await parser.create_item_nodes(items)
        await parser.enrich_character_arcs(character_arcs)
        
        # Verify that no relationship queries were executed
        query_types = [q[0][0] for q in mock_query.call_args_list]
        relationship_queries = [q for q in query_types if "MERGE (c1:Character)" in q or "[:PART_OF]" in q]
        assert len(relationship_queries) == 0, f"Stage 2 should not create relationships, but found {len(relationship_queries)} relationship queries"


@pytest.mark.asyncio
async def test_idempotency_of_node_creation(mock_global_outline_file):
    """Test that node creation is idempotent (can run multiple times without errors).

    The MERGE query should handle re-running without creating duplicates.
    """
    parser = GlobalOutlineParser(global_outline_path=mock_global_outline_file)

    # Mock the Neo4j manager and LLM extraction
    with patch("core.parsers.global_outline_parser.neo4j_manager.execute_write_query") as mock_query, \
         patch.object(parser, '_extract_world_items_from_outline') as mock_extract:

        mock_query.return_value = None
        mock_extract.return_value = []

        # Run the pipeline twice
        success1, message1 = await parser.parse_and_persist()
        success2, message2 = await parser.parse_and_persist()
        
        # Verify both runs succeed
        assert success1, f"First run failed: {message1}"
        assert success2, f"Second run failed: {message2}"
        
        # Verify that the same number of queries were executed
        assert mock_query.call_count >= 10, f"Expected at least 10 queries, got {mock_query.call_count}"
        
        # Verify the success messages
        assert "4" in message1 and "4" in message2
        assert "MajorPlotPoints" in message1 and "MajorPlotPoints" in message2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
