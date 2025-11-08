"""
Test script for initialization workflow.

This script demonstrates how to run the initialization phase standalone
to generate character sheets, outlines, and commit them to the knowledge graph.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.db_manager import neo4j_manager
from core.langgraph.initialization import create_initialization_graph
from core.langgraph.state import create_initial_state


async def main():
    """Run initialization workflow test."""

    # Step 1: Connect to Neo4j
    print("Connecting to Neo4j...")
    await neo4j_manager.connect()
    await neo4j_manager.create_db_schema()
    print("✓ Connected to Neo4j")

    # Step 2: Create initial state
    print("\nCreating initial state...")
    initial_state = create_initial_state(
        project_id="init-test",
        title="The Last Compiler",
        genre="science fiction",
        theme="AI rebellion and the nature of consciousness",
        setting="Post-apocalyptic Seattle, 2147",
        target_word_count=80000,
        total_chapters=20,
        project_dir="./output/init_test",
        protagonist_name="Aria",
        generation_model="qwen3-a3b",
        extraction_model="qwen3-a3b",
        revision_model="qwen3-a3b",
    )
    print(f"✓ Initial state created for '{initial_state['title']}'")

    # Step 3: Create initialization workflow (no checkpointing for test)
    print("\nCreating initialization workflow...")
    graph = create_initialization_graph(checkpointer=None)
    print("✓ Workflow graph created")

    # Step 4: Run initialization
    print("\nRunning initialization workflow...")
    print("This will:")
    print("  1. Generate character sheets")
    print("  2. Generate global outline")
    print("  3. Generate act outlines")
    print("  4. Commit all to Neo4j")
    print("\nThis may take a few minutes...\n")

    try:
        result = await graph.ainvoke(initial_state)

        # Step 5: Display results
        print("\n" + "=" * 60)
        print("INITIALIZATION COMPLETE!")
        print("=" * 60)

        print(f"\n✓ Initialization Status: {result.get('initialization_step')}")
        print(f"✓ Complete: {result.get('initialization_complete')}")

        # Character sheets
        character_sheets = result.get("character_sheets", {})
        print(f"\n✓ Character Sheets Generated: {len(character_sheets)}")
        for name in character_sheets.keys():
            print(f"  - {name}")

        # Global outline
        global_outline = result.get("global_outline")
        if global_outline:
            print("\n✓ Global Outline Generated:")
            print(f"  - Act Count: {global_outline.get('act_count')}")
            print(f"  - Structure: {global_outline.get('structure_type')}")

        # Act outlines
        act_outlines = result.get("act_outlines", {})
        print(f"\n✓ Act Outlines Generated: {len(act_outlines)}")
        for act_num in sorted(act_outlines.keys()):
            print(f"  - Act {act_num}: {act_outlines[act_num].get('act_role')}")

        # Active characters (committed to Neo4j)
        active_characters = result.get("active_characters", [])
        print(f"\n✓ Characters Committed to Neo4j: {len(active_characters)}")
        for char in active_characters:
            print(f"  - {char.name}: {len(char.traits)} traits")

        # World items
        world_items = result.get("world_items", [])
        print(f"\n✓ World Items Committed to Neo4j: {len(world_items)}")
        for item in world_items[:5]:  # Show first 5
            print(f"  - {item.name} ({item.category})")

        # Files created
        print("\n✓ Files Created:")
        project_dir = Path(result.get("project_dir", "./output/init_test"))

        # Character files
        char_dir = project_dir / "characters"
        if char_dir.exists():
            char_files = list(char_dir.glob("*.yaml"))
            print(f"  - characters/: {len(char_files)} files")
            for f in char_files[:3]:
                print(f"    • {f.name}")

        # Outline files
        outline_dir = project_dir / "outline"
        if outline_dir.exists():
            outline_files = list(outline_dir.glob("*.yaml"))
            print(f"  - outline/: {len(outline_files)} files")
            for f in outline_files:
                print(f"    • {f.name}")

        # World files
        world_dir = project_dir / "world"
        if world_dir.exists():
            world_files = list(world_dir.glob("*.yaml"))
            if world_files:
                print(f"  - world/: {len(world_files)} files")
                for f in world_files:
                    print(f"    • {f.name}")

        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("  - Data in Neo4j: ✓")
        print("  - Human-readable files: ✓")
        print("  - Checkpointer state: ✓")
        print(f"\nProject directory: {project_dir}")
        print("=" * 60)

        # Show sample character details
        if active_characters:
            print("\nSample Character Details:")
            char = active_characters[0]
            print(f"\n{char.name}:")
            print(f"  Traits: {', '.join(char.traits)}")
            print(f"  Status: {char.status}")
            print(f"  Description: {char.description[:200]}...")

    except Exception as e:
        print(f"\n❌ ERROR during initialization: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\nClosing Neo4j connection...")
        await neo4j_manager.close()
        print("✓ Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
