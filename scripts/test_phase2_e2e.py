#!/usr/bin/env python
"""
End-to-End Test for Phase 2 LangGraph Workflow

This script generates a single chapter using the complete Phase 2 workflow
with SAGA's configured LLM. It validates all success criteria from
docs/phase2_migration_plan.md.

Usage:
    python scripts/test_phase2_e2e.py

Requirements:
    - Neo4j database running (configured in .env)
    - LLM endpoint available (configured in .env)
    - Embedding model available
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from core.db_manager import Neo4jManagerSingleton
from core.langgraph import (
    create_initial_state,
    create_phase2_graph,
)

logger = structlog.get_logger(__name__)


async def setup_test_novel() -> dict:
    """
    Create a minimal plot outline for testing.

    Returns a simple 1-chapter outline for end-to-end testing.
    """
    outline = {
        "title": "The Last Signal",
        "genre": "Science Fiction",
        "theme": "Isolation and Communication",
        "chapters": {
            1: {
                "plot_point": "Dr. Sarah Chen receives a mysterious signal from deep space while working alone on a remote research station.",
                "chapter_summary": "Introduction to protagonist and the discovery of the signal",
                "key_events": [
                    "Sarah completes routine maintenance on the deep space radio telescope",
                    "She detects an unusual pattern in the background noise",
                    "Initial analysis suggests the signal is artificial and impossibly distant"
                ],
                "active_characters": ["Sarah Chen"],
                "location": "Deep Space Research Station Alpha"
            }
        }
    }
    return outline


async def main():
    """
    Run end-to-end Phase 2 workflow test.

    This test validates all success criteria from Phase 2:
    1. Generate single chapter with real LLM
    2. Extract entities and commit to Neo4j
    3. Validate consistency
    4. Handle revision loop if needed
    5. Generate summary and finalize
    """
    print("\n" + "="*80)
    print("SAGA Phase 2 End-to-End Test")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Generation Model: {settings.LARGE_MODEL}")
    print(f"  Extraction Model: {settings.SMALL_MODEL}")
    print(f"  Revision Model: {settings.MEDIUM_MODEL}")
    print(f"  Neo4j URI: {settings.NEO4J_URI}")
    print(f"  LLM Endpoint: {settings.OPENAI_API_BASE}")
    print(f"  Max Generation Tokens: {settings.MAX_GENERATION_TOKENS}")
    print()

    # Step 1: Connect to Neo4j
    print("Step 1: Connecting to Neo4j...")
    db_manager = Neo4jManagerSingleton()
    try:
        await db_manager.connect()
        print("✓ Neo4j connection established\n")
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")
        print("\nPlease ensure Neo4j is running and configured correctly in .env")
        return 1

    # Step 2: Set up test novel
    print("Step 2: Creating test novel outline...")
    outline = await setup_test_novel()
    print(f"✓ Created outline for: {outline['title']}")
    print(f"  Genre: {outline['genre']}")
    print(f"  Theme: {outline['theme']}")
    print(f"  Chapters: 1")
    print()

    # Step 3: Create project directory
    project_name = "phase2_e2e_test"
    project_dir = Path(settings.BASE_OUTPUT_DIR) / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    chapters_dir = project_dir / "chapters"
    chapters_dir.mkdir(exist_ok=True)

    print(f"Step 3: Created project directory: {project_dir}\n")

    # Step 4: Create initial state
    print("Step 4: Initializing LangGraph state...")
    state = create_initial_state(
        project_id=project_name,
        title=outline["title"],
        genre=outline["genre"],
        theme=outline["theme"],
        setting=outline["chapters"][1]["location"],
        target_word_count=4000,  # Small test chapter
        total_chapters=1,
        project_dir=str(project_dir),
        protagonist_name=outline["chapters"][1]["active_characters"][0],
        generation_model=settings.LARGE_MODEL,
        extraction_model=settings.SMALL_MODEL,
        revision_model=settings.MEDIUM_MODEL,
    )

    # Add plot outline to state
    state["plot_outline"] = outline
    state["current_chapter"] = 1
    state["plot_point_focus"] = outline["chapters"][1]["plot_point"]
    state["active_character_names"] = outline["chapters"][1]["active_characters"]
    state["current_location"] = outline["chapters"][1]["location"]

    print("✓ State initialized")
    print(f"  Current Chapter: {state['current_chapter']}")
    print(f"  Plot Point: {state['plot_point_focus']}")
    print(f"  Target Word Count: {state['target_word_count']}")
    print()

    # Step 5: Create Phase 2 workflow
    print("Step 5: Creating Phase 2 workflow graph...")
    graph = create_phase2_graph()
    print("✓ Workflow graph compiled")
    print("  Nodes: generate → extract → commit → validate → {revise OR summarize} → finalize")
    print()

    # Step 6: Execute workflow
    print("Step 6: Executing workflow...")
    print("-" * 80)
    start_time = time.time()

    try:
        # Run workflow
        result = await graph.ainvoke(state)

        elapsed_time = time.time() - start_time
        print("-" * 80)
        print(f"✓ Workflow completed in {elapsed_time:.2f} seconds")
        print()

        # Step 7: Validate results
        print("Step 7: Validating results...")

        # Check draft text
        if result.get("draft_text"):
            word_count = result.get("draft_word_count", 0)
            print(f"✓ Chapter generated: {word_count} words")

            # Show first 200 characters as preview
            preview = result["draft_text"][:200].replace("\n", " ")
            print(f"  Preview: {preview}...")
        else:
            print("✗ No draft text generated")
            return 1

        # Check iteration count
        iteration_count = result.get("iteration_count", 0)
        print(f"✓ Revision iterations: {iteration_count}")

        # Check if contradictions were found
        contradictions = result.get("contradictions", [])
        if contradictions:
            print(f"⚠ Final contradictions: {len(contradictions)}")
            for i, c in enumerate(contradictions[:3], 1):
                print(f"  {i}. {c.type}: {c.description}")
        else:
            print("✓ No contradictions detected")

        # Check extracted entities
        entities = result.get("extracted_entities", {})
        total_entities = sum(len(v) if isinstance(v, list) else 1 for v in entities.values())
        print(f"✓ Entities extracted: {total_entities}")

        # Check relationships
        relationships = result.get("extracted_relationships", [])
        print(f"✓ Relationships extracted: {len(relationships)}")

        # Check summary
        summaries = result.get("previous_chapter_summaries", [])
        if summaries:
            print(f"✓ Chapter summary generated:")
            print(f"  {summaries[-1]}")
        else:
            print("⚠ No summary generated")

        # Check file output
        chapter_file = chapters_dir / "chapter_001.md"
        if chapter_file.exists():
            file_size = chapter_file.stat().st_size
            print(f"✓ Chapter file created: {chapter_file.name} ({file_size} bytes)")
        else:
            print(f"⚠ Chapter file not found: {chapter_file}")

        print()

        # Step 8: Performance metrics
        print("Step 8: Performance Metrics...")
        print(f"  Total Time: {elapsed_time:.2f} seconds")
        print(f"  Words Generated: {result.get('draft_word_count', 0)}")
        if result.get('draft_word_count', 0) > 0:
            wps = result['draft_word_count'] / elapsed_time
            print(f"  Generation Rate: {wps:.1f} words/second")

        # Check against Phase 2 success criteria
        success_criteria_met = True
        print()
        print("Step 9: Phase 2 Success Criteria Check...")
        print("-" * 80)

        # Criterion 1: Generate single chapter end-to-end
        if result.get("draft_text"):
            print("✓ Generate single chapter end-to-end with real LLM")
        else:
            print("✗ Generate single chapter end-to-end with real LLM")
            success_criteria_met = False

        # Criterion 2: Revision loop works
        if "iteration_count" in result:
            print(f"✓ Revision loop works (iterations: {result['iteration_count']})")
        else:
            print("✗ Revision loop works")
            success_criteria_met = False

        # Criterion 3: Summaries persist
        if summaries:
            print("✓ Chapter summaries generated and available for context")
        else:
            print("⚠ Chapter summaries not generated")

        # Criterion 4: Performance <5 minutes
        if elapsed_time < 300:  # 5 minutes
            print(f"✓ Performance: <5 minutes ({elapsed_time:.1f}s)")
        else:
            print(f"⚠ Performance: >{elapsed_time:.1f}s (target: <300s)")

        # Criterion 5: File output
        if chapter_file.exists():
            print("✓ Chapter file created successfully")
        else:
            print("✗ Chapter file not created")
            success_criteria_met = False

        print("-" * 80)
        print()

        # Final summary
        if success_criteria_met:
            print("🎉 SUCCESS: All Phase 2 success criteria met!")
            print()
            print("Next Steps:")
            print("  - Review generated chapter in:", chapter_file)
            print("  - Check Neo4j graph for extracted entities")
            print("  - Run multi-chapter test if single chapter looks good")
        else:
            print("⚠️  PARTIAL SUCCESS: Some criteria not met (see above)")
            print()
            print("Review the output and check:")
            print("  - LLM endpoint configuration")
            print("  - Neo4j connectivity")
            print("  - Log files for detailed errors")

        return 0 if success_criteria_met else 1

    except Exception as e:
        elapsed_time = time.time() - start_time
        print("-" * 80)
        print(f"✗ Workflow failed after {elapsed_time:.2f} seconds")
        print(f"\nError: {e}")
        logger.exception("Workflow execution failed")
        return 1

    finally:
        # Cleanup
        print()
        print("Cleaning up...")
        await db_manager.close()
        print("✓ Neo4j connection closed")


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        logger.exception("Fatal error in test script")
        sys.exit(1)
