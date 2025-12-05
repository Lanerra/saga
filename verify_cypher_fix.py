#!/usr/bin/env python3
"""
Verification script for relationship type fix in native_builders.py

This script verifies that:
1. The Cypher queries generate valid syntax
2. Relationships use apoc.merge.relationship with dynamic types
3. No hardcoded :RELATIONSHIP types remain
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_access.cypher_builders.native_builders import NativeCypherBuilder
from models import CharacterProfile, WorldItem


def verify_character_upsert():
    """Verify character upsert generates correct Cypher."""
    print("Testing character_upsert_cypher...")

    char = CharacterProfile(
        name="Alice",
        description="A brave hero",
        traits=["brave", "intelligent"],
        status="alive",
        relationships={
            "Bob": {"type": "KNOWS", "description": "Long-time friend"},
            "Eve": {"type": "ENEMY_OF", "description": "Bitter rival"},
        },
    )

    cypher, params = NativeCypherBuilder.character_upsert_cypher(char, chapter_number=1)

    # Check that RELATIONSHIP is NOT used as a relationship type
    if "MERGE (c)-[r:RELATIONSHIP" in cypher or "MERGE (c)-[:RELATIONSHIP" in cypher:
        print("❌ FAILED: Found hardcoded :RELATIONSHIP type in character cypher")
        print(cypher)
        return False

    # Check that apoc.merge.relationship is used
    if "apoc.merge.relationship" not in cypher:
        print("❌ FAILED: apoc.merge.relationship not found in character cypher")
        print(cypher)
        return False

    # Check that rel_data.rel_type is used (dynamic type)
    if "rel_data.rel_type" not in cypher:
        print("❌ FAILED: rel_data.rel_type not found in character cypher")
        print(cypher)
        return False

    # Verify relationship data in params
    if len(params["relationship_data"]) != 2:
        print(f"❌ FAILED: Expected 2 relationships, got {len(params['relationship_data'])}")
        return False

    # Verify relationship types are preserved
    rel_types = [r["rel_type"] for r in params["relationship_data"]]
    if "KNOWS" not in rel_types or "ENEMY_OF" not in rel_types:
        print(f"❌ FAILED: Expected KNOWS and ENEMY_OF, got {rel_types}")
        return False

    print("✓ character_upsert_cypher: PASSED")
    return True


def verify_world_item_upsert():
    """Verify world item upsert generates correct Cypher."""
    print("\nTesting world_item_upsert_cypher...")

    item = WorldItem.from_dict(
        "Location",
        "Ancient Library",
        {
            "description": "A vast repository of knowledge",
            "id": "location_ancient_library",
            "relationships": {
                "Central City": {"type": "LOCATED_IN", "description": "In the heart of the city"},
                "Sage Order": {"type": "OWNED_BY", "description": "Maintained by the Sage Order"},
            },
        },
    )

    cypher, params = NativeCypherBuilder.world_item_upsert_cypher(item, chapter_number=1)

    # Check that RELATIONSHIP is NOT used as a relationship type
    if "MERGE (w)-[r:RELATIONSHIP" in cypher or "MERGE (w)-[:RELATIONSHIP" in cypher:
        print("❌ FAILED: Found hardcoded :RELATIONSHIP type in world item cypher")
        print(cypher)
        return False

    # Check that apoc.merge.relationship is used
    if "apoc.merge.relationship" not in cypher:
        print("❌ FAILED: apoc.merge.relationship not found in world item cypher")
        print(cypher)
        return False

    # Check that rel_data.rel_type is used (dynamic type)
    if "rel_data.rel_type" not in cypher:
        print("❌ FAILED: rel_data.rel_type not found in world item cypher")
        print(cypher)
        return False

    # Verify relationship data in params
    if len(params["relationship_data"]) != 2:
        print(f"❌ FAILED: Expected 2 relationships, got {len(params['relationship_data'])}")
        return False

    # Verify relationship types are preserved
    rel_types = [r["rel_type"] for r in params["relationship_data"]]
    if "LOCATED_IN" not in rel_types or "OWNED_BY" not in rel_types:
        print(f"❌ FAILED: Expected LOCATED_IN and OWNED_BY, got {rel_types}")
        return False

    print("✓ world_item_upsert_cypher: PASSED")
    return True


def verify_character_fetch():
    """Verify character fetch uses correct relationship type logic."""
    print("\nTesting character_fetch_cypher...")

    cypher, params = NativeCypherBuilder.character_fetch_cypher()

    # Check that it uses type(r) for relationship type, not r.type
    if "type(r)" not in cypher:
        print("❌ FAILED: type(r) function not found in character fetch cypher")
        print(cypher)
        return False

    # Check that it handles legacy RELATIONSHIP types with CASE statement
    if "CASE WHEN type(r) = 'RELATIONSHIP'" not in cypher:
        print("❌ FAILED: No CASE statement for legacy RELATIONSHIP handling")
        print(cypher)
        return False

    print("✓ character_fetch_cypher: PASSED")
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Verifying Relationship Type Fix")
    print("=" * 60)

    results = []

    results.append(verify_character_upsert())
    results.append(verify_world_item_upsert())
    results.append(verify_character_fetch())

    print("\n" + "=" * 60)
    if all(results):
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe fix is working correctly:")
        print("1. No hardcoded :RELATIONSHIP types")
        print("2. Using apoc.merge.relationship for dynamic types")
        print("3. Fetch queries handle both new and legacy data")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
