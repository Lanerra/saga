import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.db_manager import neo4j_manager  # Use the singleton instance

async def migrate_legacy_world_elements():
    """Migrate legacy WorldElements with missing core fields."""
    await neo4j_manager.connect()
    
    # Find all invalid entries (missing category, name, or id) and get their internal ID
    query = """
    MATCH (we:WorldElement)
    WHERE we.category IS NULL OR we.name IS NULL OR we.id IS NULL
    RETURN id(we) AS node_id, we.name AS name, we.category AS category, we.id AS current_id
    """
    results = await neo4j_manager.execute_read_query(query)

    migrated_count = 0
    for row in results:
        node_id = row['node_id']
        name = row['name']
        category = row['category']
        current_id = row['current_id']
        
        # Set default values for missing fields
        if not name:
            name = f"unnamed_element_{node_id}"
        if not category:
            category = 'other'
        
        # Generate normalized ID using existing utility
        from utils import _normalize_for_id
        # Use existing ID if valid, otherwise generate a new one
        if current_id and isinstance(current_id, str) and current_id.strip():
            normalized_id = current_id
        else:
            normalized_id = f"{_normalize_for_id(category)}_{_normalize_for_id(name)}"
            # Ensure the generated ID is not empty
            if not normalized_id or normalized_id == "_":
                normalized_id = f"element_{node_id}"
        
        # Update the node with required fields
        update_query = """
        MATCH (we:WorldElement)
        WHERE id(we) = $node_id
        SET we.category = $category,
            we.name = $name,
            we.id = $normalized_id,
            we.is_provisional = true
        """
        await neo4j_manager.execute_write_query(
            update_query,
            {
                "node_id": node_id,
                "category": category,
                "name": name,
                "normalized_id": normalized_id
            }
        )
        migrated_count += 1

    print(f"Successfully migrated {migrated_count} legacy WorldElements.")

if __name__ == "__main__":
    asyncio.run(migrate_legacy_world_elements())