"""Location persistence for outline parser projections."""

from core.service_context import get_services
from models.kg_models import Location


async def persist_locations(locations: list[Location]) -> None:
    """Write locations in order, propagating failure to the parser boundary."""
    query = """
    MERGE (l:Location {id: $id})
    ON CREATE SET
        l.name = $name,
        l.description = $description,
        l.category = $category,
        l.created_chapter = $created_chapter,
        l.is_provisional = $is_provisional,
        l.created_ts = timestamp(),
        l.updated_ts = timestamp()
    ON MATCH SET
        l.name = $name,
        l.description = $description,
        l.category = $category,
        l.created_chapter = $created_chapter,
        l.is_provisional = $is_provisional,
        l.updated_ts = timestamp()
    """
    parameters = [
        {
            "id": location.id,
            "name": location.name,
            "description": location.description,
            "category": location.category,
            "created_chapter": location.created_chapter,
            "is_provisional": location.is_provisional,
        }
        for location in locations
    ]
    for values in parameters:
        await get_services().database.execute_write_query(query, values)
