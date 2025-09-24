# tests/test_kg_relationship_growth.py
from unittest.mock import AsyncMock

import pytest

from data_access import kg_queries


@pytest.mark.asyncio
async def test_relationships_are_unique_per_chapter(monkeypatch):
    """Ensure the same triple across chapters produces distinct relationships.

    This guards against MERGE flattening that would overwrite prior edges and
    cause the visible KG to appear to stagnate or shrink across chapters.
    """

    captured_batches = []

    async def fake_execute_cypher_batch(statements):  # type: ignore[override]
        captured_batches.append(statements)

    monkeypatch.setattr(
        kg_queries.neo4j_manager,
        "execute_cypher_batch",
        AsyncMock(side_effect=fake_execute_cypher_batch),
    )

    triple = {
        "subject": {"name": "Elara", "type": "Character"},
        "predicate": "LOCATED_IN",
        "object_entity": {"name": "Sunken Library", "type": "Location"},
        "is_literal_object": False,
    }

    # Add for chapter 1 and 2
    await kg_queries.add_kg_triples_batch_to_db(
        [triple], chapter_number=1, is_from_flawed_draft=False
    )
    await kg_queries.add_kg_triples_batch_to_db(
        [triple], chapter_number=2, is_from_flawed_draft=False
    )

    # We expect two batches (one per call)
    assert len(captured_batches) == 2

    def extract_rel_ids(batch):
        ids = []
        for query, params in batch:
            if "MERGE (s)-[r:`LOCATED_IN` {id: $rel_id_param}]->(o)" in query:
                ids.append(params.get("rel_id_param"))
        return ids

    rel_ids_ch1 = extract_rel_ids(captured_batches[0])
    rel_ids_ch2 = extract_rel_ids(captured_batches[1])

    # Ensure we created one LOCATED_IN relationship per batch with an id
    assert len(rel_ids_ch1) == 1 and rel_ids_ch1[0]
    assert len(rel_ids_ch2) == 1 and rel_ids_ch2[0]

    # Critical: they must be different across chapters
    assert rel_ids_ch1[0] != rel_ids_ch2[0]
