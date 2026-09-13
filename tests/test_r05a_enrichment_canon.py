"""Strict synthetic regressions at database/provider transport boundaries."""
import asyncio
from copy import deepcopy
from inspect import getfile
from pathlib import Path
from typing import Any

import pytest

import config
from core.exceptions import DatabaseError, LLMServiceError
from core.langgraph.chapter_lifecycle import ChapterLifecycle
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.context_character_retrieval import get_scene_character_context
from core.langgraph.nodes.context_retrieval_node import retrieve_context
from core.langgraph.nodes.context_scene_retrieval import get_semantic_context
from core.langgraph.nodes.narrative_enrichment_node import enrich_narrative
from core.langgraph.state import NarrativeState
from core.parsers.narrative_enrichment_parser import NarrativeEnrichmentParser
from core.service_context import get_services
from data_access.scene_queries import get_act_events
from models.kg_constants import CHARACTER_EMOTIONAL_RELATIONSHIPS, CHARACTER_SOCIAL_RELATIONSHIPS
from prompts import prompt_data_getters
from tests.test_langgraph.test_chapter_lifecycle import example_state


@pytest.fixture
def graph_reads(owned_graph_cache: None, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Route only declared database reads; all consumers and converters stay real."""
    data: dict[str, Any] = {"characters": [], "relationships": [], "fail": None, "calls": [], "provisional": False}
    prompt_data_getters.clear_context_cache()

    async def read(query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        parameters = parameters or {}
        if "c.number AS number" in query:
            kind = "chapter"
            rows = [{"id": "chapter_1", "number": 1, "title": "Synthetic", "summary": "", "act_number": 1}]
        elif "RETURN c," in query or "RETURN\n            c," in query or "RETURN c\n" in query:
            kind = "profile"
            rows = [{"c": character, "traits": [], "relationships": []} for character in data["characters"] if "name" not in parameters or character["name"] == parameters["name"]]
            if "include_provisional" in parameters:
                assert parameters["include_provisional"] is False
        elif "AS current_status" in query:
            kind = "status"
            assert parameters["chapter_limit_param"] >= 0
            rows = [{"current_status": "Alive", "char_is_provisional": data["provisional"], "provisional_rel_count": 0}] if data["characters"] else []
        elif "type(r) AS predicate" in query:
            kind = "relationships"
            assert "coalesce(r.is_provisional, FALSE) = FALSE" in query
            assert "s.name = $subject_param" in query
            rows = [row for row in data["relationships"] if row["subject"] == parameters["subject_param"] and row["chapter_added"] <= parameters["chapter_limit_param"] and not row["is_provisional"]]
        elif "[r:`LOCATED_AT`]" in query:
            kind = "location"
            assert "r.is_provisional = FALSE OR r.is_provisional IS NULL" in query
            assert parameters["chapter_limit_param"] >= 0
            rows = []
        elif "MATCH (ni:NovelInfo" in query:
            kind = "novel"
            rows = []
        elif "shortestPath" in query:
            kind = "proximity"
            rows = []
        else:
            raise AssertionError(f"Unexpected database read: {query}")
        data["calls"].append(kind)
        if data["fail"] == kind:
            raise DatabaseError(f"synthetic {kind} failure")
        return deepcopy(rows)

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    return data


@pytest.mark.parametrize("header", ["Appearance", "Physical Description", "aPpEaRaNcE"])
@pytest.mark.parametrize("names", [("Alice", "Bob"), ("Alice Smith", "Bob Jones")])
async def test_repeated_headers_retain_distinct_enrichment_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph_reads: dict[str, Any], header: str, names: tuple[str, str],
) -> None:
    assert Path(getfile(NarrativeEnrichmentParser)).resolve() == Path(__file__).resolve().parents[1] / "core/parsers/narrative_enrichment_parser.py"
    monkeypatch.setattr(config, "ENABLE_PHYSICAL_DESCRIPTION_EXTRACTION", True)
    monkeypatch.setattr(config, "ENABLE_CHAPTER_EMBEDDING_EXTRACTION", False)
    graph_reads["characters"] = [{"id": identifier, "name": name, "created_chapter": 0} for identifier, name in zip(("alice-id", "bob-id"), names, strict=True)]
    text = f"{names[0]}\n{header}: tall\n{names[1]}\n{header}: short"
    state = example_state(tmp_path)
    state["draft_ref"] = ContentManager(str(tmp_path)).save_text(text, "draft", "r05a")
    original = deepcopy(state)
    assert await enrich_narrative(state) == {"current_node": "narrative_enrichment", "last_error": None}
    assert ChapterLifecycle(state).enrichment() == {
        "descriptions": [
            {"character_id": "alice-id", "character_name": names[0], "description": "tall"},
            {"character_id": "bob-id", "character_name": names[1], "description": "short"},
        ],
        "embeddings": [],
    }
    assert state == original
    assert graph_reads["characters"] == [{"id": identifier, "name": name, "created_chapter": 0} for identifier, name in zip(("alice-id", "bob-id"), names, strict=True)]
    extracted = await NarrativeEnrichmentParser(text, chapter_number=1).extract_physical_descriptions()
    assert [(row.character_name, row.extracted_description, row.source_text) for row in extracted] == [
        (names[0], "tall", f"{header}: tall"), (names[1], "short", f"{header}: short"),
    ]


def relationship(subject: str, predicate: str, target: str, *, chapter: int = 0, provisional: bool = False) -> dict[str, Any]:
    return {"subject": subject, "predicate": predicate, "object": target, "object_type": "Character", "chapter_added": chapter, "confidence": 0.9, "is_provisional": provisional}


async def test_canonical_relationships_keep_direction_multiplicity_and_cutoff(graph_reads: dict[str, Any]) -> None:
    supported = sorted((CHARACTER_SOCIAL_RELATIONSHIPS | CHARACTER_EMOTIONAL_RELATIONSHIPS) - {"LOCATED_AT"})
    graph_reads["relationships"] = [relationship("Alice", predicate, "Bob") for predicate in supported] + [
        relationship("Bob", "MENTORS", "Alice"), relationship("Alice", "TRUSTS", "Carol", provisional=True),
        relationship("Alice", "TRUSTS", "Future", chapter=2), relationship("Alice", "ally_of", "Legacy"),
        relationship("Alice", "PART_OF", "Chapter"),
    ]
    facts: list[str] = []
    await prompt_data_getters._gather_character_facts({"Alice", "Bob"}, 1, facts, 100, 100, "Alice")
    assert facts == [f"- Alice {predicate.replace('_', ' ')} Bob." for predicate in supported] + ["- Bob MENTORS Alice."]
    assert graph_reads["calls"].count("relationships") == 2


async def test_fact_prompt_preserves_both_canonical_relationships(graph_reads: dict[str, Any]) -> None:
    graph_reads["relationships"] = [relationship("Alice", "ALLIES_WITH", "Bob"), relationship("Alice", "TRUSTS", "Bob")]
    assert await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=2, protagonist_name="Alice") == (
        "**Key Reliable KG Facts (from Neo4j - up to previous chapter/initial state):**\n- Alice ALLIES WITH Bob.\n- Alice TRUSTS Bob."
    )


@pytest.mark.parametrize("failed_read", ["status", "location", "relationships", "novel"])
async def test_required_fact_failure_propagates(graph_reads: dict[str, Any], failed_read: str) -> None:
    graph_reads["fail"] = failed_read
    with pytest.raises(DatabaseError):
        await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=2, protagonist_name="Alice")
    assert failed_read in graph_reads["calls"]


@pytest.mark.parametrize("characters", [{"Alice", "Bob"}, {"Alice", "Bob", "Carol", "Dave"}])
async def test_proximity_failure_is_not_a_valid_nonmatch(graph_reads: dict[str, Any], characters: set[str]) -> None:
    graph_reads["fail"] = "proximity"
    with pytest.raises(DatabaseError):
        await prompt_data_getters._apply_protagonist_proximity_filtering(characters, "Alice")


async def test_required_fact_failure_settles_sibling_reads(graph_reads: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    database = get_services().database
    original_read = database.execute_read_query
    completed: list[str] = []

    async def read(query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if "AS current_status" in query:
            raise DatabaseError("synthetic early failure")
        await asyncio.sleep(0.01)
        result = await original_read(query, parameters)
        completed.append(query)
        return result

    monkeypatch.setattr(database, "execute_read_query", read)
    try:
        with pytest.raises(DatabaseError, match="synthetic early failure"):
            await prompt_data_getters._gather_character_facts({"Alice"}, 1, [], 2, 7, "Alice")
        assert len(completed) == 2
    finally:
        await asyncio.sleep(0.02)


async def test_profile_failure_propagates_through_scene_consumer(tmp_path: Path, graph_reads: dict[str, Any]) -> None:
    graph_reads["fail"] = "profile"
    manager = ContentManager(str(tmp_path))
    with pytest.raises(DatabaseError, match="synthetic profile failure"):
        await get_scene_character_context({}, {"characters": ["Alice"]}, 2, "synthetic", manager)


@pytest.mark.parametrize("failed_read", ["profile", "relationships"])
async def test_retrieval_node_retains_critical_context_failure(tmp_path: Path, graph_reads: dict[str, Any], monkeypatch: pytest.MonkeyPatch, failed_read: str) -> None:
    monkeypatch.setattr(config, "DEFAULT_PROTAGONIST_NAME", "Alice")
    graph_reads["fail"] = failed_read
    manager = ContentManager(str(tmp_path))
    state: NarrativeState = {
        "project_dir": str(tmp_path), "current_chapter": 2, "current_scene_index": 0,
        "chapter_plan_ref": manager.save_json([{"title": "", "characters": ["Alice"]}], "chapter_plan", "r05a"),
    }
    original = deepcopy(state)
    result = await retrieve_context(state)
    assert set(result) == {"has_fatal_error", "last_error", "error_node", "current_node"}
    assert result["has_fatal_error"] is True
    assert result["error_node"] == result["current_node"] == "retrieve_context"
    error = result["last_error"]
    assert isinstance(error, str)
    assert error.startswith("Failed to retrieve character profiles:" if failed_read == "profile" else "Failed to retrieve KG facts:")
    assert state == original


async def test_valid_empty_reads_remain_empty_context(tmp_path: Path, graph_reads: dict[str, Any]) -> None:
    assert await prompt_data_getters.get_filtered_character_profiles_for_prompt_plain_text(["Alice"]) == "No character profiles available."
    assert await get_scene_character_context({}, {"characters": ["Alice"]}, 2, "synthetic", ContentManager(str(tmp_path))) is None
    assert await prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt(chapter_number=2, protagonist_name="Alice") == (
        "No specific reliable KG facts identified as highly relevant for this chapter's current focus from Neo4j."
    )


async def test_provisional_status_is_not_advertised_as_confirmed(graph_reads: dict[str, Any]) -> None:
    graph_reads["characters"] = [{"id": "alice-id", "name": "Alice", "created_chapter": 0}]
    graph_reads["provisional"] = True
    facts: list[str] = []
    await prompt_data_getters._gather_character_facts({"Alice"}, 1, facts, 2, 7, "Alice")
    assert facts == ["- Alice's status is: Alive (provisional)."]


@pytest.mark.parametrize("failure", [None, LLMServiceError("synthetic embedding failure")])
async def test_optional_semantic_embedding_fallback_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: LLMServiceError | None) -> None:
    calls: list[str] = []

    async def embed(text: str) -> None:
        calls.append(text)
        if failure:
            raise failure
        return None

    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    assert await get_semantic_context({}, "Synthetic arrival", 2, "synthetic", ContentManager(str(tmp_path))) is None
    assert calls == ["Synthetic arrival"]


@pytest.mark.parametrize("failed", [False, True])
async def test_r01_scene_reader_dependency_distinguishes_failure_from_empty(monkeypatch: pytest.MonkeyPatch, failed: bool) -> None:
    async def read(query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        assert "Event" in query
        if failed:
            raise DatabaseError("synthetic act failure")
        return []

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    if failed:
        with pytest.raises(DatabaseError, match="Failed to fetch act events: synthetic act failure"):
            await get_act_events(1)
    else:
        assert await get_act_events(1) == {"major_plot_points": [], "act_key_events": []}
