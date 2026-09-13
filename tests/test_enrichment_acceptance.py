"""Candidate enrichment cannot acquire authority before chapter acceptance."""
from copy import deepcopy
from inspect import getfile
from pathlib import Path
from typing import Any, cast

import pytest

import config
from core.embedding_contract import embedding_identity
from core.langgraph.chapter_lifecycle import ChapterLifecycle, extraction_binding
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.narrative_enrichment_node import enrich_narrative
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from tests.fakes.quality import example_quality_state
from tests.test_langgraph.test_chapter_lifecycle import DriverExample, Rows, TransactionExample, example_state

pytestmark = pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=3, NEO4J_VECTOR_DIMENSIONS=3, ENABLE_PHYSICAL_DESCRIPTION_EXTRACTION=True, ENABLE_CHAPTER_EMBEDDING_EXTRACTION=True)

@pytest.mark.parametrize("late_failure", [False, True])
async def test_candidate_enrichment_preserves_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late_failure: bool) -> None:
    state = example_state(tmp_path)
    state["draft_ref"] = ContentManager(str(tmp_path)).save_text("Alice was tall with brown hair.", "draft", "candidate")
    character = {"id": "alice", "name": "Alice", "created_chapter": 0, "physical_description": None}
    writes: list[Any] = []

    async def read(query: str, parameters: Any = None) -> list[dict[str, Any]]:
        if "RETURN c," in query or "RETURN c\n" in query:
            return [{"c": deepcopy(character), "traits": [], "relationships": []}]
        return [{"id": "chapter", "number": 1, "title": "Synthetic", "act_number": 1,
                 "embedding": [0.0, 1.0, 0.0] if late_failure else [1.0, 0.0, 0.0],
                 "embedding_model": config.EMBEDDING_MODEL, "embedding_identity": embedding_identity()}]

    async def batch(statements: Any) -> None:
        writes.extend(statements)

    async def write(query: str, parameters: Any = None) -> None:
        writes.append((query, parameters))

    async def embed(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(get_services().database, "execute_read_query", read)
    monkeypatch.setattr(get_services().database, "execute_cypher_batch", batch)
    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    result = await enrich_narrative(state)
    assert result["last_error"] == ("Invalid embedding for chapter 1" if late_failure else None)
    assert writes == []
    assert character["physical_description"] is None


class EnrichmentTransaction(TransactionExample):
    def __init__(self, driver: "EnrichmentDriver") -> None:
        super().__init__(driver)
        self.enrichment_driver = driver
        self.description = driver.description
        self.embedding = driver.embedding.copy()

    def run(self, query: str, parameters: Any = None, **keywords: Any) -> Rows:
        parameters = parameters or keywords
        if "RETURN c," in query or "RETURN c\n" in query:
            return Rows([{"c": {"id": "alice", "name": "Alice", "created_chapter": 0, "physical_description": self.description}, "traits": [], "relationships": []}])
        if "c.number AS number" in query:
            return Rows([{"id": "chapter", "number": 1, "title": "Synthetic", "act_number": 1,
                          "embedding": self.embedding, "embedding_model": config.EMBEDDING_MODEL, "embedding_identity": embedding_identity()}])
        if "RETURN c.physical_description AS description" in query:
            return Rows([{"description": self.description}])
        if "RETURN c.embedding_vector AS embedding" in query:
            return Rows([{"embedding": self.embedding, "embedding_model": config.EMBEDDING_MODEL, "embedding_identity": embedding_identity()}])
        if "RETURN c.id AS updated_character" in query:
            self.description = parameters["physical_description"]
            self.statements.append(query)
            if self.enrichment_driver.interruption == "after_first_write":
                self.enrichment_driver.interruption = ""
                self.enrichment_driver.prospective_writes += 1
                raise RuntimeError("interrupted after first enrichment write")
            return Rows()
        result = super().run(query, parameters)
        if "MERGE (c:Chapter" in query and parameters["embedding_vector_param"] is not None:
            self.embedding = parameters["embedding_vector_param"]
        return result

    def commit(self) -> None:
        super().commit()
        self.enrichment_driver.description = self.description
        self.enrichment_driver.embedding = self.embedding
        if self.enrichment_driver.interruption == "after_acceptance":
            self.enrichment_driver.interruption = ""
            raise RuntimeError("lost acceptance acknowledgement")


class EnrichmentDriver(DriverExample):
    def __init__(self, project_id: str) -> None:
        super().__init__(project_id)
        self.description: str | None = None
        self.embedding = [1.0, 0.0, 0.0]
        self.interruption = ""
        self.prospective_writes = 0

    def begin_transaction(self) -> EnrichmentTransaction:
        return EnrichmentTransaction(self)

    def execute_read(self, callback: Any, *arguments: Any) -> Any:
        return callback(EnrichmentTransaction(self), *arguments)


@pytest.mark.parametrize("interruption", ["rejected", "after_first_write", "after_acceptance", "success"])
async def test_acceptance_owns_enrichment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: str) -> None:
    assert Path(getfile(ChapterLifecycle)).resolve() == Path(__file__).resolve().parents[1] / "core/langgraph/chapter_lifecycle.py"
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    scenes = ["Alice was tall with brown hair."]
    state["draft_ref"] = manager.save_text(scenes[0], "draft", "candidate")
    state["scene_drafts_ref"] = manager.save_list_of_texts(scenes, "scenes", "candidate")
    state["extraction_source"] = extraction_binding(state, scenes)
    state = example_quality_state(state)
    database = get_services().database
    for name in ("_project_id", "_database", "_uri"):
        monkeypatch.setattr(database, name, None)
    database.bind_project(state["graph_project_id"])
    driver = EnrichmentDriver(state["graph_project_id"])
    monkeypatch.setattr(database, "driver", driver)
    provider_calls: list[str] = []

    async def embed(text: str) -> list[float]:
        provider_calls.append(text)
        return [0.99, 0.01, 0.0]

    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    assert (await enrich_narrative(state))["last_error"] is None
    assert driver.description is None
    assert driver.embedding == [1.0, 0.0, 0.0]
    candidate = ChapterLifecycle(state).enrichment()
    assert candidate is not None
    assert candidate["descriptions"] == [{"character_id": "alice", "character_name": "Alice", "description": "tall with brown hair"}]
    assert (await enrich_narrative(state))["last_error"] is None
    assert len(provider_calls) == 1
    assert ChapterLifecycle({**state, "iteration_count": 1}).enrichment() is None
    state = {**state, **await commit_to_graph(state)}
    assert state.get("has_fatal_error") is False
    if interruption == "rejected":
        state = example_quality_state(cast(NarrativeState, {**state, "coherence_score": 0.0}))
    else:
        driver.interruption = interruption
    result = await finalize_chapter(state)
    assert result.get("has_fatal_error") is (interruption != "success")
    if interruption in {"rejected", "after_first_write"}:
        assert driver.description is None
        assert driver.embedding == [1.0, 0.0, 0.0]
        attempt_id = state["attempt_id"]
        assert attempt_id is not None
        assert driver.receipts[attempt_id]["phase"] == "committed"
        if interruption == "rejected":
            assert not ChapterLifecycle(state).stage().files.exists("chapters/chapter_001.accepted.json")
            return
        assert driver.prospective_writes == 1
    result = await finalize_chapter(state)
    assert result.get("has_fatal_error") is False
    assert driver.description == "tall with brown hair"
    assert driver.embedding == candidate["embeddings"][0]["embedding_vector"]
    assert driver.commits == 2
    assert len(provider_calls) == 1
