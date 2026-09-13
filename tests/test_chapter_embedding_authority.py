"""Chapter content embeddings keep one producer across staging and acceptance."""
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import config
from core.embedding_contract import embedding_identity
from core.entity_embedding_service import compute_entity_embedding_text
from core.langgraph.chapter_lifecycle import ChapterLifecycle, extraction_binding
from core.langgraph.content_manager import ContentManager, save_embedding, save_scene_embeddings
from core.langgraph.nodes.assemble_chapter_node import assemble_chapter
from core.langgraph.nodes.commit_graph_ops import _aggregate_scene_embeddings_to_chapter
from core.langgraph.nodes.commit_node import commit_to_graph
from core.langgraph.nodes.embedding_node import generate_scene_embeddings
from core.langgraph.nodes.finalize_node import finalize_chapter
from core.langgraph.nodes.narrative_enrichment_node import enrich_narrative
from core.langgraph.state import NarrativeState
from core.parsers.narrative_enrichment_parser import ChapterEmbeddingExtractionResult, NarrativeEnrichmentParser
from core.service_context import get_services
from data_access.validation_queries import PRIOR_ACCEPTED_ATTEMPTS_QUERY, fetch_prior_accepted_facts
from tests.fakes.quality import example_quality_state
from tests.test_langgraph.test_chapter_lifecycle import DriverExample, Rows, TransactionExample, example_state

pytestmark = pytest.mark.run_settings(EXPECTED_EMBEDDING_DIM=3, NEO4J_VECTOR_DIMENSIONS=3, EMBEDDING_DTYPE="float32", ENABLE_CHAPTER_EMBEDDING_EXTRACTION=True, ENABLE_PHYSICAL_DESCRIPTION_EXTRACTION=False)


class ContentTransaction(TransactionExample):
    def run(self, query: str, parameters: Any = None, **keywords: Any) -> Rows:
        chapter = self.nodes["chapter-1"]["properties"]
        if query == PRIOR_ACCEPTED_ATTEMPTS_QUERY:
            assert parameters == {"project_id": self.driver.project_id, "current_chapter": 2}
            return Rows([
                {**deepcopy(row), "chapter": row["chapter_number"], "chapter_attempt_id": chapter["attempt_id"],
                 "chapter_project_id": chapter["graph_project_id"], "chapter_status": chapter["generation_status"]}
                for row in self.receipts.values() if row["phase"] == "accepted"
            ])
        if "RETURN c," in query or "RETURN c\n" in query:
            return Rows([{"c": {"id": "traveler", "name": "Traveler", "created_chapter": 0}, "traits": [], "relationships": []}])
        if "c.number AS number" in query or "RETURN c.embedding_vector AS embedding" in query:
            return Rows([{**deepcopy(chapter), "act_number": 1, "embedding": chapter.get("embedding_vector"), "embedding_model": chapter.get("embedding_model"), "embedding_identity": chapter.get("embedding_identity")}])
        return super().run(query, parameters, **keywords)


class ContentDriver(DriverExample):
    def begin_transaction(self) -> ContentTransaction:
        return ContentTransaction(self)

    def execute_read(self, callback: Any, *arguments: Any) -> Any:
        return callback(ContentTransaction(self), *arguments)


def bind_driver(state: NarrativeState, monkeypatch: pytest.MonkeyPatch) -> ContentDriver:
    database = get_services().database
    for name in ("_project_id", "_database", "_uri"):
        monkeypatch.setattr(database, name, None)
    database.bind_project(state["graph_project_id"])
    driver = ContentDriver(state["graph_project_id"])
    monkeypatch.setattr(database, "driver", driver)
    return driver


async def exercise_composition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenes: list[str], vectors: list[list[float]], metadata_text: str, metadata_vector: list[float], fault: str = "") -> None:
    """Real callers/storage/transactions, with an exact offline response bank only."""
    from inspect import getfile

    assert Path(__file__).resolve().parents[1] == Path(getfile(generate_scene_embeddings)).resolve().parents[3]
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    state["scene_drafts_ref"] = manager.save_list_of_texts(scenes, "scenes", "composed")
    state["chapter_plan_scene_count"] = len(scenes)
    state["extraction_outcomes"] = [
        {"chapter_number": 1, "scene_index": index, "extraction_type": kind, "status": "succeeded", "item_count": 0, "error": "", "error_type": ""}
        for index in range(len(scenes)) for kind in ("characters", "locations", "events", "relationships")
    ]
    state["extraction_source"] = extraction_binding(state, scenes)
    driver = bind_driver(state, monkeypatch)
    chapter = driver.nodes["chapter-1"]["properties"]
    chapter["title"] = ""
    category, chapter["summary"] = metadata_text.split("\n", 1)
    assert category == "Chapter 1"
    assert compute_entity_embedding_text(name=chapter["title"], description=chapter["summary"], category=category) == metadata_text
    calls: list[str] = []
    responses = dict(zip(scenes, vectors, strict=True))
    responses[metadata_text] = metadata_vector

    async def embed(text: str) -> np.ndarray:
        calls.append(text)
        assert text in responses, "Offline replay refuses uncaptured embedding input"
        return np.asarray(responses[text], dtype=config.EMBEDDING_DTYPE)

    async def batch(texts: list[str], batch_size: int | None = None) -> list[np.ndarray | None]:
        return [await embed(text) for text in texts]

    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    monkeypatch.setattr(get_services().language_model, "async_get_embeddings_batch", batch)

    state = cast(NarrativeState, {**state, **await generate_scene_embeddings(state)})
    state = cast(NarrativeState, {**state, **await assemble_chapter(state)})
    state["quality_policy"] = {**state["quality_policy"], "identity": "strict"}
    state = example_quality_state(state)
    assert (await enrich_narrative(state))["last_error"] is None
    assert "embedding_vector" not in driver.nodes["chapter-1"]["properties"]
    state = {**state, **await commit_to_graph(state)}
    assert state.get("has_fatal_error") is False, state.get("last_error")
    expected = _aggregate_scene_embeddings_to_chapter(vectors)
    assert driver.nodes["chapter-1"]["properties"]["embedding_vector"] == expected
    if fault == "contradiction":
        driver.nodes["chapter-1"]["properties"]["embedding_vector"] = [-value for value in expected]
    elif fault == "model":
        driver.nodes["chapter-1"]["properties"]["embedding_model"] = "another-model"
    elif fault == "provider":
        driver.nodes["chapter-1"]["properties"]["embedding_identity"] = "another-provider"
    elif fault == "quality":
        state = example_quality_state({**state, "coherence_score": 0.0, "prose_quality_score": 0.0, "plot_advancement_score": 0.0, "force_continue": True})
    before = deepcopy(driver.nodes)
    result = await finalize_chapter(state)
    if fault:
        assert result.get("has_fatal_error") is True
        reason = str(result.get("last_error"))
        assert {"contradiction": "Invalid embedding", "model": "model identity mismatch", "provider": "embedding identity mismatch", "quality": "Mandatory quality gate failed"}[fault] in reason
        assert driver.nodes == before
        assert driver.commits == 1
        assert not (tmp_path / "chapters/chapter_001.accepted.json").exists()
        return
    assert result.get("has_fatal_error") is False, result.get("last_error")
    candidate = ChapterLifecycle(state).enrichment()
    assert candidate is not None
    assert candidate["embeddings"][0]["embedding_vector"] == expected
    draft_ref = state["draft_ref"]
    assert draft_ref is not None
    assert candidate["embeddings"][0]["source_text"] == manager.load_text(draft_ref)
    assert driver.nodes["chapter-1"]["properties"]["embedding_vector"] == expected
    assert calls == scenes
    assert driver.commits == 2
    assert (tmp_path / "chapters/chapter_001.accepted.json").is_file()


async def test_scene_producer_to_quality_approved_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await exercise_composition(tmp_path, monkeypatch, ["The traveler enters.", "The traveler leaves."], [[0.6, 0.8, 0.0], [0.8, 0.6, 0.0]], "Chapter 1\nA synthetic plan.", [0.0, 0.6, 0.8])


@pytest.mark.parametrize("fault", ["contradiction", "model", "provider", "quality"])
async def test_acceptance_remains_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    await exercise_composition(tmp_path, monkeypatch, ["The traveler enters.", "The traveler leaves."], [[0.6, 0.8, 0.0], [0.8, 0.6, 0.0]], "Chapter 1\nA synthetic plan.", [0.0, 0.6, 0.8], fault)


async def test_standalone_parser_embeds_narrative_not_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    bind_driver(state, monkeypatch)
    calls: list[str] = []

    async def embed(text: str) -> np.ndarray:
        calls.append(text)
        return np.array([0.6, 0.8, 0.0])

    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    narrative = "The traveler makes a choice absent from the outline."
    results = await NarrativeEnrichmentParser(narrative, 1).extract_chapter_embeddings()
    assert calls == [narrative]
    assert results[0].source_text == narrative
    assert results[0].embedding_identity == embedding_identity()


@pytest.mark.parametrize("entrypoint", ["retained", "legacy"])
async def test_metadata_vector_cannot_claim_draft_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str) -> None:
    state = example_state(tmp_path)
    driver = bind_driver(state, monkeypatch)
    writes: list[Any] = []

    async def write(query: str, parameters: Any = None) -> None:
        writes.append((query, parameters))

    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    result = ChapterEmbeddingExtractionResult(chapter_number=1, embedding_vector=[0.6, 0.8, 0.0], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity(), source_text="Chapter 1\nA plan.")
    if entrypoint == "retained":
        with pytest.raises(ValueError, match="source"):
            ChapterLifecycle(state).retain_enrichment({"descriptions": [], "embeddings": [result.model_dump()]})
    else:
        parser = NarrativeEnrichmentParser("The actual chapter draft.", 1)
        assert await parser.update_chapter_embeddings([result]) is False
    assert driver.commits == 0
    assert writes == []


async def test_legacy_writer_rejects_contradictory_chapter_vector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    driver = bind_driver(state, monkeypatch)
    writes: list[Any] = []

    async def write(query: str, parameters: Any = None) -> None:
        writes.append((query, parameters))

    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    driver.nodes["chapter-1"]["properties"].update(embedding_vector=[0.6, 0.8, 0.0], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity())
    result = ChapterEmbeddingExtractionResult(chapter_number=1, embedding_vector=[-0.6, -0.8, 0.0], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity(), source_text="Draft")
    assert await NarrativeEnrichmentParser("Draft", 1).update_chapter_embeddings([result]) is False
    assert driver.commits == 0
    assert writes == []


def test_candidate_rejects_zero_vector() -> None:
    result = ChapterEmbeddingExtractionResult(chapter_number=1, embedding_vector=[0.0, 0.0, 0.0], embedding_model=config.EMBEDDING_MODEL, embedding_identity=embedding_identity())
    with pytest.raises(ValueError, match="nonzero"):
        result.validated_vector()


async def test_legacy_finalizer_preserves_scene_producer_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = example_state(tmp_path)
    del state["lifecycle_version"]
    bind_driver(state, monkeypatch)
    manager = ContentManager(str(tmp_path))
    vectors = [[0.6, 0.8, 0.0], [0.8, 0.6, 0.0]]
    state["scene_embeddings_ref"] = save_scene_embeddings(manager, vectors, 1, embedding_model=config.EMBEDDING_MODEL)
    state["embedding_ref"] = save_embedding(manager, [0.0, 0.6, 0.8], 1, embedding_model=config.EMBEDDING_MODEL)
    writes: list[Any] = []

    async def write(query: str, parameters: Any = None) -> None:
        writes.append((query, parameters))

    async def embed(text: str) -> np.ndarray:
        raise AssertionError("Identified producer must not be re-embedded")

    monkeypatch.setattr(get_services().database, "execute_write_query", write)
    monkeypatch.setattr(get_services().language_model, "async_get_embedding", embed)
    result = await finalize_chapter(state)
    assert result.get("has_fatal_error") is not True, result.get("last_error")
    assert len(writes) == 1
    assert writes[0][1]["embedding_vector_param"] == _aggregate_scene_embeddings_to_chapter(vectors)


def candidate_for(state: NarrativeState) -> dict[str, Any]:
    reference = state["draft_ref"]
    assert reference is not None
    return {"descriptions": [], "embeddings": [ChapterEmbeddingExtractionResult(
        chapter_number=1, embedding_vector=[0.6, 0.8, 0.0], embedding_model=config.EMBEDDING_MODEL,
        embedding_identity=embedding_identity(), source_text=ContentManager(state["project_dir"]).load_text(reference), extraction_method="scene_mean",
    ).model_dump(mode="json")]}


def historical_lifecycle(lifecycle: ChapterLifecycle) -> ChapterLifecycle:
    return ChapterLifecycle({
        "lifecycle_version": 1, "project_dir": str(lifecycle.files.root), "graph_project_id": lifecycle.project_id,
        "current_chapter": lifecycle.chapter_number, "iteration_count": lifecycle.manifest.iteration_count,
    }).load(lifecycle.manifest.attempt_id)


@pytest.fixture
async def accepted_enrichment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ChapterLifecycle, ContentDriver]:
    state = example_state(tmp_path)
    manager = ContentManager(str(tmp_path))
    assert state["scene_drafts_ref"] is not None
    state["extraction_source"] = extraction_binding(state, manager.load_list_of_texts(state["scene_drafts_ref"]))
    state["extracted_entities_ref"] = manager.save_json({"characters": [{"name": "Traveler", "attributes": {"traits": ["brave"]}}], "world_items": []}, "entities", "history")
    state = example_quality_state(state)
    driver = bind_driver(state, monkeypatch)
    lifecycle = ChapterLifecycle(state)
    lifecycle.retain_enrichment(candidate_for(state))
    lifecycle.stage()
    await lifecycle.commit([])
    publication = await lifecycle.publish()
    assert publication["lifecycle_phase"] == "published"
    assert driver.commits == 2
    assert lifecycle.manuscripts.accepted(1) is not None
    return lifecycle, driver


async def test_accepted_enrichment_is_read_as_prior_chapter_facts(accepted_enrichment: tuple[ChapterLifecycle, ContentDriver]) -> None:
    lifecycle, driver = accepted_enrichment
    before = deepcopy((driver.receipts, driver.nodes, driver.writes))
    facts = await fetch_prior_accepted_facts({
        "lifecycle_version": 1, "project_dir": str(lifecycle.files.root), "graph_project_id": lifecycle.project_id,
        "current_chapter": 2, "extraction_status": "complete", "extraction_policy": "fail_closed",
    })
    assert facts == {"relationships": {}, "characters": {"Traveler": [{"first_chapter": 1, "traits": ["brave"]}]}}
    assert (driver.receipts, driver.nodes, driver.writes) == before


@pytest.mark.parametrize("operation", ["acceptance", "enrichment", "advance"])
async def test_minimal_loaded_enrichment_uses_immutable_draft(accepted_enrichment: tuple[ChapterLifecycle, ContentDriver], operation: str) -> None:
    lifecycle, driver = accepted_enrichment
    historical = historical_lifecycle(lifecycle)
    assert "draft_ref" not in historical.state
    acceptance = json.loads(lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")))
    if operation == "acceptance":
        assert historical._verify_acceptance(acceptance) == lifecycle.manuscripts.accepted(1)
    elif operation == "enrichment":
        assert historical.enrichment_path() == lifecycle.enrichment_path()
        assert historical.enrichment() == acceptance["enrichment"]
    else:
        assert historical.advance() == {"attempt_id": None, "lifecycle_phase": "advanced", "current_chapter": 2}
    assert "draft_ref" not in historical.state
    assert driver.commits == 2


@pytest.mark.parametrize("mode", ["prestage", "loaded"])
def test_enrichment_source_modes(tmp_path: Path, mode: str) -> None:
    state = example_state(tmp_path)
    candidate = candidate_for(state)
    lifecycle = ChapterLifecycle(state)
    lifecycle.retain_enrichment(candidate)
    path = lifecycle.enrichment_path()
    if mode == "loaded":
        assert state["scene_drafts_ref"] is not None
        state["extraction_source"] = extraction_binding(state, ContentManager(str(tmp_path)).load_list_of_texts(state["scene_drafts_ref"]))
        lifecycle = historical_lifecycle(lifecycle.stage())
    assert lifecycle.enrichment_path() == path
    assert lifecycle.enrichment() == candidate


@pytest.mark.parametrize("mode", ["prestage", "loaded"])
@pytest.mark.parametrize("mutation,reason", [
    ("text", "source mismatch"), ("chapter", "chapter identity"), ("identity", "embedding identity"),
    ("model", "model identity"), ("dimensions", "dimensions"), ("rank", "valid number"),
    ("nonfinite", "finite"), ("zero", "nonzero"),
])
def test_enrichment_source_rejects_invalid_candidate(tmp_path: Path, mode: str, mutation: str, reason: str) -> None:
    state = example_state(tmp_path)
    candidate = candidate_for(state)
    lifecycle = ChapterLifecycle(state)
    if mode == "loaded":
        assert state["scene_drafts_ref"] is not None
        state["extraction_source"] = extraction_binding(state, ContentManager(str(tmp_path)).load_list_of_texts(state["scene_drafts_ref"]))
        lifecycle = historical_lifecycle(lifecycle.stage())
    embedding = candidate["embeddings"][0]
    field, value = {
        "text": ("source_text", embedding["source_text"].rstrip()), "chapter": ("chapter_number", 2),
        "identity": ("embedding_identity", "other-provider"), "model": ("embedding_model", "other-model"),
        "dimensions": ("embedding_vector", [0.6, 0.8]), "rank": ("embedding_vector", [[0.6, 0.8, 0.0]]),
        "nonfinite": ("embedding_vector", [float("nan"), 0.8, 0.0]), "zero": ("embedding_vector", [0.0, 0.0, 0.0]),
    }[mutation]
    embedding[field] = value
    with pytest.raises(ValueError, match=reason):
        lifecycle.retain_enrichment(candidate)


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
async def test_loaded_acceptance_requires_immutable_draft(accepted_enrichment: tuple[ChapterLifecycle, ContentDriver], mutation: str) -> None:
    lifecycle, _ = accepted_enrichment
    historical = historical_lifecycle(lifecycle)
    acceptance = json.loads(lifecycle.files.read_bytes(lifecycle.phase_path("acceptance")))
    path = lifecycle.files.root / lifecycle.artifact_ref("draft_ref")["path"]
    if mutation == "missing":
        path.rename(path.with_suffix(".retained"))
        error: type[Exception] = FileNotFoundError
    else:
        path.write_bytes(b"corrupt immutable draft")
        error = ValueError
    with pytest.raises(error):
        historical._verify_acceptance(acceptance)
    with pytest.raises(error):
        lifecycle.enrichment()


@pytest.mark.parametrize("field,value", [("checksum", "0" * 64), ("size_bytes", 0), ("size_bytes", "float_size"), ("content_type", "application/json")])
async def test_loaded_enrichment_rejects_divergent_live_identity(accepted_enrichment: tuple[ChapterLifecycle, ContentDriver], field: str, value: Any) -> None:
    lifecycle, _ = accepted_enrichment
    reference = lifecycle.state["draft_ref"]
    assert reference is not None
    if value == "float_size":
        value = float(reference["size_bytes"])
    lifecycle.state = {**lifecycle.state, "draft_ref": cast(Any, {**reference, field: value})}
    with pytest.raises(ValueError):
        lifecycle.enrichment()


def test_prestage_enrichment_requires_live_draft(tmp_path: Path) -> None:
    state = example_state(tmp_path)
    candidate = candidate_for(state)
    del state["draft_ref"]
    lifecycle = ChapterLifecycle(state)
    with pytest.raises(ValueError, match="draft reference"):
        lifecycle.retain_enrichment(candidate)
