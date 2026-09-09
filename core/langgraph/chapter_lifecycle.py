"""Single-writer chapter attempt receipts and cross-store reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

from neo4j import Transaction
from pydantic import BaseModel, ConfigDict, Field

from core.graph_ownership import validate_project_id
from core.langgraph.content_manager import ContentRef, require_project_dir
from core.langgraph.manuscript import ManuscriptReceipt, ManuscriptStore
from core.langgraph.quality_policy import acceptance_decision, announce_acceptance, validation_decision, verify_checkpoint_quality, verify_decision
from core.langgraph.state import NarrativeState
from core.service_context import get_services
from data_access.chapter_queries import build_chapter_upsert_statement
from utils.file_io import ContainedFiles

ATTEMPT_QUERY = """
MATCH (attempt:ChapterAttempt {project_id: $project_id, chapter_number: $chapter_number})
RETURN attempt.id AS id, attempt.manifest AS manifest, attempt.phase AS phase,
       attempt.acceptance AS acceptance
"""
CREATE_ATTEMPT = """
CREATE (attempt:ChapterAttempt {id: $attempt_id, project_id: $project_id,
chapter_number: $chapter_number, manifest: $manifest, phase: 'committed'})
WITH attempt
MATCH (chapter:Chapter {number: $chapter_number})
SET chapter.attempt_id = $attempt_id, chapter.graph_project_id = $project_id
"""
UPDATE_ATTEMPT = """
MATCH (attempt:ChapterAttempt {id: $attempt_id})
SET attempt.phase = $phase, attempt.acceptance = $acceptance
"""
CHAPTER_QUERY = "MATCH (chapter:Chapter {number: $chapter_number}) RETURN chapter.generation_status AS status, chapter.attempt_id AS attempt_id"
ARTIFACT_FIELDS = ("draft_ref", "scene_drafts_ref", "extracted_entities_ref", "extracted_relationships_ref")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extraction_binding(state: NarrativeState, scenes: list[str]) -> dict[str, Any]:
    return {
        "project_id": state["graph_project_id"],
        "chapter_number": state["current_chapter"],
        "iteration_count": state.get("iteration_count", 0),
        "scenes_sha256": digest(canonical_bytes(scenes)),
    }


class ArtifactIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    content_type: str = Field(min_length=1)


class ExtractionSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    chapter_number: int = Field(gt=0)
    scene_index: int = Field(ge=0)
    extraction_type: Literal["characters", "locations", "events", "relationships"]
    status: Literal["succeeded"]
    item_count: int = Field(ge=0)
    error_type: Literal[""]
    error: Literal[""]


class AttemptManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal[1] = 1
    project_id: str
    chapter_number: int = Field(gt=0)
    iteration_count: int = Field(ge=0)
    artifacts: dict[str, ArtifactIdentity]
    extraction_outcomes: list[ExtractionSlot]
    scenes_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def encoded(self) -> bytes:
        return canonical_bytes(self.model_dump())

    @property
    def attempt_id(self) -> str:
        return digest(self.encoded)


class ChapterLifecycle:
    """Authority for one retained attempt, not a transaction across stores."""

    state: NarrativeState
    files: ContainedFiles
    project_id: str
    chapter_number: int
    manuscripts: ManuscriptStore
    manifest: AttemptManifest

    def __init__(self, state: NarrativeState) -> None:
        if type(state.get("lifecycle_version")) is not int or state["lifecycle_version"] != 1:
            raise ValueError("Unsupported chapter lifecycle version; legacy resume requires explicit migration")
        self.state = state
        self.files = ContainedFiles(Path(require_project_dir(state)), durable=True)
        self.project_id = validate_project_id(state["graph_project_id"])
        if self.files.read_bytes("graph-project-id").decode("ascii") != self.project_id:
            raise ValueError("Lifecycle project identity mismatch")
        self.chapter_number = state["current_chapter"]
        if type(self.chapter_number) is not int or self.chapter_number <= 0:
            raise ValueError("Lifecycle chapter number must be a positive integer")
        self.manuscripts = ManuscriptStore(self.files.root)

    def _retain(self, path: str, content: bytes) -> None:
        if self.files.exists(path):
            if self.files.read_bytes(path) != content:
                raise ValueError(f"Immutable attempt artifact differs: {path}")
        else:
            self.files.write_bytes(path, content)
        if self.files.read_bytes(path) != content:
            raise ValueError("Attempt artifact readback mismatch")

    def _manifest_path(self, attempt_id: str) -> str:
        if len(attempt_id) != 64 or any(character not in "0123456789abcdef" for character in attempt_id):
            raise ValueError("Invalid attempt identity")
        return f".saga/attempts/{attempt_id}/manifest.json"

    def selection_path(self) -> str:
        iteration = self.state.get("iteration_count", 0)
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Invalid attempt iteration")
        return f".saga/attempts/chapter_{self.chapter_number}/iteration_{iteration}.json"

    def phase_path(self, phase: str) -> str:
        if phase not in {"commit_started", "committed", "acceptance", "published", "advanced", "compensation_required", "compensated", "revision_result"}:
            raise ValueError("Invalid local attempt phase")
        return f".saga/attempts/{self.manifest.attempt_id}/{phase}.json"

    def observe(self, phase: str) -> None:
        self._retain(self.phase_path(phase), canonical_bytes({"schema_version": 1, "attempt_id": self.manifest.attempt_id, "phase": phase}))

    def artifact_ref(self, name: str) -> ContentRef:
        identity = self.manifest.artifacts[name]
        extension = "txt" if name == "draft_ref" else "json"
        return cast(
            ContentRef,
            {
                **identity.model_dump(),
                "version": 1,
                "path": f".saga/content/attempts/{self.manifest.attempt_id}/{name}.{extension}",
            },
        )

    def _read_artifact(self, reference: dict[str, Any]) -> bytes:
        identity = ArtifactIdentity.model_validate({name: reference.get(name) for name in ("checksum", "size_bytes", "content_type")})
        path = reference["path"]
        if not isinstance(path, str) or not path.startswith(".saga/content/"):
            raise ValueError("Attempt source must be a contained ContentRef")
        content = self.files.read_bytes(path)
        if len(content) != identity.size_bytes or digest(content) != identity.checksum:
            raise ValueError("Attempt source checksum or size mismatch")
        return content

    def load(self, attempt_id: str) -> ChapterLifecycle:
        encoded = self.files.read_bytes(self._manifest_path(attempt_id))
        self.manifest = AttemptManifest.model_validate_json(encoded)
        if self.manifest.encoded != encoded or self.manifest.attempt_id != attempt_id:
            raise ValueError("Attempt manifest identity mismatch")
        if self.manifest.project_id != self.project_id or self.manifest.chapter_number != self.chapter_number:
            raise ValueError("Attempt project/chapter mismatch")
        if self.manifest.iteration_count != self.state.get("iteration_count", 0):
            raise ValueError("Checkpoint/attempt iteration mismatch")
        if set(self.manifest.artifacts) != set(ARTIFACT_FIELDS):
            raise ValueError("Incomplete attempt artifact set")
        for name in ARTIFACT_FIELDS:
            self._read_artifact(dict(self.artifact_ref(name)))
            reference = self.state.get(name)
            if reference is not None and (not isinstance(reference, dict) or reference.get("checksum") != self.manifest.artifacts[name].checksum):
                raise ValueError("Checkpoint/attempt content identity mismatch")
        return self

    def stage(self) -> ChapterLifecycle:
        attempt_id = self.state.get("attempt_id")
        if attempt_id is not None:
            return self.load(attempt_id)
        selection = self.selection_path()
        if self.files.exists(selection):
            return self.load(self.files.read_bytes(selection).decode("ascii"))
        if self.state.get("extraction_status") != "complete" or self.state.get("extraction_policy") != "fail_closed":
            raise ValueError("Attempt requires complete fail-closed extraction")
        contents: dict[str, bytes] = {}
        identities: dict[str, ArtifactIdentity] = {}
        for name in ARTIFACT_FIELDS:
            reference = self.state.get(name)
            if not isinstance(reference, dict):
                raise ValueError("Attempt requires explicit draft and extraction refs")
            contents[name] = self._read_artifact(reference)
            identities[name] = ArtifactIdentity.model_validate({key: reference[key] for key in ("checksum", "size_bytes", "content_type")})
        scenes = json.loads(contents["scene_drafts_ref"])
        if not isinstance(scenes, list) or not scenes or any(not isinstance(scene, str) or not scene for scene in scenes):
            raise ValueError("Attempt requires nonempty scene strings")
        binding = extraction_binding(self.state, scenes)
        if self.state.get("extraction_source") != binding:
            raise ValueError("Extraction completion is not bound to this project/chapter/draft attempt")
        if contents["draft_ref"].decode("utf-8") != "\n\n# ***\n\n".join(scenes):
            raise ValueError("Draft differs from extracted source scenes")
        outcomes = [ExtractionSlot.model_validate(slot) for slot in self.state.get("extraction_outcomes", [])]
        expected = {(index, kind) for index in range(len(scenes)) for kind in ("characters", "locations", "events", "relationships")}
        if len(outcomes) != len(expected) or {(slot.scene_index, slot.extraction_type) for slot in outcomes} != expected or any(slot.chapter_number != self.chapter_number for slot in outcomes):
            raise ValueError("Attempt requires each successful scene/type extraction exactly once")
        self.manifest = AttemptManifest(
            project_id=self.project_id,
            chapter_number=self.chapter_number,
            iteration_count=self.state.get("iteration_count", 0),
            artifacts=identities,
            extraction_outcomes=outcomes,
            scenes_sha256=binding["scenes_sha256"],
        )
        if self.files.exists(f"chapters/chapter_{self.chapter_number:03d}.accepted.json"):
            raise ValueError("Accepted chapter cannot be replaced by a new attempt")
        for name, content in contents.items():
            self._retain(self.artifact_ref(name)["path"], content)
        self._retain(self._manifest_path(self.manifest.attempt_id), self.manifest.encoded)
        self._retain(selection, self.manifest.attempt_id.encode("ascii"))
        return self

    def state_update(self, phase: str) -> NarrativeState:
        return cast(
            NarrativeState,
            {
                "attempt_id": self.manifest.attempt_id,
                "lifecycle_phase": phase,
                **{name: self.artifact_ref(name) for name in ARTIFACT_FIELDS},
            },
        )

    def parameters(self) -> dict[str, Any]:
        return {"project_id": self.project_id, "chapter_number": self.chapter_number, "attempt_id": self.manifest.attempt_id, "manifest": self.manifest.encoded.decode("utf-8")}

    def _validate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        selected = [row for row in rows if row["id"] == self.manifest.attempt_id]
        if len(selected) > 1:
            raise ValueError("Duplicate graph attempt receipt")
        for row in rows:
            manifest = AttemptManifest.model_validate_json(row["manifest"])
            if manifest.attempt_id != row["id"] or manifest.project_id != self.project_id or manifest.chapter_number != self.chapter_number:
                raise ValueError("Graph attempt identity mismatch")
            if row["phase"] not in {"committed", "accepted", "compensated"}:
                raise ValueError("Unknown graph attempt phase")
            if row["id"] != self.manifest.attempt_id and row["phase"] != "compensated":
                raise ValueError("Another uncompensated attempt owns this chapter")
        if selected and selected[0]["manifest"] != self.manifest.encoded.decode("utf-8"):
            raise ValueError("Graph/local manifest mismatch")
        return selected[0] if selected else None

    async def graph_receipt(self) -> dict[str, Any] | None:
        if get_services().database.require_project_binding() != self.project_id:
            raise ValueError("Lifecycle/database ownership mismatch")
        receipt = self._validate_rows(await get_services().database.execute_read_query(ATTEMPT_QUERY, self.parameters()))
        if receipt is not None and receipt["phase"] != "compensated":
            chapters = await get_services().database.execute_read_query(CHAPTER_QUERY, self.parameters())
            if len(chapters) != 1 or chapters[0]["attempt_id"] != self.manifest.attempt_id:
                raise ValueError("Graph chapter/attempt projection mismatch")
            if (chapters[0]["status"] == "finalized") != (receipt["phase"] == "accepted"):
                raise ValueError("Graph chapter/attempt status mismatch")
        return receipt

    async def commit(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        self.observe("commit_started")

        def apply(transaction: Transaction) -> None:
            row = self._validate_rows([dict(record) for record in transaction.run(ATTEMPT_QUERY, self.parameters())])
            if row is not None:
                if row["phase"] != "committed":
                    raise ValueError("Attempt is not eligible for commit replay")
                return
            chapters = list(transaction.run(CHAPTER_QUERY, self.parameters()))
            if any(chapter["status"] == "finalized" or chapter["attempt_id"] is not None for chapter in chapters):
                raise ValueError("Existing chapter requires explicit lifecycle reconciliation")
            for query, parameters in statements:
                transaction.run(query, parameters).consume()
            transaction.run(CREATE_ATTEMPT, self.parameters()).consume()

        await self.graph_receipt()
        await get_services().database.execute_in_transaction(apply)
        row = await self.graph_receipt()
        if row is None or row["phase"] != "committed":
            raise ValueError("Graph commit readback missing")
        self.observe("committed")

    async def compensate(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        row = await self.graph_receipt()
        if row is None or row["phase"] not in {"committed", "compensated"}:
            raise ValueError("Only a committed rejected attempt may be compensated")
        self.observe("compensation_required")

        def apply(transaction: Transaction) -> None:
            current = self._validate_rows([dict(record) for record in transaction.run(ATTEMPT_QUERY, self.parameters())])
            if current is None or current["phase"] not in {"committed", "compensated"}:
                raise ValueError("Compensation phase mismatch")
            if current["phase"] == "compensated":
                return
            for query, parameters in statements:
                transaction.run(query, parameters).consume()
            transaction.run(UPDATE_ATTEMPT, {**self.parameters(), "phase": "compensated", "acceptance": None}).consume()

        if row["phase"] != "compensated":
            await get_services().database.execute_in_transaction(apply)
        verified = await self.graph_receipt()
        if verified is None or verified["phase"] != "compensated":
            raise ValueError("Compensation readback missing")
        self.observe("compensated")

    def revision_result(self) -> NarrativeState | None:
        path = self.phase_path("revision_result")
        if not self.files.exists(path):
            return None
        result = json.loads(self.files.read_bytes(path))
        self._read_artifact(result["revision_guidance_ref"])
        if result["iteration_count"] != self.manifest.iteration_count + 1 or result["attempt_id"] is not None:
            raise ValueError("Revision result identity mismatch")
        return cast(NarrativeState, result)

    def retain_revision_result(self, encoded: bytes) -> None:
        self._retain(self.phase_path("revision_result"), encoded)
        self.revision_result()

    def _acceptance(self) -> dict[str, Any]:
        path = self.phase_path("acceptance")
        if self.files.exists(path):
            acceptance = json.loads(self.files.read_bytes(path))
        else:
            quality = acceptance_decision(self.state)
            draft = self._read_artifact(dict(self.artifact_ref("draft_ref"))).decode("utf-8")
            receipt = self.manuscripts.prepare(self.chapter_number, draft)
            policy = {
                name: self.state.get(name)
                for name in (
                    "force_continue",
                    "needs_revision",
                    "iteration_count",
                    "max_iterations",
                    "coherence_score",
                    "prose_quality_score",
                    "plot_advancement_score",
                    "pacing_score",
                    "tone_consistency_score",
                    "quality_feedback",
                )
            }
            policy["contradictions"] = [item.model_dump(mode="json") for item in self.state.get("contradictions", [])]
            acceptance = {"schema_version": 1, "attempt_id": self.manifest.attempt_id, "manuscript": receipt.model_dump(), "policy": policy, "summary": self.state.get("current_summary"), "quality": quality}
            self._retain(path, canonical_bytes(acceptance))
        self._verify_acceptance(acceptance)
        return cast(dict[str, Any], acceptance)

    def _verify_acceptance(self, acceptance: dict[str, Any]) -> ManuscriptReceipt:
        if acceptance.get("schema_version") != 1 or acceptance.get("attempt_id") != self.manifest.attempt_id:
            raise ValueError("Acceptance attempt identity mismatch")
        if "quality" not in acceptance:
            raise ValueError("Legacy acceptance lacks quality evidence; explicit revalidation required")
        verify_decision(acceptance["quality"], {**self.state, **self.state_update("accepted"), "quality_policy": acceptance["quality"]["policy"]})
        receipt = ManuscriptReceipt.model_validate(acceptance["manuscript"])
        if receipt.chapter_number != self.chapter_number or receipt.body_sha256 != self.manifest.artifacts["draft_ref"].checksum:
            raise ValueError("Accepted manuscript/draft mismatch")
        self.manuscripts.read(receipt)
        return receipt

    async def publish(self) -> NarrativeState:
        if self.state.get("revision_rollback_failure") is not None:
            raise ValueError("Quality acceptance cannot bypass compensation failure")
        row = await self.graph_receipt()
        if row is None or row["phase"] not in {"committed", "accepted"} or self.files.exists(self.phase_path("compensation_required")):
            raise ValueError("Attempt is not eligible for acceptance")
        if row["phase"] == "committed":
            if not self.files.exists(self.phase_path("acceptance")):
                from core.langgraph.nodes.quality_assurance_node import assess_graph_quality

                validation_decision(self.state)
                self.state = {**self.state, "graph_quality_check": await assess_graph_quality(self.state)}
            acceptance = self._acceptance()
            verify_checkpoint_quality(self.state, acceptance["quality"])
            receipt = self._verify_acceptance(acceptance)
            self.manuscripts.resume_prepared(receipt)

            def apply(transaction: Transaction) -> None:
                current = self._validate_rows([dict(record) for record in transaction.run(ATTEMPT_QUERY, self.parameters())])
                if current is None or current["phase"] != "committed":
                    raise ValueError("Acceptance requires committed graph receipt")
                statement = build_chapter_upsert_statement(chapter_number=self.chapter_number, summary=acceptance["summary"], is_provisional=False, generation_status="finalized")
                transaction.run(*statement).consume()
                transaction.run(UPDATE_ATTEMPT, {**self.parameters(), "phase": "accepted", "acceptance": canonical_bytes(acceptance).decode("utf-8")}).consume()

            await get_services().database.execute_in_transaction(apply)
            row = await self.graph_receipt()
        if row is None or row["phase"] != "accepted":
            raise ValueError("Graph acceptance readback missing")
        acceptance = json.loads(row["acceptance"])
        if "quality" not in acceptance:
            raise ValueError("Legacy acceptance lacks quality evidence; explicit revalidation required")
        verify_checkpoint_quality(self.state, acceptance["quality"])
        if self.files.read_bytes(self.phase_path("acceptance")) != canonical_bytes(acceptance):
            raise ValueError("Graph/local acceptance mismatch")
        receipt = self._verify_acceptance(acceptance)
        if self.files.exists(receipt.accepted_path) and self.manuscripts.accepted(self.chapter_number) != receipt:
            raise ValueError("Another accepted selection requires explicit reconciliation")
        self.manuscripts.resume_prepared(receipt)
        self.manuscripts.accept(receipt)
        if self.manuscripts.accepted(self.chapter_number) != receipt:
            raise ValueError("Publication readback mismatch")
        self.observe("published")
        announce_acceptance(acceptance["quality"], self.phase_path("acceptance"))
        return {**self.state_update("published"), "graph_quality_check": acceptance["quality"]["graph_quality_check"], "current_node": "finalize", "has_fatal_error": False, "last_error": None, "needs_revision": False}

    def advance(self) -> NarrativeState:
        if not self.files.exists(self.phase_path("published")):
            raise ValueError("Cannot advance an unpublished attempt")
        acceptance = json.loads(self.files.read_bytes(self.phase_path("acceptance")))
        receipt = self._verify_acceptance(acceptance)
        if self.manuscripts.accepted(self.chapter_number) != receipt:
            raise ValueError("Cannot advance a divergent publication")
        self.observe("advanced")
        return {"attempt_id": None, "lifecycle_phase": "advanced", "current_chapter": self.chapter_number + 1}


async def reconcile_checkpoint(graph: Any, state: NarrativeState, configuration: dict[str, Any]) -> NarrativeState:
    """Reconcile durable attempts before native continuation, never rerun entry."""
    lifecycle = ChapterLifecycle(state)
    snapshot = await graph.aget_state(configuration)
    if not snapshot.values or snapshot.values.get("graph_project_id") != lifecycle.project_id:
        raise ValueError("Native checkpoint/project identity mismatch")
    if state.get("attempt_id") is None and not lifecycle.files.exists(lifecycle.selection_path()):
        # No lifecycle side effect exists to reconcile. Preserve any failure;
        # native scheduled guards/error handler must not clear it.
        return state
    lifecycle.stage()
    receipt = await lifecycle.graph_receipt()
    if receipt is not None and receipt["phase"] == "accepted":
        update = await lifecycle.publish()
        merged: NarrativeState = {**state, **update}
        # An explicit checkpoint identity preserves scheduled tasks whose successful
        # pending writes disappear from the latest effective snapshot's next tuple.
        physical_snapshot = await graph.aget_state(snapshot.config)
        if not physical_snapshot.next and not physical_snapshot.tasks and state.get("current_node") in {"check_quality", "finalize"}:
            if state["current_chapter"] < state["total_chapters"]:
                from core.langgraph.workflow import advance_chapter

                update = advance_chapter(merged)
                update["run_start_chapter"] = update["current_chapter"]
                await graph.aupdate_state(configuration, update, as_node="advance_chapter")
                return {**merged, **update}
            return merged
        if state.get("has_fatal_error") or any(node not in {"heal_graph", "check_quality", "advance_chapter"} for node in snapshot.next):
            await graph.aupdate_state(configuration, update, as_node="finalize")
        return merged
    if lifecycle.files.exists(lifecycle.phase_path("compensation_required")):
        from core.langgraph.nodes.revision_node import _rollback_chapter_data

        await _rollback_chapter_data(state["current_chapter"], lifecycle=lifecycle)
        previous_result = lifecycle.revision_result()
        if previous_result is not None:
            await graph.aupdate_state(configuration, previous_result, as_node="revise")
            return {**state, **previous_result}
    if state.get("has_fatal_error") or state.get("revision_rollback_failure") is not None:
        boundary = state.get("error_node")
        predecessors = {"commit": "normalize_relationships", "finalize": "summarize", "revise": "validate"}
        if boundary not in predecessors:
            raise ValueError("Failure is outside the chapter lifecycle recovery boundary")
        if boundary == "revise" and not lifecycle.files.exists(lifecycle.phase_path("compensated")):
            raise ValueError("Revision recovery requires verified compensation")
        update = {**lifecycle.state_update("staged" if receipt is None else receipt["phase"]), "has_fatal_error": False, "last_error": None, "error_node": None, "revision_rollback_failure": None}
        await graph.aupdate_state(configuration, update, as_node=predecessors[boundary])
        return cast(NarrativeState, {**state, **update})
    return state
