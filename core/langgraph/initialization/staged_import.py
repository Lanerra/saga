"""Local initialization retention, projection recovery and atomic graph acceptance."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from neo4j import Transaction

from core.langgraph.initialization.graph_plan import InitializationGraphPlan, load_plan, produce_plan
from core.langgraph.initialization.snapshot import ARTIFACTS, CATALOG_ARTIFACT, FrozenPayload, InitializationSnapshot, digest, encoded, require, select_snapshot
from core.langgraph.state import NarrativeState
from core.schema_readiness import verify_catalog_event_identity
from core.service_context import get_services
from utils.file_io import ContainedFiles

RECEIPT_QUERY = "MATCH (owner:SagaGraphOwner {key: 'exclusive', project_id: $project_id}) RETURN owner.initialization_plan AS identity"


class SelectedProducerPayload(FrozenPayload):
    payload: str
    checksum: str


async def retain_producer_selection(project_dir: str, stage: Literal["catalog", "relationships"], parent_identity: str, produce: Callable[[], Awaitable[FrozenPayload]]) -> str:
    """Retain one validated result before publishing its checkpoint reference."""
    require(len(parent_identity) == 64 and all(character in "0123456789abcdef" for character in parent_identity), "Malformed producer parent identity")
    importer = InitializationImport(project_dir)
    name = f"{stage}-{parent_identity}.json"
    with importer.files.exclusive_lock(f"{importer.root}/{stage}.lock"):
        if importer.files.exists(f"{importer.root}/{name}"):
            retained = SelectedProducerPayload.model_validate_json(importer.files.read_bytes(f"{importer.root}/{name}"))
            require(digest(retained.payload) == retained.checksum, "Selected producer checksum mismatch")
            return retained.payload
        payload = (await produce()).model_dump_json()
        importer.retain(name, SelectedProducerPayload(payload=payload, checksum=digest(payload)).model_dump_json())
        return payload


class InitializationImport:
    def __init__(self, project_dir: str) -> None:
        self.project_dir = Path(project_dir)
        self.files = ContainedFiles(self.project_dir, durable=True)
        self.root = ".saga/initialization"

    def retain(self, name: str, text: str) -> None:
        path = f"{self.root}/{name}"
        if self.files.exists(path):
            require(self.files.read_bytes(path) == text.encode("utf-8"), f"Immutable initialization artifact conflict: {name}")
        else:
            self.files.write_bytes(path, text.encode("utf-8"))

    def load(self) -> InitializationGraphPlan:
        identity = self.files.read_bytes(f"{self.root}/selected").decode("ascii")
        require(len(identity) == 64 and all(character in "0123456789abcdef" for character in identity), "Malformed initialization identity")
        text = self.files.read_bytes(f"{self.root}/{identity}.json").decode("utf-8")
        require(digest(text) == identity, "Frozen initialization checksum mismatch")
        plan = load_plan(text)
        require(plan.identity == identity, "Frozen initialization serialization mismatch")
        return plan

    async def prepare(self, state: NarrativeState) -> InitializationGraphPlan:
        with self.files.exclusive_lock(f"{self.root}/writer.lock"):
            if self.files.exists(f"{self.root}/selected"):
                plan = self.load()
                selected = state.get("initialization_id")
                require(selected is None or selected == plan.identity, "Selected initialization identity mismatch")
                require(plan.snapshot == select_snapshot(state), "Selected content differs from frozen initialization; reinitialization is blocked")
                return plan
            from core.langgraph.initialization.persist_files_node import assert_initialization_files_absent

            assert_initialization_files_absent(self.project_dir)
            snapshot = select_snapshot(state)
            # Input admission and model work finish before graph connection or graph writes.
            plan = await produce_plan(snapshot)
            plan = plan.model_copy(update={"projections": self.projection_manifest(snapshot)})
            self.retain(f"{plan.identity}.json", plan.model_dump_json())
            self.retain("selected", plan.identity)
            return plan

    def verify_sources(self, plan: InitializationGraphPlan) -> None:
        from core.langgraph.content_manager import ContentManager

        manager = ContentManager(str(self.project_dir))
        for artifact in plan.snapshot.artifacts:
            require(encoded(manager.load_json_strict(artifact.reference())) == artifact.payload, "Selected initialization source changed")

    def projection_manifest(self, snapshot: InitializationSnapshot) -> str:
        from core.langgraph.initialization.catalog import EntityCatalog
        from core.langgraph.initialization.persist_files_node import (
            _create_directory_structure,
            _write_character_files,
            _write_outline_files,
            _write_saga_yaml,
            _write_summaries_readme,
            _write_world_history_stub,
            _write_world_items_file,
            _write_world_rules_stub,
        )
        from models.kg_models import WorldItem

        with TemporaryDirectory(prefix="projection-", dir=self.project_dir / self.root) as temporary:
            root = Path(temporary)
            state: NarrativeState = json.loads(snapshot.metadata)
            state["total_chapters"] = snapshot.total_chapters
            _create_directory_structure(root)
            _write_character_files(root, snapshot.source("character_sheets"))
            _write_outline_files(
                root, snapshot.source("global_outline"), {act["act_number"]: act for act in snapshot.source("act_outlines")["acts"]}, state,
                selected_outlines={name: snapshot.source(name) for name in ("global_outline", "act_outlines", "chapter_outlines")},
            )
            catalog = EntityCatalog.model_validate_json(encoded(snapshot.source(CATALOG_ARTIFACT)))
            world_items = [WorldItem.model_validate_json(entity.payload) for entity in catalog.entities if entity.label in {"Item", "Location"}]
            _write_world_items_file(root, world_items, state.get("setting", ""))
            _write_saga_yaml(root, state)
            _write_world_rules_stub(root)
            _write_world_history_stub(root)
            _write_summaries_readme(root)
            return encoded({str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in sorted(root.rglob("*")) if path.is_file()})

    def publish_projections(self, plan: InitializationGraphPlan) -> None:
        name = f"{plan.identity}-projections.json"
        path = f"{self.root}/{name}"
        if not self.files.exists(path):
            from core.langgraph.initialization.persist_files_node import assert_initialization_files_absent

            assert_initialization_files_absent(self.project_dir)
            self.retain(name, plan.projections)
        text = self.files.read_bytes(path).decode("utf-8")
        require(text == plan.projections, "Projection checksum mismatch")
        manifest = json.loads(text)
        require(isinstance(manifest, dict) and bool(manifest), "Missing projection manifest")
        # Validate every known byte and reject unowned files before creating missing projections.
        known = set(manifest)
        for directory in ("characters", "outline", "world", "summaries", "chapters", "exports"):
            for name in self.files.list_names(directory):
                require(f"{directory}/{name}" in known, f"Unknown historical projection: {directory}/{name}")
        for relative, text in manifest.items():
            require(isinstance(text, str), "Malformed projection payload")
            if self.files.exists(relative):
                require(self.files.read_bytes(relative) == text.encode("utf-8"), f"User-owned projection differs: {relative}; explicit import required")
        for relative, text in manifest.items():
            if not self.files.exists(relative):
                # Publication is create-only; B31's portable filename/race guards remain active.
                from core.langgraph.initialization.persist_files_node import _publish_new_file
                from utils.file_io import write_text_file

                target = self.project_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                _publish_new_file(target, text, writer=write_text_file)
        self.retain(f"{plan.identity}-projected", digest(encoded(manifest)))

    async def receipt(self, plan: InitializationGraphPlan) -> bool:
        rows = await get_services().database.execute_read_query(RECEIPT_QUERY, {"project_id": plan.snapshot.project_id})
        require(len(rows) == 1 and set(rows[0]) == {"identity"}, "Missing or malformed initialization owner receipt")
        identity = rows[0]["identity"]
        require(identity is None or identity == plan.identity, "Conflicting accepted initialization; reinitialization is blocked")
        return identity == plan.identity

    async def accept(self, expected_identity: str) -> InitializationGraphPlan:
        with self.files.exclusive_lock(f"{self.root}/writer.lock"):
            plan = self.load()
            require(plan.identity == expected_identity, "Initialization acceptance identity mismatch")
            get_services().database.bind_project(plan.snapshot.project_id)
            # Exact graph receipt is authoritative across a database acknowledgement gap.
            if await self.receipt(plan):
                self.retain(f"{plan.identity}-accepted", plan.identity)
                return plan
            self.verify_sources(plan)
            self.publish_projections(plan)

            def apply(transaction: Transaction) -> None:
                owner = transaction.run(
                    "MATCH (owner:SagaGraphOwner {key: 'exclusive', project_id: $project_id}) "
                    "SET owner.initialization_lock = coalesce(owner.initialization_lock, 0) + 1 "
                    "RETURN owner.initialization_plan AS identity", project_id=plan.snapshot.project_id,
                ).single(strict=True)
                require(owner is not None, "Initialization graph owner missing")
                identity = owner["identity"]
                require(identity is None or identity == plan.identity, "Concurrent initialization conflict")
                if identity == plan.identity:
                    return
                if plan.snapshot.schema_version == 2:
                    verify_catalog_event_identity(transaction)
                inventory = transaction.run("MATCH (n) WHERE NOT n:SagaGraphOwner RETURN count(n) AS count").single(strict=True)
                require(inventory is not None and inventory["count"] == 0, "Unreceipted nonempty graph; reinitialization is blocked")
                for statement in plan.statements:
                    transaction.run(statement.query, json.loads(statement.parameters)).consume()
                for entity in plan.entities:
                    record = transaction.run(f"MATCH (n:{entity.label} {{id: $identity}}) RETURN count(n) AS count", identity=entity.identity).single(strict=True)
                    require(record is not None and record["count"] == 1, f"Graph identity coverage failed: {entity.identity}")
                transaction.run(
                    "MATCH (owner:SagaGraphOwner {key: 'exclusive', project_id: $project_id}) "
                    "SET owner.initialization_plan = $identity", project_id=plan.snapshot.project_id, identity=plan.identity,
                ).consume()

            await get_services().database.execute_in_transaction(apply)
            require(await self.receipt(plan), "Initialization acknowledgement requires exact graph receipt")
            self.retain(f"{plan.identity}-accepted", plan.identity)
            from data_access.cache_coordinator import clear_character_read_caches, clear_world_read_caches

            clear_character_read_caches()
            clear_world_read_caches()
            return plan

    def state(self, plan: InitializationGraphPlan) -> NarrativeState:
        state: dict[str, Any] = json.loads(plan.snapshot.metadata)
        state.update(project_dir=str(self.project_dir), graph_project_id=plan.snapshot.project_id, total_chapters=plan.snapshot.total_chapters, initialization_id=plan.identity)
        for artifact in plan.snapshot.artifacts:
            require(artifact.content_type in (*ARTIFACTS, CATALOG_ARTIFACT), "Unknown initialization source")
            state[artifact.content_type + "_ref"] = artifact.reference()
        return cast(NarrativeState, state)

    async def reconcile_checkpoint(self, graph: Any, state: NarrativeState, configuration: dict[str, Any]) -> NarrativeState:
        require(state.get("current_chapter") == 1 and not state.get("initialization_complete"), "Initialization recovery cannot reposition active chapter canon")
        require(state.get("attempt_id") is None, "Initialization recovery cannot replace a chapter attempt")
        require(state.get("error_node") in {None, "commit_initialization", "persist_files", "run_parsers"}, "Initialization recovery cannot clear an unrelated failure")
        identity = state.get("initialization_id")
        if not isinstance(identity, str):
            raise ValueError("Initialization recovery requires a selected identity")
        plan = self.load()
        require(plan.identity == identity and plan.snapshot == select_snapshot(state), "Initialization checkpoint differs from selected plan")
        await self.accept(identity)
        update: NarrativeState = {
            "has_fatal_error": False, "last_error": None, "error_node": None,
            "initialization_step": "parsers_complete", "current_node": "run_parsers",
        }
        await graph.aupdate_state(configuration, update, as_node="init_run_parsers")
        return {**state, **update}
