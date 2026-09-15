# SAGA workflow and recovery audit

**Scope:** read-only review of the requested LangGraph workflow, generation/validation/extraction subgraphs, revision/finalize/QA nodes, orchestrator, and focused tests. This is a source audit, not a database/LLM/generation run.

**Snapshot:** `/home/dlewis3/Desktop/AI/saga`, `refactor/kg-schema`, `4ce8870388b54e19ead91c86c10273e0fdd32e61`; worktree was already broadly dirty. Eleven reviewed production files had identical SHA-256 values over a two-second quiet window. The report is the only file written by this audit.

**Evidence boundary:** findings marked **source-proven** follow current control flow. Claims about the installed LangGraph runtime's resume semantics are marked **runtime-unverified**: the available `python` cannot import `langgraph`, and no dependency was installed. Parent-provided offline-suite evidence is `1639 passed / 43 failed` in `pytest.xml`; it was not independently re-run for this bounded review.

## Architecture and persistence map

```text
new run
  orchestrator._load_or_create_state()
  -> workflow route
  -> initialization
       ... -> init_run_parsers -> ParserRunner.run_all_parsers()
                              -> ChapterOutlineParser.parse_and_persist()
                              -> creates every outlined :Chapter, Scene, Event
  -> chapter_outline -> generation subgraph
       plan_scenes -> retrieve_context -> draft_scene (loop)
  -> scene-extraction subgraph
       extract_from_scenes -> consolidate
  -> embeddings -> assemble -> enrichment -> normalize
  -> commit (Neo4j chapter/entities/relationships)
  -> validation subgraph
       consistency -> LLM quality -> contradiction detection
  -> revise -> generation (when requested), else summarize -> finalize
  -> heal_graph -> check_quality -> advance_chapter or END

resume
  checkpointer.aget(thread_id) -> checkpoint.channel_values
  -> `_validate_resume_state_or_raise_async()`
  -> graph.astream(checkpoint_state, config=...)
```

**Persistent mutation points.** Scene planning can write provisional character stubs (`core/langgraph/nodes/scene_planning_node.py:129-202`); initialization parsers write the planned graph; `commit` writes a transactional chapter/entity/relationship batch (`core/langgraph/nodes/commit_node.py:437-526`); `finalize` writes chapter files then updates Neo4j (`core/langgraph/nodes/finalize_node.py:99-185`); QA can mutate relationships (`core/langgraph/nodes/quality_assurance_node.py:110-146`). No state field names a durable chapter phase, commit receipt, compensating-transaction receipt, or finalization receipt.

## Findings

### WFR-01 — P0: initialization turns future planned chapters into resume-conflict evidence

**Source-proven.**

- **Evidence:** initialization always reaches `init_run_parsers` before `init_complete` (`core/langgraph/workflow.py:452-472`). The runner invokes `ChapterOutlineParser` for the all-chapter outline artifact (`core/parser_runner.py:51-67,132-143`). Its `parse_and_persist()` iterates every entry in `all_chapter_data` (`core/parsers/chapter_outline_parser.py:1066-1115`) and creates all collected `:Chapter` nodes (`:1119-1124`). The counter used both for fresh starts and resume conflict detection is unqualified `MATCH (c:Chapter) RETURN count(c)` (`data_access/chapter_queries.py:111-125`). Resume rejects when that count is **greater than or equal to** the checkpoint's current chapter (`orchestration/langgraph_orchestrator.py:797-800`).
- **Activation:** after successful initialization, any checkpointed restart while generating a normal chapter. The graph already contains all planned Chapter nodes; for an N-chapter project, `chapter_count=N`, so it is at or ahead of every valid `current_chapter` from 1 through N.
- **Impact:** restart after initialization is systematically rejected before generation resumes. The count does not mean “finalized/generated chapters,” contrary to its use as recovery evidence.
- **Smallest repair:** separate planned-outline graph structure from generated/published chapter state. Resume against a chapter-specific durable receipt/status (for example `generation_status=committed|validated|finalized`) keyed by project and chapter, not the cardinality of all `:Chapter` nodes. Do not relax `>=` alone.
- **Acceptance test:** run initialization for outlines containing chapters 1–3, assert that all three planned Chapter nodes exist, then resume a checkpoint at chapter 1 and at chapter 2. Both must enter their defined pending phase; a generated/finalized status ahead of the checkpoint must still fail closed. Existing conflict coverage uses only a mocked count of 4 versus checkpoint 3 (`tests/test_langgraph_orchestrator.py:918-946`) and does not construct initialized planned chapters.

### WFR-02 — P0: “resume” injects checkpoint values as a new graph input rather than continuing a saved task/cursor

**Source-proven invocation; runtime-unverified precise LangGraph behavior.**

- **Evidence:** the orchestrator obtains `checkpoint["channel_values"]` and returns that mapping as state (`orchestration/langgraph_orchestrator.py:739-767`). It then calls `graph.astream(state, config=...)` with the non-empty mapping (`:443-465,450`) rather than a continuation call with no input or an explicit saved task/cursor. No code reads a LangGraph pending-node/task identifier or `get_state()`/`aget_state()` result.
- **Activation:** any interruption after a checkpoint but before a chapter reaches END, especially between side-effecting nodes.
- **Impact:** the implementation has no source-level mechanism to preserve and resume the pending node. Depending on the installed LangGraph version's non-empty-input semantics, it can restart at `route` and replay already performed work, or otherwise merge a new input over checkpoint state. This is incompatible with exactly-once persistence unless every node is phase-idempotent and recovery is explicitly reconciled.
- **Smallest repair:** choose and document one recovery contract: (a) use the checkpointer's native continuation API with no fresh input and test the saved next-node/task, or (b) deliberately replay from a recorded phase, with idempotency/compensation receipts at every write boundary. Store the workflow schema/version with the receipt.
- **Acceptance test:** instrument a checkpointer-backed graph with a controlled interrupt after each of `plan_scenes`, `commit`, `validate`, `finalize`, and `check_quality`; restart via the production orchestrator. Assert the first resumed node and exactly-once counts for each side effect. This test must run with the production locked LangGraph version; static review cannot prove its scheduler semantics. Current checkpoint coverage only checks that SQLite exists after a completed `ainvoke()` (`tests/test_langgraph/test_phase2_workflow.py:394-413`).

### WFR-03 — P0: commit precedes acceptance, with no durable phase to reconcile the commit window

**Source-proven control flow; crash timing runtime-unverified.**

- **Evidence:** the main graph routes `commit -> validate`, followed only later by `summarize -> finalize -> heal_graph -> check_quality -> advance_chapter` (`core/langgraph/workflow.py:544-610`). Commit adds a Chapter upsert to its batch and executes it (`core/langgraph/nodes/commit_node.py:514-526`); only `advance_chapter` increments `current_chapter` (`core/langgraph/workflow.py:207-240`). Resume has only `current_chapter` and a global node count, not a phase receipt (`orchestration/langgraph_orchestrator.py:769-817`).
- **Activation:** interruption after the Neo4j commit acknowledgement but before a durable checkpoint records validation/finalization/advance; this remains relevant even after WFR-01 is repaired.
- **Impact:** the recovery code cannot distinguish a pre-commit checkpoint, a committed-but-unvalidated chapter, a finalized chapter, or a chapter whose compensation is required. With the current `>=` rule it rejects equality; with a naïve rule change it could replay commit/validation against an ambiguous graph state.
- **Smallest repair:** write an atomic, project-scoped chapter attempt/phase receipt around the persistence boundary, then reconcile on restart: `planned`, `commit_started`, `committed`, `validated`, `finalized`, `advanced`, and `compensation_required/complete`. Bind it to draft/extraction content identities and transaction/attempt ID.
- **Acceptance test:** create a checkpoint at chapter N plus a matching durable `committed` receipt and Chapter N data; resume must execute validation without a second commit. A `commit_started` receipt must retry/reconcile explicitly; a finalized receipt must advance only. A mismatched receipt/content identity must stop before mutation. The focused test named `test_workflow_graph_persistence_wiring_is_after_validation` actually asserts `commit -> validate` (`tests/test_langgraph/test_phase2_workflow.py:551-562`), so it does not prove its stated ordering contract.

### WFR-04 — P1: revision explicitly continues after compensating rollback failure

**Source-proven.**

- **Evidence:** rollback removes chapter relationships, marks current-chapter entities provisional, deletes scenes, then the Chapter node (`core/langgraph/nodes/revision_node.py:59-106`). `revise_chapter()` catches every rollback exception and continues (`:146-155`); successful guidance then clears generation/extraction errors and returns to generation (`:291-300`; `core/langgraph/workflow.py:565-573`). A later commit filters entities already present in Neo4j (`core/langgraph/nodes/commit_node.py:370-391`).
- **Activation:** validation requests revision after commit and `execute_cypher_batch()` for compensation fails partially or fully.
- **Impact:** rejected-draft graph facts remain while revision regenerates. The next commit can preserve stale entities through the existing-name filter; there is no durable signal requiring reconciliation before another write.
- **Smallest repair:** make rollback failure fatal/recovery-blocked, retain the original chapter attempt identity and error, and require a successful idempotent compensation or explicit operator reconciliation before `generate` is reachable. Record compensation completion durably.
- **Acceptance test:** force the rollback batch to fail, then execute the graph. Assert the revision LLM and generation subgraph are not called, `has_fatal_error` or `compensation_required` persists, and a later restart retries/reconciles before any commit. The revision tests cover artifact clearing and guidance failure only (`tests/test_langgraph/test_revision_node.py:111-157`).

### WFR-05 — P1: failed revision re-plan can draft one scene from the rejected old plan

**Source-proven.**

- **Evidence:** revision clears generated artifacts and resets `current_scene_index`/`chapter_plan_scene_count`, but does not clear `chapter_plan_ref` (`core/langgraph/state_helpers.py:15-23`; `core/langgraph/nodes/revision_node.py:291-300`). Planner failure returns only `last_error`, not a fatal marker (`core/langgraph/nodes/scene_planning_node.py:333-338`), and the generation gate inspects only `has_fatal_error` (`core/langgraph/subgraphs/_shared.py:9-13`; `generation.py:71-85`). Context retrieval loads the retained plan (`core/langgraph/nodes/context_retrieval_node.py:74-85`); after one draft, the zero scene count makes the loop end (`core/langgraph/subgraphs/generation.py:42-54`).
- **Activation:** a revision has completed, the next `plan_scenes` exhausts/fails, and the previous plan reference remains readable.
- **Impact:** a partial single-scene chapter based on the rejected plan can progress through extraction, commit, and validation.
- **Smallest repair:** clear `chapter_plan_ref` as part of revision generation-artifact clearing; distinguish plan failure from a valid empty plan and route terminal failure to error before retrieval/drafting.
- **Acceptance test:** seed a revised state with an old plan; force all planning attempts to fail; assert no context retrieval, scene draft, extraction, or commit occurs. Also prove a legitimate zero-scene plan is either rejected by contract or handled through a distinct explicit path. Focused generation coverage is success-only (`tests/core/langgraph/subgraphs/test_generation_subgraph.py:12-67`).

### WFR-06 — P1: scene draft exception re-enters the generation loop without a budget

**Source-proven.**

- **Evidence:** `draft_scene()` catches every LLM exception and returns only `last_error` (`core/langgraph/nodes/scene_generation_node.py:155-160`). The next router ignores `last_error`; if scenes remain, it returns `continue` (`core/langgraph/subgraphs/generation.py:39-54`) and returns to context retrieval then drafting (`:76-86`).
- **Activation:** any draft LLM/serialization/filesystem exception before the final planned scene.
- **Impact:** unbounded repeated context/LLM attempts until LangGraph's outer recursion limit, with no clear failure state and potentially repeated cost/side effects.
- **Smallest repair:** set fatal state for non-retryable failure, or add a persisted per-scene retry counter and error route on exhaustion.
- **Acceptance test:** make the draft LLM raise persistently; assert an exact bounded call count, no subsequent context/draft cycle after exhaustion, and a non-success orchestrator outcome.

### WFR-07 — P1: exhausted scene extraction is silently published as a valid empty extraction

**Source-proven.**

- **Evidence:** every per-type extractor uses `max_attempts=2` then catches `LLMServiceError` and generic exceptions as `[]` (characters `core/langgraph/nodes/scene_extraction.py:176-185,243-259`; locations `:296-305,349-365`; events `:402-411,455-471`; relationships `:509-518,563-579`). `extract_from_scenes()` appends those results, consolidates, and saves normal refs (`:664-728`); the subgraph proceeds unless `has_fatal_error` is set (`core/langgraph/subgraphs/scene_extraction.py:37-42`).
- **Activation:** a per-scene/per-type extraction exhausts retries or throws.
- **Impact:** missing knowledge-graph facts are indistinguishable from a genuine empty extraction and may be committed as accepted data.
- **Smallest repair:** preserve a per-scene/type outcome ledger. Treat exhausted extraction as fatal before commit, or require an explicit partial-extraction policy/override that is visible to validation and finalization.
- **Acceptance test:** make one relationship extraction exhaust while other extractors succeed; assert extraction cannot reach `consolidate`/`commit` without an explicit partial override. The focused subgraph test only asserts that refs exist after all successful empty responses (`tests/core/langgraph/subgraphs/test_scene_extraction_subgraph.py:11-46`).

### WFR-08 — P1: validation and QA are fail-open; QA is after finalization and not a gate

**Source-proven.**

- **Evidence:** quality-evaluation exceptions return `None` scores and feedback without fatal/revision state (`core/langgraph/subgraphs/validation.py:180-194`); its routing gate checks only `has_fatal_error` (`:687-697`). Contradiction detection deliberately accepts critical/major issues at the iteration limit (`:513-535`), and the workflow also skips requested revision under `force_continue` (`core/langgraph/workflow.py:84-106`). QA runs only after `finalize` and `heal_graph` (`:575-607`), can be disabled or cadence-skipped (`core/langgraph/nodes/quality_assurance_node.py:54-71`), catches each query/write error (`:89-146`), clears `last_error` (`:174-182`), and its results are not inspected by the next-chapter router (`core/langgraph/workflow.py:243-285,600-607`).
- **Activation:** quality LLM failure, malformed/absent quality input, issue at max revision iterations, `force_continue`, QA disabled/skipped, or any QA check failure.
- **Impact:** a chapter can be finalized and the workflow can advance without successful quality evaluation or QA checks; even a detected critical validation issue can be intentionally accepted with no durable acceptance receipt. QA cannot prevent publication because it occurs after finalization.
- **Smallest repair:** declare policy explicitly. If quality/QA is mandatory, make validation completeness and QA success required gates before finalization and surface a typed fatal state. If best-effort/override is intended, persist the reason, actor/configuration, issue set, and an explicit `accepted_with_exceptions` receipt; do not clear operational errors.
- **Acceptance test:** (1) quality LLM error cannot reach summarize/finalize without an explicit override; (2) a QA query failure or issue does not advance when QA is mandatory; (3) override/max-iteration acceptance yields a durable exception receipt; (4) QA success allows advance. Existing tests assert the fail-open evaluator error (`tests/core/langgraph/subgraphs/test_validation_subgraph.py:150-162`), best-effort acceptance (`:451-479`), force-continue bypass (`:424-449`), and only QA timestamp/disabled behavior (`tests/test_langgraph/test_quality_assurance_node.py:21-88`).

### WFR-09 — P1: validation reads its own just-committed relationships without order or chapter filter

**Source-proven.**

- **Evidence:** `commit` precedes validation (`core/langgraph/workflow.py:544-562`). `_fetch_validation_data(current_chapter)` sends a query that accepts `current_chapter` but neither uses it in Cypher nor orders rows (`core/langgraph/subgraphs/validation.py:321-340`), retaining only the first row for each endpoint pair (`:346-355`). Relationship-evolution detection treats that selection as prior state (`:585-613`).
- **Activation:** a current relationship and an historical relationship share endpoints, and Neo4j returns the current row first (for example historical `HATES`, current `LOVES`).
- **Impact:** the code can compare `LOVES` to `LOVES`, miss the transition, and skip revision; outcome is result-order dependent.
- **Smallest repair:** validate against the pre-commit relationship snapshot, or query only prior chapters and order deterministically, e.g. `WHERE coalesce(r.chapter_added, -1) < $current_chapter ORDER BY r.chapter_added DESC`.
- **Acceptance test:** return current `LOVES` before historical `HATES` for the same pair; `_fetch_validation_data(5)` must select historical `HATES`, and contradiction detection must produce the expected transition. Current query tests use one row per endpoint pair or inject the desired mapping directly (`tests/test_validation_query_optimization.py:20-61`).

### WFR-10 — P2: fatal workflow termination returns normally and is logged as generation complete

**Source-proven.**

- **Evidence:** the error handler only sets `current_node` then ends the graph (`core/langgraph/workflow.py:109-126,388-390`). After stream completion, the orchestrator logs a fatal state but does not raise (`orchestration/langgraph_orchestrator.py:485-512`), then its outer method always logs `Generation Complete` if no exception escaped (`:240-250`).
- **Activation:** any node communicates failure through `has_fatal_error=True` rather than raising.
- **Impact:** callers and CLI supervision can receive a normal return/completion banner for an incomplete, possibly post-commit workflow. That obscures the recovery requirement and can create false terminal automation records.
- **Smallest repair:** raise a typed workflow-fatal exception after the stream when final state is fatal; emit a distinct failed outcome/receipt rather than completion.
- **Acceptance test:** stream a fatal node into `error_handler`; assert the production orchestration method raises/nonzero exits and does not emit the completion event. Existing fatal-routing tests stop at `graph.ainvoke()` state (`tests/test_langgraph/test_fatal_error_routing.py:43-104`), and orchestrator stream tests accept incomplete/no-event runs without a terminal failure assertion (`tests/test_langgraph_orchestrator.py:419-453`).

### WFR-11 — P2: finalization can report success after canonical prose-file failure

**Source-proven.**

- **Evidence:** finalization writes the markdown and text chapter files first (`core/langgraph/nodes/finalize_node.py:99-115,208-279`), catches any filesystem failure, and continues to Neo4j. The subsequent database write stores summary/embedding/provisional status (`:153-185`; `data_access/chapter_queries.py:131-177`), not the draft text. It then returns a normal finalization state (`core/langgraph/nodes/finalize_node.py:187-205`).
- **Activation:** any filesystem failure, including interruption/partial failure between the markdown and legacy text writes.
- **Impact:** the graph's stated source of truth does not retain canonical prose, yet the chapter can finalize and advance. There is no atomic artifact promotion or receipt to distinguish complete from absent/partial files on restart.
- **Smallest repair:** write both files to a sibling temporary location, fsync/promote atomically where required, then record and verify an artifact checksum/receipt before treating finalization as successful. If a database-only chapter is deliberate, store durable prose or explicitly model it as non-final.
- **Acceptance test:** inject failure before each file write and between the two writes; assert no finalization receipt/advance occurs and retry produces exactly one consistent pair. The current test intentionally asserts filesystem failure still succeeds (`tests/test_langgraph/test_finalize_node.py:240-252`).

## Focused test gaps and repair order

The focused suites prove graph shape, ordinary happy-path transitions, direct fatal routing, and selected node-local behavior. They do **not** prove an initialized project can resume, a checkpoint continues its pending node, or any post-write interruption is reconciled. The broad suite outcome therefore cannot establish recovery correctness.

Recommended order:

1. Define a project-scoped generated-chapter/attempt schema independent of initialization's planned Chapter nodes (WFR-01).
2. Choose native checkpoint continuation or phase-replay semantics and add an interrupt matrix against the production LangGraph version (WFR-02).
3. Make commit/validation/finalize/compensation a durable phase machine with immutable attempt and artifact identities (WFR-03, WFR-04, WFR-11).
4. Fail closed for plan/draft/extraction failures and clear all stale scene-plan state on revision (WFR-05–WFR-07).
5. Make acceptance/override policy explicit and auditable; place mandatory gates before finalization (WFR-08–WFR-10).

No source files, tests, Git state, network services, databases, LLMs, dependencies, or generated stories were modified/read by this audit.
