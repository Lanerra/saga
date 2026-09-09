# SAGA modernization backlog and acceptance gates

This is an implementation handoff, **not authorization to alter Git history or existing narrative data**. Source baseline and evidence are in `source-inventory.json`; start with `README.md` in this audit directory. Priority means consequence/ordering, not an estimated work duration. All work below is pending; the audit itself implemented none of it.

## Non-negotiable scope

Preserve SAGA's local-first Python CLI, scene-based generation, Neo4j narrative memory, human-readable artifacts, and semi-autonomous writer workflow. Do not turn the repair into a web app, a distributed system, or an architecture rewrite. Keep external LLM/embedding services configurable. Preserve existing stories and graph data independently of Git. Do not transplant DWARF repository publication restrictions into SAGA.

Do not split more monoliths just to reduce line counts. Move one coherent responsibility only after its boundaries and behavior are tested. Keep commit policy, query construction, storage, and acceptance policy distinct. Single-user operation does not excuse cross-project identity collisions, unsafe file paths, or unreliable recovery.

## Phase 0 — preserve and reproduce

| ID | Task | Primary files / evidence | Dependency | Acceptance |
|---|---|---|---|---|
| B01 | Preserve dirty source, refs, untracked helpers and user data separately | `git-analysis.json`, `git-branches-before.txt`, `disk-inventory.json` | None | Verified recoverable Git bundle + meaningful tracked patch + ten untracked helpers; story/content and Neo4j backup with restore test. No source/story disappearance. |
| B02 | Choose integration authority and quarantine mode noise | Current dirty `refactor/kg-schema`; `master` divergence; two patch-distinct old branches | B01 | Content diff reviewed independently of 284 executable-bit changes; explicit integration branch; old patches and merge-only differences accounted for. No reset, blind merge or force push. |
| B03 | Establish one supported runtime and lock dependencies | `requirements.txt`, `pyproject.toml`, `runtime-freeze.txt` | B01 | Fresh Linux Python 3.12 install; locked transitives and declared test/lint/type plugins; `pip check`; documented spaCy model setup; CLI help offline. Preserve the broken venv only until recoverability is settled. |
| B04 | Finish the existing module split as a bounded repair | Ten untracked `nodes/commit_*`, `context_*`, `scene_extraction_*`; their callers/tests | B02,B03 | Required helper files tracked together with callers; correct tuple-return annotation and imports; no unresolved F821 or Mypy errors; patch targets point to actual call providers. No extra decomposition. |
| B05 | Make the unit test boundary offline by default | `tests/conftest.py`, both LangGraph test trees, service fixtures | B03,B04 | Explicitly marked integration tests; denied-network unit test run; no accidental model/tokenizer downloads or Neo4j connections; real timeout plugin and asyncio loop scope; retain all meaningful tests and repair weak assertions. |

**Phase 0 exit:** one reproducible working-tree baseline whose failures are classified and owned, rather than falsely green through deselection. Preserve this audit's initial 1,639/43 result as baseline evidence, not as a release gate.

## Phase 1 — stop unsafe state transitions

| ID | Task | Primary files / evidence | Dependency | Acceptance |
|---|---|---|---|---|
| B06 | Stop destructive compensation after a batch that never committed or already rolled back | `nodes/commit_node.py:397-595`, `:207-267`; `core/db_manager.py` | B05 | Seed prior committed chapter, fail pre-batch and in-batch separately: original graph unchanged. Post-commit cache failure does not erase durable data. |
| B07 | Make failed revision rollback recovery-blocking | `nodes/revision_node.py:146-155`; lifecycle probe | B05 | Failed rollback cannot call revision LLM/generation, clears no evidence, and resumes only through reconciliation. |
| B08 | Separate planned chapters from finalized/generated progress | `parsers/chapter_outline_parser.py:1066-1124`, `chapter_queries.py:111-125`, orchestrator resume guard | B05 | All planned Chapter nodes may exist while generation still resumes at chapter 1; counts never substitute for contiguous finalized status. |
| B09 | Establish graph project ownership | Chapter/name merges, schemas, every data-access/cache/healing boundary | B01,B05 | Either explicit single-project database ownership with fail-closed mismatch as a narrow initial safety measure, or full project-scoped graph identities; two stories with identical names/chapter numbers cannot interact. Migration/restore tested. |
| B10 | Correct failure propagation and bounded retries | `scene_planning_node.py`, `scene_generation_node.py`, `subgraphs/generation.py`, orchestrator | B05 | Exhausted plan/draft failure stops before using stale plan or repeating without bound; state-signaled fatal completion produces a failed result/nonzero CLI exit. |
| B11 | Distinguish empty extraction from failed extraction | `nodes/scene_extraction.py`, `subgraphs/scene_extraction.py` | B05 | Per-scene/type completeness recorded; exhausted parser/model failure cannot publish a valid-empty result. Partial mode, if allowed, is explicit and visible. |
| B12 | Require durable canonical manuscript before finalization | `nodes/finalize_node.py`, `utils/file_io.py`, export | B05 | File failures do not silently finalize; prose remains recoverable; accepted artifact is checksum-bound; interrupted pair of Markdown/legacy writes has defined recovery. Decide whether legacy mirror is needed. |
| B13 | Fix actual semantic context predicates and budgets | `chapter_queries.py:348-372`, `context_scene_retrieval.py:104-160` | B05 | Both provisional predicates have correct truth table; previous-scenes total (including headings) stays within a measured token budget, irrespective of scene count. No claim that token or vector budget implies narrative quality. |
| B31 | Protect user-owned project files before any reinitialization/replay | `initialization/persist_files_node.py:384-477`, `parser_runner.py:132-143`; I-04,I-08 | B01,B05 | Existing rules/history/items are not replaced with stubs/empty lists; colliding character projection names fail without overwriting; stale v0 replay cannot overwrite an active v1 plan. Keep reinitialization blocked until the larger B18 protocol is ready. |

**Phase 1 exit:** no known failure path destroys a previous durable chapter, converts service failure into valid data, conflates planned/completed chapters, silently crosses project ownership, or claims a nonexistent manuscript is finalized.

## Phase 2 — one durable chapter and initialization lifecycle

| ID | Task | Primary files / evidence | Dependency | Acceptance |
|---|---|---|---|---|
| B14 | Define chapter attempt/acceptance/publication protocol | `workflow.py`, orchestrator, commit/revision/finalize, state | B06-B12 | Explicit project/chapter/attempt IDs; immutable draft/extraction refs; durable status/receipts distinguish staged, committed, accepted, published, advanced and compensation-required. One documented recovery authority reconciles Neo4j, files and checkpoint. Do not add a distributed transaction service. |
| B15 | Use actual LangGraph continuation semantics | orchestrator `:724-817`, `:450`; `lifecycle-probes.json` | B08,B14 | Real locked checkpointer interrupt matrix verifies first resumed node and no duplicated side effects. Distinguish mid-run resume (`None`/native continuation) from a deliberate new invocation after completed END. Do not just replace every call with `None`. |
| B16 | Validate candidate facts against prior accepted canon | `subgraphs/validation.py:321-373,544-665`, commit order | B14 | Current/rejected/planned facts cannot masquerade as prior accepted truth; input row order cannot change contradiction findings; dynamic probe 0-versus-1 inconsistency disappears. |
| B17 | Unify fact assertion provenance and relationship writers | `commit_node.py:441-464,682-696,952-1010`, native builder | B09,B14 | Profile-origin edges survive; repeated facts have chapter-scoped assertion membership; revise chapter N without removing chapter N-1 evidence; canonical IDs used across all entity labels. |
| B18 | Make initialization an explicit staged import with one acceptance boundary | initialization nodes, `parser_runner.py`, all outline parsers | B09,B14 | Frozen typed initialization payloads; semantic coverage/identity checks before writes; one rollback/retry story for partially executed parsers; accepted user edits persisted deliberately, not overwritten by a second writer. See `initialization-review.md`. |
| B19 | Verify schema and migration prerequisites, not just attempt DDL | `core/db_manager.py`, query builders, schema models | B09,B17,B18 | Missing required constraints/APOC/indexes fail before dependent writes; legacy snapshot migration and restore tests; project-scoped identity invariants validated with disposable Neo4j. |
| B32 | Round-trip the actual character domain fields | native character builder, `commit_init_node.py:195-243`, enrichment node/parser; I-07,I-09 | B17,B18 | Explicit graph-versus-file ownership for motivations/background/skills/internal conflict/protagonist status; physical-description updates persist through the actual writer and read API. No successful no-op persistence. |

**Phase 2 exit:** deterministic fake-provider novel run, interruption after every persistent boundary, process restart with SQLite checkpoint, and a disposable real Neo4j/APOC graph all converge on the same accepted state and manuscript. No real user's graph is the integration fixture.

## Phase 3 — explicit services, configuration and generation quality

| ID | Task | Primary files / evidence | Dependency | Acceptance |
|---|---|---|---|---|
| B20 | Replace global service rewiring with a per-run service context | `llm_interface_refactored.py`, `_LLM_SERVICE_PATCH_MODULES`, DB/embedding/parser imports | B04,B05; may proceed alongside Phase 1 on disjoint ownership | CLI/service bootstrap owns one managed client lifetime; nodes/parsers consume injected interfaces; no module-name patch registry; two sequential runs close and recreate clients; fakes work without patching imported aliases. |
| B21 | Consolidate validated run configuration and provider contracts | `config/`, HTTP and LLM services | B03,B20 | One immutable effective config per run; positive attempt/concurrency/deadline constraints; coherent process-over-file precedence; exact provider schemas; configured dedicated embedding credential reaches only its endpoint, with no completion-key fallback; prompt/response canaries absent from ordinary operational logs. See `inference-review.md`. |
| B22 | Enforce vector, truncation and total-request budgets | embedding service, entity/scene aggregation, context construction, HTTP retries | B13,B20,B21 | Rank/dimensions/finiteness/model identity validated at every storage/query seam; stale cache cannot cross models; total retry deadline/cancellation; retrieval and full serialized messages fit prompt+completion budget. |
| B23 | Make quality acceptance versus advisory maintenance explicit | validation subgraph, QA/healing, scene prompts | B14,B16 | Mandatory gate failure cannot finalize; intentional force-continue/cadence/max-revision decisions generate visible accepted-with-exceptions receipts; quality scores and completeness preserved; test both strict and advisory policies. |
| B24 | Honor author configuration and measure creative regressions | project config, scene_generation prompt, bootstrap | B21,B23 | Project narrative style reaches prompts instead of global default; chapter/scene target lengths remain coherent; fixed fixture stories assess continuity, fact coverage, repetition, perspective, prose fidelity and usefulness. Establish baseline with one explicitly configured local model only after safety gates. |

## Phase 4 — simplify and polish without semantic loss

| ID | Task | Primary files / evidence | Dependency | Acceptance |
|---|---|---|---|---|
| B25 | Reduce state by contract, not naive string counts | `state.py`, helpers, checkpoint migration | B14,B15,B20 | Phase-specific required state structures; truly dead fields removed only after reads, writes, telemetry and checkpoint compatibility are mapped; stored `generated_embedding` contract resolved. `state-field-uses.json` is a lookup aid, not deletion permission. |
| B26 | Deepen modules along established boundaries | commit/query construction, parsers, ContentManager, prompt getters, healing | B17-B25 | Responsibility split lowers coupling and duplicate policy, not just file size. Character/Location parser duplicate resolved. Preserve test coverage and behavior; new module code actually owns live symbols. |
| B27 | Make content storage strict and reference-safe | `content_manager.py`, `file_io.py` | B12,B14 for full storage redesign; B05 for narrow containment guard | All path-containment and immutable-reference probes pass; create-only/versioned semantics preserve old checkpoints; migration adapters only at explicit boundaries. Bring the narrow containment guard into Phase 1 before importing references or running cleanup operations; do not wait for polish to protect file boundaries. |
| B28 | Deliver predictable writer-facing CLI and export | `main.py`, project manager, Rich UI, export | B12,B15,B24 | Explicit project selection, collision-safe bootstrap/config, honest success/failure/cancel summaries; exact frontmatter/prose export; resume/reset scope visible; all README commands smoke-tested. |
| B29 | Reconcile docs and operational tools | README, AGENTS/CLAUDE, architecture docs, reset/visualization scripts | B28 | One current guide with valid links/paths/commands; destructive tools guarded and project-scoped; remove copied irrelevant test/webservice conventions; old audits clearly historical. |
| B30 | Finish Git and repository curation | Branch matrix, ignore rules, tests/prompts/docs/source boundary | B01,B02,B26-B29 | Clean reproducible checkout; required new files tracked; generated data/credentials excluded; retained tests/docs/tools included; obsolete refs removed only after evidence review and backup; no history rewrite unless separately requested. |

## First implementation slice I recommend

Begin with **B01–B05**, then the narrow safety fixes **B06–B13 and B31**, before the broader lifecycle/schema migration. Bring forward B27's containment guard before accepting imported refs or executing cleanup. These are bounded, testable and protect the value already in SAGA. B20 can start with a tiny explicit seam needed to make those tests honest, but a repo-wide injection rewrite should not precede preservation or block urgent data-safety repairs. B31 and B32 were numbered after the initial task inventory; numbering is identity, while phase placement/dependencies determine order.

Do not claim this entire backlog is complete because Ruff/Mypy/unit tests become green. The decisive release gate is an interruption-safe, project-isolated, deterministic end-to-end authoring run with actual graph semantics and honest manuscript publication.

## Suggested durable test matrix

1. Pure domain/schema/parser contracts, including malformed/empty/partial inputs.
2. Full in-memory workflow using deterministic LLM/embedding fixtures and real ContentManager/SQLite.
3. Disposable Neo4j+APOC tests for statement ordering, constraints, identity, assertion provenance, compensation and vectors.
4. Crash/restart injection at each before/after DB acknowledgement, file promotion, checkpoint and chapter advancement boundary.
5. Two synthetic projects with intentionally identical display names and chapter numbers.
6. Failed generation/quality/embedding/parse/IO paths, bounded retries and explicit degraded policy.
7. Exact manuscript export fidelity and resume artifact integrity.
8. One bounded real-model smoke and fixed-story narrative regression panel; no provider or narrative-quality certification was produced by this audit.
