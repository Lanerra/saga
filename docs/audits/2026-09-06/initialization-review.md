# SAGA initialization / parser / domain audit

**Date:** 2026-09-06
**Scope:** static, read-only trace of the initialization persistence and parser boundary requested below. No services, database, story project, network, installation, or test command was used. This report is grounded in the dirty worktree at `4ce8870` (branch `refactor/kg-schema`, ahead 1); it is not a clean-commit verdict.

## Verdict

**NOT APPROVED for retry-safe initialization.** The current path has two competing artifact stores, repeats entity extraction and graph writes, accepts partial chapter plans as successful initialization, and has no rollback boundary for parser-stage writes. It can therefore report initialization failure after writing an incomplete graph, or report success with a graph based on stale/partial artifacts.

## Verified hazards

### I-01 — Human-facing YAML and parser input JSON are competing sources of truth

**Evidence**

- `persist_initialization_files()` describes its YAML outputs as the canonical on-disk inputs for later phases (`core/langgraph/initialization/persist_files_node.py:2-10`), then writes `characters/*.yaml`, `outline/*.yaml`, `world/*.yaml`, `saga.yaml`, and stubs (`:125-148`, `:262-292`, `:301-376`, `:384-477`).
- Its post-write validator explicitly validates existence only, not YAML content or linkage to an initialization snapshot (`core/langgraph/initialization/validation.py:2-6`, `:45-71`).
- In contrast, `ParserRunner` constructs fixed `.saga/content` JSON paths with the private `ContentManager._get_content_path()` and gives those paths directly to parsers (`core/parser_runner.py:39-41`, `:120-143`). Parsers use direct `open(...); json.load(...)`, for example character sheets (`core/parsers/character_sheet_parser.py:62-83`), global outline (`core/parsers/global_outline_parser.py:70-81`), acts (`core/parsers/act_outline_parser.py:67-78`), and chapters (`core/parsers/chapter_outline_parser.py:66-77`).
- `ContentManager.save_json()` supplies a checksum-bearing `ContentRef` (`core/langgraph/content_manager.py:275-318`), but `run_initialization_parsers()` receives only `project_dir` and passes no ContentRef to the runner (`core/langgraph/initialization/run_parsers_node.py:44-49`).

**Activation / impact**

An editor changes a persisted YAML file, a prior file remains in the project, or `.saga/content/*_v*.json` is replaced after generation. The persistence node may validate its YAML representation and the parser then builds the graph from a different, unchecked JSON payload. Resume/retry behavior is also tied to static filenames rather than to the state snapshot that the workflow generated. This breaks the claimed canonical-file contract and makes the graph neither reproducible from YAML nor bound to the state ContentRefs.

**Smallest repair direction**

Define one versioned `InitializationSnapshot` manifest containing all artifact paths, versions, checksums, semantic schema version, and expected chapter/act counts. Either make the YAML projections read-only exports of that snapshot, or make YAML the parsed source and atomically regenerate a checksum-bound snapshot from it. Pass the selected `ContentRef`s (not a project directory) into parser adapters and use `load_json_strict()` before any graph write.

**Regression proof**

Create a valid snapshot, alter one byte of the selected JSON or edit YAML without regeneration, and assert no parser database method is called. Separately prove that an intentional YAML edit is either the input used to regenerate the snapshot or is rejected as an unmanaged divergence.

---

### I-02 — Parser failure occurs after dependent parsers have already committed work

**Evidence**

- `ParserRunner.run_all_parsers()` records a failed parser then deliberately continues through the remaining parsers (`core/parser_runner.py:51-67`). The existing unit test explicitly locks in this behavior (`tests/test_parser_runner.py:151-175`; inspected, not run).
- The LangGraph node learns that a parser failed only after `run_all_parsers()` returns (`core/langgraph/initialization/run_parsers_node.py:49-67`).
- Parser persistence itself is staged and non-atomic. Character node sync occurs before relationship writes (`core/parsers/character_sheet_parser.py:403-412`); global parser creates plot points, locations, items, arc updates, and possession edges in separate steps (`core/parsers/global_outline_parser.py:723-772`); chapter parser creates chapters, scenes, events, locations, then relationships in separate steps (`core/parsers/chapter_outline_parser.py:1119-1151`).
- `execute_write_query()` creates an individual driver `execute_write` transaction per call (`core/db_manager.py:264-269`). The available batch API does use one explicit transaction (`core/db_manager.py:271-339`), but these parser loops do not use it.

**Activation / impact**

If `global_outline` fails after the character parser succeeds, the act and chapter parsers still run against an incomplete prerequisite graph. If any later parser substep fails, its prior individual writes remain. The node ultimately emits `parsers_failed`, but there is no rollback or cleanup receipt for graph data already written. A retry can then merge against that contaminated partial graph.

**Smallest repair direction**

First load and validate every input into a pure, immutable `InitializationGraphPlan`; only then execute one ordered transactional batch. Until that seam exists, fail fast at the first parser failure and mark the project as needing explicit cleanup rather than invoking dependent parsers. Do not call the resulting state resumable until the transaction receipt is published.

**Regression proof**

Inject a failure in each stage and a failure in the final statement of each parser plan. Assert: (1) no dependent parser starts after an upstream failure, (2) no graph writes remain after a failed batch, and (3) a retry produces exactly the same node/edge inventory as a clean first run.

---

### I-03 — Partial all-chapter generation is marked complete and is accepted by the chapter parser

**Evidence**

- `generate_all_chapter_outlines()` logs and skips an absent chapter result (`core/langgraph/initialization/all_chapter_outlines_node.py:73-98`). It fails only when **no** outline was produced (`:100-107`); otherwise it saves the incomplete mapping and returns `all_chapter_outlines_complete` (`:109-131`).
- The existing focused test intentionally expects chapter 2 to fail while the node returns complete with no error (`tests/core/langgraph/initialization/test_all_chapter_outlines_node.py:74-106`; inspected, not run).
- `ChapterOutlineParser.parse_and_persist()` silently skips non-dict entries, requires only at least one resulting chapter, then writes all parsed entries (`core/parsers/chapter_outline_parser.py:1071-1117`, `:1119-1161`). It does not compare keys with `1..state.total_chapters` because it receives only a file path.
- The normal workflow proceeds from all-chapter generation into commit, file persistence, and parser execution (`core/langgraph/workflow.py:420-466`).

**Activation / impact**

Any single LLM timeout, empty response, or parsing error for a chapter leaves a hole in the saved v0 map but allows initialization to reach `initialization_complete`. The stage-4 graph has no chapter node for that hole; downstream on-demand generation can later create a different v1 outline without a clear recovery or reconciliation policy.

**Smallest repair direction**

For the enabled all-chapters mode, validate a complete, unique key set `{1, …, total_chapters}`, each outline's embedded `chapter_number`, and act allocation before persisting the ContentRef. Treat any omission as a failed initialization snapshot. If partial plans are a supported product mode, give them a distinct state/status and prohibit the structural parser from running until the missing-chapter set is explicitly resolved.

**Regression proof**

For `total_chapters=3`, force chapter 2 to return `None`; assert no `chapter_outlines_ref` is published, no parser runs, and no graph write occurs. Add duplicate-key, out-of-range-key, and embedded-number-mismatch cases.

---

### I-04 — Parser replay prefers stale v0 over enriched v1 and overwrites graph entities

**Evidence**

- Initial all-chapter generation saves version 0 (`core/langgraph/initialization/all_chapter_outlines_node.py:109-116`).
- Enriching a v0 skeleton later saves the updated map as version 1 while retaining the v0 file (`core/langgraph/initialization/chapter_outline_node.py:79-113`).
- `ParserRunner` selects v0 whenever it exists; it only chooses v1 when v0 is absent (`core/parser_runner.py:132-143`).
- Chapter, scene, and scene-event writes use `MERGE` plus `ON MATCH SET`, so a replay replaces their current fields (`core/parsers/chapter_outline_parser.py:399-417`, `:458-486`, `:532-560`).

**Activation / impact**

Running the parser CLI after any chapter has been enriched re-parses the old skeleton rather than the active v1 content. The replay overwrites chapter/scene/event fields with stale v0 data. The static priority is also invisible to the workflow state, which carries a ContentRef rather than a generic “latest” selector.

**Smallest repair direction**

Make the selected snapshot/ref authoritative. The runner must take the exact chapter ContentRef from state or a manifest and reject a replay whose version/checksum is not the selected published snapshot. If a latest-version policy is intended, implement it once in `ContentManager`, validate it against the manifest, and remove the hard-coded v0 preference.

**Regression proof**

Publish distinct v0 and v1 chapter payloads; assert parser construction uses the selected v1 ref and graph statements contain v1 values. Verify that a request to replay v0 against an active v1 snapshot is rejected without writes.

---

### I-05 — Initialization commits characters/world data twice, with separate LLM extraction contracts

**Evidence**

- Before parser execution, `commit_initialization_to_graph()` converts character sheets, independently extracts world items from the global outline through an LLM, and batch-commits the result (`core/langgraph/initialization/commit_init_node.py:95-143`, `:423-539`, `:625-685`).
- The subsequent parser runner always invokes `CharacterSheetParser` and `GlobalOutlineParser` (`core/parser_runner.py:51-67`). The character parser writes the same character sheets (`core/parsers/character_sheet_parser.py:378-414`). The global parser independently calls an LLM to extract world items (`core/parsers/global_outline_parser.py:230-272`) and then writes locations/items (`:696-742`; write identities at `:415-446` and `:471-502`).
- These extraction calls are not one recorded result: the initialization commit can use `state.medium_model` (`core/langgraph/initialization/commit_init_node.py:105-111`, `:449-460`), while the global parser uses `config.NARRATIVE_MODEL` (`core/parsers/global_outline_parser.py:254-265`). The commit path parses world items strictly with exact keys (`core/langgraph/initialization/commit_init_node.py:471-521`), while the parser path silently skips malformed item entries (`core/parsers/global_outline_parser.py:299-322`).
- The two writer families merge differently: the native commit path keys world items by `name` (`data_access/cypher_builders/native_builders.py:196-221`), whereas the legacy global parser keys locations/items by `id` (`core/parsers/global_outline_parser.py:415-431`, `:471-487`).

**Activation / impact**

Every normal initialization executes duplicate character persistence and two independent, non-seeded extractions of the same global outline. When those calls differ in item name, category, or description—as their separate model/configuration and calls permit—the graph can contain different results from each writer or have later parser writes overwrite the first result. Strictness also depends on which writer ran last. This is a true ownership conflict, not harmless retry idempotency.

**Smallest repair direction**

Choose one domain-plan producer and one graph committer. LLM extraction must produce a checksum-bound `WorldItemPlan` once; parser adapters may validate/normalize it but must not generate a second plan. Use one canonical identity key and one upsert implementation for every initialization entity type.

**Regression proof**

Stub the extraction service with distinguishable results and assert one invocation per initialization snapshot, one generated plan, and one writer path. Replay the same plan twice and compare complete graph node/edge/property inventories rather than only successful return messages.

---

### I-06 — Corrupt or unreadable outline-relationship input is silently omitted while commit succeeds

**Evidence**

- When `outline_relationships_ref` is set, the commit node loads it inside a broad `try`; any exception only produces a warning and leaves `outline_relationships=[]` (`core/langgraph/initialization/commit_init_node.py:61-83`).
- The node then commits whatever characters/world items remain and returns `committed_to_graph` (`:114-156`).
- Relationship statement construction itself skips incomplete records rather than reporting a contract violation (`:562-576`).

**Activation / impact**

A missing file, checksum mismatch, malformed JSON, wrong root, or malformed relationship record turns into an initialization that is marked successful but has no (or fewer) outline relationships. Because the system has already committed other entities, a later fix/retry has no receipt proving which graph revision lacked the relationships.

**Smallest repair direction**

Make a declared relationship artifact required once the extraction stage has published a ref: strict-load it, validate a typed/allowlisted relationship schema, and fail before constructing any statements. If “no relationships” is valid, publish an explicit empty, checksum-bound list rather than using `None` and exception swallowing.

**Regression proof**

Use a bad checksum, malformed root, and one malformed relationship item. Each must return a fatal pre-commit error and make zero calls to `execute_cypher_batch`.

---

### I-07 — Character initialization fields are placed in `updates` but no initialization writer persists them

**Evidence**

- The commit adapter reads `motivations`, `background`, `skills`, `internal_conflict`, and protagonist status, then puts them in `CharacterProfile.updates` (`core/langgraph/initialization/commit_init_node.py:195-243`).
- The model advertises `updates` as its flexible overflow area (`models/kg_models.py:32-67`, `:97-109`).
- The native character Cypher used by both the initialization commit and `sync_characters()` sets description, status, ids/provenance, and traits, but does not write `updates` or the four fields (`data_access/cypher_builders/native_builders.py:55-114`, `:140-156`; `data_access/character_queries.py:542-590`).
- The later character parser creates a profile from only core fields and likewise does not carry those fields forward (`core/parsers/character_sheet_parser.py:143-158`).

**Activation / impact**

For every normal sheet carrying these structured fields, the initialization code collects them but the graph persistence path discards them. A successful commit/parser result therefore does not mean the graph contains the character facts supplied by the initialization artifact. This also makes a future parser replay unable to recover them.

**Smallest repair direction**

Decide whether these are graph properties, a typed `character_metadata` payload, or intentionally file-only facts. Encode that policy in one schema and one writer; if graph-backed, add an explicit allowlisted map/property write and matching read model. Do not rely on `updates` unless the writer actually persists it.

**Regression proof**

Commit a sheet with distinctive values for all five fields, query/read the character back through the production read API, and assert exact round trip. Add a replay assertion that no field is lost or reset.

---

### I-08 — Project persistence overwrites user-owned world files and leaves stale/colliding character projections

**Evidence**

- On every persistence call, `world/items.yaml` is written with an explicit empty list (`core/langgraph/initialization/persist_files_node.py:137`, `:384-409`), and `world/rules.yaml` and `world/history.yaml` are unconditionally replaced by stubs (`:143-145`, `:454-477`).
- Character YAML names are reduced to lower-case, spaces to underscores, and apostrophes removed with no collision detection (`:262-292`).
- The node writes individual files one-by-one. Each individual write is atomic (`utils/file_io.py:12-24`, `:43-67`), but there is no staged project-tree transaction/manifest for the set.
- The validator accepts any existing `characters/*.yaml`, including stale files left from an earlier initialization (`core/langgraph/initialization/validation.py:60-66`).

**Activation / impact**

Reinitializing an existing project erases user-maintained rules/history and replaces items with an empty generated list. If a character disappears in a later run, its old YAML remains and still satisfies validation. Names such as `A B` and `A_B` target the same projection filename, so the latter replaces the former. An exception between writes can leave a mixed-generation project tree even though individual files are not torn.

**Smallest repair direction**

Separate managed generated artifacts from user-owned inputs. Create stubs only when absent; never overwrite non-empty user files without an explicit migration/force policy. Stage a complete generated tree under a sibling directory, validate its manifest/content, and atomically promote the generation pointer. Use collision-resistant filename ids and detect duplicate display-name normalization.

**Regression proof**

Seed custom rules/history/items and an obsolete character projection, rerun initialization, and assert user files remain unchanged while the generated manifest advances separately. Add `A B`/`A_B` and injected-mid-write-failure cases.

---

### I-09 — Stage-5 physical-description updates report success without writing the changed property

**Evidence**

- `NarrativeEnrichmentParser.update_character_physical_descriptions()` mutates `character.physical_description` and then invokes `sync_characters()` (`core/parsers/narrative_enrichment_parser.py:461-481`).
- `sync_characters()` delegates to `NativeCypherBuilder.character_upsert_cypher()` (`data_access/character_queries.py:574-590`), whose `SET` clauses have no `physical_description` property (`data_access/cypher_builders/native_builders.py:55-114`).
- The active narrative node has the same ineffective write path: it sets `physical_description` then calls `sync_characters()` (`core/langgraph/nodes/narrative_enrichment_node.py:96-123`).
- `NarrativeEnrichmentParser.parse_and_persist()` returns a successful “persisted” message after that call (`core/parsers/narrative_enrichment_parser.py:565-591`).

**Activation / impact**

When physical-description extraction is enabled and produces an accepted result, the in-memory model is changed but the native Cypher writer omits the property. The parser can return success while the database remains unchanged, so subsequent chapters re-extract the same fact and contradiction checks read stale data.

**Smallest repair direction**

Add an explicit, schema-backed `physical_description` assignment to the canonical character upsert (or use a dedicated update statement), then verify all profile reads map it back. Keep the enrichment node and parser on the same writer contract.

**Regression proof**

Run the production writer against a character with a new physical description, read it back through `get_character_profile_by_name()`, and assert exact persistence. Repeat an identical enrichment and assert no redundant update statement is issued.

## Refactor seams

1. **`InitializationSnapshot` / manifest boundary.** A single immutable manifest should bind generated structured content, human-readable exports, selected versions, checksums, producer/model metadata, expected act/chapter topology, and a publication status. Parsers should consume this object, never discover filenames.
2. **Pure `InitializationGraphPlan` boundary.** Split every parser into `load + strict validate -> plan` and a single `commit(plan)` implementation. No LLM calls or direct Neo4j writes belong in adapters after the plan is published.
3. **One transactional graph committer.** Use the existing transactional batch capability for the complete plan and publish a receipt only after commit. A failed initialization must leave no new graph state, or record an explicit recoverable transaction/cleanup state.
4. **Managed-project writer boundary.** Generate exports into a versioned, staged managed directory. Preserve user-owned world files; expose an explicit import/reconciliation operation rather than silently treating YAML and JSON as interchangeable canonical state.
5. **Replay policy boundary.** Select a snapshot by ref/checksum, not by “v0 exists.” Define whether replays are no-op, replace, migration, or conflict and make overwrite behavior auditable.
6. **Domain field ownership.** Promote fields such as motivations, background, skills, internal conflict, protagonist state, and physical description into one explicit graph/file ownership schema. Do not use a model overflow map without a matching persistence/read contract.

## Exact source coverage

### Requested production files

- `core/langgraph/initialization/persist_files_node.py` — all 514 lines.
- `core/langgraph/initialization/run_parsers_node.py` — all 96 lines.
- `core/langgraph/initialization/commit_init_node.py` — all 688 lines.
- `core/langgraph/initialization/all_chapter_outlines_node.py` — all 134 lines.
- `core/parser_runner.py` — all 253 lines.
- `core/parsers/character_sheet_parser.py` — all 421 lines.
- `core/parsers/global_outline_parser.py` — all 789 lines.
- `core/parsers/act_outline_parser.py` — all 1,081 lines.
- `core/parsers/chapter_outline_parser.py` — all 1,168 lines.
- `core/parsers/narrative_enrichment_parser.py` — all 602 lines.

### Direct supporting contracts inspected

- `core/langgraph/content_manager.py` — content-path, save/load/checksum, and initialization/chapter getters (`:101-318`, `:941-1118`, `:1271-1315`).
- `core/langgraph/workflow.py` — initialization ordering and error routing (`:154-180`, `:288-472`).
- `core/langgraph/initialization/{character_sheets_node,global_outline_node,act_outlines_node,chapter_outline_node,outline_relationships_node,validation}.py` — generation/output format and immediate consumers.
- `core/langgraph/nodes/narrative_enrichment_node.py` — Stage-5 production caller (`:55-159`).
- `core/langgraph/state.py` — ContentRef initialization-state fields and defaults (`:239-250`, `:286-432`).
- `models/kg_models.py` — `CharacterProfile` and `WorldItem` storage semantics (`:32-247`).
- `data_access/{character_queries.py,cypher_builders/native_builders.py}` and `core/db_manager.py` — actual graph write identities and transaction scopes.
- `utils/file_io.py` — individual file-write atomicity.
- Focused tests inspected as intent only: `tests/core/langgraph/initialization/test_all_chapter_outlines_node.py`, `tests/core/langgraph/initialization/test_run_parsers_node.py`, `tests/test_parser_runner.py`, and `tests/test_langgraph/test_commit_init_node.py`.

## Unverified / deliberately out of scope

- No Neo4j connection, schema, constraints, live graph contents, APOC availability, or rollback behavior was exercised. Static evidence establishes the transaction boundaries, not a live migration result.
- No LLM/model calls were made; actual divergence rates between duplicate extractors, generated-output quality, and retry economics remain unmeasured.
- No test command, dependency install, full repository survey, git mutation, or source edit was performed. Existing tests cited above were read only.
- No real story project output was inspected; this report does not assert that a particular existing project is already corrupted.
- The worktree is broadly dirty. Primary-file SHA-256 values observed during this review: `persist_files_node.py e87afff7`, `run_parsers_node.py de245b30`, `commit_init_node.py 429c16c3`, `all_chapter_outlines_node.py 8680b7e0`, `parser_runner.py 1e66039a`, `character_sheet_parser.py ed92d56e`, `global_outline_parser.py 727f9e7a`, `act_outline_parser.py 382d78d1`, `chapter_outline_parser.py 510c5274`, and `narrative_enrichment_parser.py 5b59f27d`. Rebind any implementation work to a fresh source snapshot before acting on line evidence.
