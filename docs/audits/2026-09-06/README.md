# SAGA current-state audit and refactor handoff

**Audit:** 2026-09-06 (Pacific)  
**Repository:** `/home/dlewis3/Desktop/AI/saga`  
**Source baseline:** dirty `refactor/kg-schema`, HEAD `4ce8870388b54e19ead91c86c10273e0fdd32e61`  
**Scope:** source/Git inventory, source-level architectural and failure-path review, isolated tests and synthetic probes. **No application source, tests, Git history/index/configuration, real stories, or database were changed.** Audit-only files were created here; a disposable Python environment/source copy lives under `/tmp/saga-audit-20260906`.

## Executive judgment

**SAGA has a coherent, valuable authoring system inside an unreliable lifecycle. Repair it incrementally; do not rewrite it.**

The central design remains recognizable: structured story planning, scene-level drafting, semantic and graph-backed context, extracted narrative facts, validation/revision, and manuscript export. This is not merely an elaborate prompt wrapper. There is substantial implemented behavior and a useful test foundation.

Its most consequential problems are not file length or formatting. They are disagreements about **which artifacts are authoritative, which facts belong to which story/chapter/attempt, when a chapter is accepted, and how work resumes after failure**. Broad exception handling and global service rewiring obscure those disagreements. More module splitting before fixing these contracts would spread the same defects across more files.

**Immediate recommendation:** preserve the dirty implementation and story data, establish a reproducible offline test baseline, finish the existing split, then repair destructive/fail-open transitions. Do not reinitialize valuable projects, trust automatic resume, run parser replay against a developed story, or point multiple projects at one shared graph until the corresponding safety issues are fixed. This is precaution based on source/probes, not a claim that existing user data is already corrupted.

## Start here next session

1. Read this document, then [BACKLOG.md](BACKLOG.md). Every backlog task has source targets, dependencies and observable acceptance criteria; none of the repair tasks were implemented by this audit.
2. Run `python docs/audits/2026-09-06/check_snapshot.py` from the repo. It only compares the audited file hashes/current Git state and prints JSON; it does not import SAGA or mutate the repo. Treat source drift as a reason to re-read affected paths, not to discard the entire inventory.
3. Read the relevant subsystem report and use its finding ID to locate the source and required regression test. Use [source-catalog.md](source-catalog.md), [symbols.json](symbols.json), and [imports.json](imports.json) rather than repeating a broad crawl.
4. Begin B01–B05 (preserve/reproduce/finish split/offline tests), then B06–B13 and B31 (data-safety fixes). Bring B27's path-containment guard forward before importing untrusted references or adding cleanup operations. Larger lifecycle/schema changes follow after those narrow repairs.
5. Keep work in SAGA. DWARF's GPU/corpus conventions and source-only publication rules do not apply. Preserve SAGA's tests, prompts, useful docs and local-first writer workflow.

The manifest covers **282 first-party Python/test/prompt files**. It is an inventory, not a claim that every line received an equally deep semantic review. The five subsystem reports identify the source paths actually traced. Focused re-reading and testing remain mandatory before changing a function. Unchanged source does not need another full census.

## What exists and should be preserved

| Layer | Current source owners | Role / assessment |
|---|---|---|
| Writer CLI and project bootstrap | `main.py`, `core/project_manager.py`, `core/project_bootstrapper.py`, `ui/rich_display.py` | Local authoring entrypoints, project configuration/review and progress. Keep the CLI; make project selection and outcome reporting explicit. |
| Workflow | `orchestration/langgraph_orchestrator.py`, `core/langgraph/workflow.py`, `state.py`, `state_helpers.py` | Owns LangGraph execution, SQLite checkpoints, initialization/chapter routing and state. This should be the lifecycle coordinator, not a module-global service patch registry. |
| Initialization | `core/langgraph/initialization/`, `core/parser_runner.py`, `core/parsers/` | Produces characters/global/act/chapter plans; writes JSON, YAML and graph structure. Valuable planning hierarchy, but duplicate extractors/writers and unchecked replay need consolidation. |
| Generation and memory | generation nodes/subgraph, `context_*` helpers, `graph_context.py`, `data_access/`, prompt getters | Scene-by-scene drafting with previous-scene text, summaries, characters, world facts, plot and semantic retrieval. Preserve this, while correcting budget, provenance and provisional filtering. |
| Extraction and canon | `scene_extraction*`, normalization services, `commit_*`, Cypher builders | Converts generated prose into typed narrative memory. Actual failure versus valid empty data and relationship identity/write ownership are presently inconsistent. |
| Acceptance and revision | validation subgraph, revision/finalize/healing/QA nodes | Has meaningful validation and corrective machinery. Validation follows graph mutation; some failures are deliberately accepted, and QA is post-finalization advisory rather than a publication gate. Policy must be explicit. |
| External services | `core/http_client_service.py`, `llm_interface_refactored.py`, embedding/spaCy services, `config/` | Configurable local LLM and embedding endpoints with retries and caching. Retain transport abstraction; replace global lifetime/config rebinding with a per-run owner. |
| Durable content and export | `core/langgraph/content_manager.py`, `utils/file_io.py`, `core/langgraph/export.py` | Checksums, version labels and atomic replacement are useful existing machinery, but do not presently guarantee immutable storage, path containment or successful manuscript publication. |
| Domain models and prompts | `models/`, schema/relationship validators, `prompts/`, `processing/` | Meaningful domain structure and StrictUndefined template rendering. Need shared response schemas and producer-to-writer field round trips, not wholesale replacement. |
| Verification | `tests/`, root verification/visualization scripts | A substantial passing core, but fakes often test call shapes rather than real graph semantics; moved imports and accidental network use undermine the baseline. |

### Actual pipeline

```text
CLI / bootstrap / select project
  -> orchestrator + run services + SQLite checkpointer
  -> initialization: characters -> global/act/all-chapter outlines
       -> initial graph commit -> human-readable files -> parser graph writes
  -> chapter outline -> scene plan
  -> retrieve context -> draft scene (repeat)
  -> extract scene facts -> consolidate -> embed -> assemble chapter
  -> enrich -> normalize -> COMMIT graph
  -> validate -> revision/rollback -> regeneration, or summarize
  -> FINALIZE manuscript + chapter metadata
  -> graph healing -> advisory QA -> next chapter / end
```

Neo4j, externalized content, human-readable project files and SQLite checkpoints all contain pieces of state. **No current record reliably binds them into one accepted chapter attempt.** In particular, Neo4j finalization metadata is not a durable copy of the full manuscript text.

## Measured scale, not cache-inflated line counts

Physical lines include comments, docstrings and blanks; these are not executable SLOC.

| Scope | Files | Physical lines |
|---|---:|---:|
| Runtime Python | 114 | 38,596 |
| Root auxiliary Python | 4 | 495 |
| All non-test first-party Python | 118 | 39,091 |
| Tests | 133 | 34,123 |
| Non-Python prompt templates/system text | 31 | 1,147 |

| Directory | Physical lines | Scope |
|---|---:|---|
| `core/` | 26,065 | Python; largest concentration |
| `data_access/` | 4,784 | Python |
| `models/` | 2,165 | Python |
| `utils/` | 1,311 | Python |
| `config/` | 1,168 | Python |
| `prompts/` | 2,304 | 1,157 Python + 1,147 template/system text |
| `orchestration/` | 822 | Python |
| `processing/` | 724 | Python |
| `ui/` | 250 | Python |

Largest files: `kg_queries.py` **1,436** lines; `content_manager.py` **1,315**; chapter parser **1,168**; act parser **1,081**; `commit_node.py` **1,064**; prompt getters **1,048**; graph healing **1,040**; KG models **1,016**. Their length is a maintenance signal, not a deletion criterion.

Complexity hotspots include entity embedding statement construction, chapter relationship construction, user-story conversion, entity validation, LLM completion policy and text deduplication. Exact symbol ranges and Radon output are in [complexity-ranked.json](complexity-ranked.json). The AST census found **182 broad exception handlers** in non-test Python; review by consequence, not by indiscriminate removal. Some optional healing/embedding degradation is intentional.

## Priority evidence ledger

Evidence labels:
- **Runtime probe:** observed using production functions and synthetic fixtures/narrow fakes; not a live database/model run.
- **Source:** exact control/data flow inspected; real service consequences remain integration-test obligations.
- **Framework probe:** actual installed LangGraph API with a toy graph; not the full SAGA SQLite restart matrix.

| Priority | Finding and consequence | Evidence / exact anchor | Backlog |
|---|---|---|---|
| Immediate | Failed commit can run destructive compensation even when its transaction never committed/already rolled back, risking an earlier successful chapter | Source; [graph P0-2](graph-review.md), `commit_node.py:567-588`, `db_manager.py:271-339` | B06 |
| Immediate | Revision proceeds after rollback fails, leaving rejected facts eligible for later work | Runtime probe: rollback failure followed by guidance generation, no fatal state; `revision_node.py:146-155` | B07 |
| Immediate | Initialization creates planned Chapter nodes; resume counts all of them as progress and rejects valid restarts | Source; [WFR-01](workflow-review.md), `chapter_queries.py:111-125`, orchestrator `:797-800` | B08 |
| Immediate | Separate project directories are not separate graph ownership domains | Source; [graph P0-1](graph-review.md), chapter merge by number, global name/edge deletion | B09 |
| Immediate | Plan/draft/extraction failures can reuse stale plans, repeat until graph recursion limit, or become valid-empty extraction | Source; [WFR-05–07](workflow-review.md) | B10,B11 |
| Immediate | Canonical manuscript write can fail while finalization continues; DB update does not store full prose | Runtime probe: no manuscript, DB call proceeds, fatal=false; `finalize_node.py:99-185` | B12 |
| Immediate | Existing rules/history/items can be overwritten during initialization; parser replay prefers stale v0 | Source; [I-04,I-08](initialization-review.md), `parser_runner.py:132-143`, `persist_files_node.py:384-477` | B31,B18 |
| High | Resume passes saved values as new input rather than continuing pending work | Framework probe on LangGraph 1.0.2: first node reruns; `None` continues pending second node. Production invocation source traced | B15 |
| High | Validation can compare newly committed facts against themselves; changing DB row order changes findings | Runtime probe: 0 vs 1 issue for same current/historical rows; `subgraphs/validation.py:321-373` | B16 |
| High | Profile-created relationships are followed by chapter-wide deletion in the same batch; repeated facts lack per-chapter assertion membership | Source; [graph P0-3,P1-1](graph-review.md), `commit_node.py:441-464,682-696,952-1010` | B17 |
| High | Initialization has duplicate extraction/writer authorities, partial-plan success and partial persistent failure | Source; [I-01–06](initialization-review.md) | B18 |
| High | Model fields/physical-description updates are accepted but omitted by the canonical character writer | Source; [I-07,I-09](initialization-review.md), native builder `:55-156` | B32 |
| High | Provisional filter is inverted; previous-scene total token budget is not enforced | Source plus runtime budget probe (10-token budget yielded 38-token rendered context under deterministic counter); `chapter_queries.py:348-372`, `context_scene_retrieval.py:104-160` | B13 |
| High | Content APIs permit path escape and same-version overwrite; immutable reference supports a mutating operation | Runtime probes; [S-01,S-02](local-boundaries-review.md) | B27 |
| High | Global service construction and patch registries leave lifetime/test seams brittle; config reload and embedding identity/validation are incoherent | Source and suite failures; [F1–F7](inference-review.md) | B20–B22 |
| Policy | Quality failure/overrides and post-finalize QA are not mandatory acceptance gates | Source; existing tests sometimes intentionally encode acceptance; [WFR-08,WFR-10](workflow-review.md) | B23 |
| Polish | Per-project narrative style is ignored in scene prompting; CLI README command is invalid; export can remove prose | Source at `scene_generation_node.py:106`; real CLI exit 2; synthetic export probe | B24,B28,B29 |

**Priority reconciliation:** subsystem severity labels were authored independently. This table and BACKLOG are the implementation ordering authority. A manuscript publication failure deserves immediate attention even where a local report called it P2. Conversely, intentionally advisory QA should not become a mandatory gate without a declared writer policy.

## Verification actually executed

| Check | Result | What it establishes |
|---|---|---|
| In-memory Python AST parsing | All 251 first-party Python/test files parsed | Syntax only, not import/behavior correctness |
| Fresh isolated Python runtime | Python 3.12.13; direct pinned requirements installed; `uv pip check` passed for 99 installed packages | Audit runtime reproducible from recorded package versions; existing local venv remains broken |
| Full existing pytest suite | **1,639 passed; 43 failed; 0 errors; 0 skipped; 1,682 cases** | Real offline baseline, not a green suite or narrative-quality verdict |
| Ruff | **56 findings** | 9 I001, 30 F401, 7 F841, 5 F821, 2 UP037, 3 B007 |
| Mypy | **11 errors in 5 files**, 114 runtime files checked | Includes incorrect parser tuple/list annotation and missing type names; exact diagnostics retained |
| Boundary probes | Traversal/read/delete, overwrite/reference mutation, project collision, chapter counting and export defects reproduced | Synthetic temporary files only |
| Lifecycle/API probes | Failed rollback/finalization, validation order, context budget and framework continuation distinctions reproduced | Narrow fakes + actual ContentManager/LangGraph; no real DB or model calls |
| Git object connectivity | `git fsck --connectivity-only --no-dangling` passed | Connectivity, not exhaustive historical security audit |
| Final snapshot/handoff verification | See [completion-verification.json](completion-verification.json) | Source and scratch hashes, baseline Git preservation, report references and recorded evidence totals |

Tests ran in a byte-identical scratch copy with sanitized environment, no real `.env`/stories/data, and Python audit hooks rejecting socket connect/DNS/sendto and subprocess launch. This is **not an OS security sandbox**; user-namespace network isolation was unavailable. The per-test timeout was intentionally **8 seconds**, overriding the configured 600 seconds. Exact runner: [offline_runner.py](offline_runner.py). JUnit elapsed time: **46.705 seconds**.

The 43 failures were classified as **26 explicitly blocked network escapes, 5 audit timeouts, 4 moved/global-service seam mismatches, 3 absent-spaCy-model failures, 1 truncation expectation mismatch, and 4 other assertions requiring focused triage**. These are assertion-envelope categories, not independent root causes. In particular, do not call an absent spaCy model a source bug or delete slow tests to claim success. See [test-failures.json](test-failures.json) and [failure-categories.json](failure-categories.json).

No fresh full novel, real model/embedding endpoint, disposable Neo4j/APOC database, historical graph migration, actual SQLite crash/restart matrix, or live narrative-quality benchmark was run. No existing story/database corruption was assessed. End-to-end creative quality, actual latency/cost and installed Neo4j/APOC compatibility remain unverified.

## Git and local repository condition

- **289 tracked files; 21 local branches; one worktree.** Current branch is ahead of its cached upstream by one commit. Remote-tracking facts are cached; no fetch occurred.
- Excluding this audit: **284 modified tracked paths, 2 tracked deletions, 20 individual untracked files**. No baseline staged changes.
- All 284 surviving modified tracked paths have executable-bit changes. Read-only status with `core.fileMode=false` reveals the real content delta: **13 modified files and 2 deletions**. Do not mistake this for 284 source rewrites.
- Ten untracked helper modules are part of the live module split. A bundle of committed refs alone would not preserve them. The deleted `knowledge_graph_service.py` and its tests need explicit review with the split, not automatic restoration or removal.
- `dev` is 63 commits behind and ancestral. `master` diverges (11 master-only, 72 current-only commits); non-merge patch equivalence does not settle merge-resolution-only changes. Two old Claude branch commits remain patch-distinct: `a363c2d`, `ce3bd37`. Per-branch ancestry is in [git-analysis.json](git-analysis.json), exact refs in [git-branches-before.txt](git-branches-before.txt).
- Git has **21.46 MiB loose objects + 31.58 MiB packed objects, zero garbage**. Nothing found justifies emergency history rewriting.
- The existing `.venv/bin/python` returns `Exec format error` and is identified as `data`; its cause was not established. That venv occupies **8,301,858,245 logical bytes**. Local caches/environments/traces dominate clutter, not the first-party source. Logical size is not allocated disk usage.

The right Git operation is **preserve → integrate → verify → prune**, not `reset --hard`, blind checkout of master, blanket branch deletion, or a force push. Back up dirty/untracked source and story/Neo4j data separately. Branch pruning and environment deletion are proposed, not performed.

### Historical audit corrections

`docs/CURRENT_STATE_AUDIT.md` calls roughly 73,200 lines across 251 files production Python. That conflates tests and application code: the current measured runtime is 38,596 lines in 114 files, with 34,123 test lines separately. Its smaller passing subset is superseded here by an all-tests offline run, not by proof that external integration now works.

`docs/field-audit-20260505.md` correctly warned about missed `.get()`/writer uses. Do not delete state fields from historical “dead” lists: current `medium_model` and `project_id` have uses, including initialization/receipt identity, and quality/QA fields may be observable outputs even if routing ignores them. [state-field-uses.json](state-field-uses.json) indexes 74 annotated state fields; it is a textual lookup index, **not an automated dead-field verdict**.

The initial `inventory-summary.json.status_counts` had a first-line whitespace capture artifact; [git-analysis.json](git-analysis.json) and the final preservation receipt are authoritative for normalized baseline Git counts. The source hashes/line census were not affected. Do not rerun `inventory.py` into this dated directory, because it overwrites baseline files; use `check_snapshot.py` for drift and create a new dated inventory for later audits.

## Target architecture and repair sequence

Retain a **single Python application** with explicit internal boundaries:

1. **Run owner:** immutable effective configuration plus managed LLM, embedding, DB and content services. No global module-name patch registry.
2. **Plan producer:** typed prompt/extraction schemas produce one versioned initialization/chapter plan. No second parser secretly regenerates the same facts.
3. **Canon repository:** stable project/entity/assertion identities; one writer contract; required schema verified; accepted history distinguishable from candidate/planned facts.
4. **Chapter lifecycle coordinator:** attempt identity and durable phase/receipt reconciliation for graph commit, validation, file publication, checkpoint and advancement. Use existing Neo4j/files/SQLite, not a distributed transaction service.
5. **Artifact store/exporter:** contained, immutable/create-only references and faithful canonical manuscript publication; user-owned edits kept separate from generated projections.
6. **Writer policy:** strict versus accepted-with-exceptions behavior visible, including generation/extraction completeness, quality overrides, revision limits and advisory maintenance.

Sequence: **preserve/reproduce → urgent safety → durable lifecycle + initialization/schema → explicit services/configuration + quality → simplification/CLI/docs/Git polish**. [BACKLOG.md](BACKLOG.md) contains the task-level implementation plan and test matrix.

The decisive release gate is not reduced LOC or a green mocked suite. It is a deterministic, project-isolated authoring run that survives interruption at every persistence boundary and resumes to the same accepted graph and manuscript, followed by a bounded real-model narrative regression check.

## Evidence package index

| Artifact | Use |
|---|---|
| [BACKLOG.md](BACKLOG.md) | Prioritized implementation tasks, dependencies and acceptance tests |
| [source-catalog.md](source-catalog.md) | Every non-test source/prompt path and module purpose |
| [source-inventory.json](source-inventory.json), [inventory-summary.json](inventory-summary.json), [scope-details.json](scope-details.json) | Per-file SHA-256/size/physical LOC/tracking; aggregate scope |
| [symbols.json](symbols.json), [imports.json](imports.json), [state-field-uses.json](state-field-uses.json) | Locate definitions, dependencies and state strings without re-crawling |
| [complexity-ranked.json](complexity-ranked.json), [exact-duplicate-functions.json](exact-duplicate-functions.json) | Structural refactor candidates; not deletion proof |
| [workflow-review.md](workflow-review.md) | WFR-01–11: routing, failure, checkpoint, acceptance and finalization |
| [graph-review.md](graph-review.md) | Project isolation, transactions, relationships, dedupe, vectors and schema |
| [initialization-review.md](initialization-review.md) | I-01–09: artifacts, parsers, replay, partial writes and lost fields |
| [inference-review.md](inference-review.md) | F1–F7: service lifetime, provider/auth/logging, caches, config and schemas |
| [local-boundaries-review.md](local-boundaries-review.md) | Storage, CLI/export, testing, Git and runtime environment |
| [pytest.xml](pytest.xml), [pytest.log](pytest.log), [test-summary.json](test-summary.json), [test-failures.json](test-failures.json) | Full test evidence and exact failed cases |
| [ruff.json](ruff.json), [mypy.txt](mypy.txt), [runtime-freeze.txt](runtime-freeze.txt), [dependency-check.txt](dependency-check.txt) | Tool findings and resolved audit runtime |
| [boundary_probes.py](boundary_probes.py), [boundary-probes.txt](boundary-probes.txt) | Synthetic local-boundary reproductions |
| [lifecycle_probes.py](lifecycle_probes.py), [lifecycle-probes.json](lifecycle-probes.json) | Synthetic lifecycle and installed-framework evidence |
| [git-analysis.json](git-analysis.json), [git-status-before.txt](git-status-before.txt), [git-branches-before.txt](git-branches-before.txt), [disk-inventory.json](disk-inventory.json) | Baseline worktree/refs and local-size inventory |
| [check_snapshot.py](check_snapshot.py), [completion-verification.json](completion-verification.json) | Read-only reuse check and final audit verification |

These artifacts are local, uncommitted and not a backup of the repository or user data. Preserve the package with the eventual recovery snapshot. Raw test logs contain synthetic fixture content; review before any public publication.
