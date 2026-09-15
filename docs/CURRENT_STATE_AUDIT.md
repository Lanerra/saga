# SAGA Current-State Audit

> Historical pre-repair audit, not the current operational guide or a release verdict.
> Findings and counts below describe that snapshot. Use the [current guide](../README.md).

## Overall verdict

SAGA is a viable, unusually ambitious system caught in an unsafe mid-refactor—not a project that needs rewriting.

The core design is coherent: a two-phase LangGraph workflow, scene-level generation, Neo4j canon, externalized/versioned content, structured prompts, and strong domain modeling. The immediate problem is that infrastructure boundaries—LLM access, Neo4j access, checkpointing, and filesystem state—are globally wired and inconsistently tested. That has made the current module split brittle and difficult to validate.

No source files were modified during the audit.

## What I found

- Approximately 73,200 lines of production Python across 251 files.
- Approximately 34,100 lines of tests and 1,682 collected cases.
- A clean subset of 1,190 tests passes in 13.5 seconds.
- The full suite does not complete reliably because several “unit” tests escape into real Neo4j, embedding, or tokenizer behavior.
- Mypy currently reports 8 errors in 4 production files.
- Ruff reports 56 issues, including undefined annotations and an incorrect return type.
- The working tree has 305 status entries. Most tracked changes are executable-bit churn, but there is also a substantive, unfinished module split.
- The three subsystems being split grew by roughly 372 lines collectively, while `commit_node.py` is still 1,064 lines despite its header describing it as approximately 250.

This means the repository currently cannot provide a trustworthy green baseline, even though most of the underlying code still works.

## What is good

Several parts are much better than “first project” would suggest:

- The 27-node workflow has recognizable phases, explicit routing, revisions, and error handling.
- Scene-by-scene generation is the correct architecture for long-form narrative continuity.
- Content externalization is thoughtful: immutable references, checksums, versioning, atomic writes, and size validation.
- Neo4j batch operations use a single transaction path.
- Configuration uses strict Pydantic models in important places.
- The project remains faithful to its local-first design.
- The test corpus is substantial and exercises real behavior. The 1,190-test passing subset proves there is a healthy core worth preserving.

The project accumulated too much policy and compatibility at its edges; it did not lose its architectural center.

## Critical problems

### 1. The current refactor is incomplete

The new commit/context/scene-extraction modules contain undefined annotations and a wrong parsing return type. Tests still patch symbols at their old module locations.

The global LLM lifecycle is maintained through a manually curated list of modules in `orchestration/langgraph_orchestrator.py`. The refactor introduced another importing module, but it was not added to that list. A test catches this exact omission.

This list is a dependency-injection system implemented through runtime monkey-patching. Every future module split can silently break it.

### 2. Tests are not hermetic

The code creates a module-level LLM service during import in `core/llm_interface_refactored.py`. Other modules similarly reach global Neo4j and embedding services.

Consequences observed during testing:

- A commit-node case hung while making a real Neo4j read.
- Token counting attempted a network download.
- Semantic-offset behavior reached the real embedding retry path.
- Refactored tests failed because they patched the former facade rather than the module now performing the call.

The suite’s size therefore overstates its current confidence. A unit suite must be incapable of contacting Neo4j or the network unless explicitly enabled.

### 3. Validation happens after mutation

The workflow orders normalization → commit → validation in `core/langgraph/workflow.py`.

If validation requests a revision, the revision node attempts to roll back. If rollback fails, it logs a warning and continues generating the revision in `core/langgraph/nodes/revision_node.py`. That can leave the invalid chapter in Neo4j and then commit regenerated material on top of it.

This is the most serious correctness issue I found.

Relationship-evolution validation is also unreliable. Its query in `core/langgraph/subgraphs/validation.py`:

- Receives `current_chapter` but does not use it.
- Does not exclude the relationship just committed.
- Has no deterministic ordering.
- Keeps the first arbitrary relationship for each character pair.

It can therefore compare the new relationship against itself or an arbitrary historical version.

### 4. Resume semantics contradict each other

Different parts of the project call different stores authoritative:

- The orchestrator says checkpoints are the source of truth.
- Finalization says Neo4j is the source of truth.
- Filesystem chapter output is treated as best effort.

The resume guard rejects a checkpoint when Neo4j already contains its current chapter in `orchestration/langgraph_orchestrator.py`. A crash after Neo4j commit but before checkpoint advancement could therefore turn a recoverable partial operation into an unrecoverable resume conflict.

SAGA needs an explicit chapter commit protocol, not merely a count comparison.

### 5. State and compatibility have expanded too far

`NarrativeState` contains 74 fields and is `total=False`, so every field is technically optional even though much of the code assumes initialization.

There are also internal type contradictions. For example, `generated_embedding` is declared as a `ContentRef` in `core/langgraph/state.py`, but the commit node accepts it only when it is a raw `list[float]` in `core/langgraph/nodes/commit_node.py`.

I counted:

- 181 broad `except Exception` blocks outside tests.
- 222 references to legacy, fallback, deprecated, or backward-compatibility behavior.
- 1,746 patch/mock references in the test tree.

Those figures describe accumulated migration layers and hidden input formats—the opposite of the repository’s current strict-boundary conventions.

### 6. Development reproducibility is weak

- README targets Python 3.12.
- The checked environment reports Python 3.13 but is not executable correctly here.
- The system interpreter is Python 3.14.
- There is no dependency lockfile or CI configuration.
- Pytest config specifies a timeout in `pyproject.toml`, but `pytest-timeout` is not installed, so the setting is ignored.
- `pytest-asyncio` warns that the default fixture loop scope is unspecified.
- Undeclared plugins are being loaded from the environment.

Reproducing a known-good run should not require interpreter surgery.

## What should be done immediately

In this order:

1. **Preserve the current work.**  
   Disable local file-mode tracking with `git config core.fileMode false`, inspect the reduced diff, and snapshot the meaningful refactor on a dedicated WIP branch or commit. Do not reset the existing work or mix the scratch/audit files into that snapshot.

2. **Freeze feature development and establish one environment.**  
   Rebuild a clean Python 3.12 environment, introduce a lockfile, install the declared test plugins, and define one canonical validation command.

3. **Finish the current split as one bounded task.**  
   Fix the 8 type errors and 56 lint findings, update the moved test seams, and ensure the new modules are included in service lifecycle management. Do not continue decomposing more files until this split is green.

4. **Make unit tests fail closed.**  
   Introduce an explicit runtime-services object containing the LLM, embedding service, graph store, and content manager. Inject it through workflow construction or LangGraph runtime context. Unit tests should receive fakes and should throw immediately if real network or Neo4j access is attempted.

5. **Repair graph consistency before cleanup work.**  
   Make rollback failure fatal. Prefer validating staged output before destructive graph mutation. At minimum, relationship validation must query the deterministically latest relationship from a chapter before the current one.

6. **Define the chapter transaction model.**  
   Decide precisely which store is authoritative and implement explicit chapter statuses such as preparing, graph committed, artifacts written, and checkpoint advanced. Recovery should reconcile idempotently from those markers.

7. **Only then reduce state and legacy behavior.**  
   Divide `NarrativeState` into phase-specific structures, eliminate contradictory/dead fields, and move compatibility handling into versioned boundary adapters. Do not perform a mass deletion based on the untracked field audit; some of its “unused” conclusions are demonstrably stale.

8. **Add a minimal automated gate.**  
   Every change should run Ruff, strict Mypy, an offline unit suite, and a separately marked Neo4j integration suite. Currently configured `unit`, `integration`, and `slow` markers are barely used.

## What I would not do

I would not rewrite SAGA, change its local-first architecture, or immediately split every large file. I also would not prioritize README cleanup or stylistic modernization while graph consistency and resume behavior remain ambiguous.

The right strategy is:

> preserve → reproduce → make green → repair transactional correctness → replace globals → simplify

There is a genuinely impressive system here. Its most valuable parts are still intact; it mainly needs a hard stabilization boundary and fewer invisible dependencies.
