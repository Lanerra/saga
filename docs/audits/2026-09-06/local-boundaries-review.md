# SAGA — local storage, CLI, Git and verification review

Audit date: 2026-09-06 (Pacific). Baseline HEAD: `4ce8870388b54e19ead91c86c10273e0fdd32e61`, dirty `refactor/kg-schema`. Source is unchanged by this review; only this audit directory was authored. Findings concern current working-tree bytes, not merely HEAD.

## Evidence and execution boundary

- `source-inventory.json` binds every first-party Python source/test and prompt in the surveyed roots by SHA-256, bytes, line count and tracking state. `symbols.json` and `imports.json` provide lookup indices. All 251 Python files parsed successfully in memory.
- There are **114 runtime Python files / 38,596 physical lines**, plus **4 root auxiliary scripts / 495 lines**. Thus all non-test Python is **118 files / 39,091 lines**. Tests are **133 files / 34,123 lines**; prompt templates/system Markdown are **31 files / 1,147 lines**. Physical lines include comments, docstrings and blanks. These are not executable-SLOC counts.
- The existing `.venv/bin/python` fails with `Exec format error`; `file` identifies it as `data`. Its origin/cause is not established. A fresh Python **3.12.13** venv was created at `/tmp/saga-audit-20260906/venv` using the pinned direct requirements, plus audit tools. Resolved transitive versions are in `runtime-freeze.txt`; `uv pip check` passes for 99 installed packages.
- The audited source/tests/prompts were copied byte-for-byte to `/tmp/saga-audit-20260906/source`. No real `.env`, credentials, project stories, database or runtime output were copied. Tests run with a sanitized environment and a Python audit hook rejecting socket connect/DNS/sendto and subprocess creation. This is not an OS security sandbox; `unshare -Urn` was unavailable due to user-namespace permissions. It is an explicit guard for these reviewed Python tests.
- **Full existing suite: 1,639 passed, 43 failed, 0 errors, 0 skipped; 1,682 cases in 46.705 seconds** (JUnit suite timing). Per-test timeout was **8 seconds**, not the configured 600 seconds. No tests were edited or omitted. See `pytest.xml`, `pytest.log`, `test-summary.json`, `test-failures.json` and `failure-categories.json`.
- Failure classification by assertion envelope: **26 explicit blocked network escapes; 5 audit timeouts; 4 moved/global service seam mismatches; 3 spaCy-model-dependent failures; 1 truncation assertion mismatch; 4 other assertions requiring focused triage.** These are not 43 independently proven production bugs. Missing spaCy model and the deliberately short timeout are audit-environment limitations. The network escapes and stale injection seams are genuine test-isolation defects.
- Ruff: **56 findings** (9 I001, 30 F401, 7 F841, 5 F821, 2 UP037, 3 B007), including tests and root tools. Mypy: **11 errors in 5 files**, checking the 114 runtime files, using current installed Mypy plus Pydantic plugin. Exact outputs and tool versions are retained. Static checks were run against the scratch snapshot, not the live runtime configuration.
- No real Neo4j queries, model requests, long-form generation, container startup/reset, Git fetch, commits, branch deletion, chmod or source changes were performed.

## S-01 — ContentManager does not enforce its path boundary

**Proven by synthetic-file runtime probes.**

`core/langgraph/content_manager.py:125-152` sanitizes identifier but not content type, creates the directory before checking containment, and uses string `startswith` rather than path ancestry. `save_text(..., '../content_sibling', ...)` writes outside `.saga/content` while passing its guard. `:392-424`, `:453-464`, `:494-505`, and `:588-594` accept relative traversal or absolute paths on read/delete without containment checks. A checksum-valid `../synthetic.txt` was successfully read by the strict API, then deleted. The only target was an audit-created temporary file.

**Impact:** malformed/imported references and maintenance callers can operate outside the intended project/content directory. This is a local file-integrity boundary defect, not a demonstrated remotely exploitable vulnerability.

**Repair:** centralize canonical path resolution for every read/write/exists/delete/version operation; validate bucket/identifier/version before mkdir; enforce `Path.is_relative_to` after symlink resolution; define whether reads may target project artifacts outside `.saga/content` with an explicit separate API. Never rely on digest validation for containment.

**Acceptance:** reject absolute, `..`, same-prefix sibling, symlink escape, invalid bucket and invalid version paths; assert no file or directory is created by rejection; preserve valid legacy references through a deliberate migration adapter.

## S-02 — “Immutable/versioned” content is not immutable storage

**Proven public-API behavior; production replay impact needs the caller-specific crash tests.**

`ContentManager.save_text` / `save_json` write the requested filename regardless of existing version (`:235-318`). `_write_bytes_atomically` (`:320-332`) uses a predictable `.tmp` path and replaces the final file. Writing the same identifier/version twice invalidated the first reference's checksum. `FrozenContentRef` overrides several dict mutators (`:30-67`) but not `__ior__`; `ref |= {'path': ...}` changed it successfully.

**Impact:** the object-level and storage-level guarantees advertised for old checkpoints are stronger than the APIs implement. Many callers correctly ask for the next version; do not call all those paths broken because a low-level overwrite is possible.

**Repair:** make content creation create-only or content-addressed; treat identical replay as idempotent and different replay at the same identity as conflict. Use exclusive temporary creation rather than a predictable `.tmp`; unify the good `utils/file_io.py:12-24` write primitive while preserving required overwrite versus create-only semantics. Define atomic visibility versus power-loss durability; directory fsync is absent from both implementations. Replace or complete the immutable mapping protocol.

**Acceptance:** retain old reference readability after later writes, reject conflicting same-version content, replay identical content without mutation, exercise failure between write/rename/checkpoint, verify supported mutation operations reject edits. Test pending checkpoint serialization with the actual LangGraph saver.

## C-01 — Project selection is implicit; title collisions overwrite configuration

**Source-proven and independently reproduced with synthetic configs.**

`main.py:39-58` selects newest candidate automatically, otherwise newest unfinished directory; `generate` has no project selector (`:82-87`). `core/project_manager.py:21-46` lowercases/normalizes/truncates titles and writes config without collision checks. Two distinct titles that normalize to the same slug overwrite the same `config.json`; the probe observed the second theme replacing the first. `find_resume_project` uses directory mtime (`:96-113`), and `count_completed_chapters` counts every `chapter_*.md` (`:115-121`), including `chapter_notes.md`, which the probe demonstrated.

**Repair:** explicit project identity/path for generate/resume; default selection only when unambiguous; reject existing project collisions unless explicit overwrite/new revision is requested; atomic validated config creation. Replace filename count with a contiguous finalized-chapter ledger reconciled with graph/checkpoint state. Single-user does not mean single-story forever.

**Acceptance:** two projects with same normalized title cannot collide; unrelated candidate never hijacks resume; malformed/gapped/draft/duplicate chapter files do not count as completion; finished projects are not reinitialized; interruption during config write preserves old configuration.

## C-02 — README invocation and storage contract do not match the CLI

**Real CLI probe confirms invalid invocation.**

`README.md:82-86` advertises `python main.py generate "premise"`; real argparse exits **2** with unrecognized arguments. Actual combined flow is `quick PROMPT` (`main.py:73-75,112-113`). Bootstrap with review writes `config.candidate.json` (`main.py:109`, `project_manager.py:42`), not the `config.json` the bootstrap instructions tell the writer to review (`README.md:74`). The orchestrator uses `checkpoints/saga.db` (`orchestration/langgraph_orchestrator.py:158`) rather than `.saga/checkpoints.db` in README (`:21,120`).

Repo guidance also names missing paths (`AGENTS.md:99,102,109-114`) and contradicts Python test discovery with a blanket “test names should not include test” convention (`:37`). These should be reconciled, not blindly obeyed during repair.

**Repair:** document/test actual public commands and artifact paths after the safety-critical lifecycle work; provide help that requires no service construction; keep one maintained contributor guide and links that exist.

**Acceptance:** run documented command parsing and help offline; bootstrap-review-generate fixture verifies candidate promotion; verify all local Markdown links and artifact examples.

## C-03 — Export can change authored prose

**Proven synthetic prose loss.**

`core/langgraph/export.py:63-80` treats any prefix `---` as YAML front matter, despite its docstring requiring an exact delimiter line. A prose line beginning `---A dialogue dash...`, followed later by a separator, lost its opening content. `:84-86` also replaces literal backslash-n throughout body text, and `:125` silently replaces invalid UTF-8. `_iter_chapter_files` (`:12-36`) does not reject gaps or duplicate numeric aliases, while no canonical files yields a path without an artifact (`:115-119`).

**Repair:** exact delimiter detection; preserve body bytes/text except explicitly documented normalization; separate complete manuscript export from partial preview; reject duplicate chapter identities and missing required chapters.

**Acceptance:** dialogue/separator/literal escape fidelity; exact front matter; ordered/gapped/duplicate/empty input; export completion receipt includes actual emitted chapters.

## T-01 — The unfinished split is not a clean-checkout deliverable

Ten helper modules under `core/langgraph/nodes/` are untracked. Their importers are modified tracked files. They must be preserved and included together in the eventual repair snapshot. Relevant symptoms:

- Moved provider names remain missing at test patch targets (`test-failures.json`).
- `_LLM_SERVICE_PATCH_MODULES` misses `core.langgraph.nodes.context_scene_retrieval`; the existing inventory test fails with that exact name.
- `commit_node.py:599-600` and `commit_validation.py:104` contain unresolved annotations. Deferred annotations mean this is not proof of ordinary import failure, but static checking and runtime type-hint evaluation are broken.
- `scene_extraction_parsing.py:17-44` declares a list of dicts but returns `(name, info)` tuples; callers unpack tuples. Correct the contract instead of changing runtime output to fit a wrong annotation.
- `tests/test_langgraph/conftest.py:186` references missing `FakeNeo4jManager` in a deferred annotation.
- `_extract_scene_characters` is still an exact duplicate in `context_character_retrieval.py:77-96` and `context_retrieval_node.py:273-292`. A second exact duplicate exists for parser `create_location_nodes`, recorded in `exact-duplicate-functions.json`.

Finish the current seam before splitting additional monoliths. Tests should use injected services or narrow providers, not ever-growing facade monkeypatches.

## T-02 — The test boundary is configured but not enforced

`tests/conftest.py:13-52` makes `--unit-stubs` skip named heavy markers. AST inventory found **no uses of `pytest.mark.unit`, `.integration`, or `.slow` in the test source**; names such as `TestFullPipelineIntegration` do not apply markers. Thus that switch cannot enforce isolation. `pyproject.toml:42` configures `timeout` but `requirements.txt` does not declare pytest-timeout. The asyncio fixture loop scope is unset; deprecation/resource warnings are broadly ignored (`pyproject.toml:54-57`). Coverage reporting is configured without selecting a `--cov` target in addopts.

The suite retains substantial working behavior: 1,639 passing cases under the denied-network harness. Preserve those tests, but prioritize (a) denied-network-by-default unit execution, (b) explicit opt-in disposable-Neo4j tests, (c) a real deterministic workflow fixture. Do not “repair” the suite by deleting failing integration-shaped tests or relaxing strict assertions.

## G-01 — Git needs preservation and integration, not a reset or history purge

`git-analysis.json` records **284 modified tracked paths, 2 tracked deletions and 20 individual untracked files**, excluding this audit directory. All 284 surviving modified tracked paths have 100644→100755 mode churn. Ignoring mode only **for a read command** reduces the tracked content changes to **13 modified files + 2 deletions**. No Git config was changed. No changes were staged at baseline.

There are **21 local branches and one worktree**. Current branch is one commit ahead of its cached upstream. `dev` is an ancestor, 63 commits behind HEAD. `master` and current HEAD diverge: 11 master-only and 72 HEAD-only commits. `git cherry HEAD master` identifies its non-merge patch as already represented, but that does **not** settle merge-only conflict resolutions. Two old Claude branches retain patch-distinct commits (`a363c2d`, `ce3bd37`); review them rather than deleting every `: gone` branch. `feature/spacy`'s differing patch is equivalent. Ref counts and equivalence outputs are retained; no remote fetch was performed, so remote-tracking information is cached.

`git fsck --connectivity-only --no-dangling` passed. `git count-objects -vH`: 21.46 MiB loose objects, 31.58 MiB packs, **zero garbage**. There is no evidence for emergency history surgery. Worktree metadata inventory shows the non-executable `.venv` at **8,301,858,245 logical bytes**; this, caches and agent traces dominate local clutter, not application source or Git history. Logical bytes are not allocated-disk measurements.

**Immediate proposed sequence (not executed):**

1. Snapshot refs with a verified Git bundle; save meaningful binary-capable tracked patch and copy all required untracked source. Separately inventory/back up stories, content and Neo4j with an actual restore test. A Git bundle alone does not contain dirty work or ignored stories/database state.
2. Establish an explicit integration branch from the current dirty implementation, preserving the source authority. Normalize accidental executable bits in a separate reviewed change or use a documented local-only fileMode policy; do not hide the semantic diff in a mass chmod commit.
3. Preserve/index the ten helper files together with their callers; separate timeout-policy changes, service deletion, state annotations and module movement into reviewable changes.
4. Check branch-only patches and merge resolutions against the intended integration tree. Only then archive/delete obsolete refs. Do not assume branch names or cached remote deletion proves no value.
5. Rebuild a Linux Python 3.12 environment from a resolved lock, then retire the unusable `.venv` and regenerable caches after preservation. Do not delete projects, Neo4j, agent history or ignored `src/` based on name alone.
6. Clean `.gitignore`: repeated self-ignore rules, ignored `src/`, and accidentally public `.claude/settings.local.json` candidate need an explicit repository boundary. SAGA should retain tests, prompts, useful docs and maintenance tools; DWARF's source-only publication policy does not apply here.

## Reuse and limitations

`boundary_probes.py` uses only generated temporary fixtures; `boundary-probes.txt` records the actual observations. `offline_runner.py` documents the sandbox/guard. No protected user data was used. The full source manifest should be checked for drift before later implementation, but unchanged files do not need a new census. Focused re-reading and behavior tests remain necessary for each edit; this report cannot certify unseen future changes or real-model narrative quality.
