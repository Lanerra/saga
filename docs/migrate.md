# 🚀 NANA → LangGraph Migration Plan

**Status**: LangGraph 85% feature-complete | NANA retirement planned
**Estimated Effort**: 4-5 weeks to full migration
**Code Reduction**: ~4,800 LOC (9.9% of codebase)

---

## Executive Summary

SAGA currently maintains two orchestration pipelines:
- **NANA** (legacy): 3,309 LOC imperative orchestrator + 1,211 LOC patch-based revision
- **LangGraph** (modern): 7,683 LOC graph-based workflow with checkpointing and resume

This document provides a detailed plan to **fully migrate to LangGraph** and **remove all NANA-specific code**, reducing codebase by ~4,800 LOC while maintaining or improving functionality.

### Migration Readiness: 85% Complete ✅

**What LangGraph Has**:
- ✅ Complete chapter generation workflow (7 nodes)
- ✅ Full initialization workflow (7 nodes)
- ✅ Automatic checkpointing & resume
- ✅ Error handling with graceful recovery
- ✅ Full rewrite-based revision system

**What's Missing**:
- ⚠️ Patch-based revision (NANA has 1,211 LOC sophisticated system)
- ⚠️ Text deduplication step
- ⚠️ Ingestion mode support
- ⚠️ User story priming

**Blocking Decision**: Port NANA's patch-based revision (1,211 LOC effort) or accept LangGraph's simpler full-rewrite approach?

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Functionality Comparison](#2-functionality-comparison)
3. [Gap Analysis](#3-gap-analysis)
4. [Migration Plan](#4-migration-plan)
5. [Code Removal Targets](#5-code-removal-targets)
6. [Risk Assessment](#6-risk-assessment)
7. [Decision Points](#7-decision-points)

---

## 1. Current State Analysis

### 1.1 NANA Code Inventory

#### Primary NANA-Specific Files (3,309 LOC)

| File | LOC | Purpose | Replacement |
|------|-----|---------|-------------|
| `orchestration/nana_orchestrator.py` | 1,912 | Main orchestrator with imperative chapter loop | `langgraph_orchestrator.py` + nodes |
| `processing/revision_logic.py` | 1,211 | Patch-based revision system | `nodes/revision_node.py` (365 LOC, full rewrite) |
| `orchestration/chapter_flow.py` | 93 | High-level pipeline wrapper | LangGraph workflow graph |
| `models/narrative_state.py` | 93 | NANA state container | `core/langgraph/state.py` |

#### Related Processing Files (1,858 LOC)

| File | LOC | NANA-Specific? | Notes |
|------|-----|----------------|-------|
| `processing/problem_parser.py` | 76 | Likely | Used by `revision_logic.py` |
| `processing/state_tracker.py` | 257 | Likely | May be NANA state tracking |
| `processing/zero_copy_context_generator.py` | 314 | Mixed | Context building (may be NANA-specific) |

#### NANA-Specific Tests (850 LOC)

- `test_orchestrator_private_methods.py` (124 LOC)
- `test_orchestrator_refresh.py` (27 LOC)
- `test_revision_patching.py` (431 LOC) - Tests patch-based revision
- `test_revision_world_ids.py` (62 LOC)
- `test_novel_generation_dynamic.py` (63 LOC)
- `test_ingestion_healing.py` (57 LOC)
- `test_ingestion_mode.py` (40 LOC)
- `test_context_snapshot.py` (46 LOC)

**Total NANA-Only Code**: ~6,017 LOC

### 1.2 LangGraph Implementation Summary

#### Core LangGraph Files (7,683 LOC)

| Component | File | LOC | Purpose |
|-----------|------|-----|---------|
| **Orchestrator** | `orchestration/langgraph_orchestrator.py` | 463 | Main entry point, manages chapter loop |
| **State Schema** | `core/langgraph/state.py` | 370 | TypedDict state definition |
| **Workflow Graph** | `core/langgraph/workflow.py` | 857 | Graph structure, routing logic |
| **Context Builder** | `core/langgraph/graph_context.py` | ~300 | Assembles KG context for prompts |

#### LangGraph Generation Nodes (3,003 LOC)

| Node | File | LOC | Replaces NANA Method |
|------|------|-----|----------------------|
| **Generation** | `nodes/generation_node.py` | 311 | `_draft_initial_chapter_text()` |
| **Extraction** | `nodes/extraction_node.py` | 553 | Calls shared `KnowledgeAgent` |
| **Commit** | `nodes/commit_node.py` | 813 | Calls shared KG service |
| **Validation** | `nodes/validation_node.py` | 445 | `_run_evaluation_cycle()` |
| **Revision** | `nodes/revision_node.py` | 365 | ⚠️ Different approach than `revision_logic.py` |
| **Summary** | `nodes/summary_node.py` | 284 | Part of `_finalize_and_save_chapter()` |
| **Finalize** | `nodes/finalize_node.py` | 232 | Part of `_finalize_and_save_chapter()` |

#### LangGraph Initialization Nodes (~3,151 LOC)

- `initialization/character_sheets_node.py` (~400 LOC)
- `initialization/global_outline_node.py` (~300 LOC)
- `initialization/act_outlines_node.py` (~300 LOC)
- `initialization/chapter_outline_node.py` (~400 LOC)
- `initialization/commit_init_node.py` (~500 LOC)
- `initialization/persist_files_node.py` (651 LOC)
- `initialization/workflow.py` (~600 LOC)

### 1.3 Shared Components (4,331 LOC)

Both pipelines use these agents and utilities:

| Component | File | LOC | Usage |
|-----------|------|-----|-------|
| **Knowledge Agent** | `agents/knowledge_agent.py` | 2,253 | Entity extraction, KG persistence |
| **Narrative Agent** | `agents/narrative_agent.py` | 657 | Scene planning, drafting |
| **Revision Agent** | `agents/revision_agent.py` | 708 | Quality evaluation |
| **Entity Deduplication** | `processing/entity_deduplication.py` | 244 | Name matching, similarity |
| **Text Deduplicator** | `processing/text_deduplicator.py` | 155 | Removes repetitive text |
| **Parsing Utils** | `processing/parsing_utils.py` | 314 | LLM output parsing |

---

## 2. Functionality Comparison

### 2.1 Chapter Generation Flow

| Step | NANA Method | LangGraph Node | Status |
|------|-------------|----------------|--------|
| **Planning** | `_prepare_chapter_prerequisites()` | `chapter_outline_node` | ✅ Migrated |
| **Context** | `_build_context_snapshot()` | `graph_context.py` + state | ✅ Migrated |
| **Draft** | `_draft_initial_chapter_text()` | `generation_node` | ✅ Migrated |
| **Evaluation** | `_run_evaluation_cycle()` | `validation_node` | ✅ Migrated |
| **Revision** | `_perform_revisions()` + `revision_logic.py` | `revision_node` | ⚠️ **Different approach** |
| **Deduplication** | `perform_deduplication()` | ❌ Not in workflow | ❌ **Missing** |
| **Finalization** | `_finalize_and_save_chapter()` | `summary_node` + `finalize_node` | ✅ Migrated |

### 2.2 Revision System Comparison

| Feature | NANA (`revision_logic.py`) | LangGraph (`revision_node.py`) |
|---------|---------------------------|-------------------------------|
| **Approach** | **Patch-based**: Targeted fixes to specific sections | **Full rewrite**: Regenerate entire chapter |
| **Complexity** | 1,211 LOC with problem grouping, patch validation | 365 LOC single prompt |
| **Pros** | Preserves good sections, precise fixes | Simpler, better for structural issues |
| **Cons** | Complex logic, harder to maintain | May lose good prose, higher token cost |
| **Status** | NANA-only | LangGraph-only |

**Key Difference**: NANA's patch-based system is sophisticated but unused by LangGraph.

**Decision Required**: Port patch system (1,211 LOC effort) or accept full rewrite approach?

### 2.3 Initialization Comparison

| Component | NANA | LangGraph | Status |
|-----------|------|-----------|--------|
| **Bootstrap** | `perform_initial_setup()` | 7-node initialization workflow | ✅ **Both use same bootstrap pipeline** |
| **Character Profiles** | Bootstrap → YAML files | `character_sheets_node` | ✅ Equivalent |
| **Plot Outline** | Bootstrap → YAML files | `global_outline_node` + `act_outlines_node` | ✅ Equivalent |
| **Chapter Planning** | On-demand in `_prepare_chapter_prerequisites()` | `chapter_outline_node` | ✅ Equivalent |
| **User Story Priming** | `_prime_from_user_story_elements()` | ❌ Not implemented | ⚠️ **NANA-only** |

### 2.4 Error Handling & Checkpointing

| Feature | NANA | LangGraph |
|---------|------|-----------|
| **Error Recovery** | Try/catch with logging | Error handler node + `has_fatal_error` flag |
| **State Persistence** | None (lost on crash) | ✅ **Automatic checkpointing via AsyncSqliteSaver** |
| **Resume After Crash** | ❌ Start from scratch | ✅ **Resume from last successful node** |
| **Progress Tracking** | Rich progress panels | Structlog events + LangGraph state |

**Winner**: LangGraph (automatic resume is major improvement)

---

## 3. Gap Analysis

### 3.1 Critical Gaps (Block NANA Removal)

| Gap | NANA Location | Impact if Missing | Priority |
|-----|---------------|-------------------|----------|
| **1. Patch-Based Revision** | `processing/revision_logic.py` (1,211 LOC) | Quality regression: lose targeted fixes | 🔴 **HIGH** |
| **2. Text Deduplication** | `perform_deduplication()` | Quality regression: repetitive prose | 🟡 **MEDIUM** |
| **3. Ingestion Mode** | `run_ingestion_process()` | Feature gap: can't import existing text | 🟡 **MEDIUM** |
| **4. User Story Priming** | `_prime_from_user_story_elements()` | UX gap: no user story input | 🟢 **LOW** |

### 3.2 Non-Critical Gaps (Nice to Have)

| Gap | NANA Location | Impact |
|-----|---------------|--------|
| **Per-Chapter Log Files** | Separate log file per chapter | Debugging: harder to trace chapter-specific issues |
| **Debug Output Saving** | `_save_debug_output()` | Debugging: no intermediate artifact saving |
| **Rich Display Updates** | `_update_rich_display()` | UX: different progress display |
| **Auto KG Healing** | Triggered every N chapters | Maintenance: manual healing required |
| **Plot Continuation** | `_generate_plot_points_from_kg()` | Edge case: extending beyond planned chapters |

### 3.3 Dependency Analysis

#### Files Importing NANA Code

**`nana_orchestrator.py`**:
- `main.py` (pipeline selection)
- `orchestration/chapter_flow.py`
- 6 test files

**`revision_logic.py`**:
- `orchestration/nana_orchestrator.py` (only usage)
- `tests/test_revision_patching.py`
- `tests/test_revision_world_ids.py`

**`models/narrative_state.py`**:
- `agents/narrative_agent.py` (type hint import: `from models.narrative_state import NarrativeState`)
- `orchestration/nana_orchestrator.py`
- `tests/test_context_snapshot.py`

**Impact**: Minimal cross-dependencies. Mostly confined to NANA orchestrator and tests.

---

## 4. Migration Plan

### Phase 1: Feature Parity (2-3 weeks)

**Goal**: Close critical gaps so LangGraph can fully replace NANA

#### Week 1: Quality Features

**Task 1.1: Add Text Deduplication to LangGraph**
- **Location**: Insert after `generation_node` and `revision_node`
- **Effort**: ~50 LOC (utility already exists in `processing/text_deduplicator.py`)
- **Implementation**:
  ```python
  # In generation_node.py and revision_node.py
  from processing.text_deduplicator import deduplicate_text

  # After draft generation
  deduplicated_text = deduplicate_text(draft_text)
  state["draft_text"] = deduplicated_text
  ```
- **Files Modified**: `nodes/generation_node.py`, `nodes/revision_node.py`
- **Tests**: Add deduplication assertion to generation tests

**Task 1.2: Revision Quality Comparison Test**
- **Goal**: Measure quality difference between patch-based (NANA) vs full-rewrite (LangGraph)
- **Method**:
  1. Generate same chapter with both pipelines
  2. Compare: word count preservation, prose quality, coherence
  3. Measure: token costs, latency
- **Deliverable**: Report with recommendation (port patch system or accept full rewrite)
- **Files**: Create `scripts/compare_revision_quality.py`

**Task 1.3: DECISION POINT - Patch-Based Revision**
- **Option A**: Port `revision_logic.py` to LangGraph node
  - Effort: ~1-2 weeks (1,211 LOC to adapt)
  - Benefit: Preserve targeted revision capability
  - Risk: Complexity increase, maintenance burden

- **Option B**: Accept full-rewrite approach
  - Effort: 0 (already implemented)
  - Benefit: Simplicity, structural fixes
  - Risk: Quality regression if patch-based is superior

- **Recommendation**: Run Task 1.2 comparison test first, then decide

#### Week 2: Ingestion & Optional Features

**Task 2.1: Port Ingestion Mode (if needed)**
- **Check**: Survey if `--ingest` mode is actually used
- **If yes**: Create `ingestion_workflow.py` in `core/langgraph/`
- **Effort**: ~200 LOC
- **Files**:
  - Create `core/langgraph/ingestion_workflow.py`
  - Add `--ingest` flag handler to `langgraph_orchestrator.py`
  - Port logic from `NANA_Orchestrator.run_ingestion_process()`

**Task 2.2: End-to-End LangGraph Test**
- **Goal**: Generate full multi-chapter novel with LangGraph
- **Method**:
  - Set `CHAPTERS_PER_RUN = 5`
  - Run `python main.py --langgraph`
  - Verify all nodes execute successfully
  - Check chapter quality, KG consistency
- **Deliverable**: Test report + test novel artifact

**Task 2.3: Performance Benchmarking**
- **Goal**: Compare NANA vs LangGraph performance
- **Metrics**:
  - Chapter generation latency
  - Token usage
  - Memory consumption
  - Resume speed (LangGraph only)
- **Files**: Create `scripts/benchmark_pipelines.py`

#### Week 3: Validation & Documentation

**Task 3.1: Checkpoint/Resume Validation**
- **Goal**: Verify LangGraph resume works correctly
- **Method**:
  1. Start generation, kill process mid-chapter
  2. Restart with same config
  3. Verify state restored correctly
  4. Check no duplicate work
- **Deliverable**: Resume test suite

**Task 3.2: Quality Comparison**
- **Goal**: Compare output quality between pipelines
- **Method**:
  - Generate same story with both pipelines
  - Compare: deduplication, character consistency, plot coherence
  - Human review of prose quality
- **Deliverable**: Quality report

**Task 3.3: Update Documentation**
- **Files to Update**:
  - `README.md`: Remove NANA references, LangGraph becomes default
  - `CLAUDE.md`: Update orchestration section
  - `docs/LANGGRAPH_USAGE.md`: Expand usage guide
  - `docs/langgraph-architecture.md`: Update with any new nodes

### Phase 2: Transition (1 week)

**Goal**: Make LangGraph the default, deprecate NANA

#### Week 4: Default Pipeline Switch

**Task 4.1: Change Default to LangGraph**
- **File**: `main.py`
- **Changes**:
  ```python
  # OLD:
  if args.langgraph:
      orchestrator = LangGraphOrchestrator()
  else:
      orchestrator = NANA_Orchestrator()

  # NEW:
  if args.use_nana:  # Deprecated flag
      warnings.warn("NANA pipeline is deprecated. Use LangGraph (default).")
      orchestrator = NANA_Orchestrator()
  else:
      orchestrator = LangGraphOrchestrator()
  ```
- **Add**: Deprecation warning in help text

**Task 4.2: Announcement**
- **Channel**: GitHub Issues + README
- **Message**: Announce NANA deprecation, LangGraph default, timeline for removal
- **Timeline**: 2 weeks notice before deletion

**Task 4.3: Final Testing**
- Run full test suite with LangGraph as default
- Fix any regressions
- Verify CI/CD passes

### Phase 3: Cleanup (1 week)

**Goal**: Remove all NANA-specific code

#### Week 5: Code Deletion

**Task 5.1: Delete NANA Orchestrator Files**
- **Files** (2,098 LOC):
  - `orchestration/nana_orchestrator.py` (1,912 LOC)
  - `orchestration/chapter_flow.py` (93 LOC)
  - `models/narrative_state.py` (93 LOC)

**Task 5.2: Delete NANA-Specific Processing**
- **Files** (depends on Task 1.3 decision):
  - `processing/revision_logic.py` (1,211 LOC) - **IF not ported**
  - `processing/problem_parser.py` (76 LOC) - **IF only used by revision_logic**
  - `processing/state_tracker.py` (257 LOC) - **IF NANA-only**

**Task 5.3: Delete NANA Tests**
- **Files** (850 LOC):
  - `tests/test_orchestrator_private_methods.py`
  - `tests/test_orchestrator_refresh.py`
  - `tests/test_revision_patching.py`
  - `tests/test_revision_world_ids.py`
  - `tests/test_novel_generation_dynamic.py`
  - `tests/test_ingestion_healing.py`
  - `tests/test_ingestion_mode.py`
  - `tests/test_context_snapshot.py`

**Task 5.4: Update Imports**
- **File**: `agents/narrative_agent.py`
- **Remove**: `from models.narrative_state import NarrativeState`
- **Impact**: Only used for type hints in NANA context (unused by LangGraph)

**Task 5.5: Clean Up main.py**
- Remove `--use_nana` flag entirely
- Remove NANA import
- Simplify orchestrator instantiation

**Task 5.6: Final Documentation Update**
- Remove all NANA references
- Update architecture docs
- Update tutorial/examples to use LangGraph only

---

## 5. Code Removal Targets

### 5.1 Definite Deletions (2,098 LOC)

After LangGraph proven stable, these files can be deleted with zero functional loss:

| File | LOC | Reason |
|------|-----|--------|
| `orchestration/nana_orchestrator.py` | 1,912 | Fully replaced by `langgraph_orchestrator.py` + nodes |
| `orchestration/chapter_flow.py` | 93 | Fully replaced by LangGraph workflow graph |
| `models/narrative_state.py` | 93 | Fully replaced by `core/langgraph/state.py` |

### 5.2 Conditional Deletions (1,858 LOC)

Delete if Task 1.3 decides NOT to port patch-based revision:

| File | LOC | Condition |
|------|-----|-----------|
| `processing/revision_logic.py` | 1,211 | **IF** not ported to LangGraph |
| `processing/problem_parser.py` | 76 | **IF** only used by `revision_logic.py` |
| `processing/state_tracker.py` | 257 | **IF** NANA state tracking only |
| `processing/zero_copy_context_generator.py` | 314 | **IF** NANA context building only |

**Verification Required**: Check each file for non-NANA usage before deletion.

### 5.3 Test Deletions (850 LOC)

| File | LOC | Depends On |
|------|-----|------------|
| `test_orchestrator_private_methods.py` | 124 | Delete when `nana_orchestrator.py` deleted |
| `test_orchestrator_refresh.py` | 27 | Delete when `nana_orchestrator.py` deleted |
| `test_revision_patching.py` | 431 | Delete when `revision_logic.py` deleted |
| `test_revision_world_ids.py` | 62 | Delete when `revision_logic.py` deleted |
| `test_novel_generation_dynamic.py` | 63 | Delete when `nana_orchestrator.py` deleted |
| `test_ingestion_healing.py` | 57 | Delete when ingestion ported or abandoned |
| `test_ingestion_mode.py` | 40 | Delete when ingestion ported or abandoned |
| `test_context_snapshot.py` | 46 | Delete when `narrative_state.py` deleted |

### 5.4 Total Removal Estimate

| Scenario | LOC Removed | % of Codebase (48,555 total) |
|----------|-------------|------------------------------|
| **Minimum** (Keep revision_logic) | 2,948 | 6.1% |
| **Maximum** (Delete revision_logic) | 4,806 | 9.9% |
| **Most Likely** (Delete revision_logic) | 4,806 | 9.9% |

**Recommendation**: Full deletion (4,806 LOC) if Task 1.2 shows LangGraph revision quality is acceptable.

---

## 6. Risk Assessment

### 6.1 Risk Matrix

| Risk | Likelihood | Impact | Severity | Mitigation |
|------|-----------|--------|----------|------------|
| **Quality regression without patch-based revision** | 🟡 Medium | 🔴 High | 🔴 **HIGH** | Run comparison test (Task 1.2); port if needed |
| **Deduplication loss degrades output** | 🟡 Medium | 🟡 Medium | 🟡 **MEDIUM** | Add dedup node (Task 1.1, low effort) |
| **LangGraph bugs in production** | 🟢 Low | 🔴 High | 🟡 **MEDIUM** | Extended testing (Phase 1), keep NANA available during transition |
| **Missing ingestion mode breaks workflows** | 🟢 Low | 🟡 Medium | 🟢 **LOW** | Survey users (Task 2.1) |
| **Performance regression** | 🟢 Low | 🟢 Low | 🟢 **LOW** | Benchmark (Task 2.3) |
| **Checkpoint corruption** | 🟢 Low | 🔴 High | 🟡 **MEDIUM** | Resume validation (Task 3.1) |

### 6.2 What Could Go Wrong?

**Scenario 1: Revision Quality Significantly Worse**
- **Trigger**: Task 1.2 comparison shows LangGraph produces lower-quality output
- **Impact**: User dissatisfaction, regression from NANA
- **Mitigation**: Port patch-based revision (1,211 LOC effort, 1-2 weeks)
- **Contingency**: Keep NANA available as `--use_nana` fallback until revision ported

**Scenario 2: Critical LangGraph Bug After NANA Deletion**
- **Trigger**: Production users hit edge case that crashes LangGraph
- **Impact**: No fallback, blocked users
- **Mitigation**: Extended testing in Phase 1, gradual rollout in Phase 2
- **Contingency**: Git revert NANA deletion, fix bug, re-attempt deletion

**Scenario 3: Deduplication Removal Causes Repetition**
- **Trigger**: Users complain about repetitive prose after migration
- **Impact**: Quality regression
- **Mitigation**: Add deduplication in Task 1.1 (preventative)
- **Contingency**: Add deduplication retroactively if issues arise

**Scenario 4: User Workflows Depend on Ingestion Mode**
- **Trigger**: User reports ingestion mode missing after migration
- **Impact**: Blocked workflows, feature gap
- **Mitigation**: Survey ingestion usage in Task 2.1
- **Contingency**: Port ingestion mode retroactively (~200 LOC, 1-2 days)

### 6.3 Rollback Plan

If migration fails catastrophically:

1. **Git Revert**: Restore NANA orchestrator files from git history
2. **Revert main.py**: Change default back to NANA
3. **Fix Issues**: Debug LangGraph issues in parallel
4. **Re-attempt**: Once stable, retry migration

**Rollback Cost**: ~1 hour (git operations only)

---

## 7. Decision Points

### 7.1 Critical Decision: Patch-Based Revision

**Question**: Port NANA's 1,211 LOC patch-based revision system or accept LangGraph's full-rewrite approach?

#### Option A: Port Patch-Based Revision

**Pros**:
- ✅ Preserves targeted fix capability
- ✅ May produce higher quality output (needs testing)
- ✅ Lower token costs (only rewrite broken sections)

**Cons**:
- ❌ 1,211 LOC of complex logic to port
- ❌ 1-2 weeks additional effort
- ❌ Increased LangGraph complexity
- ❌ Maintenance burden

**Implementation**:
- Create `nodes/patch_revision_node.py`
- Port logic from `processing/revision_logic.py`
- Adapt to LangGraph state model
- Add routing logic to choose patch vs full rewrite

#### Option B: Accept Full-Rewrite Approach

**Pros**:
- ✅ 0 additional effort (already implemented)
- ✅ Simpler codebase (365 LOC vs 1,211 LOC)
- ✅ Better for structural issues
- ✅ Easier to maintain

**Cons**:
- ❌ May lose good prose sections
- ❌ Higher token costs (regenerate entire chapter)
- ❌ Potential quality regression (needs testing)

**Implementation**:
- No changes needed
- Delete `revision_logic.py` in Phase 3

#### Recommendation Process

1. **Run Task 1.2**: Quality comparison test
2. **Measure**:
   - Prose quality (human review)
   - Word count preservation
   - Token costs
   - Latency
3. **Decision Criteria**:
   - **IF** quality difference < 10%: Choose Option B (simpler)
   - **IF** quality difference > 10%: Choose Option A (port)
   - **IF** token cost difference > 50%: Factor into decision

**Timeline**: Decision by end of Week 1 (Phase 1)

### 7.2 Secondary Decision: Ingestion Mode

**Question**: Port ingestion mode or remove feature?

**Survey First**:
- Check GitHub issues for ingestion-related questions
- Search codebase for ingestion usage examples
- Ask in community channels (if applicable)

**Decision Criteria**:
- **IF** users actively use ingestion: Port (Task 2.1, ~200 LOC)
- **IF** no evidence of usage: Remove feature

**Timeline**: Decision by Week 2 (Phase 1)

### 7.3 Tertiary Decision: User Story Priming

**Question**: Port `_prime_from_user_story_elements()` or accept loss?

**Assessment**:
- Feature allows user to provide story elements before generation
- UX improvement, not critical functionality
- LangGraph initialization already accepts character/world inputs

**Decision Criteria**:
- **IF** users request feature: Port (~100 LOC effort)
- **IF** no demand: Accept loss

**Timeline**: Decision by Week 3 (Phase 1)

---

## 8. Success Criteria

### 8.1 Phase 1 Success Criteria

- ✅ Text deduplication added to LangGraph
- ✅ Revision quality comparison test completed
- ✅ Decision made on patch-based revision
- ✅ End-to-end LangGraph test passes (5+ chapters)
- ✅ Performance benchmarks show acceptable results
- ✅ Checkpoint/resume validated

### 8.2 Phase 2 Success Criteria

- ✅ LangGraph is default pipeline in `main.py`
- ✅ Deprecation announcement published
- ✅ All tests pass with LangGraph default
- ✅ No critical bugs reported during transition period

### 8.3 Phase 3 Success Criteria

- ✅ NANA orchestrator files deleted (2,098+ LOC)
- ✅ NANA tests deleted (850 LOC)
- ✅ All imports updated (no broken references)
- ✅ Documentation updated (no NANA mentions)
- ✅ Full test suite passes
- ✅ User feedback positive

### 8.4 Quality Acceptance Criteria

Before NANA deletion is permitted:

1. **Chapter Quality**:
   - LangGraph chapters meet or exceed NANA quality
   - No significant increase in repetition
   - Character consistency maintained
   - Plot coherence preserved

2. **Reliability**:
   - LangGraph generates 10 chapters without crash
   - Checkpoint/resume works 100% of the time
   - No data loss on interruption

3. **Performance**:
   - Chapter generation latency within 20% of NANA
   - Token usage within acceptable range
   - Memory usage stable

---

## 9. Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Feature Parity** | 2-3 weeks | Deduplication added, revision decision made, testing complete |
| **Phase 2: Transition** | 1 week | LangGraph default, deprecation announced |
| **Phase 3: Cleanup** | 1 week | NANA code deleted, docs updated |
| **TOTAL** | **4-5 weeks** | **~4,800 LOC removed (9.9% reduction)** |

---

## 10. Next Steps

### Immediate Actions (This Week)

1. **Task 1.1**: Add text deduplication to LangGraph generation/revision nodes
2. **Task 1.2**: Set up revision quality comparison test infrastructure
3. **Review**: Share this migration plan with team/stakeholders

### Week 1 Actions

1. **Run**: Revision quality comparison test
2. **Decide**: Patch-based revision (Option A vs Option B)
3. **Start**: End-to-end LangGraph test (5 chapters)

### Ongoing

- Monitor LangGraph stability during testing
- Document any edge cases or bugs
- Gather user feedback on LangGraph experience

---

## Appendix A: File-by-File Deletion Checklist

### Phase 3, Task 5.1: Orchestrator Deletion

- [ ] Delete `orchestration/nana_orchestrator.py` (1,912 LOC)
- [ ] Delete `orchestration/chapter_flow.py` (93 LOC)
- [ ] Delete `models/narrative_state.py` (93 LOC)
- [ ] Update `agents/narrative_agent.py`: Remove `from models.narrative_state import NarrativeState`
- [ ] Update `main.py`: Remove NANA orchestrator import and instantiation
- [ ] Run full test suite, verify no breakage

### Phase 3, Task 5.2: Processing Deletion (Conditional)

**IF** patch-based revision NOT ported:
- [ ] Verify `revision_logic.py` only used by NANA
- [ ] Delete `processing/revision_logic.py` (1,211 LOC)
- [ ] Verify `problem_parser.py` only used by `revision_logic.py`
- [ ] Delete `processing/problem_parser.py` (76 LOC)

**IF** state_tracker.py is NANA-only:
- [ ] Verify `state_tracker.py` only used by NANA
- [ ] Delete `processing/state_tracker.py` (257 LOC)

**IF** zero_copy_context_generator.py is NANA-only:
- [ ] Check if used by LangGraph `graph_context.py`
- [ ] Delete `processing/zero_copy_context_generator.py` (314 LOC) IF NANA-only

### Phase 3, Task 5.3: Test Deletion

- [ ] Delete `tests/test_orchestrator_private_methods.py` (124 LOC)
- [ ] Delete `tests/test_orchestrator_refresh.py` (27 LOC)
- [ ] Delete `tests/test_revision_patching.py` (431 LOC)
- [ ] Delete `tests/test_revision_world_ids.py` (62 LOC)
- [ ] Delete `tests/test_novel_generation_dynamic.py` (63 LOC)
- [ ] Delete `tests/test_ingestion_healing.py` (57 LOC)
- [ ] Delete `tests/test_ingestion_mode.py` (40 LOC)
- [ ] Delete `tests/test_context_snapshot.py` (46 LOC)
- [ ] Run pytest, verify no test failures due to missing files

### Phase 3, Task 5.6: Documentation Update

- [ ] Update `README.md`: Remove NANA references, LangGraph is only option
- [ ] Update `CLAUDE.md`: Update orchestration section
- [ ] Update `docs/langgraph-architecture.md`: Reflect current state
- [ ] Update `docs/LANGGRAPH_USAGE.md`: Expand as primary guide
- [ ] Archive this file: Rename to `docs/nana-migration-completed.md`

---

## Appendix B: Grep Commands for Verification

### Check for NANA Dependencies Before Deletion

```bash
# Find all imports of nana_orchestrator
grep -r "from orchestration.nana_orchestrator" --include="*.py"
grep -r "import.*nana_orchestrator" --include="*.py"

# Find all imports of revision_logic
grep -r "from processing.revision_logic" --include="*.py"
grep -r "import.*revision_logic" --include="*.py"

# Find all imports of models.narrative_state
grep -r "from models.narrative_state" --include="*.py"
grep -r "import.*narrative_state" --include="*.py"

# Find all NANA_Orchestrator instantiations
grep -r "NANA_Orchestrator()" --include="*.py"

# Find all references to ContextSnapshot
grep -r "ContextSnapshot" --include="*.py"
```

### Verify Clean Deletion

```bash
# After deletion, verify no broken imports
python -m py_compile main.py
python -m py_compile orchestration/langgraph_orchestrator.py
python -m py_compile agents/narrative_agent.py

# Run import test
python -c "from orchestration.langgraph_orchestrator import LangGraphOrchestrator; print('✓ Import successful')"

# Run full test suite
pytest
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Status**: Migration plan ready for execution
**Next Review**: After Phase 1 completion (Week 3)
