# 🗑️ Phase 3: NANA Removal Plan

**Status**: Ready to execute (LangGraph is now default as of Phase 2)
**Created**: 2025-11-14
**Estimated Effort**: 4-6 hours
**LOC to Remove**: ~2,004 lines total (1,912 NANA + 92 chapter_flow)

---

## Prerequisites (All Complete ✅)

- [x] **Phase 1**: Text deduplication migrated with provisional tracking
- [x] **Phase 2**: LangGraph made default, NANA deprecated
- [x] **Feature Parity Analysis**: Complete (76% weighted, 100% core)
- [x] **Test Suite**: 444/465 tests passing with LangGraph

**Ready to proceed**: ✅ All prerequisites met

---

## Phase 3 Overview

Phase 3 removes the deprecated NANA orchestrator and associated feature creep while preserving valuable functionality in appropriate locations.

### Three-Stage Approach

1. **Extract & Preserve** - Move reusable features to appropriate modules
2. **Delete & Clean** - Remove NANA orchestrator and feature creep
3. **Verify & Document** - Ensure nothing broke, update docs

---

## Stage 1: Extract & Preserve (2-3 hours)

### 1.1 Extract Logging Configuration

**File to Create**: `core/logging_config.py`

**What to Extract**: Logging setup from `nana_orchestrator.py` (lines 1812-1912, ~100 LOC)

**Features to Preserve**:
- Rich logging handler setup
- File rotation configuration
- Log level management
- Suppression of verbose third-party logs

**Why**: Logging configuration is useful for any orchestrator, not NANA-specific

**Implementation**:
```python
# core/logging_config.py

import logging
import structlog
from pathlib import Path
from rich.logging import RichHandler

def setup_saga_logging(
    log_level: int = logging.INFO,
    log_dir: Path = Path("output/chapter_logs"),
    enable_rich: bool = True,
    enable_file_rotation: bool = True
) -> None:
    """
    Setup SAGA logging infrastructure with Rich console output and file rotation.

    Extracted from NANA orchestrator, now available for all orchestrators.
    """
    # Implementation from lines 1812-1912 of nana_orchestrator.py
    pass
```

**Files to Update**:
- `main.py`: Change `setup_logging_nana()` to `setup_saga_logging()`
- `orchestration/langgraph_orchestrator.py`: Use new logging setup

**Testing**:
- Verify logs still appear in console with Rich formatting
- Verify chapter logs still written to `output/chapter_logs/`
- Check log rotation works correctly

---

### 1.2 Extract Ingestion Pipeline (Optional)

**File to Create**: `ingestion/ingest_workflow.py`

**What to Extract**: Ingestion logic from `nana_orchestrator.py` (lines 1667-1738, ~80 LOC)

**Features to Preserve**:
- Text file chunking
- Per-chunk summarization
- Per-chunk embedding generation
- Per-chunk KG extraction
- Periodic KG healing
- Plot outline generation from ingestion

**Why**: Ingestion is a valuable feature, separate from generation pipeline

**Implementation Options**:

**Option A: Extract as-is** (Quick, 30 min)
```python
# ingestion/ingest_workflow.py

async def run_text_ingestion(
    file_path: str,
    chunk_size: int = 3000
) -> None:
    """
    Ingest existing text file into SAGA knowledge graph.

    Extracted from NANA orchestrator, runs independently.
    """
    # Direct port of lines 1667-1738
    pass
```

**Option B: LangGraph workflow** (Better, 2-3 hours)
```python
# ingestion/ingest_langgraph_workflow.py

# Create LangGraph workflow with nodes:
# - chunk_text_node
# - summarize_chunk_node
# - embed_chunk_node
# - extract_chunk_entities_node
# - heal_kg_node
# - generate_plot_outline_node
```

**Recommendation**: **Option A** for Phase 3 (quick extraction), Option B for future enhancement

**Files to Update**:
- `main.py`: Import from new location when `--ingest` used with `--nana`
- Consider adding `--ingest-langgraph` flag for future LangGraph ingestion

**Testing**:
- Test ingestion with sample text file
- Verify chunks created correctly
- Verify entities extracted and persisted
- Verify plot outline generated

**Decision Point**: User should decide:
- ❓ Extract ingestion now (adds ~1 hour to Phase 3)
- ❓ Leave ingestion in NANA for now, extract in Phase 4
- ❓ Skip ingestion entirely (remove feature)

---

### 1.3 Verify UI Components Already Extracted

**Status**: ✅ Already done

**Files to Check**:
- `ui/rich_display_manager.py` - Rich progress display ✅
- `ui/progress_panels.py` - Progress panel components ✅

**No Action Required**: UI components already properly separated

---

## Stage 2: Delete & Clean (1-2 hours)

### 2.1 Delete NANA Orchestrator

**Files to Delete**:

1. **`orchestration/nana_orchestrator.py`** (1,912 LOC)
   - Entire NANA orchestrator class
   - All private methods
   - Feature creep code
   - Manual cache management
   - Snapshot logic

2. **`orchestration/chapter_flow.py`** (92 LOC)
   - ✅ Confirmed: Only used by NANA orchestrator
   - ✅ Safe to delete

**Command**:
```bash
git rm orchestration/nana_orchestrator.py
git rm orchestration/chapter_flow.py
```

**LOC Removed**: 2,004 LOC (1,912 + 92)

---

### 2.2 Update main.py

**Current State** (Phase 2):
```python
# Choose orchestrator based on --nana flag (LangGraph is now default)
if args.nana:
    logger.warning("Using DEPRECATED legacy NANA pipeline...")
    orchestrator = NANA_Orchestrator()
else:
    logger.info("Using LangGraph-based workflow (default)")
    orchestrator = LangGraphOrchestrator()
```

**Phase 3 Changes**:

**Option A: Remove --nana flag entirely** (Clean break)
```python
# main.py (simplified)

def main() -> None:
    setup_saga_logging()  # Changed from setup_logging_nana()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", default=None, help="Path to text file to ingest")

    # Remove --nana flag entirely
    # LangGraph is the only orchestrator

    # Bootstrap flags remain...
    args = parser.parse_args()

    # Always use LangGraph
    orchestrator = LangGraphOrchestrator()

    # Rest of logic...
```

**Option B: Keep --nana flag with error message** (Graceful migration)
```python
# main.py (graceful deprecation)

if args.nana:
    logger.error(
        "NANA pipeline has been removed in SAGA v3.0. "
        "Please use the default LangGraph workflow. "
        "See migration guide: docs/phase3_nana_removal_plan.md"
    )
    sys.exit(1)
else:
    orchestrator = LangGraphOrchestrator()
```

**Recommendation**: **Option B** (helps users who haven't migrated)

**Files to Update**:
```python
# main.py
- from orchestration.nana_orchestrator import NANA_Orchestrator, setup_logging_nana
+ from core.logging_config import setup_saga_logging

- setup_logging_nana()
+ setup_saga_logging()
```

---

### 2.3 Handle Ingestion

**If Ingestion Extracted** (Stage 1.2 complete):
```python
# main.py

elif args.ingest:
    # Use extracted ingestion workflow
    from ingestion.ingest_workflow import run_text_ingestion
    asyncio.run(run_text_ingestion(args.ingest))
```

**If Ingestion NOT Extracted**:
```python
# main.py

elif args.ingest:
    logger.error(
        "Ingestion feature has been removed with NANA deprecation. "
        "Please use external text processing or see docs/ingestion_alternatives.md"
    )
    sys.exit(1)
```

**Decision Point**: User should decide ingestion strategy before deletion

---

### 2.4 Clean Up Imports

**Files to Check for Unused Imports**:

1. **`main.py`**
   - Remove `NANA_Orchestrator` import
   - Remove `setup_logging_nana` import
   - Add `setup_saga_logging` import

2. **Test Files**
   - `tests/test_orchestrator_*.py` - May reference NANA
   - `tests/test_nana_*.py` - Delete if exists

3. **Documentation**
   - Search for `NANA` references in all `.md` files
   - Update or remove as appropriate

**Commands**:
```bash
# Find all NANA references
grep -r "NANA" --include="*.py" --exclude-dir=.git .
grep -r "nana_orchestrator" --include="*.py" --exclude-dir=.git .

# Find all documentation references
grep -r "NANA" --include="*.md" --exclude-dir=.git .
```

---

### 2.5 Delete Feature Creep Tests

**Test Files Identified** (✅ pre-analyzed):

**Delete Entirely**:
1. `tests/test_chapter_flow.py` - Tests chapter_flow.py (being deleted)
2. `tests/test_orchestrator_private_methods.py` - Tests NANA private methods

**Review & Update** (may have NANA references):
3. `tests/test_ingestion_healing.py` - Update if has NANA imports
4. `tests/test_novel_generation_dynamic.py` - Update if has NANA imports
5. `tests/test_orchestrator_refresh.py` - Update if has NANA imports

**Actions**:
```bash
# Delete NANA-only tests
git rm tests/test_chapter_flow.py
git rm tests/test_orchestrator_private_methods.py

# Check remaining files for NANA references
grep -n "nana_orchestrator\|NANA_Orchestrator" tests/test_ingestion_healing.py
grep -n "nana_orchestrator\|NANA_Orchestrator" tests/test_novel_generation_dynamic.py
grep -n "nana_orchestrator\|NANA_Orchestrator" tests/test_orchestrator_refresh.py

# Update or delete based on findings
```

**Estimated**: 2 files to delete, 3 files to review/update

---

## Stage 3: Verify & Document (1 hour)

### 3.1 Run Full Test Suite

**Command**:
```bash
pytest --tb=short -v
```

**Expected Results**:
- All LangGraph tests pass
- NANA-specific tests removed
- No import errors
- No missing module errors

**If Failures**:
- Check for missed NANA imports
- Verify logging setup works
- Check for hardcoded paths to NANA files

---

### 3.2 Manual Verification

**Tests to Run Manually**:

1. **Basic Generation**:
   ```bash
   python main.py
   # Should start LangGraph workflow without errors
   ```

2. **Bootstrap**:
   ```bash
   python main.py --bootstrap
   # Should use LangGraph initialization workflow
   ```

3. **Logging Output**:
   - Verify Rich console output appears
   - Check `output/chapter_logs/` for log files
   - Verify log rotation works

4. **Ingestion** (if preserved):
   ```bash
   python main.py --ingest path/to/test.txt
   # Should run ingestion workflow
   ```

5. **NANA Flag** (if kept with error):
   ```bash
   python main.py --nana
   # Should show error message and exit
   ```

---

### 3.3 Update Documentation

**Files to Update**:

1. **`README.md`**
   - Remove NANA deprecation notice (no longer deprecated, it's removed)
   - Remove `--nana` flag from CLI overview
   - Update "Run with legacy NANA" section (remove or note it's removed)

2. **`CLAUDE.md`**
   - Update Essential Commands section
   - Remove NANA references
   - Note that LangGraph is the only pipeline

3. **`docs/migrate.md`**
   - Mark Phase 3 as complete
   - Update status of all migration tasks

4. **`docs/langgraph-architecture.md`**
   - Note NANA has been removed
   - Update any comparisons

5. **Create `docs/CHANGELOG.md`** (if not exists)
   - Document NANA removal in v3.0
   - List breaking changes
   - Provide migration guidance

**Example Changelog Entry**:
```markdown
## v3.0.0 - NANA Removal (2025-11-14)

### Breaking Changes

- **REMOVED**: NANA orchestrator (`orchestration/nana_orchestrator.py`)
- **REMOVED**: `--nana` command-line flag
- **REMOVED**: Feature creep code (~445 LOC)
- **REMOVED**: Manual cache management and snapshot logic

### Migration Guide

- All users must now use LangGraph workflow (default behavior)
- Remove `--nana` flag from any scripts or automation
- Ingestion feature extracted to `ingestion/ingest_workflow.py`

### Improvements

- Cleaner codebase: ~2,112 LOC removed
- Single pipeline: LangGraph only
- Better logging: Extracted to `core/logging_config.py`
```

---

### 3.4 Update Help Text

**File**: `main.py`

Remove NANA deprecation message from help text:
```python
# OLD (Phase 2):
parser.add_argument(
    "--nana",
    action="store_true",
    help="[DEPRECATED] Use legacy NANA pipeline instead of LangGraph workflow. "
    "NANA pipeline will be removed in SAGA v3.0. Please migrate to LangGraph.",
)

# NEW (Phase 3 - Option B: Keep with error):
parser.add_argument(
    "--nana",
    action="store_true",
    help="[REMOVED in v3.0] NANA pipeline has been removed. Use LangGraph (default).",
)
```

---

## Phase 3 Execution Checklist

### Stage 1: Extract & Preserve

- [ ] **1.1 Extract Logging Configuration**
  - [ ] Create `core/logging_config.py`
  - [ ] Move logging setup from NANA (lines 1812-1912)
  - [ ] Update `main.py` to use `setup_saga_logging()`
  - [ ] Test logging still works (console + file)

- [ ] **1.2 Extract Ingestion Pipeline** (DECISION REQUIRED)
  - [ ] User decides: Extract now | Leave in NANA | Remove feature
  - [ ] If extracting: Create `ingestion/ingest_workflow.py`
  - [ ] If extracting: Move ingestion logic (lines 1667-1738)
  - [ ] If extracting: Test ingestion works
  - [ ] If removing: Document in changelog

- [ ] **1.3 Verify UI Already Extracted**
  - [ ] Confirm `ui/rich_display_manager.py` exists
  - [ ] Confirm `ui/progress_panels.py` exists
  - [ ] No action needed ✅

### Stage 2: Delete & Clean

- [ ] **2.1 Delete NANA Orchestrator**
  - [ ] `git rm orchestration/nana_orchestrator.py` (1,912 LOC)
  - [ ] `git rm orchestration/chapter_flow.py` (92 LOC - ✅ confirmed safe)
  - [ ] Verify files deleted

- [ ] **2.2 Update main.py**
  - [ ] Choose: Remove --nana entirely OR keep with error (recommend: keep with error)
  - [ ] Update orchestrator selection logic
  - [ ] Update imports (remove NANA, add logging_config)
  - [ ] Update logging setup call

- [ ] **2.3 Handle Ingestion**
  - [ ] Update `--ingest` flag handling based on Stage 1.2 decision
  - [ ] Test ingestion works (if preserved)
  - [ ] Document removal (if removed)

- [ ] **2.4 Clean Up Imports**
  - [ ] Find all NANA references: `grep -r "NANA" --include="*.py" .`
  - [ ] Remove unused imports from Python files
  - [ ] Remove NANA test imports

- [ ] **2.5 Delete Feature Creep Tests**
  - [ ] Delete: `git rm tests/test_chapter_flow.py`
  - [ ] Delete: `git rm tests/test_orchestrator_private_methods.py`
  - [ ] Review/update: `tests/test_ingestion_healing.py`
  - [ ] Review/update: `tests/test_novel_generation_dynamic.py`
  - [ ] Review/update: `tests/test_orchestrator_refresh.py`
  - [ ] Verify no broken imports

### Stage 3: Verify & Document

- [ ] **3.1 Run Full Test Suite**
  - [ ] `pytest --tb=short -v`
  - [ ] Fix any import errors
  - [ ] Fix any missing module errors
  - [ ] Achieve similar pass rate as Phase 2 (444/465 or better)

- [ ] **3.2 Manual Verification**
  - [ ] Test basic generation: `python main.py`
  - [ ] Test bootstrap: `python main.py --bootstrap`
  - [ ] Test logging output (console + files)
  - [ ] Test ingestion (if preserved)
  - [ ] Test --nana flag shows error (if kept)

- [ ] **3.3 Update Documentation**
  - [ ] Update `README.md` (remove deprecation, update CLI)
  - [ ] Update `CLAUDE.md` (remove NANA references)
  - [ ] Update `docs/migrate.md` (mark Phase 3 complete)
  - [ ] Update `docs/langgraph-architecture.md`
  - [ ] Create `docs/CHANGELOG.md` entry for v3.0

- [ ] **3.4 Update Help Text**
  - [ ] Update `--nana` help text (or remove flag)
  - [ ] Verify `python main.py --help` looks correct
  - [ ] Test help text displays properly

### Final Commit

- [ ] **Commit Phase 3 Changes**
  - [ ] Review all changes: `git status`
  - [ ] Stage changes: `git add .`
  - [ ] Commit with detailed message
  - [ ] Push to branch

---

## Commit Message Template

```
feat(phase3): remove NANA orchestrator, complete migration to LangGraph

BREAKING CHANGES:
- Removed NANA orchestrator (orchestration/nana_orchestrator.py, 1,912 LOC)
- Removed chapter_flow.py (orchestration/chapter_flow.py, 92 LOC)
- Removed --nana flag [OR: --nana flag now returns error]
- Total LOC removed: 2,004

Extracted Features:
- Logging setup → core/logging_config.py (setup_saga_logging)
- [Ingestion → ingestion/ingest_workflow.py] [IF EXTRACTED]
- UI components already in ui/ (no changes needed)

Changes:
- main.py: Use LangGraphOrchestrator exclusively
- main.py: Update logging to use setup_saga_logging()
- [main.py: Ingestion now uses extracted workflow] [IF EXTRACTED]
- Removed all NANA-specific imports
- Deleted NANA-only tests
- Updated documentation (README, CLAUDE.md, migrate.md)

Migration Guide:
- All users now use LangGraph workflow (default)
- No --nana flag needed (or shows error if used)
- [Ingestion available via ingestion module] [IF EXTRACTED]
- [Ingestion feature removed] [IF NOT EXTRACTED]

Test Results: [INSERT RESULTS]
- Full test suite: X/465 passing
- Manual verification: ✅ Generation works
- Manual verification: ✅ Bootstrap works
- Manual verification: ✅ Logging works

Closes Phase 3 of LangGraph migration.
Codebase now 100% LangGraph, 0% NANA.
```

---

## Risk Assessment

### Low Risk ✅

- **Deleting NANA orchestrator**: Full parity achieved, LangGraph tested
- **Removing feature creep**: Features were optional/unused
- **Extracting logging**: Straightforward refactor

### Medium Risk ⚠️

- **Ingestion extraction**: Complex feature, needs careful testing
- **Test suite updates**: May break some tests temporarily

### High Risk ❌

- **None identified**: All prerequisites met, Phase 2 transition successful

### Mitigation Strategies

1. **Create backup branch** before deletion
   ```bash
   git checkout -b backup-nana-phase3
   git checkout claude/langgraph-deduplication-logic-01D5Vyn65CYi8R1rV4usjQ7s
   ```

2. **Incremental commits**: Commit after each stage
   - Stage 1 commit: "Extract logging and ingestion"
   - Stage 2 commit: "Delete NANA orchestrator"
   - Stage 3 commit: "Update documentation and tests"

3. **Test between stages**: Run test suite after each stage

4. **Keep --nana flag with error** (recommended): Helps users who haven't migrated

---

## Success Criteria

### Must Have ✅

- [ ] NANA orchestrator deleted (1,912+ LOC removed)
- [ ] Test suite passing (similar rate to Phase 2)
- [ ] LangGraph workflow runs end-to-end
- [ ] Logging works correctly (console + files)
- [ ] Documentation updated

### Should Have 🎯

- [ ] Ingestion extracted and working (or removal documented)
- [ ] All NANA references removed from codebase
- [ ] --nana flag shows helpful error message
- [ ] Changelog created for v3.0

### Nice to Have 🌟

- [ ] Code quality improved (linting passes)
- [ ] Test coverage maintained or improved
- [ ] Performance benchmarks documented

---

## Post-Phase 3 Opportunities

Once NANA is removed, consider:

1. **Rename LangGraph orchestrator** → `Orchestrator` (it's the only one)
2. **Simplify initialization** - Remove NANA compatibility layers
3. **Remove dead code** - Find unused utilities that only NANA used
4. **Optimize imports** - Clean up circular dependencies
5. **Refactor tests** - Remove NANA comparison tests

---

## Timeline Estimate

| Stage | Task | Time Estimate |
|-------|------|---------------|
| **1.1** | Extract logging | 30-45 min |
| **1.2** | Extract ingestion (optional) | 0-2 hours |
| **1.3** | Verify UI extracted | 5 min |
| **2.1** | Delete NANA files | 10 min |
| **2.2** | Update main.py | 15 min |
| **2.3** | Handle ingestion | 10-30 min |
| **2.4** | Clean up imports | 20 min |
| **2.5** | Delete tests | 15 min |
| **3.1** | Run test suite | 10 min |
| **3.2** | Manual verification | 20 min |
| **3.3** | Update docs | 30-45 min |
| **3.4** | Update help text | 10 min |
| **-** | **Total** | **3-6 hours** |

**Fastest Path** (skip ingestion): ~3 hours
**Complete Path** (extract ingestion): ~6 hours

---

## Decision Points for User

Before executing Phase 3, user must decide:

### 1. Ingestion Strategy (Stage 1.2)

- **Option A**: Extract to `ingestion/ingest_workflow.py` (adds 1-2 hours)
- **Option B**: Leave in NANA for now, extract later (skip for Phase 3)
- **Option C**: Remove ingestion feature entirely (document removal)

**Recommendation**: **Option B** (defer ingestion to Phase 4)

### 2. --nana Flag Handling (Stage 2.2)

- **Option A**: Remove --nana flag entirely (clean break)
- **Option B**: Keep --nana flag, show error message (graceful)

**Recommendation**: **Option B** (helps users migrate)

### 3. chapter_flow.py Deletion (Stage 2.1)

- ✅ **Confirmed**: LangGraph does NOT use `chapter_flow.py`
- ✅ **Action**: Delete it (92 LOC)
- Only imported by `nana_orchestrator.py` and `tests/test_chapter_flow.py`

---

## Questions for User

Before starting Phase 3:

1. ❓ **Ingestion**: Extract now, defer, or remove?
   - Option A: Extract to `ingestion/ingest_workflow.py` (adds 1-2 hours)
   - Option B: Defer to Phase 4 (recommended)
   - Option C: Remove feature entirely

2. ❓ **--nana flag**: Remove or keep with error?
   - Option A: Remove entirely (clean break)
   - Option B: Keep with error message (recommended - helps users)

3. ❓ **Timeline**: Execute now or in future session?

4. ❓ **Testing**: Run full end-to-end test before deletion?

**Already Answered** ✅:
- chapter_flow.py: Confirmed safe to delete (only used by NANA)

---

**Document Status**: Ready for execution
**Prerequisites**: All met ✅
**Blockers**: None
**Ready to Start**: ✅ Awaiting user decisions on ingestion and --nana flag
