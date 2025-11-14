# 🔍 Text Deduplication: NANA vs LangGraph Implementation Comparison

**Analysis Date**: 2025-11-14
**Purpose**: Identify gaps and determine safe NANA code removal

---

## Executive Summary

### ✅ Successfully Ported
- ✅ Core deduplication logic (uses same `TextDeduplicator` utility)
- ✅ Post-generation deduplication
- ✅ Post-revision deduplication
- ✅ Logging of deduplication results

### ⚠️ Implementation Differences
| Aspect | NANA | LangGraph | Impact |
|--------|------|-----------|--------|
| **Segmentation Level** | `sentence` | `paragraph` | 🟡 MEDIUM - Different granularity |
| **Provisional Flagging** | Tracks via `is_from_flawed_source_for_kg` | Hardcoded `False` | 🔴 HIGH - Data quality tracking lost |
| **Debug Artifacts** | Saves deduplicated text to debug files | Not saved | 🟢 LOW - Optional debugging feature |
| **Config Explicit** | Passes all 3 config params | Uses defaults | 🟢 LOW - Config still works |

### 🔴 Critical Gap: Provisional Entity Tracking
NANA marks entities as `is_provisional=True` when deduplication removes text, signaling potentially flawed extraction. LangGraph currently hardcodes this to `False`, losing this data quality indicator.

---

## 1. Detailed Code Comparison

### 1.1 NANA Implementation

**Location**: `orchestration/nana_orchestrator.py:544-581`

```python
async def perform_deduplication(
    self, text_to_dedup: str, chapter_number: int
) -> tuple[str, int]:
    logger.info(f"SAGA: Performing de-duplication for Chapter {chapter_number}...")
    if not text_to_dedup or not text_to_dedup.strip():
        logger.info(
            f"De-duplication for Chapter {chapter_number}: Input text is empty. No action taken."
        )
        return text_to_dedup, 0
    try:
        deduper = TextDeduplicator(
            similarity_threshold=config.DEDUPLICATION_SEMANTIC_THRESHOLD,
            use_semantic_comparison=config.DEDUPLICATION_USE_SEMANTIC,
            min_segment_length_chars=config.DEDUPLICATION_MIN_SEGMENT_LENGTH,
        )
        deduplicated_text, chars_removed = await deduper.deduplicate(
            text_to_dedup, segment_level="sentence"  # ⚠️ SENTENCE level
        )
        if chars_removed > 0:
            method = (
                "semantic"
                if config.DEDUPLICATION_USE_SEMANTIC
                else "normalized string"
            )
            logger.info(
                f"De-duplication for Chapter {chapter_number} removed {chars_removed} text characters using {method} matching."
            )
        else:
            logger.info(
                f"De-duplication for Chapter {chapter_number}: No significant duplicates found."
            )
        return deduplicated_text, chars_removed
    except Exception as e:
        logger.error(
            f"Error during de-duplication for Chapter {chapter_number}: {e}",
            exc_info=True,
        )
        return text_to_dedup, 0  # Return original text on error
```

**Key Features:**
1. ✅ Explicitly passes all 3 config parameters
2. ✅ Uses `segment_level="sentence"` for finer granularity
3. ✅ Returns chars_removed for tracking
4. ✅ Exception handling returns original text
5. ✅ Logs deduplication method used

---

### 1.2 LangGraph Implementation

**Location**: `core/langgraph/nodes/generation_node.py:236-257`

```python
# Step 7: Deduplicate text to remove repetitive segments
deduplicator = TextDeduplicator()
deduplicated_text, removed_chars = await deduplicator.deduplicate(
    draft_text, segment_level="paragraph"  # ⚠️ PARAGRAPH level
)

if removed_chars > 0:
    final_word_count = len(deduplicated_text.split())
    logger.info(
        "generate_chapter: deduplication applied",
        chapter=chapter_number,
        chars_removed=removed_chars,
        original_words=word_count,
        final_words=final_word_count,
    )
else:
    deduplicated_text = draft_text
    final_word_count = word_count
    logger.info(
        "generate_chapter: no duplicates detected",
        chapter=chapter_number,
    )
```

**Location**: `core/langgraph/nodes/revision_node.py:192-215` (identical pattern)

**Key Features:**
1. ✅ Uses default `TextDeduplicator()` constructor (inherits config)
2. ⚠️ Uses `segment_level="paragraph"` (coarser than NANA)
3. ✅ Returns chars_removed for tracking
4. ❌ No exception handling (will propagate up)
5. ✅ Logs chars_removed and word count changes

---

## 2. NANA Deduplication Call Sites

### 2.1 Call Site #1: Post-Draft (When Revisions Disabled)
**Location**: `nana_orchestrator.py:990-1008`

```python
# When MAX_REVISION_CYCLES_PER_CHAPTER = 0
deduplicated_text, removed_char_count = await self.perform_deduplication(
    initial_draft_text, novel_chapter_number
)
is_from_flawed_source_for_kg = removed_char_count > 0  # ⚠️ FLAG SET
if is_from_flawed_source_for_kg:
    logger.info(
        f"SAGA: Ch {novel_chapter_number} - Text marked as flawed for KG due to de-duplication removing {removed_char_count} characters."
    )
    await self._save_debug_output(
        novel_chapter_number,
        "deduplicated_text_no_eval_path",
        deduplicated_text,
    )
```

**Purpose**: Deduplicate draft when skipping evaluation/revision entirely

---

### 2.2 Call Site #2: Post-Draft (Before Evaluation)
**Location**: `nana_orchestrator.py:1015-1042`

```python
# The de-duplication step is a single, definitive cleaning step after drafting and before evaluation.
self._update_rich_display(
    step=f"Ch {novel_chapter_number} - Post-Draft De-duplication"
)
logger.info(
    f"SAGA: Ch {novel_chapter_number} - Applying post-draft de-duplication."
)
(
    deduplicated_text,
    removed_char_count,
) = await self.perform_deduplication(
    current_text_to_process, novel_chapter_number
)
if removed_char_count > 0:
    is_from_flawed_source_for_kg = True  # ⚠️ FLAG SET
    logger.info(
        f"SAGA: Ch {novel_chapter_number} - De-duplication removed {removed_char_count} characters. Text marked as potentially flawed for KG."
    )
    current_text_to_process = deduplicated_text
    await self._save_debug_output(
        novel_chapter_number,
        "deduplicated_text_after_draft",
        current_text_to_process,
    )
else:
    logger.info(
        f"SAGA: Ch {novel_chapter_number} - Post-draft de-duplication found no significant changes."
    )
```

**Purpose**: Deduplicate before evaluation, flag if text modified

---

### 2.3 Call Site #3: Post-Revision
**Location**: `nana_orchestrator.py:1168-1181`

```python
dedup_text_after_rev, removed_after_rev = await self.perform_deduplication(
    current_text_to_process, novel_chapter_number
)
if removed_after_rev > 0:
    logger.info(
        f"SAGA: Ch {novel_chapter_number} - De-duplication after revisions removed {removed_after_rev} characters."
    )
    current_text_to_process = dedup_text_after_rev
    is_from_flawed_source_for_kg = True  # ⚠️ FLAG SET
    await self._save_debug_output(
        novel_chapter_number,
        "deduplicated_text_after_revision",
        current_text_to_process,
    )
```

**Purpose**: Final deduplication after revision loop completes

---

## 3. The `is_from_flawed_source_for_kg` Flag

### 3.1 How NANA Uses This Flag

**Purpose**: Track whether chapter text has issues that may impact entity extraction quality

**Conditions That Set Flag**:
1. ✅ Deduplication removed text (any call site)
2. ✅ Chapter length < `MIN_ACCEPTABLE_DRAFT_LENGTH` (line 1183-1187)
3. ✅ Revision LLM failed to improve evaluation score (line 1103)

**How Flag Is Used**:
```python
# In _finalize_and_save_chapter (line 1242)
await chapter_queries.save_chapter_to_db(
    chapter_number=novel_chapter_number,
    text=final_text_to_process,
    raw_llm_output=final_raw_llm_output or "",
    summary=result.get("summary"),
    embedding_array=result.get("embedding"),
    is_provisional=is_from_flawed_source_for_kg,  # ⚠️ STORED IN NEO4J
)
```

**Downstream Effects**:
- Neo4j Chapter node gets `is_provisional=true` property
- Entities extracted from provisional chapters are marked `is_provisional=true`
- Relationships from provisional chapters are marked `is_provisional=true`
- Prompts can filter out provisional entities for higher quality context
- Users can query for provisional data to manually review

**Database Schema**:
```cypher
CREATE INDEX entity_is_provisional_idx IF NOT EXISTS
  FOR (e:Entity) ON (e.is_provisional)

CREATE INDEX chapter_is_provisional IF NOT EXISTS
  FOR (c:`Chapter`) ON (c.is_provisional)
```

---

### 3.2 How LangGraph Handles This

**Current Implementation**:
```python
# In commit_node.py:135
is_from_flawed_draft=False,  # ⚠️ HARDCODED FALSE
```

**Impact**:
- 🔴 **All entities marked as high-quality, even when deduplication removed text**
- 🔴 **No way to identify potentially flawed extractions**
- 🔴 **Loss of data quality metadata**

---

## 4. Configuration Comparison

### 4.1 Deduplication Config Settings
**Location**: `config/settings.py:208-210`

```python
DEDUPLICATION_USE_SEMANTIC: bool = False
DEDUPLICATION_SEMANTIC_THRESHOLD: float = 0.45
DEDUPLICATION_MIN_SEGMENT_LENGTH: int = 150
```

### 4.2 How NANA Uses Config
✅ Explicitly passes all 3 parameters to `TextDeduplicator()`:
```python
deduper = TextDeduplicator(
    similarity_threshold=config.DEDUPLICATION_SEMANTIC_THRESHOLD,
    use_semantic_comparison=config.DEDUPLICATION_USE_SEMANTIC,
    min_segment_length_chars=config.DEDUPLICATION_MIN_SEGMENT_LENGTH,
)
```

### 4.3 How LangGraph Uses Config
⚠️ Uses defaults from `TextDeduplicator.__init__()`:
```python
deduplicator = TextDeduplicator()  # Uses defaults from config
```

**TextDeduplicator Constructor**:
```python
def __init__(
    self,
    similarity_threshold: float = config.DEDUPLICATION_SEMANTIC_THRESHOLD,
    use_semantic_comparison: bool = config.DEDUPLICATION_USE_SEMANTIC,
    min_segment_length_chars: int = config.DEDUPLICATION_MIN_SEGMENT_LENGTH,
    prefer_newer: bool = False,
) -> None:
```

**Result**: ✅ Both approaches use same config values (LangGraph via defaults)

---

## 5. Segmentation Level Difference

### 5.1 NANA: Sentence-Level
```python
await deduper.deduplicate(text_to_dedup, segment_level="sentence")
```

**Behavior**:
- Splits text at sentence boundaries (`.`, `!`, `?`)
- Finer granularity = more aggressive deduplication
- Can catch repeated sentences even in different paragraphs
- Example: "She walked away." repeated in different paragraphs → removed

### 5.2 LangGraph: Paragraph-Level
```python
await deduplicator.deduplicate(draft_text, segment_level="paragraph")
```

**Behavior**:
- Splits text at paragraph boundaries (`\n\n`)
- Coarser granularity = less aggressive deduplication
- Only catches repeated paragraphs
- Example: Same sentence in different paragraphs → NOT removed

### 5.3 Impact Assessment

| Aspect | Sentence-Level (NANA) | Paragraph-Level (LangGraph) | Verdict |
|--------|----------------------|---------------------------|---------|
| **Deduplication Aggressiveness** | More aggressive | Less aggressive | 🟡 Different quality trade-off |
| **False Positive Risk** | Higher (may remove intentional repetition) | Lower | 🟢 LangGraph safer |
| **Effectiveness on Bad Output** | Better at catching LLM loops | May miss sentence-level loops | 🟡 NANA better for pathological cases |
| **Prose Quality Preservation** | May harm intentional patterns | Preserves more structure | 🟢 LangGraph better for quality prose |

**Recommendation**:
- 🟡 **Test both approaches** - Task 1.2 quality comparison should evaluate this
- 🟢 **Paragraph-level is safer default** for quality prose
- 🟡 **Sentence-level may be needed** if LLMs produce sentence-level loops

---

## 6. Debug Artifact Saving

### 6.1 NANA Debug Saves
```python
await self._save_debug_output(
    novel_chapter_number,
    "deduplicated_text_after_draft",
    current_text_to_process,
)
```

**Saves To**: `output/debug_outputs/chapter_{N}_deduplicated_text_after_draft.txt`

**Artifacts Created**:
1. `deduplicated_text_no_eval_path` - When revisions disabled
2. `deduplicated_text_after_draft` - After initial draft dedup
3. `deduplicated_text_after_revision` - After revision dedup

### 6.2 LangGraph Debug Saves
❌ Not implemented

**Impact**: 🟢 LOW - Debugging convenience only, not critical

---

## 7. Exception Handling

### 7.1 NANA
```python
try:
    deduper = TextDeduplicator(...)
    deduplicated_text, chars_removed = await deduper.deduplicate(...)
    return deduplicated_text, chars_removed
except Exception as e:
    logger.error(f"Error during de-duplication: {e}", exc_info=True)
    return text_to_dedup, 0  # ✅ Returns original text on error
```

**Behavior**: Gracefully degrades to no-op on error

### 7.2 LangGraph
```python
deduplicator = TextDeduplicator()
deduplicated_text, removed_chars = await deduplicator.deduplicate(...)
# ❌ No try/catch - exception propagates up
```

**Behavior**: Exception propagates to node level, triggers error handler

**Impact**: 🟢 LOW - LangGraph error handler will catch at node level

---

## 8. Gap Analysis Summary

### 8.1 Critical Gaps (Must Fix)

| # | Gap | NANA Behavior | LangGraph Behavior | Priority | Effort |
|---|-----|---------------|-------------------|----------|--------|
| **1** | **Provisional flagging** | Tracks `is_from_flawed_source_for_kg`, stores in Neo4j | Hardcoded `False` | 🔴 HIGH | 1-2 hours |

---

### 8.2 Medium Gaps (Should Fix)

| # | Gap | NANA Behavior | LangGraph Behavior | Priority | Effort |
|---|-----|---------------|-------------------|----------|--------|
| **2** | **Segmentation level** | Uses `sentence` level | Uses `paragraph` level | 🟡 MEDIUM | 5 minutes (config) |

---

### 8.3 Low Gaps (Optional)

| # | Gap | NANA Behavior | LangGraph Behavior | Priority | Effort |
|---|-----|---------------|-------------------|----------|--------|
| **3** | **Debug artifacts** | Saves 3 debug files | Doesn't save | 🟢 LOW | 30 minutes |
| **4** | **Exception handling** | Try/catch returns original | Propagates to node handler | 🟢 LOW | 10 minutes |

---

## 9. Recommended Fixes for LangGraph

### 9.1 Critical Fix: Add Provisional Flagging

**File**: `core/langgraph/nodes/generation_node.py` & `revision_node.py`

**Change**:
```python
# Step 7: Deduplicate text to remove repetitive segments
deduplicator = TextDeduplicator()
deduplicated_text, removed_chars = await deduplicator.deduplicate(
    draft_text, segment_level="paragraph"
)

# NEW: Track if deduplication modified text
is_from_flawed_draft = removed_chars > 0

if removed_chars > 0:
    final_word_count = len(deduplicated_text.split())
    logger.info(
        "generate_chapter: deduplication applied",
        chapter=chapter_number,
        chars_removed=removed_chars,
        original_words=word_count,
        final_words=final_word_count,
    )
else:
    deduplicated_text = draft_text
    final_word_count = word_count
    logger.info(
        "generate_chapter: no duplicates detected",
        chapter=chapter_number,
    )

return {
    **state,
    "draft_text": deduplicated_text,
    "draft_word_count": final_word_count,
    "is_from_flawed_draft": is_from_flawed_draft,  # NEW: Add to state
    ...
}
```

**File**: `core/langgraph/nodes/commit_node.py`

**Change**:
```python
# OLD:
is_from_flawed_draft=False,

# NEW:
is_from_flawed_draft=state.get("is_from_flawed_draft", False),
```

**Effort**: 1-2 hours
**Files Modified**: 3 (generation_node.py, revision_node.py, commit_node.py)

---

### 9.2 Optional Fix: Configurable Segmentation Level

**File**: `config/settings.py`

**Add**:
```python
DEDUPLICATION_SEGMENT_LEVEL: str = "paragraph"  # or "sentence"
```

**File**: `core/langgraph/nodes/generation_node.py` & `revision_node.py`

**Change**:
```python
deduplicated_text, removed_chars = await deduplicator.deduplicate(
    draft_text, segment_level=config.DEDUPLICATION_SEGMENT_LEVEL
)
```

**Effort**: 5 minutes
**Files Modified**: 3

---

### 9.3 Optional Fix: Exception Handling

**File**: `core/langgraph/nodes/generation_node.py` & `revision_node.py`

**Change**:
```python
try:
    deduplicator = TextDeduplicator()
    deduplicated_text, removed_chars = await deduplicator.deduplicate(
        draft_text, segment_level="paragraph"
    )
except Exception as e:
    logger.error(
        "generate_chapter: deduplication failed, using original text",
        error=str(e),
        exc_info=True,
    )
    deduplicated_text = draft_text
    removed_chars = 0
```

**Effort**: 10 minutes
**Files Modified**: 2

---

## 10. What Can Be Safely Removed from NANA

### 10.1 After Gap Fixes Applied

Once the critical gap (#1 provisional flagging) is fixed in LangGraph:

#### Safe to Remove:
✅ `orchestration/nana_orchestrator.py:544-581` (perform_deduplication method)
- **Reason**: Functionality fully replicated in LangGraph nodes
- **Dependencies**:
  - 3 call sites within `nana_orchestrator.py` (lines 990, 1025, 1168)
  - All calls are within NANA orchestrator only
- **Safe**: YES (when NANA orchestrator is deleted)

#### Must Keep:
❌ `processing/text_deduplicator.py` (TextDeduplicator class)
- **Reason**: Shared utility used by both NANA and LangGraph
- **Safe to Remove**: NO (until NANA fully deprecated)

---

### 10.2 Deletion Strategy

**When LangGraph is default**:
1. ✅ Delete `orchestration/nana_orchestrator.py` entirely (includes perform_deduplication)
2. ❌ Keep `processing/text_deduplicator.py` (used by LangGraph)

**No partial extraction needed** - NANA's perform_deduplication is just a wrapper around TextDeduplicator

---

## 11. Testing Recommendations

### 11.1 Verify LangGraph Deduplication Works
```bash
# Run existing deduplication tests
pytest tests/test_langgraph/test_generation_node.py::TestGenerationDeduplication -v
```

### 11.2 Test Provisional Flagging (After Fix)
```python
# New test case needed
async def test_provisional_flag_set_when_deduplication_removes_text():
    """Test that is_from_flawed_draft flag is set when dedup removes chars."""
    duplicate_text = "Same paragraph.\n\n" * 10

    result = await generate_chapter(state_with_duplicate_text)

    assert result["is_from_flawed_draft"] is True
    assert result.get("chars_removed", 0) > 0
```

### 11.3 Quality Comparison (Task 1.2)
```bash
# Compare sentence vs paragraph segmentation
python scripts/compare_deduplication_levels.py
```

---

## 12. Conclusion

### 12.1 Feature Parity Status

| Feature | NANA | LangGraph | Status |
|---------|------|-----------|--------|
| Core deduplication | ✅ | ✅ | ✅ **COMPLETE** |
| Post-generation | ✅ | ✅ | ✅ **COMPLETE** |
| Post-revision | ✅ | ✅ | ✅ **COMPLETE** |
| Provisional flagging | ✅ | ❌ | 🔴 **CRITICAL GAP** |
| Configurable segmentation | ✅ | ⚠️ | 🟡 **IMPLICIT (via defaults)** |
| Debug artifacts | ✅ | ❌ | 🟢 **OPTIONAL** |
| Exception handling | ✅ | ⚠️ | 🟢 **HANDLED AT NODE LEVEL** |

### 12.2 Safe Removal Answer

**Question**: Can NANA's deduplication code be removed?

**Answer**:
- ✅ **YES**, after fixing Gap #1 (provisional flagging)
- ✅ **YES**, when NANA orchestrator is deleted entirely
- ❌ **NO**, cannot remove `TextDeduplicator` utility (shared code)
- ⚠️ **OPTIONAL**: Fix gaps #2-4 first for full parity

### 12.3 Action Items

1. 🔴 **HIGH PRIORITY**: Fix provisional flagging in LangGraph (1-2 hours)
2. 🟡 **MEDIUM PRIORITY**: Test sentence vs paragraph segmentation (Task 1.2)
3. 🟢 **LOW PRIORITY**: Add exception handling wrapper (10 minutes)
4. 🟢 **LOW PRIORITY**: Add debug artifact saving (30 minutes)

### 12.4 LOC Impact

**When NANA deleted**:
- `perform_deduplication` method: ~38 LOC
- 3 call sites: ~45 LOC
- **Total removable**: ~83 LOC (part of 1,912 LOC `nana_orchestrator.py`)

**Shared code kept**:
- `processing/text_deduplicator.py`: 155 LOC (kept for LangGraph)

---

**End of Analysis**
