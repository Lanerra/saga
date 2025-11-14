# SAGA LangGraph Migration - Phase 2 Plan

## Executive Summary

Phase 2 completes the MVP LangGraph workflow by adding generation and revision capabilities to the Phase 1 foundation (extraction, commit, validation). This enables end-to-end single-chapter and multi-chapter novel generation.

**Timeline**: 2-3 weeks
**Prerequisites**: Phase 1 complete (✅ Done)
**Deliverables**: Full generation → extraction → commit → validation → revision workflow

---

## Phase 1 Recap (Completed ✅)

| Component | Status | Lines | Tests |
|-----------|--------|-------|-------|
| State Schema | ✅ Complete | 335 | 21 |
| Extraction Node | ✅ Complete | 527 | 21 |
| Commit/Deduplication Node | ✅ Complete | 524 | 12 |
| Query Wrapper (Context) | ✅ Complete | 439 | 16 |
| Validation Node | ✅ Complete | 385 | 17 |
| Basic Workflow Graph | ✅ Complete | 245 | 11 |
| **Total** | **✅** | **2,455** | **98** |

---

## Phase 2 Goals

### Primary Objectives:
1. ✅ Complete MVP workflow (generate → extract → commit → validate → revise loop)
2. ✅ Enable single-chapter generation with full quality control
3. ✅ Enable multi-chapter generation with narrative continuity
4. ✅ Maintain all Phase 1 functionality without regressions

### Success Criteria:
- [ ] Generate single chapter end-to-end with real LLM
- [ ] Revision loop works with contradiction detection
- [ ] Multi-chapter generation with context from previous chapters
- [ ] Chapter summaries persist to Neo4j and guide future generation
- [ ] Full test coverage (>90%) for all new nodes
- [ ] Performance: <5 minutes per chapter (4000 words, Q4 model)
- [ ] Memory: <4GB RAM during generation

---

## Phase 2 Components

### 2.1 Generation Node (Week 1, Days 1-3)

**Source Location:**
- `agents/generation_agent.py` (lines 50-650)
- `agents/generation_agent.py:generate_chapter()` - Main orchestrator
- `core/context_construction/` - Context building helpers

**Target:** `core/langgraph/nodes/generation_node.py` (~600 lines)

**Key Functionality:**
1. **Context Construction** (Port from existing):
   - Query Neo4j for active characters
   - Get previous chapter summaries (last 5 chapters)
   - Retrieve key events from last 10 chapters
   - Get character relationships for current scene
   - Get location details if specified

2. **Prompt Construction**:
   - Genre-specific instructions
   - Character context (personalities, traits, motivations)
   - Relationship context (who knows/likes/opposes whom)
   - Recent events (for continuity)
   - Outline entry (scene goal, key beats)
   - Style guidelines

3. **LLM Generation**:
   - Reuse `core/llm_interface_refactored.py:llm_service`
   - Model selection from state (`generation_model`)
   - Temperature, max_tokens configuration
   - Stream handling (optional, for UI progress)

4. **Post-Processing**:
   - Word count calculation
   - Basic cleanup (trailing whitespace, markdown formatting)
   - State update

**Implementation Steps:**

#### Step 2.1.1: Create Context Builder (8 hours)
```python
# core/langgraph/nodes/generation_node.py

async def generate_chapter(state: NarrativeState) -> NarrativeState:
    """
    Generate chapter prose from outline and Neo4j context.

    PORTED FROM: GenerationAgent.generate_chapter()
    USES: build_context_from_graph() (Phase 1 - already implemented)
    """

    # Step 1: Build context from knowledge graph
    context = await build_context_from_graph(
        current_chapter=state["current_chapter"],
        active_character_names=state.get("active_character_names", []),
        location_id=state.get("current_location"),
        lookback_chapters=5,
        max_characters=10,
        max_world_items=10
    )

    # Step 2: Construct generation prompt
    prompt = _construct_generation_prompt(
        chapter_num=state["current_chapter"],
        outline_entry=state["outline"][state["current_chapter"]],
        context=context,
        genre=state["genre"],
        theme=state["theme"],
        protagonist=state.get("protagonist_name", "the protagonist")
    )

    # Step 3: Generate via LLM
    draft_text, usage = await llm_service.async_call_llm(
        model_name=state["generation_model"],
        prompt=prompt,
        temperature=config.Temperatures.CHAPTER_GENERATION,
        max_tokens=config.MAX_CHAPTER_TOKENS,
        allow_fallback=True,
        stream_to_disk=False
    )

    # Step 4: Update state
    return {
        **state,
        "draft_text": draft_text,
        "draft_word_count": len(draft_text.split()),
        "current_node": "generate",
        "last_error": None
    }
```

**Deliverable:**
- generation_node.py with full context integration
- Reuses existing prompt templates from `prompts/generation_agent/`
- Test with mock LLM (return canned chapter text)

---

### 2.2 Revision Node (Week 1, Days 4-5)

**Source Location:**
- `agents/revision_agent.py` (all 800+ lines)
- `agents/revision_agent.py:revise_chapter()` - Main revision logic
- `core/contradiction_finder.py` - Contradiction analysis

**Target:** `core/langgraph/nodes/revision_node.py` (~400 lines)

**Key Functionality:**
1. **Revision Prompt Construction**:
   - Original chapter text
   - List of contradictions with severity
   - Suggested fixes from validation
   - Outline constraints (maintain scene goals)

2. **LLM Revision**:
   - Use `revision_model` (can be different/better than generation model)
   - Lower temperature (0.5 vs 0.7) for consistency
   - Track revision history

3. **Iteration Control**:
   - Max attempts (default: 3)
   - Escalation if max reached (user intervention)
   - State updates for re-validation

**Implementation Steps:**

#### Step 2.2.1: Create Revision Node (6 hours)
```python
# core/langgraph/nodes/revision_node.py

async def revise_chapter(state: NarrativeState) -> NarrativeState:
    """
    Revise chapter based on validation feedback.

    PORTED FROM: RevisionAgent.revise_chapter()

    Strategy:
    1. Build revision prompt with contradictions
    2. Use revision model (potentially better/different)
    3. Track iteration count
    4. Update state for re-validation
    """

    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        logger.warning(
            "revise_chapter: max iterations reached",
            iteration_count=state["iteration_count"],
            max_iterations=state["max_iterations"]
        )
        return {
            **state,
            "needs_revision": False,  # Stop loop
            "last_error": f"Max revisions ({state['max_iterations']}) reached. Manual review needed.",
            "current_node": "revise_failed"
        }

    # Build revision prompt
    prompt = _construct_revision_prompt(
        original_text=state["draft_text"],
        contradictions=state["contradictions"],
        chapter_number=state["current_chapter"],
        outline_entry=state["outline"][state["current_chapter"]]
    )

    # Call revision model
    revised_text, usage = await llm_service.async_call_llm(
        model_name=state["revision_model"],
        prompt=prompt,
        temperature=config.Temperatures.REVISION,  # Lower temp
        max_tokens=config.MAX_CHAPTER_TOKENS,
        allow_fallback=True
    )

    # Update state
    return {
        **state,
        "draft_text": revised_text,
        "draft_word_count": len(revised_text.split()),
        "iteration_count": state["iteration_count"] + 1,
        "contradictions": [],  # Will be re-validated
        "current_node": "revise"
    }
```

**Deliverable:**
- revision_node.py with iteration control
- Reuses existing revision prompts
- Test with mock contradictions

---

### 2.3 Summarization Node (Week 2, Days 1-2)

**Source Location:**
- New functionality (not in SAGA 1.0, but needed for long context)
- Logic similar to `agents/generation_agent.py:_summarize_chapter()`

**Target:** `core/langgraph/nodes/summary_node.py` (~200 lines)

**Key Functionality:**
1. **Summary Generation**:
   - Extract 2-3 sentence summary of chapter
   - Focus on: plot events, character decisions, key revelations
   - Use fast extraction model

2. **Neo4j Persistence**:
   - Update Chapter node with summary
   - Make available for future context queries

3. **State Management**:
   - Add summary to `previous_chapter_summaries` list
   - Keep last 5 summaries in state (rolling window)

**Implementation Steps:**

#### Step 2.3.1: Create Summary Node (4 hours)
```python
# core/langgraph/nodes/summary_node.py

async def summarize_chapter(state: NarrativeState) -> NarrativeState:
    """
    Generate chapter summary for context in future chapters.

    NEW FUNCTIONALITY (not in SAGA 1.0)
    """

    # Generate summary using fast model
    summary_prompt = f"""Summarize the following chapter in 2-3 sentences.

Focus on:
- Key plot events
- Character actions and decisions
- Important revelations or conflicts

Chapter {state['current_chapter']}:
{state['draft_text']}

Summary (2-3 sentences):"""

    summary, usage = await llm_service.async_call_llm(
        model_name=state["extraction_model"],  # Use fast model
        prompt=summary_prompt,
        temperature=0.3,
        max_tokens=200
    )

    # Store in Neo4j
    await chapter_queries.save_chapter_summary(
        chapter_number=state["current_chapter"],
        summary=summary.strip()
    )

    # Update state
    previous_summaries = state.get("previous_chapter_summaries", [])[-4:]
    previous_summaries.append(summary.strip())

    return {
        **state,
        "previous_chapter_summaries": previous_summaries,
        "current_node": "summarize"
    }
```

**Deliverable:**
- summary_node.py with Neo4j integration
- Test with mock chapter text

---

### 2.4 Finalization Node (Week 2, Day 3)

**Source Location:**
- File I/O from `core/file_system_manager.py`
- Chapter persistence logic

**Target:** `core/langgraph/nodes/finalize_node.py` (~150 lines)

**Key Functionality:**
1. **File Persistence**:
   - Write chapter to Markdown file
   - Include frontmatter (chapter number, word count, model, timestamp)
   - Create chapter directory if needed

2. **State Cleanup**:
   - Clear draft_text (no longer needed in state)
   - Reset iteration_count for next chapter
   - Clear contradictions list
   - Advance current_chapter counter

3. **Progress Tracking**:
   - Log completion
   - Update project metadata

**Implementation Steps:**

#### Step 2.4.1: Create Finalize Node (3 hours)
```python
# core/langgraph/nodes/finalize_node.py

from pathlib import Path
from datetime import datetime

async def finalize_chapter(state: NarrativeState) -> NarrativeState:
    """
    Persist chapter to disk and prepare for next chapter.

    PORTED FROM: file_system_manager.save_chapter()
    """

    # Write chapter file
    chapter_path = Path(state["chapters_dir"]) / f"chapter_{state['current_chapter']:03d}.md"
    chapter_path.parent.mkdir(parents=True, exist_ok=True)

    frontmatter = f"""---
chapter: {state['current_chapter']}
word_count: {state['draft_word_count']}
generated_at: {datetime.now().isoformat()}
model: {state['generation_model']}
iterations: {state['iteration_count']}
---

"""

    chapter_path.write_text(frontmatter + state["draft_text"])

    logger.info(
        "finalize_chapter: chapter saved",
        chapter=state["current_chapter"],
        path=str(chapter_path),
        word_count=state["draft_word_count"]
    )

    # Prepare for next chapter
    return {
        **state,
        "current_chapter": state["current_chapter"] + 1,
        "iteration_count": 0,
        "contradictions": [],
        "draft_text": None,
        "draft_word_count": 0,
        "extracted_entities": {},
        "extracted_relationships": [],
        "needs_revision": False,
        "current_node": "finalize"
    }
```

**Deliverable:**
- finalize_node.py with file I/O
- Test chapter file creation

---

### 2.5 Full Workflow Integration (Week 2, Days 4-5)

**Target:** `core/langgraph/workflow.py` (update existing)

**Updates Needed:**
1. Add new nodes to graph
2. Wire complete flow: generate → extract → commit → validate → revise/summarize → finalize
3. Add multi-chapter loop conditional edge
4. Update exports in __init__.py

**Implementation Steps:**

#### Step 2.5.1: Update Workflow Graph (4 hours)
```python
# core/langgraph/workflow.py (UPDATE)

def create_phase2_graph(checkpointer=None) -> StateGraph:
    """
    Create complete Phase 2 LangGraph workflow.

    Workflow:
    START → generate → extract → commit → validate → {revise OR summarize}
                                                            ↓          ↓
                                                          extract   finalize
                                                            ↓          ↓
                                                          (loop)    {next OR end}
    """
    workflow = StateGraph(NarrativeState)

    # Add all nodes
    workflow.add_node("generate", generate_chapter)
    workflow.add_node("extract", extract_entities)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", validate_consistency)
    workflow.add_node("revise", revise_chapter)
    workflow.add_node("summarize", summarize_chapter)
    workflow.add_node("finalize", finalize_chapter)

    # Linear flow through main pipeline
    workflow.add_edge("generate", "extract")
    workflow.add_edge("extract", "commit")
    workflow.add_edge("commit", "validate")

    # Conditional: revise or proceed to summary
    workflow.add_conditional_edges(
        "validate",
        should_revise,
        {
            "revise": "revise",
            "proceed": "summarize"
        }
    )

    # Revision loop: revise → extract → commit → validate
    workflow.add_edge("revise", "extract")

    # After summary, finalize chapter
    workflow.add_edge("summarize", "finalize")

    # Multi-chapter: next chapter or end
    workflow.add_conditional_edges(
        "finalize",
        should_continue_to_next_chapter,
        {
            "next": "generate",
            "end": END
        }
    )

    # Set entry point
    workflow.set_entry_point("generate")

    # Compile
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()

def should_continue_to_next_chapter(state: NarrativeState) -> Literal["next", "end"]:
    """Determine if we should generate next chapter or finish."""
    if state["current_chapter"] <= state["total_chapters"]:
        return "next"
    return "end"
```

**Deliverable:**
- Updated workflow.py with 7 nodes
- Multi-chapter support
- Backward compatible with Phase 1 tests

---

### 2.6 Testing Suite (Week 3)

**Test Coverage:**

#### Unit Tests (Days 1-2):
- `test_generation_node.py` (~250 lines, 15 tests)
  - Context construction
  - Prompt building
  - LLM integration (mocked)
  - Error handling

- `test_revision_node.py` (~200 lines, 12 tests)
  - Revision prompt construction
  - Iteration limit enforcement
  - Model escalation

- `test_summary_node.py` (~150 lines, 10 tests)
  - Summary generation
  - Neo4j persistence
  - State updates

- `test_finalize_node.py` (~150 lines, 10 tests)
  - File I/O
  - Frontmatter formatting
  - State cleanup

#### Integration Tests (Day 3):
- `test_full_workflow.py` (~300 lines, 8 tests)
  - Single chapter end-to-end
  - Revision loop with contradictions
  - Multi-chapter generation (3 chapters)
  - Checkpointing and recovery

#### End-to-End Tests (Day 4):
- Generate 5-chapter short story with real LLM
- Verify Neo4j population
- Verify chapter files
- Check for contradictions
- Measure performance

---

## Phase 2 Summary

### Components Added:
1. ✅ Generation Node (~600 lines)
2. ✅ Revision Node (~400 lines)
3. ✅ Summarization Node (~200 lines)
4. ✅ Finalization Node (~150 lines)
5. ✅ Full Workflow Graph (updated ~300 lines)
6. ✅ Test Suite (~1,050 lines, 55 tests)

### Total Phase 2 Additions:
- **Code**: ~2,700 lines (production)
- **Tests**: ~1,050 lines
- **Total Tests**: 98 (Phase 1) + 55 (Phase 2) = **153 tests**

### File Structure:
```
core/langgraph/
├── __init__.py (updated)
├── state.py (Phase 1)
├── workflow.py (updated for Phase 2)
├── graph_context.py (Phase 1)
└── nodes/
    ├── __init__.py (updated)
    ├── extraction_node.py (Phase 1)
    ├── commit_node.py (Phase 1)
    ├── validation_node.py (Phase 1)
    ├── generation_node.py (NEW - Phase 2)
    ├── revision_node.py (NEW - Phase 2)
    ├── summary_node.py (NEW - Phase 2)
    └── finalize_node.py (NEW - Phase 2)

tests/test_langgraph/
├── conftest.py (Phase 1)
├── test_state.py (Phase 1)
├── test_extraction_node.py (Phase 1)
├── test_commit_node.py (Phase 1)
├── test_graph_context.py (Phase 1)
├── test_validation_node.py (Phase 1)
├── test_workflow.py (Phase 1)
├── test_generation_node.py (NEW - Phase 2)
├── test_revision_node.py (NEW - Phase 2)
├── test_summary_node.py (NEW - Phase 2)
├── test_finalize_node.py (NEW - Phase 2)
└── test_full_workflow.py (NEW - Phase 2)
```

---

## Success Metrics

### Functional:
- [ ] Generate 1 chapter end-to-end (generate → extract → commit → validate → finalize)
- [ ] Revision loop works (validate finds contradiction → revise → re-validate)
- [ ] Generate 3 chapters with narrative continuity
- [ ] Summaries persist to Neo4j and appear in next chapter context
- [ ] Max iterations respected (stops after N attempts)
- [ ] Force continue bypasses validation

### Performance:
- [ ] Single chapter generation: <5 minutes (Q4 model, 4000 words)
- [ ] Memory usage: <4GB RAM during generation
- [ ] Neo4j queries: <100ms for context building
- [ ] Checkpoint save/load: <1 second

### Quality:
- [ ] Test coverage: >90% for all new nodes
- [ ] No regressions in Phase 1 tests (all 98 still pass)
- [ ] Code quality: Passes ruff linting
- [ ] Documentation: All nodes have docstrings

---

## Risk Mitigation

### Risk 1: LLM Integration Complexity
**Mitigation:**
- Reuse existing `llm_interface_refactored.py` (battle-tested)
- Mock LLM calls in tests
- Add retry logic for transient failures

### Risk 2: State Management Growth
**Mitigation:**
- LangGraph checkpointer handles persistence automatically
- Keep state schema lean (don't store full chapter text long-term)
- Test checkpoint recovery explicitly

### Risk 3: Revision Loop Convergence
**Mitigation:**
- Hard limit on max_iterations (default: 3)
- Track revision history to detect oscillation
- Provide user override (force_continue)

### Risk 4: Performance at Scale
**Mitigation:**
- Profile with 10-chapter generation
- Optimize Neo4j queries (indexes on chapter numbers)
- Consider async parallelism for extraction in Phase 3

---

## Next Steps After Phase 2

**Phase 3 (Optimization & Features):**
- Parallel execution for extraction subtasks
- Quality scoring models (prose, coherence, plot)
- Multi-model voting (generate 3 variants, pick best)
- Character arc tracking across chapters
- Real-time graph visualization

**Phase 4 (UI & Polish):**
- Web UI with HTMX
- Interactive outline editor
- Export to EPUB/MOBI
- Plugin system for custom validators

---

## Conclusion

Phase 2 transforms SAGA from a tested component library into a fully functional narrative generation system. By porting generation and revision logic into LangGraph nodes, we enable:

1. **End-to-end generation** with quality control
2. **Iterative refinement** through validation loops
3. **Multi-chapter continuity** via summaries and context
4. **Crash recovery** through checkpointing
5. **Extensibility** for Phase 3+ features

**Estimated Timeline:**
- Week 1: Generation + Revision nodes
- Week 2: Summary + Finalize + Workflow integration
- Week 3: Testing + Documentation + E2E validation

**Total Effort:** 2-3 weeks for complete Phase 2 MVP
