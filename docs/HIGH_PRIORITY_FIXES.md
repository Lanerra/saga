# High-Priority Implementation Plan
## SAGA 2.0 LangGraph - Post-Critical Bug Fixes

**Document Version**: 1.0
**Date**: 2025-11-09
**Status**: Ready for Implementation
**Prerequisites**: Critical bugs #1 and #2 must be fixed first

---

## Overview

This document provides detailed implementation plans for the 4 high-priority issues identified in the LangGraph evaluation. These issues don't block basic execution but significantly impact reliability, maintainability, and user experience.

**Priority Order**:
1. **P1.1**: Add error handling for empty `plot_outline` in generation_node
2. **P1.2**: Add Neo4j transaction rollback in commit_node
3. **P1.3**: Improve error propagation from nodes to workflow edges
4. **P1.4**: Consolidate `plot_outline` vs `chapter_outlines` duplication

**Total Estimated Effort**: 3-4 days
**Complexity**: Medium
**Risk Level**: Low (incremental improvements)

---

## Issue P1.1: Add Error Handling for Empty plot_outline

### Problem Statement

**Current Behavior**:
- `generate_chapter` checks for empty `plot_outline` (line 60)
- Returns error message in state but doesn't signal failure to workflow
- Workflow continues as if nothing is wrong

**Risk Scenario**:
```
chapter_outline (fails silently) → generate (detects empty plot_outline)
  → returns error in state → extract runs anyway → data corruption
```

**Impact**:
- Silent failures cascade through pipeline
- User sees no clear error message
- Wasted compute on invalid state
- Potential database corruption

### Current Code

**Location**: `core/langgraph/nodes/generation_node.py:60-66`

```python
# Validate we have the necessary inputs
if not state.get("plot_outline"):
    logger.error("generate_chapter: no plot outline available")
    return {
        **state,
        "last_error": "No plot outline available for generation",
        "current_node": "generate",
    }
```

**Problem**: Sets `last_error` but workflow doesn't check it.

### Proposed Solution

**Approach**: Add explicit error state that workflow edges can check.

#### Step 1: Add Error Flag to State

**File**: `core/langgraph/state.py`

Add new field to `NarrativeState`:

```python
class NarrativeState(TypedDict):
    # ... existing fields ...

    # Error handling
    last_error: str | None
    has_fatal_error: bool  # NEW FIELD
    error_node: str | None  # NEW FIELD - which node failed
    retry_count: int
```

**Rationale**:
- `has_fatal_error` is a clear boolean signal for conditional edges
- `error_node` helps debugging and error reporting
- Separates recoverable errors from fatal ones

#### Step 2: Update generation_node.py

**File**: `core/langgraph/nodes/generation_node.py`

```python
def generate_chapter(state: NarrativeState) -> NarrativeState:
    """Generate chapter prose from outline and context."""
    logger.info(
        "generate_chapter: starting generation",
        chapter=state["current_chapter"],
        model=state["generation_model"],
    )

    # Validate we have the necessary inputs
    if not state.get("plot_outline"):
        error_msg = "No plot outline available for generation"
        logger.error("generate_chapter: fatal error", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,  # NEW
            "error_node": "generate",  # NEW
            "current_node": "generate",
        }

    chapter_number = state["current_chapter"]
    plot_outline = state["plot_outline"]

    # Validate chapter exists in outline
    if chapter_number not in plot_outline:
        error_msg = f"Chapter {chapter_number} not found in plot outline"
        logger.error("generate_chapter: fatal error", error=error_msg, chapter=chapter_number)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,  # NEW
            "error_node": "generate",  # NEW
            "current_node": "generate",
        }

    # ... rest of generation logic ...
```

#### Step 3: Add Error-Checking Conditional Edge

**File**: `core/langgraph/workflow.py`

Add new routing function:

```python
def should_handle_error(state: NarrativeState) -> Literal["error", "continue"]:
    """
    Check if a fatal error occurred and needs handling.

    Routes to:
    - "error": If has_fatal_error=True (stop workflow gracefully)
    - "continue": If no fatal error (proceed normally)

    Args:
        state: Current narrative state

    Returns:
        "error" or "continue"
    """
    if state.get("has_fatal_error", False):
        logger.error(
            "should_handle_error: fatal error detected",
            error=state.get("last_error"),
            node=state.get("error_node"),
        )
        return "error"

    return "continue"
```

Add error handler node:

```python
def handle_fatal_error(state: NarrativeState) -> NarrativeState:
    """
    Handle fatal errors gracefully.

    Logs detailed error information and prepares state for clean exit.
    """
    logger.error(
        "handle_fatal_error: workflow terminated due to fatal error",
        error=state.get("last_error"),
        failed_node=state.get("error_node"),
        chapter=state.get("current_chapter"),
    )

    # Prepare user-friendly error message
    error_details = {
        "error": state.get("last_error", "Unknown error"),
        "failed_node": state.get("error_node", "unknown"),
        "chapter": state.get("current_chapter"),
        "timestamp": "2025-11-09T12:00:00Z",  # Use real timestamp
    }

    # Could write error to file for debugging
    # Path(state["project_dir"]) / ".saga" / "last_error.json"

    return {
        **state,
        "current_node": "error_handler",
    }
```

Update workflow graph in `create_full_workflow_graph()`:

```python
# Add error handler node
workflow.add_node("error_handler", handle_fatal_error)

# Add error check after generate
workflow.add_conditional_edges(
    "generate",
    should_handle_error,
    {
        "error": "error_handler",
        "continue": "extract",
    },
)

# Error handler terminates workflow
workflow.add_edge("error_handler", END)
```

#### Step 4: Update Initialization create_initial_state

**File**: `core/langgraph/state.py`

```python
def create_initial_state(...) -> NarrativeState:
    """Create initial narrative state."""
    return {
        # ... existing fields ...
        "last_error": None,
        "has_fatal_error": False,  # NEW
        "error_node": None,  # NEW
        "retry_count": 0,
    }
```

### Testing Strategy

**Unit Tests** (`tests/test_langgraph/test_generation_node.py`):

```python
@pytest.mark.asyncio
async def test_generate_chapter_empty_plot_outline():
    """Test that generate_chapter sets fatal error flag when plot_outline is empty."""
    state = create_test_state()
    state["plot_outline"] = {}  # Empty outline

    result = generate_chapter(state)

    assert result["has_fatal_error"] is True
    assert result["error_node"] == "generate"
    assert "No plot outline available" in result["last_error"]

@pytest.mark.asyncio
async def test_generate_chapter_missing_chapter_in_outline():
    """Test that generate_chapter sets fatal error when chapter not in outline."""
    state = create_test_state()
    state["current_chapter"] = 5
    state["plot_outline"] = {1: {...}, 2: {...}}  # Chapter 5 missing

    result = generate_chapter(state)

    assert result["has_fatal_error"] is True
    assert "Chapter 5 not found" in result["last_error"]
```

**Workflow Tests** (`tests/test_langgraph/test_workflow.py`):

```python
@pytest.mark.asyncio
async def test_workflow_handles_fatal_error():
    """Test that workflow stops gracefully on fatal error."""
    from core.langgraph.workflow import create_phase2_graph, create_checkpointer

    checkpointer = create_checkpointer(":memory:")
    graph = create_phase2_graph(checkpointer=checkpointer)

    # Create state with missing plot_outline
    state = create_test_state()
    state["plot_outline"] = {}

    result = await graph.ainvoke(state)

    # Should stop at error_handler node
    assert result["current_node"] == "error_handler"
    assert result["has_fatal_error"] is True
    # Should NOT have proceeded to extract/commit
    assert "extracted_entities" not in result or len(result["extracted_entities"]) == 0
```

### Rollout Plan

1. **Phase 1**: Add new state fields (1 hour)
2. **Phase 2**: Update generation_node with error flags (1 hour)
3. **Phase 3**: Add error checking conditional edge (2 hours)
4. **Phase 4**: Write and run tests (2 hours)
5. **Phase 5**: Update documentation (1 hour)

**Total**: 7 hours (1 day)

**Risk**: Low - additive changes, doesn't break existing functionality

---

## Issue P1.2: Add Neo4j Transaction Rollback

### Problem Statement

**Current Behavior**:
- `commit_to_graph` makes multiple Neo4j write operations
- If one fails, previous writes remain committed
- Database left in inconsistent state

**Example Failure Scenario**:
```
1. Create 3 characters ✓
2. Create 5 relationships ✓
3. Create chapter node ✗ (fails)
Result: Characters and relationships in DB, but no chapter
```

**Impact**:
- Database corruption on failures
- Orphaned entities
- Difficult recovery
- Data integrity violations

### Current Code Pattern

**Location**: `core/langgraph/nodes/commit_node.py`

```python
async def commit_to_graph(state: NarrativeState) -> NarrativeState:
    """Commit extracted entities to Neo4j."""

    # Multiple separate operations
    await _deduplicate_and_commit_characters(...)
    await _deduplicate_and_commit_locations(...)
    await _commit_relationships(...)
    await _create_chapter_node(...)  # If this fails, previous writes are committed!

    return state
```

**Problem**: No transaction boundary - each operation auto-commits.

### Proposed Solution

**Approach**: Wrap all Neo4j operations in explicit transaction with rollback.

#### Step 1: Create Transaction Helper

**File**: `core/db_manager.py`

Add transaction context manager:

```python
from contextlib import asynccontextmanager
from neo4j import AsyncSession, AsyncDriver
import structlog

logger = structlog.get_logger(__name__)

@asynccontextmanager
async def neo4j_transaction(session: AsyncSession):
    """
    Async context manager for Neo4j transactions with automatic rollback.

    Usage:
        async with neo4j_transaction(session) as tx:
            await tx.run("CREATE (n:Node {id: $id})", id=123)
            # If exception occurs, transaction is rolled back

    Args:
        session: Neo4j async session

    Yields:
        Neo4j transaction object
    """
    tx = await session.begin_transaction()
    try:
        yield tx
        await tx.commit()
        logger.debug("neo4j_transaction: committed successfully")
    except Exception as e:
        await tx.rollback()
        logger.error(
            "neo4j_transaction: rolled back due to error",
            error=str(e),
            exc_info=True,
        )
        raise  # Re-raise to let caller handle
```

#### Step 2: Refactor commit_node to Use Transactions

**File**: `core/langgraph/nodes/commit_node.py`

```python
from core.db_manager import neo4j_transaction

async def commit_to_graph(state: NarrativeState) -> NarrativeState:
    """
    Commit extracted entities to Neo4j knowledge graph with transaction safety.

    All operations are wrapped in a single transaction. If any operation fails,
    all changes are rolled back to maintain database consistency.
    """
    logger.info(
        "commit_to_graph: starting commit",
        chapter=state["current_chapter"],
        entities=len(state.get("extracted_entities", {})),
        relationships=len(state.get("extracted_relationships", [])),
    )

    from core.db_manager import neo4j_manager

    try:
        # Get Neo4j driver (not session yet)
        driver = neo4j_manager.driver

        # Create session and transaction
        async with driver.session() as session:
            async with neo4j_transaction(session) as tx:
                # All operations in single transaction
                char_mappings = await _deduplicate_and_commit_characters(
                    tx,  # Pass transaction instead of session
                    state.get("extracted_entities", {}).get("characters", []),
                    state["current_chapter"],
                )

                loc_mappings = await _deduplicate_and_commit_locations(
                    tx,
                    state.get("extracted_entities", {}).get("locations", []),
                    state["current_chapter"],
                )

                await _commit_relationships(
                    tx,
                    state.get("extracted_relationships", []),
                    char_mappings,
                    state["current_chapter"],
                )

                await _create_chapter_node(
                    tx,
                    state["current_chapter"],
                    state.get("draft_text", ""),
                    state.get("draft_word_count", 0),
                )

                # If we get here, all operations succeeded
                # Transaction will auto-commit when exiting context manager

        logger.info("commit_to_graph: all entities committed successfully")

        return {
            **state,
            "current_node": "commit",
            "last_error": None,
            "has_fatal_error": False,
        }

    except Exception as e:
        error_msg = f"Failed to commit entities to Neo4j: {str(e)}"
        logger.error(
            "commit_to_graph: transaction failed and rolled back",
            error=error_msg,
            chapter=state["current_chapter"],
            exc_info=True,
        )

        return {
            **state,
            "current_node": "commit",
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "commit",
        }
```

#### Step 3: Update Helper Functions to Use Transaction

**File**: `core/langgraph/nodes/commit_node.py`

Update function signatures to accept transaction instead of session:

```python
async def _deduplicate_and_commit_characters(
    tx,  # Changed from session to tx
    extracted_chars: list[ExtractedEntity],
    current_chapter: int,
) -> dict[str, str]:
    """
    Deduplicate and commit characters within a transaction.

    Args:
        tx: Neo4j transaction object
        extracted_chars: List of extracted character entities
        current_chapter: Current chapter number

    Returns:
        Mapping of extracted names to Neo4j IDs
    """
    char_mappings = {}

    for char in extracted_chars:
        # Use tx.run() instead of session.run()
        neo4j_id = await deduplicate_and_get_id(
            char.name,
            char.description,
            "character",
            current_chapter,
            tx=tx,  # Pass transaction
        )

        char_mappings[char.name] = neo4j_id

    return char_mappings

# Similar updates for:
# - _deduplicate_and_commit_locations
# - _commit_relationships
# - _create_chapter_node
```

**File**: `processing/entity_deduplication.py`

Update `deduplicate_and_get_id` to accept transaction:

```python
async def deduplicate_and_get_id(
    name: str,
    description: str,
    entity_type: str,
    chapter: int,
    tx=None,  # NEW: Optional transaction parameter
) -> str:
    """
    Deduplicate entity and return Neo4j ID.

    Args:
        name: Entity name
        description: Entity description
        entity_type: Type of entity (character, location, etc.)
        chapter: Chapter number
        tx: Optional Neo4j transaction. If provided, operations run in transaction.

    Returns:
        Neo4j ID of entity (existing or newly created)
    """
    from core.db_manager import neo4j_manager

    # If transaction provided, use it; otherwise create session
    if tx:
        result = await tx.run(
            """
            MATCH (e {name: $name, type: $type})
            RETURN e.id AS id
            LIMIT 1
            """,
            name=name,
            type=entity_type,
        )
        # ... rest of logic using tx.run() ...
    else:
        # Fallback to session (for backward compatibility)
        async with neo4j_manager.driver.session() as session:
            result = await session.run(...)
            # ... rest of logic ...
```

### Testing Strategy

**Unit Tests** (`tests/test_langgraph/test_commit_node.py`):

```python
@pytest.mark.asyncio
async def test_commit_rollback_on_failure(mock_neo4j):
    """Test that transaction is rolled back when any operation fails."""
    from core.langgraph.nodes.commit_node import commit_to_graph

    # Mock Neo4j to succeed on first 2 operations, fail on 3rd
    mock_tx = AsyncMock()
    mock_tx.run = AsyncMock(side_effect=[
        Mock(),  # Character dedup succeeds
        Mock(),  # Location dedup succeeds
        Exception("Simulated failure"),  # Relationship commit fails
    ])
    mock_tx.commit = AsyncMock()
    mock_tx.rollback = AsyncMock()

    state = create_test_state()
    state["extracted_entities"] = {
        "characters": [create_test_character()],
        "locations": [create_test_location()],
    }
    state["extracted_relationships"] = [create_test_relationship()]

    result = await commit_to_graph(state)

    # Should have rolled back
    assert mock_tx.rollback.called
    assert not mock_tx.commit.called
    assert result["has_fatal_error"] is True
    assert "Failed to commit" in result["last_error"]
```

**Integration Tests** (`tests/test_langgraph/test_commit_node.py`):

```python
@pytest.mark.asyncio
async def test_commit_leaves_no_orphans_on_failure(neo4j_test_db):
    """Test that failed commit leaves no partial data in database."""
    from core.langgraph.nodes.commit_node import commit_to_graph
    from core.db_manager import neo4j_manager

    # Setup: Clear database
    await neo4j_test_db.clear()

    # Create state that will fail on chapter node creation
    state = create_test_state()
    state["extracted_entities"] = {"characters": [create_test_character("Alice")]}
    state["draft_text"] = None  # Will cause chapter node creation to fail

    # Execute (should fail)
    result = await commit_to_graph(state)

    # Verify: No characters were committed
    async with neo4j_manager.driver.session() as session:
        count_result = await session.run("MATCH (c:Character) RETURN count(c) AS count")
        count = await count_result.single()

    assert count["count"] == 0, "No characters should be committed after rollback"
    assert result["has_fatal_error"] is True
```

### Rollout Plan

1. **Phase 1**: Add transaction helper to db_manager (2 hours)
2. **Phase 2**: Refactor commit_node to use transactions (3 hours)
3. **Phase 3**: Update helper functions (2 hours)
4. **Phase 4**: Write unit tests (2 hours)
5. **Phase 5**: Write integration tests (3 hours)
6. **Phase 6**: Test with real Neo4j instance (2 hours)

**Total**: 14 hours (2 days)

**Risk**: Medium - touches critical data path, requires thorough testing

---

## Issue P1.3: Improve Error Propagation from Nodes to Workflow

### Problem Statement

**Current Behavior**:
- Nodes set `last_error` in state
- Workflow has no conditional edges to check errors
- Errors propagate silently through pipeline
- User sees no clear error indication

**Example**:
```
generate (sets last_error) → extract (runs anyway) → commit (commits bad data)
```

**Impact**:
- Silent failures accumulate
- Bad data committed to database
- Poor user experience
- Difficult debugging

### Current Architecture Gap

**Workflow Definition**: `core/langgraph/workflow.py`

```python
# Current edges - no error checking
workflow.add_edge("generate", "extract")
workflow.add_edge("extract", "commit")
workflow.add_edge("commit", "validate")
```

**Node Pattern**: All nodes return state with `last_error` but workflow ignores it.

### Proposed Solution

**Approach**: Add error-checking conditional edges after critical nodes.

#### Step 1: Define Critical Nodes

Identify nodes where failures should stop workflow:
- ✅ `generate` - Already covered in P1.1
- ✅ `extract` - Entity extraction failures
- ✅ `commit` - Database commit failures
- ❌ `validate` - Validation failures are expected (trigger revision)
- ✅ `revise` - Revision failures
- ❌ `summarize` - Summary failures are non-critical
- ✅ `finalize` - File write failures

#### Step 2: Add Error Checks to Extract Node

**File**: `core/langgraph/nodes/extraction_node.py`

```python
async def extract_entities(state: NarrativeState) -> NarrativeState:
    """Extract entities and relationships from generated chapter text."""
    logger.info(
        "extract_entities: starting extraction",
        chapter=state["current_chapter"],
    )

    # Validate input
    if not state.get("draft_text"):
        error_msg = "No draft text available for entity extraction"
        logger.error("extract_entities: fatal error", error=error_msg)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "extract",
            "current_node": "extract",
        }

    try:
        # Run extraction logic
        results = await asyncio.gather(
            _extract_characters(state["draft_text"], state["extraction_model"]),
            _extract_locations(state["draft_text"], state["extraction_model"]),
            # ... other extractions ...
        )

        # Validate results
        if all(len(r) == 0 for r in results):
            logger.warning(
                "extract_entities: no entities extracted (unusual)",
                chapter=state["current_chapter"],
            )
            # This is suspicious but not fatal - might be valid chapter with no new entities

        return {
            **state,
            "extracted_entities": {
                "characters": results[0],
                "locations": results[1],
                # ...
            },
            "current_node": "extract",
            "last_error": None,
            "has_fatal_error": False,
        }

    except Exception as e:
        error_msg = f"Entity extraction failed: {str(e)}"
        logger.error(
            "extract_entities: fatal error",
            error=error_msg,
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "extract",
            "current_node": "extract",
        }
```

#### Step 3: Add Error Checks to Revise Node

**File**: `core/langgraph/nodes/revision_node.py`

```python
async def revise_chapter(state: NarrativeState) -> NarrativeState:
    """Revise chapter based on validation feedback."""

    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        error_msg = f"Max revision attempts ({state['max_iterations']}) reached"
        logger.error(
            "revise_chapter: iteration limit exceeded",
            iterations=state["iteration_count"],
            max_iterations=state["max_iterations"],
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "needs_revision": False,
            "current_node": "revise",
        }

    try:
        # Revision logic
        revised_text = await _call_llm(...)

        if not revised_text or len(revised_text.strip()) < 100:
            raise ValueError("Revision produced insufficient text")

        return {
            **state,
            "draft_text": revised_text,
            "iteration_count": state["iteration_count"] + 1,
            "last_error": None,
            "has_fatal_error": False,
            "current_node": "revise",
        }

    except Exception as e:
        error_msg = f"Revision failed: {str(e)}"
        logger.error("revise_chapter: fatal error", error=error_msg, exc_info=True)
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "revise",
            "current_node": "revise",
        }
```

#### Step 4: Add Error Checks to Finalize Node

**File**: `core/langgraph/nodes/finalize_node.py`

```python
async def finalize_chapter(state: NarrativeState) -> NarrativeState:
    """Finalize chapter by writing to filesystem and database."""

    logger.info(
        "finalize_chapter: finalizing",
        chapter=state["current_chapter"],
    )

    try:
        # Write chapter file
        chapter_path = Path(state["chapters_dir"]) / f"chapter_{state['current_chapter']:03d}.md"
        chapter_path.parent.mkdir(parents=True, exist_ok=True)

        chapter_content = _format_chapter(state)
        chapter_path.write_text(chapter_content, encoding="utf-8")

        # Save to database
        from data_access.chapter_queries import save_chapter
        await save_chapter(
            state["current_chapter"],
            state["draft_text"],
            state.get("draft_word_count", 0),
        )

        logger.info(
            "finalize_chapter: chapter finalized successfully",
            chapter=state["current_chapter"],
            path=str(chapter_path),
        )

        return {
            **state,
            "current_node": "finalize",
            "last_error": None,
            "has_fatal_error": False,
        }

    except Exception as e:
        error_msg = f"Failed to finalize chapter: {str(e)}"
        logger.error(
            "finalize_chapter: fatal error",
            error=error_msg,
            chapter=state["current_chapter"],
            exc_info=True,
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "finalize",
            "current_node": "finalize",
        }
```

#### Step 5: Update Workflow Edges

**File**: `core/langgraph/workflow.py`

```python
def create_phase2_graph(checkpointer=None) -> CompiledGraph:
    """Create Phase 2 workflow graph with error handling."""

    workflow = StateGraph(NarrativeState)

    # Add nodes
    workflow.add_node("generate", generate_chapter)
    workflow.add_node("extract", extract_entities)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", validate_consistency)
    workflow.add_node("revise", revise_chapter)
    workflow.add_node("summarize", summarize_chapter)
    workflow.add_node("finalize", finalize_chapter)
    workflow.add_node("error_handler", handle_fatal_error)

    # Entry point
    workflow.set_entry_point("generate")

    # Add error-checking edges
    workflow.add_conditional_edges(
        "generate",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "extract",
        },
    )

    workflow.add_conditional_edges(
        "extract",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "commit",
        },
    )

    workflow.add_conditional_edges(
        "commit",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "validate",
        },
    )

    # Validate has special routing (not just error check)
    workflow.add_conditional_edges(
        "validate",
        should_revise_or_handle_error,  # NEW: Combined check
        {
            "error": "error_handler",
            "revise": "revise",
            "continue": "summarize",
        },
    )

    workflow.add_conditional_edges(
        "revise",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": "extract",  # Re-extract after revision
        },
    )

    # Summarize failures are non-critical (continue anyway)
    workflow.add_edge("summarize", "finalize")

    workflow.add_conditional_edges(
        "finalize",
        should_handle_error,
        {
            "error": "error_handler",
            "continue": END,
        },
    )

    # Error handler terminates
    workflow.add_edge("error_handler", END)

    return workflow.compile(checkpointer=checkpointer)


def should_revise_or_handle_error(
    state: NarrativeState
) -> Literal["error", "revise", "continue"]:
    """
    Combined check for fatal errors and revision needs.

    Priority:
    1. Fatal error → "error"
    2. Needs revision → "revise"
    3. Otherwise → "continue"
    """
    # Check for fatal errors first
    if state.get("has_fatal_error", False):
        return "error"

    # Check if revision needed
    if state.get("needs_revision", False):
        return "revise"

    return "continue"
```

### Testing Strategy

**Workflow Tests** (`tests/test_langgraph/test_phase2_workflow.py`):

```python
@pytest.mark.asyncio
async def test_error_stops_at_extract():
    """Test that error in extract stops workflow."""
    from core.langgraph.workflow import create_phase2_graph

    graph = create_phase2_graph()

    # Create state that will fail at extract
    state = create_test_state()
    state["draft_text"] = None  # Missing draft text

    result = await graph.ainvoke(state)

    assert result["current_node"] == "error_handler"
    assert result["error_node"] == "extract"
    assert "draft text" in result["last_error"].lower()

@pytest.mark.asyncio
async def test_error_stops_at_commit():
    """Test that error in commit stops workflow."""
    graph = create_phase2_graph()

    # Mock commit to fail
    with patch("core.langgraph.nodes.commit_node.commit_to_graph") as mock:
        mock.return_value = {
            "has_fatal_error": True,
            "error_node": "commit",
            "last_error": "Neo4j connection lost",
        }

        state = create_test_state()
        result = await graph.ainvoke(state)

        assert result["current_node"] == "error_handler"
        assert result["error_node"] == "commit"

@pytest.mark.asyncio
async def test_validate_error_vs_revision():
    """Test that validation distinguishes between errors and needed revisions."""
    graph = create_phase2_graph()

    # Case 1: Fatal error in validate
    state1 = create_test_state()
    with patch("core.langgraph.nodes.validation_node.validate_consistency") as mock:
        mock.return_value = {
            "has_fatal_error": True,
            "error_node": "validate",
        }
        result1 = await graph.ainvoke(state1)
        assert result1["current_node"] == "error_handler"

    # Case 2: Needs revision (not error)
    state2 = create_test_state()
    with patch("core.langgraph.nodes.validation_node.validate_consistency") as mock:
        mock.return_value = {
            "has_fatal_error": False,
            "needs_revision": True,
            "contradictions": [create_test_contradiction()],
        }
        result2 = await graph.ainvoke(state2)
        # Should go to revise, not error_handler
        assert result2["current_node"] == "revise" or "extract" in str(result2)
```

### Rollout Plan

1. **Phase 1**: Add error flags to extract, revise, finalize nodes (3 hours)
2. **Phase 2**: Update workflow edges with error checks (2 hours)
3. **Phase 3**: Create combined should_revise_or_handle_error function (1 hour)
4. **Phase 4**: Write workflow tests (3 hours)
5. **Phase 5**: Integration testing (2 hours)

**Total**: 11 hours (1.5 days)

**Risk**: Medium - changes workflow routing logic, needs careful testing

---

## Issue P1.4: Consolidate plot_outline vs chapter_outlines Duplication

### Problem Statement

**Current Behavior**:
- State maintains TWO outline structures:
  - `plot_outline: dict[int, dict]` (used by generation)
  - `chapter_outlines: dict[int, dict]` (populated by initialization)
- Manual bridging code in `chapter_outline_node.py` lines 101-109

**Why This Exists**:
- `plot_outline` is legacy from SAGA 1.0
- `chapter_outlines` is new initialization framework
- Bridge maintains backward compatibility

**Impact**:
- Two sources of truth for same data
- Maintenance burden (keep in sync)
- Developer confusion
- Code duplication

### Current Bridge Code

**Location**: `core/langgraph/initialization/chapter_outline_node.py:101-109`

```python
# Update chapter_outlines dict
updated_outlines = {**existing_outlines, chapter_number: chapter_outline}

# Also update plot_outline for compatibility with existing generation code
plot_outline = state.get("plot_outline", {})
plot_outline[chapter_number] = {
    "chapter": chapter_number,
    "act": act_number,
    "scene_description": chapter_outline.get("scene_description", ""),
    "key_beats": chapter_outline.get("key_beats", []),
    "plot_point": chapter_outline.get("plot_point", ""),
}
```

### Analysis: Which Structure to Keep?

#### Option A: Keep plot_outline (Simpler)

**Pros**:
- Simpler structure (flat dict)
- Used by generation_node (fewer changes)
- Legacy compatibility

**Cons**:
- Less expressive (no nested structure for acts/scenes)
- Loses initialization framework's richer schema
- Doesn't align with modern design

#### Option B: Keep chapter_outlines (Recommended)

**Pros**:
- Richer schema (supports nested structure)
- Aligns with initialization framework
- More extensible for future features
- Better separation of concerns

**Cons**:
- Requires updating generation_node
- More complex data access

**Recommendation**: **Option B** - Modernize around chapter_outlines

### Proposed Solution

**Approach**:
1. Deprecate `plot_outline`
2. Update all consumers to use `chapter_outlines`
3. Add migration helper for smooth transition

#### Step 1: Define Canonical chapter_outlines Schema

**File**: `core/langgraph/state.py`

Update schema documentation:

```python
class NarrativeState(TypedDict):
    # ... other fields ...

    # Plot structure
    # DEPRECATED: plot_outline - use chapter_outlines instead
    plot_outline: dict[int, dict[str, Any]]  # Legacy, will be removed in v3.0

    # CANONICAL: chapter_outlines - primary source of truth
    chapter_outlines: dict[int, dict[str, Any]]  # {chapter_num: outline_dict}
    # Schema per chapter:
    # {
    #   "chapter": int,
    #   "act": int,
    #   "scene_description": str,
    #   "key_beats": list[str],
    #   "plot_point": str,
    #   "characters": list[str],  # NEW: character names in this chapter
    #   "location": str | None,   # NEW: primary location
    # }
```

#### Step 2: Update generation_node to Use chapter_outlines

**File**: `core/langgraph/nodes/generation_node.py`

```python
def generate_chapter(state: NarrativeState) -> NarrativeState:
    """Generate chapter prose from outline and context."""

    # OLD CODE (deprecated):
    # if not state.get("plot_outline"):
    #     return error...
    # plot_outline = state["plot_outline"]

    # NEW CODE:
    chapter_outlines = state.get("chapter_outlines")
    if not chapter_outlines:
        # Fallback to plot_outline for backward compatibility
        chapter_outlines = state.get("plot_outline")
        if not chapter_outlines:
            error_msg = "No chapter outlines available for generation"
            logger.error("generate_chapter: fatal error", error=error_msg)
            return {
                **state,
                "last_error": error_msg,
                "has_fatal_error": True,
                "error_node": "generate",
                "current_node": "generate",
            }

    chapter_number = state["current_chapter"]

    # Validate chapter exists in outlines
    if chapter_number not in chapter_outlines:
        error_msg = f"Chapter {chapter_number} not found in chapter outlines"
        logger.error(
            "generate_chapter: fatal error",
            error=error_msg,
            chapter=chapter_number,
            available_chapters=list(chapter_outlines.keys()),
        )
        return {
            **state,
            "last_error": error_msg,
            "has_fatal_error": True,
            "error_node": "generate",
            "current_node": "generate",
        }

    outline_entry = chapter_outlines[chapter_number]

    # Extract data from outline (supporting both old and new schema)
    scene_description = outline_entry.get("scene_description", "")
    key_beats = outline_entry.get("key_beats", [])
    plot_point = outline_entry.get("plot_point", "")
    characters = outline_entry.get("characters", [])  # NEW
    location = outline_entry.get("location")  # NEW

    # Build context from outline
    context = await _build_context_from_graph(
        current_chapter=chapter_number,
        character_names=characters,  # Use outline characters if available
        location_name=location,
    )

    # ... rest of generation logic ...
```

#### Step 3: Remove Bridge Code from chapter_outline_node

**File**: `core/langgraph/initialization/chapter_outline_node.py`

```python
async def generate_chapter_outline(state: NarrativeState) -> NarrativeState:
    """Generate outline for a specific chapter on-demand."""

    # ... outline generation logic ...

    # OLD CODE (remove this):
    # # Also update plot_outline for compatibility with existing generation code
    # plot_outline = state.get("plot_outline", {})
    # plot_outline[chapter_number] = {
    #     "chapter": chapter_number,
    #     "act": act_number,
    #     "scene_description": chapter_outline.get("scene_description", ""),
    #     "key_beats": chapter_outline.get("key_beats", []),
    #     "plot_point": chapter_outline.get("plot_point", ""),
    # }

    # NEW CODE (single source of truth):
    updated_outlines = {**existing_outlines, chapter_number: chapter_outline}

    return {
        **state,
        "chapter_outlines": updated_outlines,
        # plot_outline no longer updated
        "current_node": "chapter_outline",
        "initialization_step": f"chapter_outline_{chapter_number}_complete",
    }
```

#### Step 4: Add Migration Helper

**File**: `core/langgraph/state.py`

```python
def migrate_plot_outline_to_chapter_outlines(state: NarrativeState) -> NarrativeState:
    """
    Migrate legacy plot_outline to chapter_outlines.

    This helper ensures backward compatibility during transition period.
    Can be removed in v3.0 after all users have migrated.

    Args:
        state: Current narrative state

    Returns:
        State with chapter_outlines populated from plot_outline if needed
    """
    # If chapter_outlines already populated, nothing to do
    if state.get("chapter_outlines"):
        return state

    # If plot_outline exists, migrate it
    plot_outline = state.get("plot_outline")
    if plot_outline:
        logger.info(
            "migrate_plot_outline_to_chapter_outlines: migrating legacy plot_outline",
            chapters=len(plot_outline),
        )

        # Copy plot_outline to chapter_outlines
        # (they have same schema for now)
        return {
            **state,
            "chapter_outlines": dict(plot_outline),
        }

    return state
```

**Usage in orchestrator**:

```python
# File: orchestration/langgraph_orchestrator.py

async def _load_or_create_state(self) -> NarrativeState:
    """Load existing state or create new initial state."""

    # ... existing logic ...

    state = create_initial_state(...)

    # Migrate legacy plot_outline if present
    state = migrate_plot_outline_to_chapter_outlines(state)

    return state
```

#### Step 5: Update create_initial_state

**File**: `core/langgraph/state.py`

```python
def create_initial_state(...) -> NarrativeState:
    """Create initial narrative state."""
    return {
        # ... other fields ...

        # Plot structure
        "plot_outline": {},  # DEPRECATED: kept for backward compatibility only
        "chapter_outlines": {},  # PRIMARY: use this
        "act_outlines": {},
        "global_outline": {},

        # ... rest of fields ...
    }
```

#### Step 6: Add Deprecation Warnings

**File**: `core/langgraph/nodes/generation_node.py`

```python
def generate_chapter(state: NarrativeState) -> NarrativeState:
    """Generate chapter prose from outline and context."""

    # Check if using deprecated plot_outline
    if state.get("plot_outline") and not state.get("chapter_outlines"):
        logger.warning(
            "generate_chapter: using deprecated plot_outline field. "
            "Please migrate to chapter_outlines. "
            "plot_outline will be removed in SAGA v3.0",
            deprecation=True,
        )

    # ... rest of generation logic using chapter_outlines ...
```

### Testing Strategy

**Migration Tests** (`tests/test_langgraph/test_state.py`):

```python
def test_migrate_plot_outline_to_chapter_outlines():
    """Test migration of legacy plot_outline to chapter_outlines."""
    from core.langgraph.state import migrate_plot_outline_to_chapter_outlines

    # Test: plot_outline present, chapter_outlines empty
    state = {
        "plot_outline": {
            1: {"chapter": 1, "scene_description": "Opening"},
            2: {"chapter": 2, "scene_description": "Conflict"},
        },
        "chapter_outlines": {},
    }

    result = migrate_plot_outline_to_chapter_outlines(state)

    assert len(result["chapter_outlines"]) == 2
    assert result["chapter_outlines"][1]["scene_description"] == "Opening"
    assert result["chapter_outlines"][2]["scene_description"] == "Conflict"

def test_migrate_preserves_chapter_outlines():
    """Test that migration doesn't overwrite existing chapter_outlines."""
    state = {
        "plot_outline": {1: {"chapter": 1, "scene_description": "Old"}},
        "chapter_outlines": {1: {"chapter": 1, "scene_description": "New"}},
    }

    result = migrate_plot_outline_to_chapter_outlines(state)

    # Should keep chapter_outlines unchanged
    assert result["chapter_outlines"][1]["scene_description"] == "New"
```

**Generation Tests** (`tests/test_langgraph/test_generation_node.py`):

```python
@pytest.mark.asyncio
async def test_generate_uses_chapter_outlines():
    """Test that generation uses chapter_outlines (not plot_outline)."""
    state = create_test_state()
    state["chapter_outlines"] = {
        1: {
            "chapter": 1,
            "scene_description": "Test scene",
            "key_beats": ["Beat 1", "Beat 2"],
        }
    }
    state["plot_outline"] = {}  # Empty (deprecated)
    state["current_chapter"] = 1

    result = generate_chapter(state)

    # Should succeed using chapter_outlines
    assert result["has_fatal_error"] is False
    assert result["draft_text"] is not None

@pytest.mark.asyncio
async def test_generate_backward_compatibility_with_plot_outline():
    """Test backward compatibility with legacy plot_outline."""
    state = create_test_state()
    state["plot_outline"] = {
        1: {
            "chapter": 1,
            "scene_description": "Legacy scene",
            "key_beats": ["Beat 1"],
        }
    }
    state["chapter_outlines"] = {}  # Empty (not yet migrated)
    state["current_chapter"] = 1

    result = generate_chapter(state)

    # Should still work via fallback
    assert result["has_fatal_error"] is False
    assert result["draft_text"] is not None
```

### Rollout Plan

**Phase 1: Preparation** (Day 1)
1. Document chapter_outlines schema (1 hour)
2. Create migration helper function (1 hour)
3. Write migration tests (1 hour)

**Phase 2: Update Consumers** (Day 1-2)
4. Update generation_node to use chapter_outlines (2 hours)
5. Add backward compatibility fallback (1 hour)
6. Add deprecation warnings (1 hour)

**Phase 3: Remove Duplication** (Day 2)
7. Remove bridge code from chapter_outline_node (1 hour)
8. Update all tests to use chapter_outlines (2 hours)

**Phase 4: Testing** (Day 2-3)
9. Run full test suite (1 hour)
10. Integration testing with real workflow (2 hours)
11. Test migration with existing projects (1 hour)

**Phase 5: Documentation** (Day 3)
12. Update architecture docs (1 hour)
13. Add migration guide for users (1 hour)
14. Update code comments (1 hour)

**Total**: 17 hours (2-3 days)

**Risk**: Low - has backward compatibility, gradual migration

---

## Implementation Order Recommendation

Based on dependencies and impact:

### Week 1 (Days 1-3)
1. **P1.1** - Error handling for plot_outline (1 day)
2. **P1.3** - Error propagation in workflow (1.5 days)

### Week 2 (Days 4-6)
3. **P1.2** - Neo4j transaction rollback (2 days)
4. **P1.4** - Consolidate outline structures (2-3 days)

**Total**: 6-7 days (allowing for buffer)

---

## Success Criteria

### P1.1 Success Metrics
- ✅ Empty plot_outline triggers error handler
- ✅ Workflow stops gracefully, doesn't corrupt data
- ✅ Clear error message shown to user
- ✅ Tests verify error handling works

### P1.2 Success Metrics
- ✅ All Neo4j operations in single transaction
- ✅ Failure rolls back all changes (no orphans)
- ✅ Integration tests verify rollback behavior
- ✅ No partial commits in error scenarios

### P1.3 Success Metrics
- ✅ All critical nodes set has_fatal_error correctly
- ✅ Workflow routes to error_handler on failures
- ✅ Tests cover error paths for each node
- ✅ User sees clear error messages

### P1.4 Success Metrics
- ✅ All nodes use chapter_outlines (not plot_outline)
- ✅ Backward compatibility maintained
- ✅ No duplicate outline data in state
- ✅ Tests verify migration works

---

## Post-Implementation

After completing all 4 issues:

1. **Update architecture docs** to reflect new error handling
2. **Create migration guide** for users on older versions
3. **Add observability** - metrics on error rates by node
4. **Performance testing** - verify transaction overhead acceptable
5. **User documentation** - explain error messages and recovery

---

## Questions / Decisions Needed

1. **Error Recovery**: Should workflow support retry logic for transient failures (e.g., Neo4j connection timeout)?
   - **Recommendation**: Yes, add retry for network errors (separate from fatal errors)

2. **Error Notifications**: Should errors trigger notifications (email, Slack, etc.)?
   - **Recommendation**: Not in P1, add in P2 (future work)

3. **Partial Progress**: Should we save partial progress before rolling back?
   - **Recommendation**: No - transaction rollback is cleaner

4. **Schema Migration**: Should we force migration of plot_outline → chapter_outlines?
   - **Recommendation**: No - maintain backward compatibility for 1-2 releases

---

## Appendix: Code Review Checklist

Before merging each fix:

- [ ] All tests pass (unit + integration)
- [ ] No performance regression (benchmark critical paths)
- [ ] Documentation updated
- [ ] Deprecation warnings added where applicable
- [ ] Error messages are user-friendly
- [ ] Logging is comprehensive but not excessive
- [ ] Code follows existing style conventions
- [ ] No security vulnerabilities introduced

---

**End of High-Priority Implementation Plan**
