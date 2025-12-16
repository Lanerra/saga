# Comprehensive LangGraph Workflow Analysis

This report provides an exhaustive walkthrough of the LangGraph-based narrative generation system in SAGA. It details the state management, workflow structure, subgraphs, context flow, and routing logic.

## 1. State Management (`NarrativeState`)

The central data structure driving the workflow is `NarrativeState` (defined in `core/langgraph/state.py`). It is a TypedDict that persists across graph executions via a SQLite checkpointer.

**Key Components:**
*   **Metadata**: `project_id`, `title`, `genre`, `theme`, `setting`, `target_word_count`.
*   **Progress Tracking**: `current_chapter`, `total_chapters`, `current_act`, `initialization_complete`.
*   **Data Models**:
    *   `character_sheets` (Init phase)
    *   `global_outline`, `act_outlines`, `chapter_outlines` (Planning)
    *   `draft_text`, `scene_drafts` (Generation)
    *   `extracted_entities`, `extracted_relationships` (Extraction)
    *   `contradictions`, `quality_feedback` (Validation)
*   **Context**: `active_characters`, `previous_chapter_summaries`, `key_events`, `hybrid_context` (for generation prompt).
*   **Control Flow**: `current_node`, `iteration_count`, `max_iterations`, `force_continue`, `has_fatal_error`.

## 2. Workflow Overview

The system uses a hierarchical graph structure:
1.  **Main Workflow** (`create_full_workflow_graph` in `core/langgraph/workflow.py`): Orchestrates the high-level phases.
2.  **Initialization Subgraph**: Handles project setup (characters, outlines).
3.  **Generation Loop**: Handles chapter-by-chapter creation using nested subgraphs.

### Top-Level Flow
```mermaid
graph TD
    START --> route[route]
    route -->|initialization_complete=False| init_sheets[init_character_sheets]
    route -->|initialization_complete=True| chap_outline[chapter_outline]

    subgraph Initialization Phase
        init_sheets --> init_global[init_global_outline]
        init_global --> init_acts[init_act_outlines]
        init_acts --> init_commit[init_commit_to_graph]
        init_commit --> init_persist[init_persist_files]
        init_persist --> init_complete[init_complete]
    end

    init_complete --> chap_outline

    subgraph Generation Loop
        chap_outline --> generate[generate (Subgraph)]
        generate --> gen_embedding[gen_embedding]
        gen_embedding --> extract[extract (Subgraph)]
        extract --> commit[commit]
        commit --> validate[validate (Subgraph)]
        
        validate -->|needs_revision=True| revise[revise]
        validate -->|needs_revision=False| summarize[summarize]
        
        revise --> extract
        summarize --> finalize[finalize]
        finalize --> heal[heal_graph]
    end

    heal --> END
```

## 3. Detailed Component Analysis

### 3.1 Initialization Phase
**Entry Point**: `route` node checks `state["initialization_complete"]`.

*   **`init_character_sheets`**: Generates 3-5 main characters using LLM. Stores in `state["character_sheets"]`.
*   **`init_global_outline`**: Creates a 3 or 5-act structure based on `total_chapters`. Stores in `state["global_outline"]`.
*   **`init_act_outlines`**: Expands global outline into chapter-level beats. Stores in `state["act_outlines"]`.
*   **`init_commit_to_graph`**:
    *   Parses character sheets into `CharacterProfile` objects.
    *   Extracts world items.
    *   Persists to Neo4j.
*   **`init_persist_files`**: Writes YAML/Markdown files to disk (output directory structure).
*   **`init_complete`**: Sets `initialization_complete=True`.

### 3.2 Generation Phase (Chapter Loop)

#### A. Chapter Outline
*   **Node**: `chapter_outline`
*   **Logic**: Generates a detailed scene-by-scene plan for the current chapter if one doesn't exist.
*   **Context**: Uses act outline, previous summaries, and character sheets.

#### B. Generation Subgraph (`core/langgraph/subgraphs/generation.py`)
This subgraph breaks chapter generation into scenes.

*   **Nodes**:
    1.  `plan_scenes`: Breaks chapter outline into specific scenes (`state["chapter_plan"]`).
    2.  `retrieve_context`: Builds `hybrid_context` for the current scene.
    3.  `draft_scene`: Generates text for one scene.
    4.  `assemble_chapter`: Combines scene drafts into `state["draft_text"]`.
*   **Flow**: `plan_scenes` -> `retrieve_context` -> `draft_scene` -> (loop or) `assemble_chapter`.
*   **Context Retrieval** (`core/langgraph/nodes/context_retrieval_node.py`):
    *   **Character Context**: Profiles of characters in the scene.
    *   **KG Facts**: Reliable facts filtered by scene entities.
    *   **Summaries**: Last 3 chapter summaries.
    *   **Previous Scenes**: Token-aware context from earlier scenes in the same chapter.
    *   **Location**: Details from Neo4j.
    *   **Semantic Search**: Vector search for thematically similar past events.

#### C. Extraction Subgraph (`core/langgraph/subgraphs/extraction.py`)
Parallelizes entity extraction.

*   **Nodes**: `extract_router`, `extract_characters`, `extract_locations`, `extract_events`, `extract_relationships`, `consolidate`.
*   **Logic**: Runs extraction nodes in parallel (using `asyncio.gather` conceptually in LangGraph).
*   **Output**: Populates `state["extracted_entities"]` and `state["extracted_relationships"]`.

#### D. Commit Node (`commit`)
*   **Logic**:
    1.  Deduplicates characters (fuzzy matching against Neo4j).
    2.  Resolves entity IDs.
    3.  Writes nodes and relationships to Neo4j.
    4.  Updates `state["active_characters"]` for the next step.

#### E. Validation Subgraph (`core/langgraph/subgraphs/validation.py`)
Ensures quality and consistency.

*   **Nodes**:
    1.  `validate_consistency`: Checks against graph constraints (e.g., character traits).
    2.  `evaluate_quality`: LLM-based scoring of prose, pacing, and coherence.
    3.  `detect_contradictions`: Checks timeline, world rules, and relationship evolution.
*   **Decision**: Updates `state["contradictions"]` and `state["needs_revision"]`.

#### F. Revision Loop (`revise`)
*   **Condition**: `should_revise_or_continue`.
*   **Logic**: If validation fails (and `iteration_count < max_iterations`), the `revise` node uses an LLM to fix specific contradictions.
*   **Loop**: Returns to `extract` to re-process the revised text.

#### G. Finalization
*   **`summarize`**: Generates a summary of the validated chapter.
*   **`finalize`**: Writes the final markdown file to disk.
*   **`heal_graph`**: Performs graph maintenance (merging provisional nodes).

## 4. Edge Cases & Routing Logic

1.  **Initialization Skip**: If `initialization_complete` is True (e.g., resuming a run), the `route` node jumps directly to `chapter_outline`.
2.  **Revision Limits**: `should_revise_or_continue` checks `iteration_count`. If max iterations are reached, it forces flow to `summarize` to prevent infinite loops, logging a warning.
3.  **Fatal Errors**: `should_handle_error` (used in Phase 2 graph) checks `state["has_fatal_error"]` and routes to `error_handler` to gracefully terminate.
4.  **Force Continue**: `state["force_continue"]` overrides validation failures.

## 5. Context Flow Summary

1.  **Init -> Planning**: Global/Act outlines inform Chapter Outlines.
2.  **Planning -> Generation**: Chapter Outline + KG Context -> Scene Drafts.
3.  **Generation -> Extraction**: Draft Text -> Extracted Entities.
4.  **Extraction -> Commit**: Extracted Entities -> Neo4j Database.
5.  **Commit -> Validation**: Neo4j Data + Draft Text -> Consistency Check.
6.  **Validation -> Revision**: Contradictions -> Revised Draft (if needed).
7.  **Finalization -> Future**: Chapter Summary -> Context for next chapter.

## Conclusion
The architecture is robust, utilizing LangGraph's state persistence to handle long-running narrative generation. The separation of concerns into subgraphs (Generation, Extraction, Validation) makes the system modular and easier to debug. The hybrid context retrieval system ensures the LLM has relevant information without overflowing context windows.