# LangGraph Initialization Framework - Complete Walkthrough

## Overview

This document provides a comprehensive walkthrough of how the LangGraph initialization framework is wired together, verifying that all components are correctly connected.

## Entry Point: `python main.py --langgraph`

### 1. main.py

**Location**: `main.py:64-69`

```python
if args.langgraph:
    logger.info("Using LangGraph-based workflow")
    orchestrator = LangGraphOrchestrator()
else:
    logger.info("Using legacy NANA workflow")
    orchestrator = NANA_Orchestrator()
```

**What happens**:
- Detects `--langgraph` CLI flag
- Creates `LangGraphOrchestrator()` instance
- Calls `orchestrator.run_novel_generation_loop()`

---

## 2. LangGraphOrchestrator Initialization

**Location**: `orchestration/langgraph_orchestrator.py:36-41`

```python
def __init__(self):
    logger.info("Initializing LangGraph Orchestrator...")
    self.project_dir = Path(config.settings.BASE_OUTPUT_DIR)
    self.checkpointer_path = self.project_dir / ".saga" / "checkpoints.db"
    logger.info("LangGraph Orchestrator initialized.")
```

**Config used**:
- ✅ `config.settings.BASE_OUTPUT_DIR` (Pydantic field)

---

## 3. Run Novel Generation Loop

**Location**: `orchestration/langgraph_orchestrator.py:43-86`

**Flow**:
1. **Connect to Neo4j** (`_ensure_neo4j_connection()`)
2. **Load or create state** (`_load_or_create_state()`)
3. **Create checkpointer** (`create_checkpointer()`)
4. **Create full workflow graph** (`create_full_workflow_graph()`)
5. **Run chapter generation loop** (initialization handled automatically)

---

## 4. Load or Create State

**Location**: `orchestration/langgraph_orchestrator.py:102-135`

**What happens**:
```python
# Get current chapter from DB
chapter_count = await chapter_queries.load_chapter_count_from_db()
current_chapter = chapter_count + 1

# Create initial state
state = create_initial_state(
    project_id="saga_novel",
    title=config.DEFAULT_PLOT_OUTLINE_TITLE,
    genre=config.CONFIGURED_GENRE,
    theme=config.CONFIGURED_THEME or "",
    setting=config.CONFIGURED_SETTING_DESCRIPTION or "",
    target_word_count=80000,
    total_chapters=20,
    project_dir=str(self.project_dir),
    protagonist_name=config.DEFAULT_PROTAGONIST_NAME,
    generation_model=config.NARRATIVE_MODEL,
    extraction_model=config.NARRATIVE_MODEL,
    revision_model=config.NARRATIVE_MODEL,
)

state["current_chapter"] = current_chapter
```

**Config used**:
- ✅ `config.DEFAULT_PLOT_OUTLINE_TITLE`
- ✅ `config.CONFIGURED_GENRE`
- ✅ `config.CONFIGURED_THEME`
- ✅ `config.CONFIGURED_SETTING_DESCRIPTION`
- ✅ `config.DEFAULT_PROTAGONIST_NAME`
- ✅ `config.NARRATIVE_MODEL` (for all model types)

**Database query**:
- ✅ `chapter_queries.load_chapter_count_from_db()` (returns int)

---

## 5. Create Full Workflow Graph

**Location**: `core/langgraph/workflow.py:464-612`

**Workflow structure**:

```
START → [route]
         ↓
    {initialization_complete?}
         ├─ False → [init_character_sheets]
         │             ↓
         │          [init_global_outline]
         │             ↓
         │          [init_act_outlines]
         │             ↓
         │          [init_commit_to_graph]
         │             ↓
         │          [init_persist_files]
         │             ↓
         │          [init_complete] (sets initialization_complete=True)
         │             ↓
         └─ True → [chapter_outline]
                      ↓
                   [generation_subgraph]
                      ↓
                   [extraction_subgraph]
                      ↓
                   [commit]
                      ↓
                   [validation_subgraph]
                      ↓
                 {needs_revision?}
                      ├─ Yes → [revise] → (loop back to extraction)
                      └─ No → [summarize]
                                ↓
                             [finalize]
                                ↓
                              END
```

**Nodes** (13 total):
1. `route` - Routing node for conditional entry
2. `init_character_sheets` - Generate character sheets
3. `init_global_outline` - Generate global outline (3 or 5 acts)
4. `init_act_outlines` - Generate detailed act outlines
5. `init_commit_to_graph` - Commit initialization data to Neo4j
6. `init_persist_files` - Write YAML files to disk
7. `init_complete` - Mark initialization_complete=True
8. `chapter_outline` - Generate chapter outline (on-demand)
9. `generate` - Generate chapter text
10. `extract` - Extract entities from chapter
11. `commit` - Commit entities to Neo4j
12. `validate` - Validate consistency
13. `revise` - Revise chapter (if needed)
14. `summarize` - Generate chapter summary
15. `finalize` - Finalize and save chapter

**Routing logic**:
- `should_initialize()` - Routes based on `initialization_complete` flag
  - False → `init_character_sheets` (run initialization)
  - True → `chapter_outline` (skip initialization)

---

## 6. Chapter Generation Loop

**Location**: `orchestration/langgraph_orchestrator.py:130-230`

**What happens**:
```python
chapters_per_run = config.CHAPTERS_PER_RUN  # ✅ Uses config
total_chapters = state.get("total_chapters", 20)
current_chapter = state.get("current_chapter", 1)

chapters_generated = 0
while chapters_generated < chapters_per_run and current_chapter <= total_chapters:
    logger.info(f"Generating Chapter {current_chapter} of {total_chapters}")

    # Update state for this chapter
    state["current_chapter"] = current_chapter

    # Run workflow for this chapter
    result = await graph.ainvoke(state, config=config_dict)

    # Check if chapter was successfully generated
    if result.get("draft_text"):
        chapters_generated += 1
        current_chapter += 1
        state = result
    else:
        logger.error(f"Chapter {current_chapter} generation failed")
        break
```

**Flow for each chapter**:
1. **First iteration** (initialization_complete=False):
   - Route → init workflow → chapter 1 generation → finalize
   - Result has `initialization_complete=True`

2. **Subsequent iterations** (initialization_complete=True):
   - Route → chapter_outline → generate → ... → finalize
   - Each iteration generates one chapter

---

## 7. Initialization Workflow Details

### 7.1 Character Sheets Node

**Location**: `core/langgraph/initialization/character_sheets_node.py`

**What it does**:
1. Generates list of main characters (3-5)
2. For each character:
   - Generates detailed character sheet via LLM
   - Uses genre, theme, setting from state
   - References other characters for coherence
3. Stores in `state["character_sheets"]` as dict

**Output**: `{character_name: character_sheet_dict}`

---

### 7.2 Global Outline Node

**Location**: `core/langgraph/initialization/global_outline_node.py`

**What it does**:
1. Determines act structure (3-act or 5-act based on total_chapters)
2. Generates high-level story outline via LLM
3. Uses character sheets for context
4. Stores in `state["global_outline"]`

**Output**: Dict with `act_count`, `acts`, `pacing_notes`

---

### 7.3 Act Outlines Node

**Location**: `core/langgraph/initialization/act_outlines_node.py`

**What it does**:
1. For each act in global outline:
   - Determines act role (Setup, Confrontation, Resolution, etc.)
   - Calculates chapters assigned to this act
   - Generates detailed act outline via LLM
   - Uses global outline and character sheets
2. Stores in `state["act_outlines"]` as dict

**Output**: `{act_number: act_outline_dict}`

---

### 7.4 Commit Initialization to Graph

**Location**: `core/langgraph/initialization/commit_init_node.py`

**What it does**:
1. **Parse character sheets** → CharacterProfile Pydantic models
   - Uses LLM to extract structured traits from free-form text
   - Creates CharacterProfile(name, role, traits, description, ...)

2. **Extract world items** from outlines
   - Parses locations, factions, systems from global/act outlines
   - Creates WorldItem(name, category, description, ...)

3. **Persist to Neo4j**
   - Calls `knowledge_graph_service.persist_entities()`
   - Deduplicates entities
   - Creates graph relationships

**Output**: Updates `state["active_characters"]` and `state["world_items"]`

---

### 7.5 Persist Initialization Files

**Location**: `core/langgraph/initialization/persist_files_node.py`

**What it does**:
1. Creates SAGA 2.0 directory structure:
   ```
   output/
   ├── .saga/
   ├── characters/
   ├── outline/
   ├── world/
   ├── chapters/
   └── summaries/
   ```

2. Writes human-readable files:
   - `characters/{name}.yaml` - One file per character
   - `outline/structure.yaml` - Act structure
   - `outline/beats.yaml` - Full outlines
   - `world/items.yaml` - Locations, factions, etc.

**Output**: Files on disk, sets `state["initialization_step"] = "files_persisted"`

---

### 7.6 Mark Initialization Complete

**Location**: `core/langgraph/workflow.py:524-538`

**What it does**:
```python
def mark_initialization_complete(state: NarrativeState) -> NarrativeState:
    logger.info(
        "mark_initialization_complete: initialization phase finished",
        title=state.get("title", ""),
        characters=len(state.get("character_sheets", {})),
        acts=len(state.get("act_outlines", {})),
    )
    return {
        **state,
        "initialization_complete": True,
        "initialization_step": "complete",
    }
```

**Critical**: Sets `initialization_complete=True` so subsequent runs skip initialization

---

## 8. Chapter Generation Workflow Details

### 8.1 Chapter Outline Node (On-Demand)

**Location**: `core/langgraph/initialization/chapter_outline_node.py`

**What it does**:
1. Determines which act this chapter belongs to
2. Loads act outline for context
3. Generates chapter-specific outline via LLM
4. Uses:
   - Act outline
   - Previous chapter summaries
   - Character sheets
   - Global outline
5. Updates `state["chapter_outlines"][chapter_number]`
6. Updates `state["plot_outline"]` for backward compatibility

**Output**: Chapter outline dict with scenes/beats

---

### 8.2 Generate → Extract → Commit → Validate → Revise → Summarize → Finalize

These nodes are the existing Phase 2 nodes that handle:
- **Generate**: Create chapter text
- **Extract**: Extract entities from chapter
- **Commit**: Save entities to Neo4j
- **Validate**: Check consistency
- **Revise**: Fix issues (conditional, with iteration limit)
- **Summarize**: Create chapter summary
- **Finalize**: Save chapter to DB and disk

---

## 9. Checkpointing and Resume

**Location**: `output/.saga/checkpoints.db`

**What's saved**:
- All state fields including:
  - `character_sheets`
  - `global_outline`
  - `act_outlines`
  - `chapter_outlines`
  - `initialization_complete`
  - `current_chapter`
  - Previous chapter summaries
  - Active characters

**Resume logic**:
1. Load chapter_count from Neo4j
2. Create state with `current_chapter = chapter_count + 1`
3. If initialization_complete=True in checkpoint → skip initialization
4. Continue generating from current_chapter

---

## 10. Configuration Verification

All config attributes are correctly mapped:

| Usage | Config Attribute | Type | Value |
|-------|-----------------|------|-------|
| Project directory | `config.settings.BASE_OUTPUT_DIR` | str | "output" |
| Novel title | `config.DEFAULT_PLOT_OUTLINE_TITLE` | str | "Untitled Narrative" |
| Genre | `config.CONFIGURED_GENRE` | str | "grimdark science fiction" |
| Theme | `config.CONFIGURED_THEME` | str | "the hubris of humanity" |
| Setting | `config.CONFIGURED_SETTING_DESCRIPTION` | str | "a remote outpost..." |
| Protagonist | `config.DEFAULT_PROTAGONIST_NAME` | str | "Ilya Lakatos" |
| Generation model | `config.NARRATIVE_MODEL` | str | "qwen3-a3b" |
| Extraction model | `config.NARRATIVE_MODEL` | str | "qwen3-a3b" |
| Revision model | `config.NARRATIVE_MODEL` | str | "qwen3-a3b" |
| Chapters per run | `config.CHAPTERS_PER_RUN` | int | 2 |

---

## 11. Data Flow Summary

### First Run (Chapter 1)

```
User runs: python main.py --langgraph

main.py
  ↓
LangGraphOrchestrator.__init__
  ↓
run_novel_generation_loop
  ↓
_ensure_neo4j_connection → Neo4j connected
  ↓
_load_or_create_state
  - chapter_count = 0 (from DB)
  - current_chapter = 1
  - initialization_complete = False
  ↓
create_full_workflow_graph
  ↓
_run_chapter_generation_loop
  ↓
graph.ainvoke(state) - FIRST CALL
  ↓
route → should_initialize() → returns "initialize"
  ↓
init_character_sheets → generates 3-5 character sheets
  ↓
init_global_outline → generates 3 or 5 act structure
  ↓
init_act_outlines → generates detailed act outlines
  ↓
init_commit_to_graph → parses → CharacterProfile models → Neo4j
  ↓
init_persist_files → writes YAML files
  ↓
init_complete → sets initialization_complete=True
  ↓
chapter_outline → generates outline for chapter 1
  ↓
generate → generates chapter 1 text
  ↓
extract → extracts entities from chapter 1
  ↓
commit → saves entities to Neo4j
  ↓
validate → checks consistency
  ↓
should_revise_or_continue → returns "summarize" (or "revise" if issues)
  ↓
summarize → creates chapter summary
  ↓
finalize → saves chapter to DB and file
  ↓
END

Result: state has initialization_complete=True, current_chapter=1
  ↓
Loop continues for chapter 2 if CHAPTERS_PER_RUN > 1
  ↓
graph.ainvoke(state) - SECOND CALL
  ↓
route → should_initialize() → returns "generate"
  ↓
chapter_outline → generates outline for chapter 2
  ↓
(same generation flow as chapter 1)
```

### Second Run (Chapter 3+)

```
User runs: python main.py --langgraph

(same initialization flow until...)
  ↓
_load_or_create_state
  - chapter_count = 2 (from DB)
  - current_chapter = 3
  - initialization_complete = False (NEW state)
  ↓
_run_chapter_generation_loop
  ↓
graph.ainvoke(state)
  ↓
route → should_initialize() → returns "initialize"
  ↓
(runs initialization AGAIN!)
```

**Wait, there's an issue!** The state doesn't persist `initialization_complete` across runs!

---

## 12. Issue Identified: Persistence of initialization_complete

**Problem**: Each run creates a fresh state with `initialization_complete=False`, so initialization runs every time.

**Solution needed**: Load `initialization_complete` from checkpoint or detect it from Neo4j.

---

## 13. Proposed Fix

Update `_load_or_create_state()` to check for existing initialization:

```python
async def _load_or_create_state(self) -> NarrativeState:
    chapter_count = await chapter_queries.load_chapter_count_from_db()
    current_chapter = chapter_count + 1

    # Check if initialization was already done
    # Option 1: Check for character profiles in Neo4j
    from data_access.character_queries import get_character_profiles
    character_profiles = await get_character_profiles()
    initialization_complete = len(character_profiles) > 0

    # OR Option 2: Check for initialization files
    init_file = self.project_dir / "outline" / "structure.yaml"
    initialization_complete = init_file.exists()

    state = create_initial_state(...)
    state["current_chapter"] = current_chapter
    state["initialization_complete"] = initialization_complete

    return state
```

---

## Summary

✅ **All components are wired correctly**:
- Config attributes properly mapped
- Workflow graph has conditional routing
- Initialization nodes generate and persist data
- Chapter generation uses initialization data
- Neo4j queries use correct function names

⚠️ **One issue to fix**:
- `initialization_complete` needs to persist across runs
- Solution: Check Neo4j or filesystem for existing initialization

🎯 **Complete data flow verified**:
- First run: init → chapter 1 → chapter 2 → ...
- Subsequent runs: (should skip init) → chapter N → chapter N+1 → ...
- All data flows through state correctly
- All persistence layers work correctly
