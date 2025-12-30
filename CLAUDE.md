# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎨 Code Style

- Don't write forgiving code
  - Don't permit multiple input formats
    - In TypeScript, this means avoiding Union Type (the `|` in types)
  - Use preconditions
    - Use schema libraries
    - Assert that inputs match expected formats
    - When expectations are violated, throw, don't log
  - Don't add defensive try/catch blocks
    - Usually we let exceptions propagate out
- Don't use abbreviations or acronyms
  - Choose `number` instead of `num` and `greaterThan` instead of `gt`
- Emoji and unicode characters are welcome
  - Use them at the beginning of comments, commit messages, and in headers in docs
- Use comments sparingly
- Don't comment out code
  - Remove it instead
- Don't add comments that describe the process of changing code
  - Comments should not include past tense verbs like added, removed, or changed
  - Example: `this.timeout(10_000); // Increase timeout for API calls`
  - This is bad because a reader doesn't know what the timeout was increased from, and doesn't care about the old behavior
- Don't add comments that emphasize different versions of the code, like "this code now handles"
- Do not use end-of-line comments
  - Place comments above the code they describe
- Prefer editing an existing file to creating a new one
- Never create documentation files (`*.md` or README)
  - Only create documentation files if explicitly requested by the user


## 🧪 Tests

- Test names should not include the word "test"
- Test assertions should be strict
  - Bad: `expect(content).to.include('valid-content')`
  - Better: `expect(content).to.equal({ key: 'valid-content' })`
  - Best: `expect(content).to.deep.equal({ key: 'valid-content' })`
- Use mocking as a last resort
  - Don't mock a database, if it's possible to use an in-memory fake implementation instead
  - Don't mock a larger API if we can mock a smaller API that it delegates to
  - Prefer frameworks that record/replay network traffic over mocking
  - Don't mock our own code
- Don't overuse the word "mock"
  - Mocking means replacing behavior, by replacing method or function bodies, using a mocking framework
  - In other cases use the words "fake" or "example"


## Project Overview

SAGA (Semantic And Graph-enhanced Authoring) is a local-first Python CLI application for AI-driven long-form fiction generation using Neo4j knowledge graphs and LangGraph workflow orchestration.

**Core Philosophy**: Single-user, single-machine, local-first. No web servers, microservices, or distributed systems.

## Essential Commands

### Searching the Codebase
```bash
# Use ripgrep (rg) for all code searches - required for 50K+ line codebase
rg "pattern"                          # Search all files
rg "pattern" -t py                    # Search Python files only
rg "pattern" -g "*.py"                # Glob pattern filter
rg "class MyClass" -A 5               # Show 5 lines after match
rg "def method" --files-with-matches  # List files only
```

## Architecture Overview

### Orchestration

**LangGraph Orchestrator** (`orchestration/langgraph_orchestrator.py`)
- Manages workflow execution, LLM lifecycle, and checkpointing
- Built-in checkpointing to SQLite (`output/.saga/checkpoints.db`)
- Automatic resume from interruption
- Workflow definition in `core/langgraph/workflow.py`

### Core LangGraph Components

**Main Workflow** (`core/langgraph/workflow.py`)
- Entry point: `create_full_workflow_graph()`
- 20 nodes total with conditional routing
- Two phases: initialization and chapter generation
- Integrates all subgraphs and nodes

**State Management** (`core/langgraph/state.py`)
- `NarrativeState` TypedDict: Central state container
- Content externalization via `ContentRef` to reduce checkpoint bloat
- Includes: metadata, progress tracking, extraction results, validation feedback, quality scores, error handling

**Content Manager** (`core/langgraph/content_manager.py`)
- Externalizes large content (drafts, outlines, scene text) to files in `.saga/content/`
- Stores only file references in state to prevent SQLite bloat
- Reduces checkpoint size from megabytes to kilobytes

**Graph Context** (`core/langgraph/graph_context.py`)
- Manages Neo4j query context for generation nodes
- Retrieves active characters, relationships, events, locations
- Builds hybrid context from KG + previous summaries

**Workflow Visualization** (`core/langgraph/visualization.py`)
- Generates visual graph representations (PNG, SVG)
- Debug state transitions and routing logic

**Export System** (`core/langgraph/export.py`)
- Exports complete novel to markdown
- Combines all finalized chapters

### LangGraph Nodes (`core/langgraph/nodes/`)

**Generation Nodes:**
- `scene_planning_node.py`: Breaks chapter into 4-6 discrete scenes
- `scene_generation_node.py`: Generates individual scene text (loops per scene)
- `assemble_chapter_node.py`: Combines scene drafts into chapter
- `context_retrieval_node.py`: Retrieves relevant KG context for scenes
- `generation_node.py`: Legacy single-shot chapter drafting (deprecated)

**Extraction Nodes:**
- `extraction_nodes.py`: Consolidates and externalizes extraction results
- Scene-level extraction handled by `subgraphs/scene_extraction.py`

**Graph Management:**
- `commit_node.py`: Pre-commit deduplication and atomic Neo4j writes
- `relationship_normalization_node.py`: Normalizes relationship types to canonical vocabulary
- `graph_healing_node.py`: Provisional node enrichment, graduation, merging, cleanup

**Quality Assurance:**
- `validation_node.py`: Consistency checking and contradiction detection
- `quality_assurance_node.py`: Periodic QA checks (every N chapters)
- `revision_node.py`: Full chapter regeneration with validation feedback
- `summary_node.py`: Chapter summary generation for future context
- `finalize_node.py`: Persistence to disk and database

**Embeddings:**
- `embedding_node.py`: Generates embeddings for semantic search

### LangGraph Initialization (`core/langgraph/initialization/`)

**Initialization Nodes:**
- `character_sheets_node.py`: Generates 3-5 protagonist/antagonist/supporting profiles
- `global_outline_node.py`: Creates 3 or 5-act story structure
- `act_outlines_node.py`: Expands acts into chapter-level beats
- `chapter_outline_node.py`: On-demand detailed chapter scene plans
- `commit_init_node.py`: Persists initialization data to Neo4j
- `persist_files_node.py`: Writes YAML artifacts to disk

**Supporting Modules:**
- `workflow.py`: Initialization workflow orchestration
- `validation.py`: Validates character sheets, outlines, and world items
- `chapter_allocation.py`: Chapter allocation utilities

### LangGraph Subgraphs (`core/langgraph/subgraphs/`)

**Generation Subgraph** (`generation.py`)
```
plan_scenes → retrieve_context → draft_scene (loop) → assemble_chapter
```
- Scene-by-scene generation with context retrieval per scene

**Scene Extraction Subgraph** (`scene_extraction.py`)
```
extract_from_scenes → consolidate
```
- Processes each scene individually for entity extraction
- Consolidates and externalizes results

**Validation Subgraph** (`validation.py`)
```
validate_consistency → evaluate_quality → detect_contradictions
```
- LLM-based quality evaluation with 5 quality scores
- Contradiction detection: timeline violations, world rule violations, relationship evolution

### Core Systems (`core/`)

**Database & Knowledge Graph:**
- `db_manager.py`: Neo4j connection management, schema creation
- `knowledge_graph_service.py`: Entity/relationship CRUD operations
- `graph_healing_service.py`: Entity enrichment, merging, graduation, and cleanup

**Relationship Management:**
- `relationship_normalization_service.py`: Normalizes relationship types to canonical vocabulary
- `relationship_validation.py`: Relationship constraint enforcement

**LLM Interface:**
- `llm_interface_refactored.py`: OpenAI-compatible API client for local LLMs
  - Three-layer service: Completion, Embedding, combined RefactoredLLMService
  - Supports JSON object/array parsing with retries
  - Fallback model support on primary failure

**Entity & Embedding Services:**
- `entity_embedding_service.py`: Entity embedding generation and persistence
- `schema_validator.py`: Schema validation utilities

**Text Processing:**
- `text_processing_service.py`: Token counting, text cleaning, chunking

**Logging:**
- `logging_config.py`: Structured logging with Rich console and file rotation

**Other Services:**
- `http_client_service.py`: HTTP client for LLM/embedding endpoints
- `lightweight_cache.py`: Simple caching layer
- `exceptions.py`: Custom exception types

### Data Access Layer (`data_access/`)

**Query Builders:**
- `cypher_builders/native_builders.py`: Native Cypher query construction
- `kg_queries.py`: Knowledge graph query operations

**Repository Pattern:**
- `character_queries.py`: Character-specific queries
- `chapter_queries.py`: Chapter metadata and retrieval
- `plot_queries.py`: Plot point and narrative arc queries
- `world_queries.py`: World-building element queries
- `cache_coordinator.py`: Query result caching

### Prompts (`prompts/`)

**Jinja2 Templates:** Organized by workflow phase
- `initialization/`: Character sheets, outlines, story structure (5 templates)
- `knowledge_agent/`: Entity extraction, enrichment, summarization, disambiguation (9 templates)
- `narrative_agent/`: Scene planning and drafting (4 templates)
- `revision_agent/`: Full chapter rewrite prompts
- `validation_agent/`: Quality evaluation template

**Prompt Utilities:**
- `prompt_data_getters.py`: Context assembly for prompt templates
- `prompt_renderer.py`: Jinja2 template rendering

### Processing (`processing/`)

- `entity_deduplication.py`: Fuzzy matching and entity merging
- `text_deduplicator.py`: Content deduplication
- `parsing_utils.py`: YAML parsing, entity extraction helpers

### Models (`models/`)

**Pydantic Models:**
- `kg_models.py`: CharacterProfile, WorldItem, entity schemas
- `agent_models.py`: EvaluationResult, SceneDetail, PatchInstruction
- `user_input_models.py`: User story and input validation
- `validation_utils.py`: Validation helper functions
- `kg_constants.py`: Ontology constants and type definitions
- `db_extraction_utils.py`: Database extraction utilities

### UI (`ui/`)

- `rich_display.py`: Rich CLI display, progress panels, event streaming

### Neo4j Knowledge Graph

**Node Types:**
- `Character`: name, description, traits, motivations, first_appearance, is_provisional, confidence
- `Location`: name, description, rules
- `Event`: description, importance, chapter
- `Chapter`: number, word_count, summary, embedding_vector
- `PlotPoint`, `WorldFact`, `Object`, `Item`

**Key Relationships:**
- Character relationships: LOVES, HATES, TRUSTS, WORKS_FOR, ALLIES_WITH, ENEMIES_WITH, KNOWS
- Character-Location: LIVES_IN, VISITED, LOCATED_IN
- Character-Event: PARTICIPATED_IN, CAUSED, WITNESSED
- Event-Chapter: OCCURRED_IN, MENTIONED_IN
- Chapter-Chapter: FOLLOWS

**Vector Indexes:**
- `chapterEmbeddings` on Chapter nodes
- Entity embedding indexes for Character, Location, Item, Event nodes

## Configuration

Primary config file: `config/settings.py` (uses Pydantic BaseSettings)

## LangGraph State & Workflow

**State Object** (`core/langgraph/state.py`)
- All workflow state persists automatically via LangGraph checkpointer
- Checkpoint location: `output/.saga/checkpoints.db`
- Content externalization via `ContentRef` reduces checkpoint bloat

**Key State Categories:**
- Project metadata: `project_id`, `title`, `genre`, `theme`, `setting`
- Position tracking: `current_chapter`, `total_chapters`, `current_act`
- Generated content: `draft_ref`, `embedding_ref`, `draft_word_count`
- Extraction results: `extracted_entities`, `extracted_relationships` (with `*_ref` alternatives)
- Quality scores: `coherence_score`, `prose_quality_score`, `plot_advancement_score`, `pacing_score`, `tone_consistency_score`
- Validation: `contradictions`, `needs_revision`, `revision_feedback`
- Workflow control: `iteration_count`, `max_iterations`, `force_continue`
- Error handling: `last_error`, `has_fatal_error`, `workflow_failed`, `error_node`
- Graph healing: `nodes_graduated`, `nodes_merged`, `nodes_enriched`, `nodes_removed`

**Workflow Structure:**
```
Initialization (First Run):
START → route → character_sheets → global_outline → act_outlines →
commit_to_graph → persist_files → init_complete → chapter_outline

Chapter Generation (Main Loop):
chapter_outline → [Generation Subgraph] → gen_embedding → [Scene Extraction Subgraph] →
normalize_relationships → commit → [Validation Subgraph] →
(revise → extract)* → summarize → finalize → heal_graph → check_quality → [next chapter or END]

Generation Subgraph:
plan_scenes → retrieve_context → draft_scene (loop per scene) → assemble_chapter

Scene Extraction Subgraph:
extract_from_scenes → consolidate

Validation Subgraph:
validate_consistency → evaluate_quality → detect_contradictions → routing decision
```

**Routing Functions:**
- `should_initialize()`: Checks `initialization_complete` flag
- `should_continue_init()`: Error handling between init nodes
- `should_revise_or_continue()`: Checks `needs_revision`, `iteration_count`, `max_iterations`
- `should_generate_chapter_outline()`: On-demand outline generation
- `should_continue_scenes()`: Loop control for scene generation

## Output Structure

```
output/
├── .saga/
│   ├── checkpoints.db          # LangGraph state persistence
│   ├── content/                # Externalized draft text, outlines, scene text
│   └── logs/
├── characters/
│   └── *.yaml                  # Character profiles (human-readable)
├── outline/
│   ├── structure.yaml          # Act structure
│   └── beats.yaml              # Detailed outlines
├── world/
│   └── items.yaml              # Locations, objects
├── chapters/
│   └── chapter_*.md            # Final chapter text
├── chapter_logs/
│   └── chapter_*.log           # Per-chapter generation logs
├── summaries/
│   └── *.txt                   # Chapter summaries
├── debug_outputs/
│   └── *.json                  # Debug artifacts (prompts, validations)
└── exports/
    └── novel_full.md           # Compiled manuscript
```

## Testing Strategy

**Test Organization** (`tests/`)
- 77 test files across unit, integration, and LangGraph-specific tests
- `tests/test_langgraph/`: 40+ files for LangGraph node and workflow tests
- `tests/core/`: Core module tests

**Test Markers** (defined in `pyproject.toml`):
- `unit`: Fast, isolated unit tests
- `integration`: Tests requiring Neo4j/LLM
- `slow`: Long-running tests
- `performance`: Performance benchmarks
- `langgraph`: LangGraph-specific tests

**Test Configuration:**
- `conftest.py` and `tests/test_langgraph/conftest.py`: Shared fixtures
- `--asyncio-mode=auto`: Tests use async/await extensively

## Project Constraints (CRITICAL)

From `docs/PROJECT_CONSTRAINTS.md`:

**Hard Constraints:**
- Single user, single machine only
- No databases beyond Neo4j/file storage
- No web servers, APIs, or network services
- Consumer hardware target
- Local-first architecture

**NOT Needed:**
- Authentication/authorization
- Horizontal scaling
- Microservices, message queues, load balancers
- Container orchestration

**Neo4j Usage:**
- Local embedded instance only
- Think "personal knowledge base" not "web-scale backend"

## Development Workflow

### When Adding New Features

1. **Initialization Phase**: Add nodes to `core/langgraph/initialization/` and update `core/langgraph/initialization/workflow.py`
2. **Generation Nodes**: Add nodes to `core/langgraph/nodes/` and wire into `core/langgraph/workflow.py`
3. **Subgraphs**: For complex multi-node features, create subgraphs in `core/langgraph/subgraphs/`
4. **KG Operations**: For Neo4j queries/schema changes, work in `data_access/` or `core/knowledge_graph_service.py`
5. **Workflow Changes**: For graph structure/routing, edit `core/langgraph/workflow.py`
6. **Prompts**: Add/edit Jinja2 templates in `prompts/` (organized by phase)

## Important Implementation Details

### Entity Deduplication
- Exact name match (case-insensitive)
- Fuzzy matching (Levenshtein distance, SequenceMatcher)
- Optional embedding similarity for semantic matching
- Phase 2 deduplication: relationship-aware merging

### Graph Healing
- Provisional nodes: `is_provisional = true` until graduated
- Confidence scoring: relationships (40%), attributes (20-30%), age (20%)
- Graduation threshold: 0.75 confidence, age ≥ 1 chapter
- Merge detection: name similarity + optional embedding similarity
- Orphan cleanup: nodes with no relationships after 3 chapters

### Validation & Revision
- Max iterations configurable (`MAX_REVISION_CYCLES_PER_CHAPTER`, default: 2)
- Validation identifies contradictions (character traits, relationships, timeline)
- Full chapter regeneration with validation feedback (not patch-based)
- Quality scores below 0.7 threshold trigger revision

### Content Externalization
- Large content stored in `.saga/content/`
- State contains only `ContentRef` objects with file paths
- Reduces SQLite checkpoint size from megabytes to kilobytes

## Documentation

Key docs in `docs/`:
- `langgraph-architecture.md`: Detailed LangGraph design and architecture
- `WORKFLOW_WALKTHROUGH.md`: Complete data flow walkthrough
- `WORKFLOW_VISUALIZATION.md`: Visual representation of workflow graphs
- `PROJECT_CONSTRAINTS.md`: Hard constraints and architectural decisions
- `critical-audit.md`: Critical code analysis and technical debt

## When Working on This Codebase

1. **Use ripgrep (`rg`) for searching**: Never use `grep` - codebase is 50K+ lines and ripgrep is required for efficient searching
2. **Respect the local-first constraint**: No web frameworks, no auth, no scaling
3. **Use existing patterns**: Follow LangGraph node structure, Neo4j session management, LLM interface
4. **Test with pytest**: Write tests for new logic, especially KG operations and LangGraph nodes
5. **Check Neo4j schema**: Don't introduce new node/relationship types without constraint updates
6. **Use type hints**: Codebase uses mypy strict mode (`disallow_untyped_defs = true`)
7. **Log with structlog**: Use structured logging, not print statements
8. **Configuration over hardcoding**: Use `config/settings.py` for tunable parameters
9. **Async/await**: Most operations are async; use `asyncio.run()` for entry points
10. **State immutability**: LangGraph nodes should return partial state dicts, not mutate existing state
11. **Content externalization**: For large content, use ContentManager instead of storing in state
12. **Scene-level generation**: Chapters use scene-by-scene generation, not monolithic drafting
13. **Graceful degradation**: Non-critical operations (healing, embeddings) should fail gracefully
