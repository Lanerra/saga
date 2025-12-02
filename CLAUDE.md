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
- Prefer editing an existing file to creating a new one.
- Never create documentation files (`*.md` or README).
  - Only create documentation files if explicitly requested by the user.


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

### Development Setup
```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit Neo4j and LLM endpoint configuration

# Start Neo4j (if using Docker)
docker-compose up -d
```

### Running SAGA
```bash
# Run generation loop
python main.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run single test file
pytest tests/test_langgraph/test_extraction_node.py

# Run LangGraph tests only
pytest -m langgraph
```

### Code Quality
```bash
# Lint
ruff check .

# Format
ruff format .

# Type checking
mypy .
```

### Useful Utilities
```bash
# Reset Neo4j database (destructive)
python reset_neo4j.py

# Visualize LangGraph workflow
python visualize_workflow.py
```

## Architecture Overview

### Orchestration

**LangGraph Workflow System** (`orchestration/langgraph_orchestrator.py`)
- Graph-based workflow with declarative nodes
- Built-in checkpointing to SQLite (`.saga/checkpoints.db`)
- Automatic resume from interruption
- Workflow definition in `core/langgraph/workflow.py`
- **Legacy NANA orchestrator has been completely removed**

### Core LangGraph Components

**Main Workflow** (`core/langgraph/workflow.py`)
- Entry point: `create_full_workflow_graph()`
- Conditional routing between initialization and generation phases
- Manages chapter generation loop with revision cycles
- Integrates all subgraphs and nodes

**State Management** (`core/langgraph/state.py`)
- `NarrativeState` TypedDict: Central state container
- Automatic persistence after each node
- Content externalization via `ContentRef` to reduce checkpoint bloat
- Includes metadata, progress tracking, extraction results, validation feedback

**Content Manager** (`core/langgraph/content_manager.py`)
- Externalizes large content (drafts, outlines, scene text) to files
- Stores only file references in state to prevent SQLite bloat
- Enables content versioning and efficient diffing
- Reduces checkpoint size from megabytes to kilobytes

**Graph Context** (`core/langgraph/graph_context.py`)
- Manages Neo4j query context for generation nodes
- Retrieves active characters, relationships, events, locations
- Builds hybrid context from KG + previous summaries

**Workflow Visualization** (`core/langgraph/visualization.py`)
- Generates visual graph representations (PNG, SVG)
- Debug state transitions and routing logic
- Command: `python visualize_workflow.py`

**Export System** (`core/langgraph/export.py`)
- Exports complete novel to markdown
- Combines all finalized chapters
- Handles state inspection and debugging

### LangGraph Nodes (`core/langgraph/nodes/`)

**Generation Nodes:**
- `generation_node.py`: Chapter drafting from outline and KG context
- `scene_planning_node.py`: Breaks chapter into discrete scenes
- `scene_generation_node.py`: Generates individual scene text
- `assemble_chapter_node.py`: Combines scene drafts into chapter

**Extraction Nodes:**
- `extraction_node.py`: Orchestrates parallel entity extraction
- `extraction_nodes.py`: Specialized extractors (characters, locations, events, relationships)

**Graph Management:**
- `commit_node.py`: Deduplication and persistence to Neo4j
- `graph_healing_node.py`: Provisional node enrichment and entity merging
- `context_retrieval_node.py`: Retrieves relevant KG context for scenes

**Quality Assurance:**
- `validation_node.py`: Consistency checking and contradiction detection
- `revision_node.py`: Full chapter regeneration with validation feedback
- `summary_node.py`: Chapter summary generation for future context
- `finalize_node.py`: Persistence to disk and database

**Embeddings:**
- `embedding_node.py`: Generates embeddings for semantic search

### LangGraph Initialization (`core/langgraph/initialization/`)

**Initialization Nodes:**
- `character_sheets_node.py`: Generates 3-5 protagonist profiles
- `global_outline_node.py`: Creates 3 or 5-act story structure
- `act_outlines_node.py`: Expands acts into chapter-level beats
- `chapter_outline_node.py`: On-demand detailed chapter scene plans
- `commit_init_node.py`: Persists initialization data to Neo4j
- `persist_files_node.py`: Writes YAML artifacts to disk

**Initialization Workflow** (`core/langgraph/initialization/workflow.py`)
- Entry point: `create_initialization_workflow()`
- Sequential initialization node execution
- Sets `initialization_complete` flag on success

**Validation** (`core/langgraph/initialization/validation.py`)
- Validates character sheets, outlines, and world items
- Ensures initialization artifacts are well-formed

### LangGraph Subgraphs (`core/langgraph/subgraphs/`)

**Generation Subgraph** (`generation.py`)
- Scene planning → context retrieval → scene generation → assembly
- Handles chapter-to-scene decomposition and reassembly

**Extraction Subgraph** (`extraction.py`)
- Parallel execution of specialized extractors
- Consolidates results into unified entity/relationship lists

**Validation Subgraph** (`validation.py`)
- Consistency validation against KG constraints
- Quality scoring and contradiction detection
- Decides if revision is needed

### Core Systems (`core/`)

**Database & Knowledge Graph:**
- `db_manager.py`: Neo4j connection management, schema creation
- `knowledge_graph_service.py`: Entity/relationship CRUD operations
- `graph_healing_service.py`: Entity enrichment, merging, and maintenance

**LLM Interface:**
- `llm_interface_refactored.py`: OpenAI-compatible API client for local LLMs
- Supports multiple model tiers (narrative, reasoning, extraction)
- Integrates GBNF grammar-constrained generation

**Type Inference & Validation:**
- `simple_type_inference.py`: Lightweight entity type inference (replaced `intelligent_type_inference.py`)
- `relationship_validation.py`: Relationship constraint enforcement
- `schema_validator.py`: Schema validation utilities

**Text Processing:**
- `text_processing_service.py`: Chunking, deduplication, text utilities
- `triple_processor.py`: RDF triple parsing and processing

**Logging:**
- `logging_config.py`: Structured logging with Rich console and file rotation

**Other Services:**
- `http_client_service.py`: HTTP client for LLM/embedding endpoints
- `lightweight_cache.py`: Simple caching layer
- `exceptions.py`: Custom exception types

### Data Access Layer (`data_access/`)

**Query Builders:**
- `cypher_builders/`: Native Cypher query construction
- `kg_queries.py`: Knowledge graph query operations

**Repository Pattern:**
- `character_queries.py`: Character-specific queries
- `chapter_queries.py`: Chapter metadata and retrieval
- `plot_queries.py`: Plot point and narrative arc queries
- `world_queries.py`: World-building element queries

### Prompts (`prompts/`)

**Jinja2 Templates:** Organized by workflow phase
- `initialization/`: Character sheets, outlines, story structure
- `knowledge_agent/`: Entity extraction (characters, locations, events, relationships)
- `narrative_agent/`: Scene planning and drafting
- `revision_agent/`: Full chapter rewrite prompts

**Grammar-Based Constrained Generation:**
- `grammars/`: GBNF grammar definitions for structured output
  - `common.gbnf`: Shared grammar rules
  - `extraction.gbnf`: Entity extraction grammars
  - `initialization.gbnf`: Initialization phase grammars
  - `healing.gbnf`: Graph healing grammars
- `grammar_loader.py`: Loads and combines grammar files

**Prompt Utilities:**
- `prompt_data_getters.py`: Context assembly for prompt templates
- `prompt_renderer.py`: Jinja2 template rendering

### Processing (`processing/`)

**Entity Processing:**
- `entity_deduplication.py`: Fuzzy matching and entity merging
- `text_deduplicator.py`: Content deduplication

**Text Processing:**
- `parsing_utils.py`: YAML parsing, entity extraction helpers

### Models (`models/`)

**Pydantic Models:**
- `kg_models.py`: CharacterProfile, WorldItem, entity schemas
- `agent_models.py`: EvaluationResult, SceneDetail, PatchInstruction
- `user_input_models.py`: User story and input validation
- `validation_utils.py`: Validation helper functions
- `kg_constants.py`: Ontology constants and type definitions

### UI (`ui/`)

**Rich CLI Display:**
- `rich_display.py`: Progress panels, live displays, event streaming

### Neo4j Knowledge Graph

**Node Types:**
- `Character`: name, description, traits, motivations, first_appearance
- `Location`: name, description, rules
- `Event`: description, importance, chapter
- `Chapter`: number, word_count, summary, embedding_vector
- `PlotPoint`, `WorldFact`, `Object`

**Key Relationships:**
- Character relationships: LOVES, HATES, TRUSTS, WORKS_FOR, ALLIES_WITH, ENEMIES_WITH, KNOWS
- Character-Location: LIVES_IN, VISITED, LOCATED_IN
- Character-Event: PARTICIPATED_IN, CAUSED, WITNESSED
- Event-Chapter: OCCURRED_IN, MENTIONED_IN
- Chapter-Chapter: FOLLOWS

**Vector Index:** `chapterEmbeddings` on Chapter nodes for semantic search

## Configuration

Primary config file: `config/settings.py` (uses Pydantic BaseSettings)

Key environment variables (`.env`):
- `OPENAI_API_BASE`: Local LLM endpoint (e.g., `http://127.0.0.1:8080/v1`)
- `EMBEDDING_API_BASE`: Local embedding service (e.g., `http://127.0.0.1:11434`)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection
- `LARGE_MODEL`, `MEDIUM_MODEL`, `SMALL_MODEL`, `NARRATIVE_MODEL`: Model names
- `CHAPTERS_PER_RUN`, `TARGET_CHAPTERS`: Generation behavior
- `MAX_REVISION_CYCLES_PER_CHAPTER`: Revision loop limit (currently defaults to 0, disabled)

## LangGraph State & Workflow

**State Object** (`core/langgraph/state.py`)
- All workflow state persists automatically via LangGraph checkpointer
- State includes: character profiles, outlines, draft references, validation results
- Checkpoint location: `output/.saga/checkpoints.db`
- Content externalization via `ContentRef` reduces checkpoint bloat

**Workflow Structure:**
```
Initialization (First Run):
START → route → character_sheets → global_outline → act_outlines →
commit_to_graph → persist_files → initialization_complete

Chapter Generation (Main Loop):
route → chapter_outline → [Generation Subgraph] → generate_embedding →
[Extraction Subgraph] → commit → [Validation Subgraph] →
(revise → extract)* → summarize → finalize → heal_graph → [next chapter or END]

Generation Subgraph:
plan_scenes → retrieve_context → draft_scene (loop per scene) → assemble_chapter

Extraction Subgraph:
extract_characters | extract_locations | extract_events | extract_relationships (parallel) → consolidate

Validation Subgraph:
validate_consistency → evaluate_quality → detect_contradictions → routing decision
```

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
- Unit tests: Individual functions, entity extraction, validation logic
- Integration tests: LangGraph node workflows, Neo4j operations, LLM interactions
- LangGraph-specific tests: `tests/test_langgraph/` (5,764 lines of tests)

**Test Markers** (defined in `pyproject.toml`):
- `unit`: Fast, isolated unit tests
- `integration`: Tests requiring Neo4j/LLM
- `slow`: Long-running tests
- `performance`: Performance benchmarks
- `langgraph`: LangGraph-specific tests

**Test Configuration:**
- `conftest.py` and `tests/test_langgraph/conftest.py`: Shared fixtures
- `--asyncio-mode=auto`: Tests use async/await extensively

**Key Test Files:**
- `tests/test_langgraph/test_workflow.py`: End-to-end workflow tests
- `tests/test_langgraph/test_extraction_node.py`: Extraction logic tests
- `tests/test_langgraph/test_generation_node.py`: Generation node tests
- `tests/test_langgraph/test_commit_node.py`: Commit and deduplication tests
- `tests/test_langgraph/test_validation_node.py`: Validation logic tests
- `tests/test_langgraph/test_revision_node.py`: Revision flow tests
- `tests/test_langgraph/test_state.py`: State management tests
- `tests/test_initialization_gbnf.py`: Grammar-constrained initialization tests
- `tests/test_extraction_healing_gbnf.py`: Grammar-constrained extraction tests

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

**LangGraph Node Architecture:**
- Sequential processing pipeline orchestrated by LangGraph state machine
- Nodes are async functions that transform state, not separate processes
- Declarative graph definition with conditional routing

## Development Workflow

### When Adding New Features

1. **Initialization Phase**: Add nodes to `core/langgraph/initialization/` and update `core/langgraph/initialization/workflow.py`
2. **Generation Nodes**: Add nodes to `core/langgraph/nodes/` and wire into `core/langgraph/workflow.py`
3. **Subgraphs**: For complex multi-node features, create subgraphs in `core/langgraph/subgraphs/`
4. **KG Operations**: For Neo4j queries/schema changes, work in `data_access/` or `core/knowledge_graph_service.py`
5. **Workflow Changes**: For graph structure/routing, edit `core/langgraph/workflow.py`
6. **Prompts**: Add/edit Jinja2 templates in `prompts/` (organized by phase)
7. **Grammars**: For constrained generation, add GBNF grammars to `prompts/grammars/`

### When Debugging

- Check `output/chapter_logs/` for per-chapter logs
- Inspect `output/debug_outputs/` for saved prompts/validations
- Review externalized content in `output/.saga/content/`
- Use `python visualize_workflow.py` to see LangGraph state transitions
- Query Neo4j directly: `MATCH (n) RETURN n LIMIT 25`
- Enable structlog debug logging in code

### Common Patterns

**Neo4j Sessions**: Always use context manager
```python
async with neo4j_manager.get_session() as session:
    result = await session.run(query, params)
```

**LLM Calls**: Use `core/llm_interface_refactored.py`
```python
from core.llm_interface_refactored import call_llm_async
response = await call_llm_async(
    prompt=prompt,
    model=settings.NARRATIVE_MODEL,
    temperature=settings.TEMPERATURE_DRAFTING,
    grammar=grammar  # Optional: GBNF grammar for constrained generation
)
```

**State Updates in LangGraph**: Return modified state dict
```python
def my_node(state: NarrativeState) -> NarrativeState:
    # Process...
    return {**state, "new_field": value}
```

**Content Externalization**: Use ContentManager for large content
```python
from core.langgraph.content_manager import ContentManager
manager = ContentManager(project_dir)
ref = manager.save_text(content, "draft", chapter_num)
# Store ref in state instead of content itself
return {**state, "draft_ref": ref}
```

## Important Implementation Details

### Entity Deduplication
- Exact name match (case-insensitive)
- Fuzzy matching (Levenshtein distance, SequenceMatcher)
- Embedding similarity for semantic matching
- Always conservative: create new entity if ambiguous

### Context Assembly
- Query Neo4j for active characters, relationships, recent events
- Retrieve last N chapter summaries (default: 5)
- Pull world/location rules if applicable
- Assemble into structured prompt context via `graph_context.py`

### Revision Loop
- Max iterations configurable (`MAX_REVISION_CYCLES_PER_CHAPTER`, currently defaults to 0)
- Validation identifies contradictions (character traits, relationships, timeline)
- Full chapter regeneration with validation feedback (not patch-based)
- Revision prompt includes specific issues + suggested fixes
- Re-extract and re-validate after revision

### Checkpoint Resume
- LangGraph checkpointer tracks all state in SQLite
- On restart, loads latest checkpoint and continues from last node
- No manual state reconstruction needed
- Content externalization prevents checkpoint bloat

### Content Externalization
- Large content (drafts, outlines, scene text) stored in `.saga/content/`
- State contains only `ContentRef` objects with file paths
- Reduces SQLite checkpoint size from megabytes to kilobytes
- Enables efficient versioning and diffing

### Grammar-Based Generation (GBNF)
- Constrains LLM output to valid structured formats
- Used for initialization (character sheets, outlines)
- Used for extraction (entities, relationships)
- Grammar files in `prompts/grammars/`, loaded via `grammar_loader.py`

## Known Issues & Current State

Current state:
- `MAX_REVISION_CYCLES_PER_CHAPTER` defaults to 0 (disabled, being refactored)
- Legacy NANA orchestrator completely removed
- LangGraph is the only orchestration system

## Documentation

Key docs in `docs/`:
- `langgraph-architecture.md`: Detailed LangGraph design and architecture
- `WORKFLOW_WALKTHROUGH.md`: Complete data flow walkthrough
- `WORKFLOW_VISUALIZATION.md`: Visual representation of workflow graphs
- `PROJECT_CONSTRAINTS.md`: Hard constraints and architectural decisions
- `schema-map.md`: Neo4j schema documentation
- `ontology.md`: Entity type ontology and relationship types
- `content-externalization-implementation.md`: Content externalization design
- `gbnf-implementation-plan.md`: Grammar-constrained generation design
- `proper-noun-preference.md`: Entity name handling guidelines
- `complexity-hotspots.md`: Performance and complexity analysis
- `critcodeanalysis.md`: Critical code analysis and technical debt

## When Working on This Codebase

1. **Respect the local-first constraint**: No web frameworks, no auth, no scaling
2. **Use existing patterns**: Follow LangGraph node structure, Neo4j session management, LLM interface
3. **Test with pytest**: Write tests for new logic, especially KG operations and LangGraph nodes
4. **Check Neo4j schema**: Don't introduce new node/relationship types without constraint updates
5. **Use type hints**: Codebase uses mypy strict mode (`disallow_untyped_defs = true`)
6. **Log with structlog**: Use structured logging, not print statements
7. **Configuration over hardcoding**: Use `config/settings.py` for tunable parameters
8. **Async/await**: Most operations are async; use `asyncio.run()` for entry points
9. **State immutability**: LangGraph nodes should return new state dicts, not mutate existing state
10. **Content externalization**: For large content, use ContentManager instead of storing in state
11. **GBNF grammars**: For structured output, define grammars in `prompts/grammars/`
12. **Scene-level generation**: New chapters use scene-by-scene generation, not monolithic drafting
