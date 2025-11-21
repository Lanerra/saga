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

# Bootstrap world/characters/plot (standalone)
python main.py --bootstrap
python main.py --bootstrap --bootstrap-phase world  # or characters|plot
python main.py --bootstrap --bootstrap-reset-kg     # destructive: wipes Neo4j first
```

### Testing
```bash
# Run all tests
pytest

# Run specific test markers
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run single test file
pytest tests/test_knowledge_agent.py
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

# Test initialization only
python init_test.py              # full test
python init_test_minimal.py      # minimal test
```

## Architecture Overview

### Orchestration

**LangGraph Pipeline** (`orchestration/langgraph_orchestrator.py`)
- Graph-based workflow with declarative nodes
- Built-in checkpointing (SQLite)
- Automatic resume from interruption
- Defined in `core/langgraph/workflow.py`
- Phase 3: Legacy NANA orchestrator removed

### Core Components

**LangGraph Nodes** (`core/langgraph/nodes/`)
- `generation_node.py`: Chapter drafting from outline and KG context
- `extraction_node.py`: Entity and relationship extraction from generated text
- `commit_node.py`: Deduplication and persistence to Neo4j
- `validation_node.py`: Consistency checking and contradiction detection
- `revision_node.py`: Full chapter regeneration with validation feedback
- `summary_node.py`: Chapter summary generation
- `finalize_node.py`: Persistence to disk and database

**LangGraph Initialization** (`core/langgraph/initialization/`)
- `character_sheets_node.py`: Character sheet generation
- `global_outline_node.py`: 3/5-act story outline generation
- `act_outlines_node.py`: Detailed act-level outlines
- `chapter_outline_node.py`: On-demand chapter outlines
- `commit_init_node.py`: Persist initialization to Neo4j
- `persist_files_node.py`: Write YAML artifacts to disk

**Core Systems** (`core/`)
- `db_manager.py`: Neo4j connection management, schema creation
- `knowledge_graph_service.py`: Entity/relationship CRUD operations
- `llm_interface_refactored.py`: OpenAI-compatible API client for local LLMs
- `simple_type_inference.py`: Lightweight entity type inference
- `relationship_validator.py`: Relationship constraint enforcement
- `langgraph/`: LangGraph workflow definitions, state management, visualization

**Data Access** (`data_access/`)
- Cypher query builders for Neo4j operations
- Repository pattern for entities, relationships, chapters

**Initialization** (`initialization/`)
- `bootstrap_pipeline.py`: Multi-phase bootstrap (world → characters → plot)
- `bootstrappers/`: World, character, and plot generators with validation

**Orchestration** (`orchestration/`)
- `langgraph_orchestrator.py`: LangGraph-based workflow orchestrator (only orchestrator)

**Logging** (`core/`)
- `logging_config.py`: SAGA logging setup with Rich console and file rotation

**Processing** (`processing/`)
- Text processing utilities, embeddings, semantic search

**Models** (`models/`)
- Pydantic models for entities, relationships, validation schemas

**Prompts** (`prompts/`)
- Jinja2 templates for LLM prompts, organized by agent type

**UI** (`ui/`)
- Rich CLI progress panels and live displays

### Neo4j Knowledge Graph

**Node Types**:
- `Character`: name, description, traits, motivations, first_appearance
- `Location`: name, description, rules
- `Event`: description, importance, chapter
- `Chapter`: number, word_count, summary, embedding_vector
- `PlotPoint`, `WorldFact`, `Object`

**Key Relationships**:
- Character relationships: LOVES, HATES, TRUSTS, WORKS_FOR, ALLIES_WITH
- Character-Location: LIVES_IN, VISITED
- Character-Event: PARTICIPATED_IN, CAUSED, WITNESSED
- Event-Chapter: OCCURRED_IN
- Chapter-Chapter: FOLLOWS

**Vector Index**: `chapterEmbeddings` on Chapter nodes for semantic search

## Configuration

Primary config file: `config/settings.py` (uses Pydantic BaseSettings)

Key environment variables (`.env`):
- `OPENAI_API_BASE`: Local LLM endpoint (e.g., `http://127.0.0.1:8080/v1`)
- `EMBEDDING_API_BASE`: Local embedding service (e.g., `http://127.0.0.1:11434`)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection
- `LARGE_MODEL`, `MEDIUM_MODEL`, `SMALL_MODEL`, `NARRATIVE_MODEL`: Model names
- `CHAPTERS_PER_RUN`, `TARGET_CHAPTERS`: Generation behavior
- `BOOTSTRAP_*`: Bootstrap configuration flags

## LangGraph State & Workflow

**State Object** (`core/langgraph/state.py`)
- All workflow state persists automatically via LangGraph checkpointer
- State includes: chapter metadata, character profiles, outlines, draft text, validation results
- Checkpoint location: `output/.saga/checkpoints.db`

**Workflow Nodes** (`core/langgraph/workflow.py`)
- `initialize`: Setup characters, outlines, world-building
- `generate_chapter`: Draft chapter prose from outline + KG context
- `extract_entities`: Parse entities/relationships from generated text
- `commit_to_graph`: Deduplicate and persist to Neo4j
- `validate_consistency`: Check for contradictions
- `revise_chapter`: Fix issues based on validation feedback
- `summarize_chapter`: Generate brief summary
- `finalize_chapter`: Write to disk, advance to next chapter

**Graph Structure**:
```
START → initialize → generate → extract → commit → validate
                        ↑                             ↓
                        └─── revise ← [needs_revision?]
                                        ↓ [valid]
                                     summarize → finalize → [next chapter or END]
```

## Output Structure

```
output/
├── .saga/
│   └── checkpoints.db          # LangGraph state persistence
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
- LangGraph tests: Workflow node execution, state transitions, initialization

**Test Markers** (defined in `pyproject.toml`):
- `unit`: Fast, isolated unit tests
- `integration`: Tests requiring Neo4j/LLM
- `slow`: Long-running tests
- `performance`: Performance benchmarks
- `langgraph`: LangGraph-specific tests

**Test Configuration**:
- `conftest.py`: Shared fixtures for Neo4j connections, mock LLMs
- `--asyncio-mode=auto`: Tests use async/await extensively

## Project Constraints (CRITICAL)

From `docs/PROJECT_CONSTRAINTS.md`:

**Hard Constraints**:
- Single user, single machine only
- No databases beyond Neo4j/file storage
- No web servers, APIs, or network services
- Consumer hardware target
- Local-first architecture

**NOT Needed**:
- Authentication/authorization
- Horizontal scaling
- Microservices, message queues, load balancers
- Container orchestration

**Neo4j Usage**:
- Local embedded instance only
- Think "personal knowledge base" not "web-scale backend"

**LangGraph Node Architecture**:
- Sequential processing pipeline orchestrated by LangGraph state machine
- Nodes are async functions that transform state, not separate processes
- Declarative graph definition with conditional routing

## Development Workflow

### When Adding New Features

1. **Bootstrap Phase**: If adding world/character/plot generation logic, work in `initialization/bootstrappers/`
2. **LangGraph Nodes**: For narrative generation/revision, work in `core/langgraph/nodes/`
3. **Initialization Nodes**: For initialization workflow changes, work in `core/langgraph/initialization/`
4. **KG Operations**: For Neo4j queries/schema changes, work in `data_access/` or `core/knowledge_graph_service.py`
5. **Workflow Changes**: For graph structure/routing, edit `core/langgraph/workflow.py`
6. **Prompts**: Add/edit Jinja2 templates in `prompts/` (organized by node type)

### When Debugging

- Check `output/chapter_logs/` for per-chapter logs
- Inspect `output/debug_outputs/` for saved prompts/validations
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
    temperature=settings.TEMPERATURE_DRAFTING
)
```

**State Updates in LangGraph**: Return modified state dict
```python
def my_node(state: NarrativeState) -> NarrativeState:
    # Process...
    return {**state, "new_field": value}
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
- Assemble into structured prompt context

### Revision Loop
- Max iterations configurable (`MAX_REVISION_CYCLES_PER_CHAPTER`, currently disabled)
- Validation identifies contradictions (character traits, relationships, timeline)
- Full chapter regeneration with validation feedback (not patch-based)
- Revision prompt includes specific issues + suggested fixes
- Re-extract and re-validate after revision

### Checkpoint Resume
- LangGraph checkpointer tracks all state in SQLite
- On restart, loads latest checkpoint and continues from last node
- No manual state reconstruction needed

## Known Issues & Current State

Current state:
- `MAX_REVISION_CYCLES_PER_CHAPTER` defaults to 0 (disabled, broken, being refactored)
- Phase 3 complete: Legacy NANA orchestrator removed, LangGraph is the only pipeline

## Documentation

Key docs in `docs/`:
- `langgraph-architecture.md`: Detailed LangGraph design (comprehensive)
- `LANGGRAPH_USAGE.md`: Usage guide for LangGraph workflow
- `WORKFLOW_WALKTHROUGH.md`: Complete data flow walkthrough
- `PROJECT_CONSTRAINTS.md`: Hard constraints and architectural decisions
- `initialization-framework.md`: Bootstrap pipeline details
- `schema-map.md`: Neo4j schema documentation
- `migrate.md`: NANA→LangGraph migration progress tracker

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
