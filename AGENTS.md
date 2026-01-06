# AGENTS.md

This file provides guidance to agents when working with code in this repository.

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

## Configuration

Primary config file: `config/settings.py` (uses Pydantic BaseSettings)

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

## Documentation

Key docs in `docs/`:
- `bootstrapper.md`: Documentation for SAGA's bootstrap mode
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
