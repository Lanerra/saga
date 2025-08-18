# Technical Context

## Core Technologies
- **Python 3.10+**: Primary language for all agent implementations
- **Neo4j Graph Database**: Used for knowledge graph storage (via `kg_queries.py`)
- **Jinja2 Templates**: For prompt management across agents (`prompts/` directory)
- **Git Version Control**: Repository at `/home/dlewis3/Desktop/AI/saga`

## Development Setup
- **Project Structure**:
  - Agents organized by function in `agents/`
  - Prompts grouped by agent type in `prompts/{agent_name}/`
  - Data access layer in `data_access/` with Cypher builders

- **Output Management**:
  - All persistent outputs stored in `novel_output/`
  - Log files configured via `config.py` (e.g., `LOG_FILE = "novel_output/saga_run.log"`)

## Key Technical Decisions
1. **Agent Consolidation**:
   - Merged `kg_maintainer_agent` into `knowledge_agent` to reduce complexity
   - Eliminated redundant modules while preserving functionality

2. **Naming Standardization**:
   - Adopted `[agent_name]_MODEL` pattern for all agent-specific constants
   - Example: `NARRATIVE_MODEL` replaces `NARRATOR_MODEL`

3. **Output Organization**:
   - Implemented directory-based output structure (`novel_output/`)
   - Ensured all log paths updated to reflect new location

## Dependency Notes
- **Backward Compatibility**: 
  - All existing workflow paths maintained via adapter patterns in `finalize_agent.py`
  - No changes required to external integration points

- **Test Coverage**:
  - All refactored modules have corresponding test updates in `tests/`
  - Critical path validation completed (see `test_kg_heal.py`, `test_novel_generation_dynamic.py`)

## Implementation Patterns
- **Unified Data Flow**: 
  - Chapter generation follows consistent pattern: planning → drafting → revision → knowledge update
  - All phases track usage data for performance monitoring

- **Modular Design**:
  - Each agent has clear responsibilities with well-defined interfaces
  - Dependencies are explicitly managed through import statements
