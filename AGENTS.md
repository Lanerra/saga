# AGENTS.md - SAGA Novel Generation System

This file provides guidance for OpenAI Codex and other AI agents working with the SAGA codebase.

## Project Overview

**SAGA** is an autonomous, agentic creative-writing system designed to generate entire novels. Powered by the NANA engine, SAGA leverages Large Language Models (LLMs), narrative context through embeddings, and a Neo4j graph database to create rich, coherent, and evolving narratives.

**Core Technologies**: Python 3.9+, Neo4j (Graph Database), Cypher (Query Language), OpenAI-compatible LLM APIs, Ollama (Embeddings), Docker, Pydantic (Configuration)

## Project Structure

```
/
├── agents/                           # Specialized AI agents
│   ├── planner_agent.py             # Scene-by-scene planning
│   ├── drafting_agent.py            # Initial prose generation
│   ├── comprehensive_evaluator_agent.py  # Draft evaluation
│   ├── patch_validation_agent.py    # Patch instruction validation
│   ├── kg_maintainer_agent.py       # Knowledge graph management
│   └── finalize_agent.py            # Chapter finalization
├── core/                            # Core infrastructure
│   ├── db_manager.py                # Neo4j database management
│   └── llm_interface.py             # LLM API abstraction
├── data_access/                     # Database access layer
│   ├── chapter_queries.py           # Chapter-related database operations
│   └── world_queries.py             # World data queries
├── orchestration/                   # Main orchestration
│   ├── nana_orchestrator.py         # Primary orchestrator
│   └── chapter_flow.py              # Chapter generation flow
├── processing/                      # Text processing pipeline
│   ├── revision_manager.py          # Coordinates revisions
│   ├── patch_generator.py           # Generates targeted patches
│   └── text_deduplicator.py         # Reduces repetitive content
├── chapter_generation/              # Chapter generation services
│   ├── context_orchestrator.py      # Provider-based context system
│   └── context_kg_utils.py          # KG context helpers
├── initialization/                  # Story setup and genesis
│   ├── genesis.py                   # Initial story generation
│   ├── models.py                    # Story element models
│   └── bootstrapper/                # Plot/character/world generation
├── config.py                       # Pydantic configuration management
├── main.py                         # Entry point
├── user_story_elements.yaml         # User-provided story elements
├── requirements.txt                 # Python dependencies
├── docker-compose.yml              # Neo4j container setup
└── .env                            # Environment configuration
```

## Coding Standards

### Python Conventions
- **Python Version**: 3.9+ required
- **File Headers**: All source files must include their relative path as a comment at the top (e.g., `# agents/drafting_agent.py`)
- **Code Style**: Use `ruff` for formatting and linting (replaces Black/flake8)
- **Type Hints**: Mandatory for all function parameters and return values
- **Docstrings**: Mandatory Google-style docstrings for all classes and methods
- **Import Organization**: Group imports (stdlib, third-party, local) with blank lines
- **Variable Naming**: Descriptive snake_case names, avoid abbreviations
- **Class Naming**: PascalCase for classes, especially agent classes ending in "Agent"

```python
# agents/kg_maintainer_agent.py
"""Knowledge graph maintenance agent for SAGA."""

import structlog
from typing import Dict, Any

logger = structlog.get_logger(__name__)

class KGMaintainerAgent:
    """Manages the novel's evolving knowledge graph in Neo4j.
    
    Handles chapter summarization, knowledge extraction, and periodic
    healing cycles to resolve duplicates and enrich sparse data.
    """
    
    def extract_knowledge_from_chapter(
        self, 
        chapter_text: str, 
        chapter_number: int
    ) -> Dict[str, Any]:
        """Extract new knowledge from finalized chapter text.
        
        Args:
            chapter_text: The final approved chapter content
            chapter_number: Sequential chapter number for context
            
        Returns:
            Dictionary containing extracted entities, relationships, and updates
        """
        logger.info(
            "Starting knowledge extraction",
            chapter_number=chapter_number,
            text_length=len(chapter_text)
        )
```

### Agent Architecture Patterns
- **Agent Classes**: All agents should inherit from a base agent pattern
- **Method Naming**: Use descriptive action verbs (e.g., `plan_chapter`, `evaluate_draft`, `extract_knowledge`)
- **Error Handling**: Implement robust error handling for LLM API failures and Neo4j connection issues
- **Logging**: Use `structlog` for all logging operations with structured logging patterns
- **Configuration**: Access configuration through the centralized `config.py` system

```python
# agents/revision_agent.py
"""Chapter revision agent for SAGA."""

import structlog
from typing import Dict, Any

logger = structlog.get_logger(__name__)

def process_chapter_draft(
    self,
    draft_text: str,
    context_data: Dict[str, Any],
    plot_focus: str
) -> ProcessingResult:
    """Process a chapter draft through the agent's pipeline.
    
    Args:
        draft_text: Raw chapter content to process
        context_data: Contextual information from knowledge graph
        plot_focus: Current plot point focus for the chapter
        
    Returns:
        ProcessingResult with success status and any generated content
        
    Raises:
        LLMAPIError: When LLM API calls fail
        Neo4jConnectionError: When database operations fail
    """
    logger.info(
        "Processing chapter draft",
        draft_length=len(draft_text),
        plot_focus=plot_focus,
        context_keys=list(context_data.keys())
    )
```

### Neo4j/Cypher Best Practices
- **Query Organization**: Store complex queries in the `data_access/` module
- **Parameter Binding**: Always use parameterized queries (`$parameter_name`)
- **Node Labels**: Use PascalCase (e.g., `Character`, `PlotPoint`, `WorldElement`)
- **Relationship Types**: Use SCREAMING_SNAKE_CASE (e.g., `DEVELOPS_INTO`, `APPEARS_IN`, `RELATES_TO`)
- **Property Names**: Use camelCase for consistency with the existing schema
- **Vector Operations**: Handle embeddings through the established vector index system
- **Entity Normalization**: Strip leading articles ("the", "a", "an") before ID creation

```python
# Good Cypher query pattern for SAGA
GET_CHAPTER_CONTEXT = """
MATCH (c:Chapter)-[:CONTAINS]->(element)
WHERE c.chapterNumber < $current_chapter
WITH element, c
ORDER BY c.chapterNumber DESC
LIMIT $context_limit
RETURN element.name, element.description, element.importance, c.chapterNumber
"""

# Vector similarity search pattern
FIND_SIMILAR_CONTENT = """
CALL db.index.vector.queryNodes('chapter_embeddings', $top_k, $query_embedding)
YIELD node AS chapter, score
WHERE chapter.isProvisional = false AND score > $similarity_threshold
RETURN chapter.text, chapter.summary, chapter.chapterNumber, score
ORDER BY score DESC
"""
```

### Configuration Management
- **Environment Variables**: Define all configuration in `.env` and maintain `.env.example` as an example of configuration 
- **Pydantic Models**: Use the existing `config.py` pattern for all new configuration
- **Model Aliases**: Use the established model alias system (`LARGE_MODEL`, `MEDIUM_MODEL`, etc.)
- **Feature Flags**: Follow the pattern of boolean configuration flags (e.g., `ENABLE_PATCH_BASED_REVISION`)

```python
# Configuration access pattern
from config import (
    OPENAI_API_BASE,
    LARGE_MODEL,
    ENABLE_PATCH_BASED_REVISION,
    MAX_REVISION_CYCLES_PER_CHAPTER
)

# Use configuration values consistently
if ENABLE_PATCH_BASED_REVISION and revision_count < MAX_REVISION_CYCLES_PER_CHAPTER:
    # Proceed with patch-based revision
```

## Testing Requirements

### Test Framework Setup
- **Primary Framework**: pytest for all testing
- **Coverage**: Use pytest-cov for coverage reporting
- **Neo4j Testing**: Mock database operations or use test database instance
- **LLM API Testing**: Mock LLM responses for deterministic testing

### Test Organization
```bash
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_agents/        # Agent-specific tests
│   ├── test_core/          # Core infrastructure tests
│   └── test_processing/    # Processing pipeline tests
├── integration/            # Integration tests
│   ├── test_neo4j_integration.py
│   └── test_llm_integration.py
└── fixtures/               # Test data and fixtures
    ├── sample_chapters.py
    └── mock_responses.py
```

### Test Execution Commands
```bash
# Run all tests with coverage
pytest -v --cov=. --cov-report=term-missing

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run tests for specific agents
pytest tests/unit/test_agents/test_drafting_agent.py -v
```

## Development Workflow

### Code Quality Checks
```bash
# Format and lint with ruff
ruff check .
ruff format --check .

# Type checking with mypy
mypy .

# Run full quality suite
ruff check . && ruff format --check . && mypy . && pytest -v --cov=. --cov-report=term-missing
```

### Environment Setup
```bash
# Virtual environment setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Neo4j setup (Docker required)
docker-compose up -d neo4j

# Configuration
cp .env.example .env
# Edit .env with your specific configuration
```

### Development Dependencies
```bash
# Install development tools
pip install ruff mypy pytest-cov
```

## Neo4j Database Management

### Connection and Schema
- **Connection**: Use `core.db_manager` for all database interactions
- **Schema Creation**: Automatic constraint and index creation on first run
- **Vector Index**: Configured for chapter embeddings with dimensions from `NEO4J_VECTOR_DIMENSIONS`

### Database Operations Patterns
```python
# data_access/knowledge_graph_queries.py
"""Knowledge graph query operations."""

import structlog
from typing import Dict, Any, List
from core.db_manager import get_neo4j_driver

logger = structlog.get_logger(__name__)

async def query_knowledge_graph(cypher_query: str, parameters: Dict[str, Any]) -> List[Dict]:
    """Execute Cypher query against the knowledge graph.
    
    Args:
        cypher_query: Parameterized Cypher query string
        parameters: Query parameters for safe execution
        
    Returns:
        List of result records as dictionaries
    """
    logger.info(
        "Executing knowledge graph query",
        query_length=len(cypher_query),
        parameter_count=len(parameters),
        parameters=list(parameters.keys())
    )
    
    async with get_neo4j_driver() as driver:
        async with driver.session() as session:
            result = await session.run(cypher_query, parameters)
            records = [record.data() for record in result]
            
            logger.info(
                "Knowledge graph query completed",
                result_count=len(records),
                execution_successful=True
            )
            
            return records
```

### Knowledge Graph Maintenance
- **Healing Cycles**: Perform maintenance every `KG_HEALING_INTERVAL` chapters
- **Entity Deduplication**: Handle duplicate entities during healing
- **Provisional Data**: Track provisional status for draft-derived data
- **Relationship Promotion**: Convert dynamic relationships to static when appropriate

## LLM Integration Patterns

### API Configuration
- **Base URL**: Configure via `OPENAI_API_BASE` environment variable
- **Model Selection**: Use model aliases (`LARGE_MODEL`, `NARRATOR_MODEL`, etc.)
- **Error Handling**: Implement retry logic and graceful degradation
- **Rate Limiting**: Respect API rate limits and implement backoff strategies

### LLM Interface Pattern
```python
# core/llm_interface.py
"""LLM API interface utilities."""

import structlog
from core.llm_interface import get_llm_response

logger = structlog.get_logger(__name__)

async def generate_agent_response(
    system_prompt: str,
    user_prompt: str,
    model_alias: str = "LARGE_MODEL",
    temperature: float = 0.7
) -> str:
    """Generate response using configured LLM.
    
    Args:
        system_prompt: System-level instructions for the LLM
        user_prompt: User/task-specific prompt
        model_alias: Model configuration key from config.py
        temperature: Sampling temperature for generation
        
    Returns:
        Generated text response from the LLM
    """
    logger.info(
        "Generating LLM response",
        model_alias=model_alias,
        system_prompt_length=len(system_prompt),
        user_prompt_length=len(user_prompt),
        temperature=temperature
    )
    
    response = await get_llm_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model_alias,
        temperature=temperature
    )
    
    logger.info(
        "LLM response generated",
        model_alias=model_alias,
        response_length=len(response)
    )
    
    return response
```

## Processing Pipeline Patterns

### Revision Management
- **Patch-Based Revision**: Use `processing.revision_manager.RevisionManager`
- **Problem Grouping**: Group related evaluation problems for efficient patching
- **Validation**: Use `PatchValidationAgent` when `AGENT_ENABLE_PATCH_VALIDATION` is enabled
- **Full Rewrite Fallback**: Trigger complete rewrite when patching is insufficient

### Text Processing
- **De-duplication**: Apply text de-duplication before evaluation
- **Similarity Checking**: Use both string and semantic similarity for content comparison
- **Length Validation**: Enforce minimum chapter length (`MIN_ACCEPTABLE_DRAFT_LENGTH`)

## Orchestration Patterns

### Chapter Generation Flow
1. **Prerequisites**: Plot point focus, planning (optional), context generation
2. **Drafting**: Initial prose generation with contextual guidance
3. **Evaluation**: Parallel evaluation by multiple specialized agents
4. **Revision**: Patch-based fixes or full rewrite based on evaluation results
5. **Finalization**: Knowledge graph updates, summarization, embedding generation

### Agent Coordination
```python
# orchestration/chapter_flow.py
"""Chapter generation orchestration flow."""

import structlog
from typing import Dict, Any

logger = structlog.get_logger(__name__)

async def coordinate_chapter_generation(chapter_number: int) -> ChapterResult:
    """Coordinate the full chapter generation pipeline.
    
    Args:
        chapter_number: Sequential chapter number to generate
        
    Returns:
        ChapterResult with final text, evaluations, and metadata
    """
    logger.info("Starting chapter generation", chapter_number=chapter_number)
    
    # 1. Prerequisites and planning
    plot_focus = await get_plot_focus(chapter_number)
    context = await generate_hybrid_context(chapter_number)
    logger.info(
        "Chapter prerequisites complete",
        chapter_number=chapter_number,
        plot_focus=plot_focus,
        context_length=len(context)
    )
    
    # 2. Initial drafting
    draft = await drafting_agent.generate_draft(plot_focus, context)
    logger.info("Initial draft complete", chapter_number=chapter_number, draft_length=len(draft))
    
    # 3. Evaluation and revision cycle
    for cycle in range(MAX_REVISION_CYCLES_PER_CHAPTER):
        evaluations = await evaluate_draft(draft)
        logger.info(
            "Draft evaluation complete",
            chapter_number=chapter_number,
            revision_cycle=cycle,
            evaluation_count=len(evaluations)
        )
        
        if not requires_revision(evaluations):
            logger.info("Draft approved", chapter_number=chapter_number, final_cycle=cycle)
            break
            
        draft = await revise_draft(draft, evaluations)
        logger.info("Draft revision complete", chapter_number=chapter_number, revision_cycle=cycle)
    
    # 4. Finalization
    result = await finalize_chapter(draft, chapter_number)
    logger.info(
        "Chapter generation complete",
        chapter_number=chapter_number,
        final_length=len(result.text),
        revision_cycles=cycle + 1
    )
    
    return result
```

## Error Handling and Logging

### Logging Configuration
SAGA uses `structlog` for structured logging with enhanced console output via Rich (when available).

```python
# Import structlog in all modules
import structlog

# Get logger for the current module
logger = structlog.get_logger(__name__)

# Use structured logging with context
logger.info(
    "Agent operation completed",
    agent_name=self.__class__.__name__,
    chapter_number=chapter_num,
    processing_time=elapsed_time,
    success=True
)

logger.error(
    "LLM API call failed",
    model=model_name,
    attempt=retry_count,
    error_type=type(e).__name__,
    error_message=str(e)
)
```

### Logging Setup
- **Initialization**: Call `setup_logging_nana()` from `utils.logging` at application startup
- **Configuration**: Logging levels and formats controlled via `settings` from config
- **File Logging**: Automatic file rotation (10MB max, 5 backups)
- **Console Logging**: Rich-enhanced output when available, fallback to standard formatting
- **Logger Levels**: Neo4j, httpx, and httpcore loggers automatically set to WARNING to reduce noise

### Structured Logging Patterns
```python
# Good: Structured context with relevant fields
logger.info(
    "Knowledge graph query executed",
    query_type="similarity_search",
    results_count=len(results),
    threshold=similarity_threshold,
    execution_time_ms=duration
)

# Good: Error logging with context
logger.error(
    "Neo4j connection failed",
    uri=neo4j_uri,
    database=database_name,
    retry_attempt=attempt_num,
    exc_info=True  # Include stack trace
)

# Avoid: Unstructured string formatting
logger.info(f"Processing chapter {chapter_num} with {len(context)} context items")
```

### Error Handling Patterns
```python
# core/database_operations.py
"""Database operation utilities."""

import structlog
from typing import Dict, Any

logger = structlog.get_logger(__name__)

# LLM API error handling
try:
    response = await get_llm_response(prompt, model="LARGE_MODEL")
    logger.info(
        "LLM response received",
        model="LARGE_MODEL",
        prompt_length=len(prompt),
        response_length=len(response)
    )
except LLMAPIError as e:
    logger.error(
        "LLM API call failed",
        model="LARGE_MODEL",
        error_type=type(e).__name__,
        error_message=str(e),
        exc_info=True
    )
    # Implement fallback strategy or raise with context
    raise AgentProcessingError(f"Failed to generate response: {e}")

# Neo4j connection error handling
try:
    result = await execute_cypher_query(query, parameters)
    logger.info(
        "Cypher query executed successfully",
        query_type=query.__name__ if hasattr(query, '__name__') else "dynamic",
        result_count=len(result),
        parameters=parameters
    )
except Neo4jConnectionError as e:
    logger.error(
        "Neo4j query failed",
        query=query[:100] + "..." if len(query) > 100 else query,
        parameters=parameters,
        retry_count=retry_count,
        error_message=str(e),
        exc_info=True
    )
    # Implement retry logic or graceful degradation
    if retry_count < MAX_RETRIES:
        await asyncio.sleep(RETRY_DELAY)
        logger.info("Retrying Neo4j operation", retry_count=retry_count + 1)
        return await execute_cypher_query(query, parameters, retry_count + 1)
    raise DatabaseError(f"Failed to execute query after {MAX_RETRIES} retries: {e}")
```

## Performance Considerations

### LLM Optimization
- **Model Selection**: Use appropriate model sizes for different tasks
- **Context Management**: Keep prompts within `MAX_CONTEXT_TOKENS` limits
- **Parallel Processing**: Run independent evaluations in parallel
- **Caching**: Cache repeated LLM calls where appropriate

### Neo4j Optimization
- **Vector Search**: Optimize vector similarity searches with appropriate thresholds
- **Query Efficiency**: Use `LIMIT` clauses and avoid Cartesian products
- **Batch Operations**: Group related database operations when possible
- **Index Usage**: Ensure queries use existing indexes effectively

### Processing Pipeline
- **Lazy Loading**: Load context data only when needed
- **Memory Management**: Clear large objects after processing
- **Async Operations**: Use async/await for I/O bound operations

## Security and Data Management

### API Security
- **API Keys**: Store securely in environment variables, never in code
- **Input Validation**: Validate all user inputs and LLM responses
- **Rate Limiting**: Implement proper rate limiting for API calls

### Data Integrity
- **Provisional Data**: Clearly mark and handle provisional data from unrevised drafts
- **Backup Strategy**: Regular Neo4j database backups
- **Version Control**: Track changes in story elements and generated content

## Pull Request Guidelines

### PR Title Format
`[COMPONENT] Brief description`

Components: `AGENT`, `CORE`, `ORCHESTRATION`, `PROCESSING`, `CONFIG`, `TESTS`, `DOCS`

### PR Description Requirements
1. **Summary**: Brief description of changes and motivation
2. **Agent Changes**: New agents added or existing agent modifications
3. **Database Changes**: Schema modifications, new queries, or index changes
4. **Configuration Changes**: New environment variables or config options
5. **Testing Done**: 
   - Unit tests added/updated
   - Integration tests verified
   - Manual testing with actual LLM generation
6. **Performance Impact**: Any expected changes to generation speed or resource usage

### Review Checklist
- [ ] Code follows SAGA conventions and patterns
- [ ] All tests pass (`pytest -v --cov=.`)
- [ ] Code quality checks pass (`ruff check . && mypy .`)
- [ ] Type hints present for new code
- [ ] Logging appropriately configured
- [ ] Error handling implemented
- [ ] Configuration properly externalized
- [ ] Documentation updated for new features
- [ ] Neo4j schema changes documented
- [ ] LLM integration follows established patterns

## Useful Development Commands

### Project Management
```bash
# Start complete development environment
docker-compose up -d neo4j
python main.py

# Reset for fresh story generation
docker-compose down -v
docker-compose up -d neo4j
# Wait for Neo4j initialization, then:
python main.py

# Ingest existing text into knowledge graph
python main.py --ingest path/to/novel.txt
```

### Testing and Quality
```bash
# Quick quality check
ruff check . && ruff format . && mypy .

# Full test suite with coverage
pytest -v --cov=. --cov-report=html

# Test specific components
pytest tests/unit/test_agents/ -v
pytest tests/integration/test_neo4j_integration.py -v
```

### Neo4j Management
```bash
# View Neo4j logs
docker-compose logs -f neo4j

# Access Neo4j browser
# Navigate to http://localhost:7474
# Login: neo4j / saga_password (or your configured password)

# Direct Cypher shell access
docker-compose exec neo4j cypher-shell -u neo4j -p saga_password
```

## Configuration Patterns

### Environment Variables
- **LLM Configuration**: `OPENAI_API_BASE`, `OPENAI_API_KEY`, model aliases
- **Neo4j Configuration**: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- **Embedding Configuration**: `OLLAMA_EMBED_URL`, `EMBEDDING_MODEL`
- **Agent Behavior**: `AGENT_LOG_LEVEL`, `AGENT_ENABLE_PATCH_VALIDATION`
- **Generation Parameters**: `CHAPTERS_PER_RUN`, `MAX_REVISION_CYCLES_PER_CHAPTER`

### Model Configuration Pattern
```python
# Define model usage by capability requirement
MODELS = {
    'planning': LARGE_MODEL,      # Complex reasoning for scene planning
    'evaluation': LARGE_MODEL,    # Critical analysis capabilities
    'drafting': NARRATOR_MODEL,   # Creative writing optimized
    'patches': MEDIUM_MODEL,      # Targeted text modifications
    'summaries': SMALL_MODEL,     # Efficient summarization
    'kg_updates': MEDIUM_MODEL    # Structured data extraction
}
```

---

**Important Notes**:
- This AGENTS.md should be updated as SAGA evolves
- Pay special attention to the multi-agent orchestration patterns
- Maintain consistency with the established Neo4j schema and vector operations
- Follow the revision pipeline patterns for any text processing work
- Respect the configuration management system for all new features
