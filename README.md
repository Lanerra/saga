# SAGA - Semantic And Graph-enhanced Authoring
# Autonomous Novel-Writing System

**NOTE**: SAGA is currently in a state of (mostly) functional flux as it is undergoing a significant refactoring and overhaul. Things may not work as intended. Ingestion, for certain, doesn't work at the moment.

SAGA is a novel-writing system that leverages a knowledge graph and multiple specialized agents to autonomously create and refine stories. It is designed to handle complex narrative structures while maintaining coherence and consistency.


## Progress Window
*(This is a representation of the Rich CLI progress window)*

![SAGA Progress Window](https://github.com/Lanerra/saga/blob/master/SAGA.png)


## Example Knowledge Graph Visualization (4 Chapters)

![SAGA KG Visualization](https://github.com/Lanerra/saga/blob/master/SAGA-KG-Ch4.png)


## Features

- **Knowledge Graph (KG)**: Stores and manages all entities, relationships, and narrative elements using Neo4j.
- **Modular Agent Architecture**: Includes agents for narrative drafting, consistency checks, revisions, and knowledge extraction.
- **LLM Integration**: Utilizes large language models for creative and analytical tasks.
- **Configurable Generation Parameters**: Fine-tune the novel generation process through configuration settings.
- **Robust Testing Framework**: Comprehensive test coverage using `pytest` with custom markers for test categorization.
- **Code Quality**: Enforces PEP8 and bug-free code with `ruff` linter and formatter.
- **Vector Search**: Uses vector embeddings for enhanced search and similarity detection.
- **Agentic Planning**: Supports scene planning and chapter drafting through agentic workflows.

## Installation

To set up and run SAGA, follow these steps:

### 1. Clone the Repository

```bash
git clone git@github.com:Lanerra/saga
cd saga
```

### 2. Set Up a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the Project

```bash
pip install -e .
```

### 5. Set Up Configuration

Copy the `.env.example` file to `.env` and update the configuration values as needed:

```bash
cp .env.example .env
```

Example configuration values:

```
# API and Model Configuration
OLLAMA_EMBED_URL=http://127.0.0.1:11434
OPENAI_API_BASE=http://127.0.0.1:8080/v1
OPENAI_API_KEY=sk-nope
EMBEDDING_MODEL="nomic-embed-text:latest"
EXPECTED_EMBEDDING_DIM=768

# Neo4j Connection Settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=saga_password
NEO4J_DATABASE=neo4j
NEO4J_VECTOR_DIMENSIONS=768

# Model Aliases
LARGE_MODEL="qwen3-a3b"
MEDIUM_MODEL="qwen3-a3b"
SMALL_MODEL="qwen3-a3b"
NARRATIVE_MODEL="qwen3-a3b"

# SAGA Generation Parameters
ENABLE_LLM_NO_THINK_DIRECTIVE=True
UNHINGED_PLOT_MODE=False

# Generation Run Settings
TARGET_PLOT_POINTS_INITIAL_GENERATION=18
CHAPTERS_PER_RUN=6
TARGET_SCENES_MIN=4
TARGET_SCENES_MAX=6

# Token Limits
MAX_CONTEXT_TOKENS=40960
MAX_GENERATION_TOKENS=32768
MAX_SUMMARY_TOKENS=4096
MAX_KG_TRIPLE_TOKENS=8192
MAX_PREPOP_KG_TOKENS=16384
MAX_PLANNING_TOKENS=16384

# Draft Quality & Length
MIN_ACCEPTABLE_DRAFT_LENGTH=12000

# Revision Process
ENABLE_PATCH_BASED_REVISION=True
AGENT_ENABLE_PATCH_VALIDATION=True
MAX_PATCH_INSTRUCTIONS_TO_GENERATE=10
REVISION_COHERENCE_THRESHOLD=0.60
REVISION_SIMILARITY_ACCEPTANCE=0.995

# LLM Call Robustness
LLM_RETRY_ATTEMPTS=3

# Logging
LOG_LEVEL=INFO

# Rich Progress Display
ENABLE_RICH_PROGRESS=True
```

### 6. Start the System

Run the system using the `main.py` entry point:

```bash
python main.py
```
#### WARNING: Ingestion is currently broken and needs refactoring
To ingest a novel, use the `--ingest` flag with the path to your text file:

```bash
python main.py --ingest path/to/novel.txt
```

If the system is running in Docker, ensure you have the Neo4j database running via `docker-compose`:

```bash
docker-compose up -d
```

## Project Structure

```
saga/
├── agents/                # Specialized agents for narrative tasks
├── core/                  # Core components for data management and LLM interaction
├── data_access/           # Data access layer for Neo4j operations
├── docs/                  # Documentation (currently empty)
├── initialization/        # Initialization scripts for data loading and setup
├── models/                # Pydantic models for validation and data structure
├── novel_output/          # Output directory for generated novel content
├── orchestration/         # Orchestrator logic for managing agent workflows
├── processing/            # Tools for text processing, context generation, and similarity checks
├── prompts/               # J2 templates for generating prompts for agents
├── tests/                 # Comprehensive test suite for validation
├── ui/                    # UI components for display and interaction
├── utils/                 # Utility functions for ingestion, similarity checks, etc.
├── .env.example           # Example configuration file
├── main.py                # Entry point for the system
├── pyproject.toml         # Project configuration for dependencies and tools
├── requirements.txt       # Python dependencies for the project
├── LICENSE                # License information
└── README.md              # This file
```

## Usage

### Ingestion Mode
### WARNING: Ingestion is currently broken and needs refactoring

Ingest a novel to populate the knowledge graph and start the system's internal modeling process:

```bash
python main.py --ingest novel.txt
```

### Generation Mode

Start the autonomous novel generation loop:

```bash
python main.py
```

The system will generate chapters and refine them using the defined agents and workflows.

## Development Practices

The project uses the following tools to ensure code quality and maintainability:

- **Ruff**: Linter and formatter for PEP8 compliance and bug detection.
- **pytest**: Testing framework with custom markers for test categorization.
- **Neo4j**: Knowledge graph for narrative structure, character relationships, and world elements.
- **LLM Integration**: Uses LLMs for creative and analytical tasks in the novel generation process.

## License

SAGA is licensed under the [![Apache-2.0 License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
