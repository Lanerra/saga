# SAGA: Semantic And Graph-enhanced Authoring 🌌📚
## Let NANA (Next-gen Autonomous Narrative Architecture) tell you a story, just like old times!

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

SAGA is an autonomous, agentic creative-writing system designed to generate entire novels. Powered by the **NANA** engine, SAGA leverages Large Language Models (LLMs), a sophisticated understanding of narrative context through embeddings, and a Neo4j graph database to create rich, coherent, and evolving narratives.

### Progress Window
*(This is a representation of the Rich CLI progress window)*
![SAGA Progress Window](https://github.com/Lanerra/saga/blob/master/SAGA.png)

### Example Knowledge Graph Visualization (4 Chapters)
![SAGA KG Visualization](https://github.com/Lanerra/saga/blob/master/SAGA-KG-Ch4.png)

## Overview

SAGA, with its NANA engine, is an ambitious project designed to autonomously craft entire novels. It transcends simple text generation by employing a streamlined team of specialized AI agents:

*   **`agents.NarrativeAgent`:** Combines planning and drafting capabilities, creating detailed scene-by-scene plans and generating prose for chapters.
*   **`agents.RevisionAgent`:** Handles evaluation (plot coherence, thematic alignment, character consistency), continuity checks, and patch-based revisions.
*   **`agents.KnowledgeAgent`:** Manages the knowledge graph in Neo4j, summarizing chapters, extracting new information from final text, and performing periodic "healing" cycles to resolve duplicates and enrich sparse data.

SAGA constructs a dynamic, interconnected understanding of the story's world, characters, and plot. This evolving knowledge, stored and queried from a Neo4j graph database, enables the system to maintain greater consistency, depth, and narrative cohesion as the story unfolds over many chapters.

## Key Features

*   **Consolidated Agent Architecture:** Streamlined agent roles reduce complexity while enhancing coordination (NarrativeAgent for drafting/planning, RevisionAgent for evaluation/revision, KnowledgeAgent for KG maintenance).
*   **Autonomous Multi-Chapter Novel Generation:** Capable of generating batches of chapters (e.g., 3 chapters in ~11 minutes) in a single run, producing substantial narrative content (~13K+ tokens per chapter).
*   **Sophisticated Agentic Pipeline:** Orchestrates the creative writing process through NarrativeAgent, RevisionAgent, and KnowledgeAgent working in concert.
*   **Deep Knowledge Graph Integration (Neo4j):**
    *   Persistently stores and retrieves story canon, character profiles, detailed world-building elements, and plot points via a dedicated `data_access` layer.
    *   Features a robust, predefined schema with constraints and vector index for semantic search, automatically created on first run.
    *   Supports complex queries for consistency checking, context retrieval, and graph maintenance.
*   **Hybrid Semantic & Factual Context Generation:**
    *   Leverages text embeddings (via Ollama) and Neo4j vector similarity search to construct semantically relevant context from previous chapters.
    *   Integrates key factual data extracted from the Knowledge Graph to ground the LLM in established canon.
*   **Iterative Drafting, Evaluation, & Revision Cycle:** Chapters undergo a rigorous process of drafting, multi-faceted evaluation, and intelligent revision (patch-based or full rewrite).
*   **Dynamic Knowledge Graph Updates & Healing:** The system "learns" from generated content. `KnowledgeAgent` extracts new information (character updates, world-building changes) and performs periodic maintenance to resolve duplicate entities and enrich sparse nodes.
*   **Provisional Data Handling:** Explicitly tracks and manages provisional data derived from unrevised drafts, ensuring clear distinction between canonical and tentative information in the knowledge graph.
*   **Flexible Configuration (`config.py` & `.env`):**
    *   Extensive options for LLM endpoints, model selection per task, API keys, Neo4j connection details, generation parameters, and more.
    *   Supports "Unhinged Mode" for highly randomized initial story elements when user input is minimal.
*   **User-Driven Initialization:** Accepts user-supplied story elements via `user_story_elements.yaml`, allowing customized starting points with `[Fill-in]` placeholders for SAGA to generate missing elements.
*   **Rich Console Progress Display:** Optional live progress updates using Rich library for clear generation tracking.

## Architecture & Pipeline

SAGA's NANA engine orchestrates a refined pipeline for novel generation:

1.  **Initialization & Setup (First Run or Reset):**
    *   **Connect & Verify Neo4j:** Establishes connection and ensures database schema (indexes, constraints, vector index) is in place.
    *   **Load Existing State (if any):** Attempts to load plot outline and chapter count from Neo4j.
    *   **Initial Story Generation (`initialization.genesis`):**
        *   If `user_story_elements.yaml` provided, parses it to bootstrap story elements.
        *   Otherwise, `bootstrapper` modules fill in missing elements via targeted LLM calls.
    *   **KG Pre-population:** `KnowledgeAgent` syncs foundational story data to Neo4j.

2.  **Chapter Generation Loop (up to `CHAPTERS_PER_RUN` chapters):**
    *   **(A) Prerequisites (`orchestration.chapter_flow`):**
        *   Retrieves current **Plot Point Focus** for the chapter.
        *   **Narrative Planning & Drafting:** `NarrativeAgent` creates scene plans and drafts chapters.
        *   **Context Generation:** `processing.context_generator` assembles hybrid context by:
            *   Querying Neo4j for semantically similar past chapter summaries (vector search).
            *   Fetching key facts from Knowledge Graph via `prompt_data_getters`.
    *   **(B) Revision & Evaluation:**
        *   Draft undergoes de-duplication via `processing.text_deduplicator`.
        *   `RevisionAgent` assesses draft against multiple criteria.
    *   **(C) Revision (if needed):**
        *   `revision_logic` applies targeted text patches or full rewrites.
        *   Evaluation steps repeated on revised text.
    *   **(D) Knowledge Graph Update:**
        *   `KnowledgeAgent` processes final chapter text, extracts new information, and updates Neo4j.
        *   Periodic healing cycles resolve duplicates and enrich sparse nodes.

## Setup

### Prerequisites

*   Python 3.9+
*   Ollama instance for text embeddings (`ollama serve`)
*   OpenAI-API compatible LLM server (e.g., LM Studio, oobabooga)
*   Neo4j Database (v5.x recommended). Docker setup provided.

### 1. Clone Repository

```bash
git clone https://github.com/Lanerra/saga.git
cd saga
```

### 2. Install Dependencies

Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure SAGA

Create `.env` file in project root:

```dotenv
# LLM API Configuration
OPENAI_API_BASE="http://127.0.0.1:8080/v1"
OPENAI_API_KEY="nope"

# Embedding Model (Ollama)
OLLAMA_EMBED_URL="http://127.0.0.1:11434"
EMBEDDING_MODEL="nomic-embed-text:latest"
EXPECTED_EMBEDDING_DIM="768"

# Neo4j Connection
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="saga_password"
NEO4J_DATABASE="neo4j"

# Model Aliases
LARGE_MODEL="Qwen3-14B-Q4"    # Planning, Evaluation
MEDIUM_MODEL="Qwen3-8B-Q4"   # KG updates, Patches
SMALL_MODEL="Qwen3-4B-Q4"    # Summaries
NARRATOR_MODEL="Qwen3-14B-Q4" # Drafting, Full Revision
```

Refer to `config.py` for full configurable options.

### 4. Set Up Neo4j Database

Docker setup provided in `docker-compose.yml`:

```bash
docker-compose up -d        # Start Neo4j
docker-compose down         # Stop Neo4j
docker-compose logs -f neo4j-apoc  # View logs
```

First run of SAGA auto-creates necessary database constraints and indexes.

### 5. Provide Initial Story Elements (Optional)

Create `user_story_elements.yaml` using `user_story_elements.yaml.example` as template:

```yaml
title: "My Novel"
genre: "[Fill-in]"
protagonist: "[Fill-in]"
```

### 6. Configure "Unhinged Mode" (Optional)

Customize `unhinged_data/` JSON files for randomized initial story elements.

## Running SAGA

```bash
python main.py
```

*   **First Run:** Initial setup (plot, world, characters) and KG pre-population.
*   **Subsequent Runs:** Load existing state from Neo4j and continue generating chapters.
*   Output files saved in `novel_output/` (ignored by Git).

**Performance:** Generates 3 chapters (~13K tokens each) in ~11 minutes using Qwen3 models.

### Ingestion Mode

```bash
python main.py --ingest path/to/novel.txt
```

Text split into pseudo-chapters and processed through pipeline. Knowledge graph heals every `KG_HEALING_INTERVAL` chapters.

## Resetting Database

**⚠️ WARNING: Deletes ALL Neo4j data!**

```bash
docker-compose down -v  # Remove data volume
docker-compose up -d neo4j
```

Next `python main.py` will re-initialize.

## Complexity Analysis

```bash
python complexity_report.py
```

Runs `radon cc . -nc` to highlight functions needing refactoring.

## License

Licensed under Apache 2.0. See `LICENSE` file.
