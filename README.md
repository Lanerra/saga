# SAGA: Semantic And Graph-enhanced Authoring 🌌📚
## Let NANA (Next-gen Autonomous Narrative Architecture) tell you a story, just like old times!

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

SAGA is an autonomous, agentic creative-writing system designed to generate entire novels. Powered by the **NANA** engine, SAGA leverages Large Language Models (LLMs), a sophisticated understanding of narrative context through embeddings, and a Neo4j graph database to create rich, coherent, and evolving narratives.

### Progress Window
*(Your existing image of the Rich CLI progress window - ensure the link is still valid or update it)*
![SAGA Progress Window](https://github.com/Lanerra/saga/blob/master/SAGA.png)

### Example Knowledge Graph Visualization (12 Chapters)
![SAGA KG Visualization](https://github.com/Lanerra/saga/blob/master/SAGA-KG-Ch12.png)

## Overview

SAGA, with its NANA engine, is an ambitious project designed to autonomously craft entire novels. It transcends simple text generation by employing a collaborative team of specialized AI agents:

*   **`PlannerAgent`:** Strategically outlines detailed scene-by-scene plans for each chapter, ensuring plot progression.
*   **`DraftingAgent`:** Weaves the initial prose for chapters, guided by the `PlannerAgent`'s blueprints and rich contextual information. It uses two Jinja templates: `draft_chapter_from_plot_point.j2` for single-pass chapter drafting and `draft_scene.j2` when generating chapters scene by scene.
*   **`ComprehensiveEvaluatorAgent`:** Critically assesses drafts for plot coherence, thematic alignment, character consistency, narrative depth, and overall quality.
*   **`WorldContinuityAgent`:** Performs targeted checks to ensure consistency with established world-building rules, lore, and character backstories within the narrative.
*   **`ChapterRevisionLogic`:** Implements sophisticated revisions based on evaluation feedback, capable of performing targeted, patch-based fixes or full chapter rewrites.
*   **`PatchValidationAgent`:** Verifies generated patch instructions to ensure they resolve the issues before being applied.
*   **`KGMaintainerAgent`:** Intelligently manages the novel's evolving knowledge graph in Neo4j. It summarizes chapters, pre-populates the graph with initial story data, and exposes an `extract_and_merge_knowledge` method to parse new chapters and persist updates.
*   **`kg_maintainer` utilities:** Provide dataclass models and helper functions for parsing, merging, and generating Cypher statements used by `KGMaintainerAgent`.

SAGA constructs a dynamic, interconnected understanding of the story's world, characters, and plot. This evolving knowledge, stored and queried from a Neo4j graph database, enables the system to maintain greater consistency, depth, and narrative cohesion as the story unfolds over many chapters.

## Key Features

*   **Autonomous Multi-Chapter Novel Generation:** Capable of generating batches of chapters (e.g., 3 chapters in ~11 minutes) in a single run, producing substantial narrative content (e.g., ~13K+ tokens per chapter).
*   **Sophisticated Agentic Architecture:** Utilizes a suite of specialized AI agents, each responsible for distinct phases of the creative writing pipeline, orchestrated by `NANA_Orchestrator`.
*   **Deep Knowledge Graph Integration (Neo4j):**
    *   Persistently stores and retrieves story canon, character profiles (including development over time), detailed world-building elements, and plot points.
    *   Supports complex queries for consistency checking and context retrieval.
    *   Features a robust schema with constraints and a vector index for semantic search.
*   **Hybrid Semantic & Factual Context Generation:**
    *   Leverages text embeddings (via Ollama) and Neo4j vector similarity search to construct semantically relevant context from previous chapters, ensuring narrative flow and tone.
    *   Integrates key factual data extracted from the Knowledge Graph to ground the LLM in established canon.
*   **Iterative Drafting, Evaluation, & Revision Cycle:** Chapters undergo a rigorous process of drafting, multi-faceted evaluation, and intelligent revision (patch-based or full rewrite) to enhance quality and coherence.
*   **Dynamic Knowledge Graph Updates:** The system "learns" from the generated content, with the `KGMaintainerAgent` extracting and merging new information (character updates, world-building changes, KG triples) into the Neo4j database.
*   **Provisional Data Handling:** Explicitly tracks and manages the provisional status of data derived from unrevised or flawed drafts, ensuring a distinction between canonical and tentative information.
*   **Flexible Configuration (`config.py` & `.env`):**
    *   Extensive options for LLM endpoints, model selection per task, API keys, Neo4j connection details, generation parameters, and more.
    *   Supports "Unhinged Mode" for highly randomized and surprising initial story elements if user input is minimal.
*   **User-Driven Initialization:** Accepts user-supplied story elements via a `user_story_elements.md` Markdown file, allowing for a customized starting point. The `[Fill-in]` placeholder system allows users to specify which elements SAGA should generate.
*   **Rich Console Progress Display:** Optional live progress updates using the Rich library, providing a clear view of the generation process.
*   **Text De-duplication:** Implements mechanisms to reduce repetitive content in generated drafts using both string and semantic comparisons.

## Architecture & Pipeline

SAGA's NANA engine orchestrates a sophisticated pipeline for novel generation:

1.  **Initialization & Setup (First Run or Reset):**
    *   **Connect & Verify Neo4j:** Establishes connection and ensures the database schema (indexes, constraints, vector index) is in place.
    *   **Load Existing State (if any):** Attempts to load plot outline, character profiles, world-building, and chapter count from Neo4j.
    *   **Initial Story Generation (if needed):**
        *   If `user_story_elements.md` is provided and contains content, it's parsed by `MarkdownStoryParser` to bootstrap the plot, characters, and world.
        *   Otherwise, or if key elements are marked `[Fill-in]`, `run_genesis_phase` fills in the plot outline, character profiles, and world-building details via targeted LLM calls. This can be influenced by "Unhinged Mode" or default configurations.
    *   **KG Pre-population:** The `KGMaintainerAgent` populates the Neo4j graph with this foundational story data.

2.  **Chapter Generation Loop (Iterates for `CHAPTERS_PER_RUN`):**
    *   **(A) Prerequisites:**
        *   Retrieves the current **Plot Point Focus** for the chapter.
        *   **Planning (if enabled):** The `PlannerAgent` creates a detailed scene-by-scene plan.
        *   **Context Generation:** `ContextGenerationLogic` assembles a "hybrid context" string by:
            *   Querying Neo4j for semantically similar past chapter summaries/text snippets (vector search).
            *   Fetching key reliable facts from the Knowledge Graph via `PromptDataGetters`.
    *   **(B) Drafting:**
        *   The `DraftingAgent` writes the initial draft, guided by the scene plan (if available), plot point focus, hybrid context, and filtered character/world data.
    *   **(C) De-duplication & Evaluation:**
        *   The draft undergoes de-duplication via `utils.deduplicate_text_segments` to reduce repetitiveness.
        *   The `ComprehensiveEvaluatorAgent` assesses the draft against multiple criteria (plot, theme, depth, consistency using character/world profiles and previous context).
        *   The `WorldContinuityAgent` performs a focused consistency check using the KG and world-building data.
    *   **(D) Revision (if `needs_revision` is true):**
        *   `ChapterRevisionLogic` attempts to fix identified issues.
        *   If `ENABLE_PATCH_BASED_REVISION` is true, it generates and applies targeted text patches for specific problems.
            See `docs/step5_redesign.md` for a proposed improvement of this phase.
        *   If patching is insufficient or disabled, or problems are extensive, a full chapter rewrite may be performed.
        *   The de-duplication and evaluation steps are repeated on the revised text.
    *   **(E) Finalization & Knowledge Update:**
        *   The `KGMaintainerAgent` summarizes the final approved chapter text.
        *   An embedding is generated for the final chapter text (via `llm_interface`).
        *   The chapter data (text, summary, embedding, provisional status) is saved to Neo4j by `chapter_queries`.
        *   The `KGMaintainerAgent` extracts new knowledge (character updates, world-building changes, new KG triples) from the final chapter text. This involves parsing LLM output and merging changes into the in-memory `novel_props_cache` and then persisting these updates to Neo4j via `character_queries`, `world_queries`, and `kg_queries`.
    *   The orchestrator's main state dictionaries (`plot_outline`, `character_profiles`, `world_building`) are updated to reflect changes merged by the `KGMaintainerAgent`.

## Setup

### Prerequisites

*   Python 3.8+ (preferably 3.9+ for some newer type hints if used)
*   An Ollama instance for generating text embeddings (e.g., running `ollama serve`).
*   An OpenAI-API compatible LLM server (e.g., running via LM Studio, oobabooga's text-generation-webui with the OpenAI extension, a local vLLM/TGI instance, or a cloud provider).
*   Neo4j Database (v4.4+ or v5.x recommended for vector index support). Docker setup is provided.

### 1. Clone the Repository

```bash
git clone https://github.com/Lanerra/saga.git
cd saga
```

### 2. Install Python Dependencies

It's highly recommended to use a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# neomodel is included for higher-level graph models
python -m spacy download en_core_web_sm
```

### 3. Configure SAGA

Core configuration is managed in `config.py`, which loads values from environment variables (e.g., from a `.env` file) or uses defaults. Create a `.env` file in the root of the project or set environment variables directly.

**Key settings to configure in your `.env` file:**

```dotenv
# LLM API Configuration
OPENAI_API_BASE="http://127.0.0.1:8080/v1" # URL of your OpenAI-compatible LLM API
OPENAI_API_KEY="nope"                       # API key (can be any string if server doesn't require auth)

# Embedding Model (Ollama) Configuration
OLLAMA_EMBED_URL="http://127.0.0.1:11434"     # URL of your Ollama API
EMBEDDING_MODEL="nomic-embed-text:latest"   # Ensure this model is pulled in Ollama (ollama pull nomic-embed-text)
EXPECTED_EMBEDDING_DIM="768"                # Dimension of your embedding model (e.g., 768 for nomic-embed-text)

# Neo4j Connection (Defaults usually work with the provided Docker setup)
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="saga_password"
NEO4J_DATABASE="neo4j" # Or your specific database name if not default
# neomodel is automatically configured with these settings

# Model Aliases (Set to the names of models available on your OPENAI_API_BASE server)
LARGE_MODEL="Qwen3-14B-Q4"    # Or your preferred large model
MEDIUM_MODEL="Qwen3-8B-Q4"   # Or your preferred medium model
SMALL_MODEL="Qwen3-4B-Q4"    # Or your preferred small model
NARRATOR_MODEL="Qwen3-14B-Q4" # Model for drafting
MAX_REVISION_CYCLES_PER_CHAPTER="2"  # Max revision loops per chapter

# Other important settings in config.py (review defaults)
# MAX_CONTEXT_TOKENS, CHAPTERS_PER_RUN, LOG_LEVEL, etc.
```

Refer to `config.py` for a full list of configurable options and their defaults.

### 4. Set up Neo4j Database

SAGA uses Neo4j for its knowledge graph. A `docker-compose.yml` file is provided for easy setup.

*   **Ensure Docker and Docker Compose are installed.**
*   **Manage Neo4j container (from the project root directory):**
    *   **Start Neo4j:**
        ```bash
        docker-compose up -d neo4j
        ```
        Wait a minute or two for Neo4j to fully initialize. You can access the Neo4j Browser at `http://localhost:7474`. Login with the credentials you configured (default: `neo4j` / `saga_password`).
    *   **Stop Neo4j:**
        ```bash
        docker-compose down
        ```
    *   **View Logs:**
        ```bash
        docker-compose logs -f neo4j
        ```

The first time SAGA runs (`python nana_orchestrator.py`), it will automatically attempt to create necessary constraints and indexes in Neo4j, including the vector index for chapter embeddings.

### 5. (Optional) Provide Initial Story Elements

To guide SAGA with your own story ideas, create a `user_story_elements.md` file in the project's root directory.
You can use `user_story_elements.md.example` as a template.
Use the `[Fill-in]` placeholder for any elements you want SAGA to generate. If this file is not present or empty, SAGA will generate these elements based on its configuration (including "Unhinged Mode" if active).

### 6. (Optional) Configure "Unhinged Mode" Data

For "Unhinged Mode" (which generates highly randomized initial story elements if no user input is provided), SAGA can use custom lists from JSON files in the `unhinged_data/` directory (e.g., `unhinged_genres.json`). If these files are missing, defaults from `config.py` are used.

## Running SAGA

Once Neo4j is running and your configuration is set:

```bash
python nana_orchestrator.py
```

*   **First Run:** SAGA will perform initial setup (plot, world, characters based on `user_story_elements.md` or generation) and pre-populate the Neo4j knowledge graph.
*   **Subsequent Runs:** It will load the existing state from Neo4j and continue generating chapters from where it left off.
*   The number of chapters generated per run is controlled by `CHAPTERS_PER_RUN` in `config.py`.

Output files (chapters, logs, debug information) will be saved in the directory specified by `BASE_OUTPUT_DIR` (default: `novel_output`).

**Performance Example:**
Using a local setup with the following GGUF models (via an OpenAI-compatible server like llama-cpp-python):
*   `LARGE_MODEL = Qwen3-14B-Q4`
*   `MEDIUM_MODEL = Qwen3-8B-Q4`
*   `SMALL_MODEL = Qwen3-4B-Q4`
*   `NARRATOR_MODEL = Qwen3-14B-Q4`

SAGA can generate a batch of **3 chapters** (each ~13,000+ tokens of narrative) in approximately **11 minutes**, involving significant processing for planning, context generation, evaluation, and knowledge graph updates.

All models used for sample generation are Unsloth's *-UD-Q4_K_XL.gguf

## Resetting the Database

To start SAGA from a completely fresh state (e.g., new story, after testing):

**⚠️ WARNING: This will delete ALL data in the Neo4j database targeted by your configuration.**

1.  Ensure SAGA (`nana_orchestrator.py`) is not running.
2.  Stop the Neo4j Docker container if you intend to also remove its volume:
    ```bash
    docker-compose down -v # The -v flag removes the data volume
    ```
    Then restart it: `docker-compose up -d neo4j`
3.  Alternatively, to clear data *within* an existing Neo4j instance, run the `reset_neo4j.py` script:
    ```bash
    python reset_neo4j.py
    ```
    You will be prompted for confirmation. To bypass confirmation (e.g., for scripting):
    ```bash
    python reset_neo4j.py --force
    ```
    You can specify Neo4j connection details with `--uri`, `--user`, and `--password` if they differ from your `config.py` / `.env` settings.

After resetting (and ensuring Neo4j is running), the next execution of `python nana_orchestrator.py` will re-initialize the story and KG.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.