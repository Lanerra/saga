# SAGA - Semantic And Graph‑enhanced Authoring

**NOTE**: `MAX_REVISION_CYCLES_PER_CHAPTER` currently defaults to `0`, effectively disabling the revision cycle during chapter generation. It is currently broken and imminently going to be refactored.

SAGA is a local‑first, single‑process Python CLI that uses a Neo4j knowledge graph, LangGraph workflow orchestration, and a small set of cooperating nodes to plan, draft, and revise long‑form fiction while preserving continuity across chapters.


## What SAGA Does

- Local knowledge graph continuity
  - Persists entities, relationships, plot points, and chapter metadata in a local Neo4j database.
  - Maintains coherence via duplicate prevention/merging.
- LangGraph workflow pipeline
  - Initialization: Generates character sheets, global outline, and act-level outlines
  - Chapter generation: Creates chapter outlines on-demand, drafts prose, extracts entities
  - Quality assurance: Validates consistency, revises via full-rewrite when needed
- Multi‑phase bootstrap (optional)
  - World → Characters → Plot, each with lightweight validation.
  - Can run standalone or as an integrated prelude to generation.
- Semantic context and search
  - Embeddings stored on chapter nodes with a Neo4j vector index for fast similarity search.
  - Context assembly pulls relevant entities, relationships, and summaries for each chapter.
- Rich CLI progress
  - Live panel shows novel title, chapter progress, current step, elapsed time, and request rate.


## Quick Start

Prereqs
- Python 3.12
- Neo4j 5.x running locally (standalone or via `docker-compose`)
- A local LLM endpoint for completions and embeddings (OpenAI‑compatible HTTP, e.g., local gateway). Configure endpoints in `.env`.

*Note: SAGA also supports connecting to cloud endpoints as well, but is local-first by design.*

Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit values as needed
```

Start Neo4j (optional helper)
```bash
docker-compose up -d   # uses docker-compose.yml in this repo
```

Run the generator
```bash
python main.py
```

Optional: Run the bootstrap independent of the novel generation cycle (world → characters → plot)
```bash
# Full pipeline
python main.py --bootstrap

# Limit to a phase
python main.py --bootstrap --bootstrap-phase world     # or characters|plot

# Tuning and safety
python main.py --bootstrap --bootstrap-level basic     # basic|enhanced|max
python main.py --bootstrap --bootstrap-dry-run
python main.py --bootstrap --bootstrap-reset-kg
```



## Key Features

- Knowledge Graph backbone (Neo4j)
  - Schema creation with constraints/indexes, including a vector index on chapter embeddings.
  - Entity deduplication and enrichment to reduce duplication and maintain coherence.
- LangGraph workflow orchestration
  - Declarative graph-based workflow with automatic state persistence
  - Initialization workflow: Character sheets → Global outline → Act outlines → On-demand chapter outlines
  - Generation workflow: Generate → Extract → Commit → Validate → Revise (if needed) → Summarize → Finalize
  - Built-in checkpointing allows resume from interruption
- Output artifacts
  - Chapter text files, per‑chapter logs, and YAML initialization artifacts under `output/`.
  - Structured outputs: `characters/`, `outline/`, `world/`, `chapters/`, `summaries/`
- Local‑first architecture
  - No web servers or distributed components; single user on a single machine.


## CLI Overview

`python main.py` — start the chapter generation loop using LangGraph workflow.

`python main.py --bootstrap [options]` — run the multi‑phase bootstrap and exit.
- `--bootstrap-phase {world|characters|plot|all}`
- `--bootstrap-level {basic|enhanced|max}`
- `--bootstrap-dry-run` (validate only; do not write to Neo4j)
- `--bootstrap-reset-kg` (wipe Neo4j before bootstrapping; destructive)


## Configuration

- Edit `.env` or adjust `config/settings.py`. Important keys:
  - `OPENAI_API_BASE`, `OPENAI_API_KEY` — OpenAI‑compatible completion endpoint (local recommended)
  - `EMBEDDING_API_BASE`, `EMBEDDING_MODEL`, `EXPECTED_EMBEDDING_DIM` — embedding service and dimensions
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — local Neo4j connection
  - `CHAPTERS_PER_RUN`, `CONTEXT_CHAPTER_COUNT`, `TARGET_PLOT_POINTS_INITIAL_GENERATION` — core behavior
  - Bootstrap toggles: `BOOTSTRAP_*` settings (healing, levels, fail‑fast, etc.)

Outputs are written under `output/`:
- `output/chapters/` — final chapter text
- `output/chapter_logs/` — chapter‑specific logs
- `output/debug_outputs/` — saved prompts, scene plans, validation reports, etc.


## How It Works (High Level)

1) Bootstrap (optional, independent loop)
- Generate character sheets, global story outline (3 or 5 acts), and detailed act outlines
- Validate structure and persist to Neo4j knowledge graph
- Creates human-readable YAML files in `output/characters/`, `output/outline/`, `output/world/`

2) Initialization (first run only)
- If not bootstrapped, LangGraph initialization workflow generates minimal required structure
- Creates character sheets, global outline, act outlines automatically
- Persists to both Neo4j and YAML files for human review/editing

3) Per‑chapter workflow (LangGraph orchestrated)
- **Chapter Outline**: Generate chapter-specific outline from act outline and context
- **Generate**: Build KG‑aware context and draft prose
- **Extract**: Extract entities and relationships from generated text
- **Commit**: Deduplicate and persist entities to Neo4j
- **Validate**: Check for continuity contradictions
- **Revise**: If validation fails, regenerate chapter with feedback
- **Summarize**: Create chapter summary for future context
- **Finalize**: Save chapter to disk and database, advance to next chapter

4) State persistence
- All workflow state automatically checkpointed to SQLite
- Resume from interruption without data loss
- Chapter progress tracked in Neo4j


## Screenshots

Progress window (Rich CLI):

![SAGA Progress Window](SAGA.png)

Example KG snapshot (4 chapters):

![SAGA KG Visualization](SAGA-KG-Ch4.png)


## Light Dev Notes

- Single user, single machine; no web servers or remote services are introduced by SAGA.
- Run tests with `pytest`; lint/format with `ruff check .` and `ruff format .`.


## License

Apache-2.0 — see `LICENSE`.
