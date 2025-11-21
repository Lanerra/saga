# LangGraph Pipeline Usage Guide

## Overview

SAGA uses a LangGraph-based workflow orchestration system for all chapter generation. The graph-based architecture provides:
- **Automatic state persistence** via SQLite checkpointing
- **Resume from interruption** without data loss
- **Declarative workflow** with clear node structure
- **Initialization on first run** with human-readable YAML outputs

## Quick Start

### Running SAGA
```bash
# Standard generation (runs initialization if needed)
python main.py

# Bootstrap independently (optional, before generation)
python main.py --bootstrap
```

## What Happens on First Run

### Automatic Initialization
```
1. Connects to Neo4j
2. Creates initial state
3. Detects initialization is needed (no chapters in DB)
4. Runs initialization workflow:
   - Generates character sheets (3-5 characters)
   - Generates global outline (3 or 5 acts based on chapter count)
   - Generates detailed act outlines
   - Commits CharacterProfile models to Neo4j
   - Writes human-readable YAML files
5. Generates CHAPTERS_PER_RUN chapters
6. Saves checkpoint for resumption
```

### Subsequent Runs
```
1. Connects to Neo4j
2. Loads last chapter count from database
3. Creates state for next chapter
4. Generates next CHAPTERS_PER_RUN chapters
5. Updates progress in Neo4j
```

## Generated Files

SAGA creates a structured output directory:

```
output/
├── .saga/
│   └── checkpoints.db          # LangGraph state persistence
├── characters/
│   ├── dr-elena-vasquez.yaml   # One file per character
│   ├── marcus-chen.yaml
│   └── ...
├── outline/
│   ├── structure.yaml          # Act structure (3 or 5 acts)
│   └── beats.yaml              # Detailed act outlines
├── world/
│   └── items.yaml              # Locations, objects, factions
├── chapters/
│   ├── chapter_001.md          # Final chapter text
│   ├── chapter_002.md
│   └── ...
├── summaries/
│   └── ...                     # Chapter summaries
├── chapter_logs/
│   └── chapter_*.log           # Per-chapter generation logs
└── saga_run.log                # Main application log
```

## Configuration

Key environment variables (`.env`):

**LLM Endpoints:**
- `OPENAI_API_BASE`: Local LLM endpoint (e.g., `http://127.0.0.1:8080/v1`)
- `EMBEDDING_API_BASE`: Embedding service endpoint

**Models:**
- `LARGE_MODEL`, `MEDIUM_MODEL`, `SMALL_MODEL`: Model names for your local LLM
- `NARRATIVE_MODEL`: Primary model for chapter generation

**Generation Behavior:**
- `CHAPTERS_PER_RUN`: Chapters to generate per execution (default: 2)
- `TARGET_CHAPTERS`: Total chapters for novel (default: 20)
- `MIN_CHAPTER_LENGTH_CHARS`: Minimum chapter length (default: 12000 ≈ 2500-3000 words)

**Novel Settings:**
- `DEFAULT_PLOT_OUTLINE_TITLE`: Novel title
- `CONFIGURED_GENRE`: Genre (e.g., "grimdark science fiction")
- `CONFIGURED_THEME`: Central theme
- `DEFAULT_PROTAGONIST_NAME`: Main character name

## LangGraph Workflow Per Chapter

Each chapter follows this workflow:

```
                    ┌─────────────────────┐
                    │  Route Entry Point  │
                    └──────────┬──────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Initialization       │◄─── First run only
                    │ Complete?            │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        [Yes - Skip Init]            [No - Run Init Workflow]
                │                             │
                │                    ┌────────▼────────┐
                │                    │ Character       │
                │                    │ Sheets          │
                │                    └────────┬────────┘
                │                             ▼
                │                    ┌────────────────┐
                │                    │ Global Outline │
                │                    └────────┬───────┘
                │                             ▼
                │                    ┌────────────────┐
                │                    │ Act Outlines   │
                │                    └────────┬───────┘
                │                             ▼
                │                    ┌────────────────┐
                │                    │ Commit to Neo4j│
                │                    └────────┬───────┘
                │                             ▼
                │                    ┌────────────────┐
                │                    │ Persist YAML   │
                │                    └────────┬───────┘
                │                             │
                └─────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Chapter Outline      │
                    │ (on-demand)          │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Generate Chapter     │
                    │ (draft prose)        │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Extract Entities     │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Commit to Neo4j      │
                    │ (with deduplication) │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Validate Consistency │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
         [Has Issues]                  [Valid]
                │                             │
                ▼                             ▼
        ┌───────────────┐          ┌─────────────────┐
        │ Revise        │          │ Summarize       │
        │ (full-rewrite)│          └────────┬────────┘
        └───────┬───────┘                   │
                │                            ▼
                └───────►[Re-extract]   ┌────────────┐
                                        │ Finalize   │
                                        └─────┬──────┘
                                              ▼
                                     [Next Chapter or END]
```

### Workflow Steps Explained:

1. **Chapter Outline**: Generates chapter-specific outline from act outline and context
2. **Generate**: Builds KG-aware context and drafts prose (3000-4000 words target)
3. **Extract**: Extracts entities (characters, locations, events) and relationships
4. **Commit**: Deduplicates entities using fuzzy matching and persists to Neo4j
5. **Validate**: Checks for continuity contradictions
6. **Revise** (if needed): Full chapter regeneration with validation feedback
7. **Summarize**: Creates chapter summary for future context
8. **Finalize**: Writes chapter to disk and database, advances to next chapter

## Resume After Interruption

If interrupted (Ctrl+C, crash, or power loss):

```bash
# Just run again - it resumes automatically
python main.py
```

The system:
- Queries Neo4j for chapter count
- Creates state for next chapter
- Continues generation seamlessly

**Note**: Initialization only runs once. If you have chapters in Neo4j, it skips init.

## Configuration Examples

### Generate 6 Chapters

```bash
# In .env or config/settings.py
CHAPTERS_PER_RUN=6
```

Then run:
```bash
python main.py
```

**First run:**
- Runs initialization (5-10 minutes)
- Generates chapters 1-6 (30-60 minutes)

**Second run:**
- Skips initialization
- Generates chapters 7-12 (25-45 minutes)

### Customize Novel Settings

```bash
# In .env
DEFAULT_PLOT_OUTLINE_TITLE="The Quantum Heist"
CONFIGURED_GENRE="cyberpunk thriller"
CONFIGURED_THEME="the cost of immortality"
DEFAULT_PROTAGONIST_NAME="Kai Nakamura"
TARGET_CHAPTERS=30
```

## Bootstrap vs Initialization

SAGA provides two ways to set up a story:

### 1. Bootstrap (Optional, Standalone)
```bash
python main.py --bootstrap
```
- Runs **before** generation
- More thorough validation
- Writes to Neo4j and YAML
- Use when you want to review/edit before generating

### 2. Automatic Initialization (Default)
```bash
python main.py  # Runs init automatically if needed
```
- Integrated into first chapter generation
- Faster startup
- Sufficient for most use cases
- Use for quick starts

**Both produce the same outputs:**
- Character sheets in `output/characters/`
- Story outline in `output/outline/`
- World items in `output/world/`
- Entities in Neo4j

## Testing the Framework

For quick testing without full generation:

```bash
# Test initialization workflow only
python init_test.py

# Minimal test (no Neo4j needed)
python init_test_minimal.py

# Visualize workflow graph
python visualize_workflow.py
```

## Troubleshooting

### Initialization Runs Every Time
**Symptom**: "Initialization needed" on every run

**Cause**: No chapters found in Neo4j

**Solution**:
- Check Neo4j connection
- Query: `MATCH (c:Chapter) RETURN count(c)`
- If count is 0, initialization runs again (expected)

### Chapter Generation Fails
**Symptom**: Error during generation node

**Solutions**:
1. Check LLM endpoint: `curl $OPENAI_API_BASE/v1/models`
2. Verify Neo4j connection: `MATCH (n) RETURN count(n)`
3. Check logs: `tail -f output/saga_run.log`
4. Verify character sheets exist: `ls output/characters/`

### Missing min_length Error
**Symptom**: `'min_length' is undefined`

**Solution**: Update to latest version (fixed in recent commits)

### Files Not Created
**Symptom**: No YAML files in `output/`

**Solutions**:
1. Check `BASE_OUTPUT_DIR` in config
2. Verify permissions: `ls -la output/`
3. Check initialization completed: `ls output/characters/`

## Advanced: Inspecting State

### View Checkpoints
```python
import sqlite3
conn = sqlite3.connect('output/.saga/checkpoints.db')
cursor = conn.cursor()

# List checkpoints
cursor.execute("SELECT * FROM checkpoints")
for row in cursor.fetchall():
    print(row)
```

### Query Neo4j Directly
```cypher
// Count chapters generated
MATCH (c:Chapter) RETURN count(c) as chapter_count

// View character profiles
MATCH (char:CharacterProfile)
RETURN char.name, char.role, char.description

// See recent entities
MATCH (n)
WHERE n.first_appearance >= 1
RETURN labels(n)[0] as type, n.name, n.first_appearance
ORDER BY n.first_appearance
LIMIT 20
```

## Performance Expectations

**Hardware:** Consumer-grade (16GB RAM, 8GB VRAM GPU)

**Timings** (with Q4 quantized models):
- Initialization: 5-10 minutes
- Chapter generation: 3-5 minutes per chapter (3000-4000 words)
- Total for 20-chapter novel: 2-3 hours

**Optimization tips:**
- Use larger `CHAPTERS_PER_RUN` for batch generation
- Increase `MIN_CHAPTER_LENGTH_CHARS` for longer chapters
- Adjust `TEMPERATURE_DRAFTING` for creativity vs consistency

## Next Steps

After your first run:

1. ✅ **Review initialization outputs**
   - Check `output/characters/*.yaml`
   - Review `output/outline/structure.yaml`
   - Read `output/outline/beats.yaml`

2. ✅ **Verify Neo4j data**
   - Query characters: `MATCH (c:CharacterProfile) RETURN c`
   - Check relationships: `MATCH ()-[r]->() RETURN type(r), count(r)`

3. ✅ **Read generated chapters**
   - `cat output/chapters/chapter_001.md`
   - Check continuity across chapters

4. ✅ **Review logs**
   - `tail -f output/saga_run.log` (during generation)
   - `cat output/chapter_logs/chapter_001.log` (per-chapter details)

5. ✅ **Edit and regenerate** (if needed)
   - Edit YAML files directly
   - Delete chapters from Neo4j to regenerate: `MATCH (c:Chapter {number: 5}) DETACH DELETE c`

## Additional Documentation

- **Architecture**: `docs/langgraph-architecture.md` - Comprehensive design document
- **Workflow Details**: `docs/WORKFLOW_WALKTHROUGH.md` - Complete data flow walkthrough
- **Migration Progress**: `docs/migrate.md` - NANA→LangGraph transition tracker
- **Schema**: `docs/schema-map.md` - Neo4j schema documentation

---

**Note**: NANA orchestrator has been removed. LangGraph is the only generation pipeline.
