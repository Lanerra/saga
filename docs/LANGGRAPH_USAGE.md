# LangGraph Pipeline Usage Guide

## Overview

SAGA now supports two generation pipelines:
1. **Legacy NANA Pipeline** (default) - The original orchestrator
2. **LangGraph Pipeline** (new) - Graph-based workflow with initialization

## Quick Start

### Using Legacy Pipeline (Default)
```bash
python main.py
```

### Using LangGraph Pipeline
```bash
python main.py --langgraph
```

## What Happens with `--langgraph`

### First Run (Initialization)
```
1. Connects to Neo4j
2. Creates initial state
3. Detects initialization is needed
4. Runs initialization workflow:
   - Generates character sheets
   - Generates global outline (3 or 5 acts)
   - Generates detailed act outlines
   - Commits to Neo4j (CharacterProfile models)
   - Writes human-readable YAML files
5. Generates CHAPTERS_PER_RUN chapters
6. Saves checkpoint for resumption
```

### Subsequent Runs (Resume)
```
1. Connects to Neo4j
2. Loads state from checkpoint
3. Detects initialization is complete
4. Resumes from last chapter
5. Generates next CHAPTERS_PER_RUN chapters
6. Updates checkpoint
```

## Generated Files

When using `--langgraph`, you'll see:

```
output/
├── .saga/
│   └── checkpoints.db          # LangGraph state
├── characters/
│   ├── aria.yaml
│   ├── marcus.yaml
│   └── ...
├── outline/
│   ├── structure.yaml          # Act structure
│   └── beats.yaml              # Full outlines
├── world/
│   └── items.yaml              # Locations, objects
├── chapters/
│   ├── chapter_001.md
│   └── ...
└── summaries/
    └── ...
```

## Configuration

Uses existing SAGA config variables:
- `CHAPTERS_PER_RUN`: Chapters to generate per run (default: 2)
- `TARGET_CHAPTERS`: Total chapters for novel (default: 20)
- `DEFAULT_MODEL_NAME`: Model for all generation
- `PROJECT_ID`, `NOVEL_TITLE`, `NOVEL_GENRE`, etc.

## Workflow Per Chapter

Each chapter goes through:
```
1. Generate chapter outline (on-demand, uses act outlines)
2. Generate chapter text (from outline + Neo4j context)
3. Extract entities (characters, locations, events)
4. Commit to Neo4j (with deduplication)
5. Validate consistency (check contradictions)
6. Revise if needed (up to max_iterations)
7. Summarize chapter
8. Finalize (write to disk)
```

## Resume After Interruption

If interrupted (Ctrl+C or crash):
```bash
# Just run again - it will resume from checkpoint
python main.py --langgraph
```

The checkpointer tracks:
- Which chapter was being processed
- All initialization data
- Previous chapter summaries
- Character profiles in state

## Comparison: Legacy vs LangGraph

| Feature | Legacy (NANA) | LangGraph |
|---------|---------------|-----------|
| Initialization | Bootstrap pipeline | Structured init workflow |
| State Management | Custom NarrativeState | LangGraph checkpointer |
| Resume | Manual chapter tracking | Automatic via checkpoint |
| Data Persistence | Neo4j only | Neo4j + YAML files |
| Workflow | Procedural code | Graph-based nodes |
| Revision Loop | Manual logic | Graph conditional edges |
| Context Building | Custom functions | Same (reused) |
| Neo4j Queries | Same | Same |

## Example: Generate 6 Chapters

Set in `.env` or config:
```
CHAPTERS_PER_RUN=6
```

Then run:
```bash
python main.py --langgraph
```

First run:
- Runs initialization (once)
- Generates chapters 1-6
- Total time: ~30-60 min (depending on model)

Second run:
- Skips initialization
- Generates chapters 7-12
- Total time: ~25-45 min

## Testing the Framework

For quick testing without full generation:
```bash
# Test just initialization
python init_test.py

# Test minimal (no Neo4j needed)
python init_test_minimal.py
```

## Troubleshooting

**Issue**: "Initialization not complete" every run
- Check `.saga/checkpoints.db` exists
- Verify `initialization_complete` field in state

**Issue**: Chapter generation fails
- Check Neo4j connection
- Verify characters exist in graph
- Check logs for specific errors

**Issue**: Files not created
- Check `OUTPUT_DIR` in config
- Verify permissions on output directory
- Check logs for file write errors

## Advanced: Inspecting State

```python
# View checkpoint state
import sqlite3
conn = sqlite3.connect('output/.saga/checkpoints.db')
# Query checkpoints table
```

## Next Steps

After testing with `--langgraph`:
1. Compare output quality with legacy pipeline
2. Check YAML files for initialization data
3. Verify Neo4j has all entities
4. Review generated chapters in `output/chapters/`
5. Provide feedback on initialization vs bootstrap

## Notes

- LangGraph pipeline is experimental
- Legacy pipeline remains default
- Both use same Neo4j database
- Both respect same config variables
- Can switch between runs (though not recommended)

---

For more details on the initialization framework:
See `docs/initialization-framework.md`

For LangGraph architecture:
See `docs/langgraph-architecture.md`
