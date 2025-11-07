# Phase 2 End-to-End Test

This directory contains the end-to-end test script for validating Phase 2 LangGraph workflow with real LLM calls.

## Test Script: `test_phase2_e2e.py`

### Purpose

Validates all Phase 2 success criteria from `docs/phase2_migration_plan.md`:

- ✅ Generate single chapter end-to-end with real LLM
- ✅ Revision loop works with contradiction detection
- ✅ Chapter summaries persist to Neo4j and guide future generation
- ✅ Performance: <5 minutes per chapter
- ✅ Full test coverage for all nodes
- ✅ File output and persistence

### Prerequisites

1. **Neo4j Database Running**
   ```bash
   # Check if Neo4j is accessible
   docker ps | grep neo4j
   # Or if running locally
   sudo systemctl status neo4j
   ```

2. **LLM Endpoint Configured**
   - Check `.env` file for `OPENAI_API_BASE`
   - Ensure LLM server is running (e.g., llama.cpp, vLLM, OpenAI API)

3. **Environment Setup**
   ```bash
   # From project root
   source venv/bin/activate  # or your virtualenv
   pip install -r requirements.txt
   ```

### Usage

```bash
# From project root
python scripts/test_phase2_e2e.py
```

**OR** (if made executable):

```bash
./scripts/test_phase2_e2e.py
```

### What It Does

The script executes a complete single-chapter generation workflow:

1. **Setup**
   - Connects to Neo4j
   - Creates test novel outline ("The Last Signal" - sci-fi short story)
   - Initializes project directory in `output/phase2_e2e_test/`

2. **Workflow Execution**
   - `generate_chapter`: Generate prose from outline
   - `extract_entities`: Extract characters, locations, objects
   - `commit_to_graph`: Deduplicate and persist to Neo4j
   - `validate_consistency`: Check for contradictions
   - `{revise OR summarize}`: Fix issues or generate summary
   - `finalize_chapter`: Save to file and Neo4j

3. **Validation**
   - Verifies chapter text generated
   - Checks word count
   - Validates entity extraction
   - Confirms file output
   - Measures performance

4. **Reporting**
   - Shows detailed progress
   - Reports success criteria status
   - Displays performance metrics
   - Provides next steps

### Expected Output

```
================================================================================
SAGA Phase 2 End-to-End Test
================================================================================

Configuration:
  Generation Model: qwen3-a3b
  Extraction Model: qwen3-a3b
  Revision Model: qwen3-a3b
  Neo4j URI: bolt://localhost:7687
  LLM Endpoint: http://127.0.0.1:8080/v1

Step 1: Connecting to Neo4j...
✓ Neo4j connection established

Step 2: Creating test novel outline...
✓ Created outline for: The Last Signal
  Genre: Science Fiction
  Theme: Isolation and Communication
  Chapters: 1

Step 3: Created project directory: output/phase2_e2e_test

Step 4: Initializing LangGraph state...
✓ State initialized
  Current Chapter: 1
  Plot Point: Dr. Sarah Chen receives a mysterious signal...
  Target Word Count: 4000

Step 5: Creating Phase 2 workflow graph...
✓ Workflow graph compiled
  Nodes: generate → extract → commit → validate → {revise OR summarize} → finalize

Step 6: Executing workflow...
--------------------------------------------------------------------------------
[... workflow execution logs ...]
--------------------------------------------------------------------------------
✓ Workflow completed in 45.32 seconds

Step 7: Validating results...
✓ Chapter generated: 4,127 words
  Preview: Dr. Sarah Chen floated through the narrow corridor...
✓ Revision iterations: 0
✓ No contradictions detected
✓ Entities extracted: 12
✓ Relationships extracted: 8
✓ Chapter summary generated:
  Sarah Chen detects a mysterious artificial signal from deep space...
✓ Chapter file created: chapter_001.md (23456 bytes)

Step 8: Performance Metrics...
  Total Time: 45.32 seconds
  Words Generated: 4127
  Generation Rate: 91.1 words/second

Step 9: Phase 2 Success Criteria Check...
--------------------------------------------------------------------------------
✓ Generate single chapter end-to-end with real LLM
✓ Revision loop works (iterations: 0)
✓ Chapter summaries generated and available for context
✓ Performance: <5 minutes (45.3s)
✓ Chapter file created successfully
--------------------------------------------------------------------------------

🎉 SUCCESS: All Phase 2 success criteria met!

Next Steps:
  - Review generated chapter in: output/phase2_e2e_test/chapters/chapter_001.md
  - Check Neo4j graph for extracted entities
  - Run multi-chapter test if single chapter looks good

Cleaning up...
✓ Neo4j connection closed
```

### Output Files

After successful execution:

```
output/phase2_e2e_test/
├── chapters/
│   └── chapter_001.md          # Generated chapter with frontmatter
└── saga_run.log                # Detailed execution logs
```

### Troubleshooting

#### Neo4j Connection Failed
```
✗ Neo4j connection failed: Neo4j service unavailable
```
**Fix:** Ensure Neo4j is running and accessible at configured URI

#### LLM Call Failed
```
Error: Connection refused when calling LLM
```
**Fix:** Check LLM endpoint in `.env` and ensure server is running

#### Out of Memory
```
Error: CUDA out of memory / System killed process
```
**Fix:** Reduce `MAX_GENERATION_TOKENS` in `.env` or use smaller model

#### Performance Too Slow
```
⚠ Performance: >300s (target: <300s)
```
**Fix:** Use faster model, GPU acceleration, or reduce token budget

### Configuration Options

You can modify the test by editing `test_phase2_e2e.py`:

```python
# Line 104: Change target word count
target_word_count=4000,  # Reduce for faster testing

# Line 107-109: Change models
generation_model=settings.LARGE_MODEL,   # Use different model
extraction_model=settings.SMALL_MODEL,
revision_model=settings.MEDIUM_MODEL,

# Line 121: Enable force_continue (skip validation)
state["force_continue"] = True  # Skip revision loop
```

### Next Steps After Success

1. **Review Generated Chapter**
   ```bash
   cat output/phase2_e2e_test/chapters/chapter_001.md
   ```

2. **Check Neo4j Graph**
   - Open Neo4j Browser: http://localhost:7474
   - Run query: `MATCH (c:Character) RETURN c LIMIT 25`

3. **Multi-Chapter Test**
   - Modify script to generate 3-5 chapters
   - Verify narrative continuity
   - Check summary context propagation

4. **Performance Optimization**
   - Profile with different models
   - Test with GPU vs CPU
   - Measure memory usage

### Related Documentation

- Phase 2 Migration Plan: `docs/phase2_migration_plan.md`
- LangGraph Architecture: `docs/langgraph-architecture.md`
- Success Criteria: `docs/phase2_migration_plan.md` (lines 35-42, 585-604)

### Support

If you encounter issues:

1. Check logs in `output/phase2_e2e_test/saga_run.log`
2. Enable debug logging: `LOG_LEVEL=DEBUG python scripts/test_phase2_e2e.py`
3. Review Phase 1 tests to ensure foundation is working: `pytest tests/test_langgraph/`
