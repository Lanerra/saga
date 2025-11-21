# LangGraph Workflow Visualization

Visual debugging tools for SAGA's LangGraph workflows.

## Overview

The visualization module provides tools to generate visual representations of LangGraph workflows, making it easier to understand the control flow, debug routing logic, and document the system architecture.

## Supported Formats

- **Mermaid Diagrams** (`.md`, `.mmd`) - Text-based diagrams that can be rendered in GitHub, VS Code, and documentation sites
- **PNG Images** (`.png`) - Requires graphviz installation
- **ASCII Text** (`.txt`) - Simple text-based representation for quick debugging

## Quick Start

### Generate All Workflows

Generate Mermaid diagrams for all workflows:

```bash
python visualize_workflow.py --all --output-dir docs/workflows
```

This creates:
- `docs/workflows/workflow_phase1.md` - Extract, Commit, Validate workflow
- `docs/workflows/workflow_phase2.md` - Full chapter generation workflow
- `docs/workflows/workflow_full.md` - Complete SAGA workflow with initialization

### Generate Single Workflow

Generate a specific workflow:

```bash
# Mermaid diagram
python visualize_workflow.py --workflow full --output docs/workflow_full.md

# PNG image (requires graphviz)
python visualize_workflow.py --workflow phase2 --output docs/workflow_phase2.png --format png

# ASCII text
python visualize_workflow.py --workflow phase1 --output debug/workflow.txt --format ascii
```

### Print Summary to Console

Quickly inspect workflow structure without creating files:

```bash
python visualize_workflow.py --workflow full --summary
```

Output:
```
============================================================
  Full SAGA Workflow (Initialization + Generation)
============================================================

Nodes (14):
  1. route
  2. init_error
  3. init_character_sheets
  4. init_global_outline
  5. init_act_outlines
  6. init_commit_to_graph
  7. init_persist_files
  8. init_complete
  9. chapter_outline
  10. generate
  11. extract
  12. commit
  13. validate
  14. revise
  15. summarize
  16. finalize

Edges (16):
  1. route → init_character_sheets [conditional]
  2. route → chapter_outline [conditional]
  3. init_character_sheets → init_global_outline
  4. init_global_outline → init_act_outlines
  ...
```

## Programmatic Usage

### Python API

```python
from core.langgraph.visualization import visualize_workflow, print_workflow_summary
from core.langgraph.workflow import create_full_workflow_graph

# Create graph
graph = create_full_workflow_graph()

# Generate Mermaid diagram
visualize_workflow(
    graph,
    "docs/workflow.md",
    format="mermaid",
    title="Full SAGA Workflow"
)

# Print summary to console
print_workflow_summary(graph, title="Full Workflow")
```

### In Jupyter Notebooks

```python
from core.langgraph.workflow import create_phase2_graph
from core.langgraph.visualization import visualize_workflow

# Create graph
graph = create_phase2_graph()

# Generate Mermaid
visualize_workflow(graph, "workflow.md", format="mermaid")

# Display in notebook (if you have mermaid extension)
from IPython.display import display, Markdown

with open("workflow.md") as f:
    display(Markdown(f.read()))
```

## Workflow Types

### Phase 1: Extract → Commit → Validate

Minimal workflow focusing on entity extraction and validation:

```bash
python visualize_workflow.py --workflow phase1 --output docs/phase1.md
```

**Nodes**: extract, commit, validate, revise
**Flow**: Linear with optional revision loop

### Phase 2: Full Chapter Generation

Complete chapter generation with all quality control:

```bash
python visualize_workflow.py --workflow phase2 --output docs/phase2.md
```

**Nodes**: generate, extract, commit, validate, revise, summarize, finalize
**Flow**: Linear with conditional revision and finalization

### Full Workflow: Initialization + Generation

Complete SAGA workflow including initialization:

```bash
python visualize_workflow.py --workflow full --output docs/full.md
```

**Nodes**:
- Initialization: route, init_character_sheets, init_global_outline, init_act_outlines, init_commit_to_graph, init_persist_files, init_complete
- Generation: chapter_outline, generate, extract, commit, validate, revise, summarize, finalize

**Flow**: Conditional entry (init vs generation), complex routing logic

## Viewing Mermaid Diagrams

### GitHub
Mermaid diagrams render automatically in GitHub markdown files and READMEs.

### VS Code
Install the "Markdown Preview Mermaid Support" extension:
1. Open Extensions (Ctrl+Shift+X)
2. Search for "Markdown Preview Mermaid"
3. Install and reload
4. Open `.md` file and preview (Ctrl+Shift+V)

### Command Line
Use `mmdc` (Mermaid CLI):
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i docs/workflow_full.md -o docs/workflow_full.png
```

### Online
Copy the Mermaid code and paste into:
- https://mermaid.live/
- https://mermaid.ink/

## PNG Export (Optional)

PNG export requires graphviz:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz graphviz-dev
pip install pygraphviz

# macOS
brew install graphviz
pip install pygraphviz

# Then generate PNG
python visualize_workflow.py --workflow full --output workflow.png --format png
```

## Troubleshooting

### "No module named 'structlog'"

Ensure you're running in the correct Python environment:

```bash
# If using poetry
poetry run python visualize_workflow.py --workflow full --summary

# If using venv
source venv/bin/activate
python visualize_workflow.py --workflow full --summary
```

### "graphviz not found" (PNG export)

Install graphviz system package and Python bindings (see PNG Export section above).

### Empty or incorrect diagram

Make sure you're visualizing a compiled graph:

```python
from core.langgraph.workflow import create_full_workflow_graph

# ✓ Correct - compiled graph
graph = create_full_workflow_graph()

# ✗ Wrong - StateGraph builder (not compiled)
from langgraph.graph import StateGraph
workflow = StateGraph(NarrativeState)  # Not compiled yet
```

## Integration with Development Workflow

### Pre-commit Hook

Generate diagrams automatically before commits:

```bash
# .git/hooks/pre-commit
#!/bin/bash
python visualize_workflow.py --all --output-dir docs/workflows
git add docs/workflows/*.md
```

### CI/CD

Generate diagrams in CI for documentation:

```yaml
# .github/workflows/docs.yml
- name: Generate workflow diagrams
  run: |
    python visualize_workflow.py --all --output-dir docs/workflows
    git add docs/workflows
    git commit -m "docs: update workflow diagrams" || true
```

### Documentation Sites

Include diagrams in MkDocs, Sphinx, or other doc generators:

```markdown
## Workflow Architecture

```mermaid
{{< include docs/workflows/workflow_full.md >}}
```
```

## API Reference

### `visualize_workflow(graph, output_path, format, title)`

Generate a visual representation of a workflow graph.

**Parameters:**
- `graph` - Compiled LangGraph StateGraph instance
- `output_path` (str|Path) - Path to save visualization file
- `format` ("mermaid"|"png"|"ascii") - Output format
- `title` (str, optional) - Diagram title

**Returns:** Path - Path to created file

**Raises:**
- `ImportError` - If required dependencies missing (e.g., graphviz for PNG)
- `ValueError` - If format not supported

### `print_workflow_summary(graph, title)`

Print workflow structure to console.

**Parameters:**
- `graph` - Compiled LangGraph StateGraph instance
- `title` (str, optional) - Title to display

**Returns:** None (prints to stdout)

## Examples

### Debug Conditional Routing

```python
from core.langgraph.workflow import create_full_workflow_graph
from core.langgraph.visualization import print_workflow_summary

graph = create_full_workflow_graph()
print_workflow_summary(graph)

# Look for conditional edges to understand routing logic
# Example: route → init_character_sheets [if initialization_complete=False]
#          route → chapter_outline [if initialization_complete=True]
```

### Document API Changes

```bash
# Before making changes
python visualize_workflow.py --workflow phase2 --output docs/workflow_before.md

# Make workflow changes
# ... edit core/langgraph/workflow.py ...

# After changes
python visualize_workflow.py --workflow phase2 --output docs/workflow_after.md

# Compare
diff docs/workflow_before.md docs/workflow_after.md
```

### Generate Presentation Slides

```bash
# Generate PNGs for slides
python visualize_workflow.py --workflow phase1 --output slides/phase1.png --format png
python visualize_workflow.py --workflow phase2 --output slides/phase2.png --format png
python visualize_workflow.py --workflow full --output slides/full.png --format png
```

## Related Documentation

- [LangGraph Architecture](langgraph-architecture.md) - Overall design
- [Workflow Walkthrough](WORKFLOW_WALKTHROUGH.md) - Detailed node descriptions
- [Phase 2 Migration Plan](phase2_migration_plan.md) - Migration guide
