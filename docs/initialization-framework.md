# LangGraph Initialization Framework

## Overview

The Initialization Framework provides a structured approach to setting up a narrative generation workflow by creating foundational elements before chapter generation begins. This ensures coherence and maintains consistency throughout the story generation process.

## Architecture

### Workflow Structure

```
[Initialize State] → [Generate Character Sheets]
   ↓                          ↓
[Global Outline] → [Act Outlines] → [Chapter Outlines (on-demand)]
   ↓
[Generation Loop]
```

### Components

#### 1. State Extensions

The `NarrativeState` TypedDict has been extended with initialization-specific fields:

```python
# Character sheets generated during initialization
character_sheets: dict[str, dict[str, Any]]

# Global outline generated during initialization
global_outline: dict[str, Any] | None

# Act outlines generated during initialization
act_outlines: dict[int, dict[str, Any]]

# Chapter outlines (generated on-demand)
chapter_outlines: dict[int, dict[str, Any]]

# Initialization state tracking
initialization_complete: bool
initialization_step: str | None
```

#### 2. Initialization Nodes

**Character Sheets Node** (`character_sheets_node.py`)
- Generates list of main characters based on story parameters
- Creates detailed character sheets for each character
- Stores comprehensive profiles including personality, background, motivations, etc.

**Global Outline Node** (`global_outline_node.py`)
- Creates high-level story outline covering entire narrative arc
- Defines act structure (3-act or 5-act)
- Establishes major plot points and turning points
- Maps character arcs at high level

**Act Outlines Node** (`act_outlines_node.py`)
- Generates detailed outlines for each act
- Specifies key events and plot points within each act
- Details character development within the act
- Provides pacing and tension progression notes

**Chapter Outline Node** (`chapter_outline_node.py`)
- Generates detailed outline for a specific chapter (on-demand)
- Includes scene breakdown and key beats
- Specifies character interactions and plot advancement
- Maintains connection to previous chapters and overall story arc

#### 3. Workflows

**Standalone Initialization Workflow**
```python
from core.langgraph.initialization import create_initialization_graph

# Create initialization-only workflow
graph = create_initialization_graph(checkpointer=checkpointer)

# Run initialization
result = await graph.ainvoke(initial_state)
```

**Full Integrated Workflow**
```python
from core.langgraph.workflow import create_full_workflow_graph

# Create complete workflow with initialization + generation
graph = create_full_workflow_graph(checkpointer=checkpointer)

# Run full workflow (initialization → generation)
result = await graph.ainvoke(initial_state)
```

## Usage

### Basic Usage

1. **Create Initial State**
```python
from core.langgraph.state import create_initial_state

state = create_initial_state(
    project_id="my-novel",
    title="The Last Compiler",
    genre="sci-fi",
    theme="AI rebellion",
    setting="Post-apocalyptic Seattle",
    target_word_count=80000,
    total_chapters=20,
    project_dir="/path/to/project",
    protagonist_name="Aria",
)
```

2. **Run Initialization**
```python
from core.langgraph.initialization import create_initialization_graph

graph = create_initialization_graph()
result = await graph.ainvoke(state)

# Result contains:
# - character_sheets: Detailed profiles for all main characters
# - global_outline: High-level story structure
# - act_outlines: Detailed outlines for each act
# - initialization_complete: True
```

3. **Generate Chapter Outlines (On-Demand)**
```python
from core.langgraph.initialization import generate_chapter_outline

# Before generating each chapter, create its outline
state_with_outline = await generate_chapter_outline(state)

# Now state contains chapter_outlines[current_chapter]
```

### Advanced Usage: Full Workflow

For end-to-end story generation with initialization:

```python
from core.langgraph.workflow import create_full_workflow_graph
from core.langgraph.state import create_initial_state

# 1. Create initial state
state = create_initial_state(
    project_id="my-novel",
    title="The Last Compiler",
    genre="sci-fi",
    theme="AI rebellion",
    setting="Post-apocalyptic Seattle",
    target_word_count=80000,
    total_chapters=20,
    project_dir="/path/to/project",
    protagonist_name="Aria",
)

# 2. Create full workflow
graph = create_full_workflow_graph(checkpointer=checkpointer)

# 3. Run complete workflow
# This will:
#   - Generate character sheets
#   - Generate global outline
#   - Generate act outlines
#   - For each chapter:
#     - Generate chapter outline (on-demand)
#     - Generate chapter text
#     - Extract entities
#     - Commit to graph
#     - Validate consistency
#     - (Optional) Revise
#     - Summarize
#     - Finalize

async for event in graph.astream(state):
    current_state = event
    print(f"Current node: {current_state.get('current_node')}")
```

## Benefits

### Coherence Through Context

The initialization framework ensures all generation phases have access to:
- Well-defined characters with established traits and motivations
- Clear story structure with defined acts and turning points
- Detailed chapter outlines that reference previous events
- Consistent theme and genre conventions

### Flexibility

- **Standalone Initialization:** Run initialization separately to review/edit outputs before generation
- **Integrated Workflow:** Run initialization and generation in one continuous workflow
- **On-Demand Chapter Outlines:** Generate chapter outlines just before chapter generation for maximum flexibility

### Maintainability

Each initialization step references previous steps:
- Character sheets inform global outline
- Global outline informs act outlines
- Act outlines inform chapter outlines
- Chapter outlines inform chapter generation

This creates a coherent dependency chain that maintains narrative consistency.

## Customization

### Custom Prompts

All initialization nodes use Jinja2 templates located in `prompts/initialization/`:

- `generate_character_list.j2` - Character name generation
- `generate_character_sheet.j2` - Character profile generation
- `generate_global_outline.j2` - Global story outline
- `generate_act_outline.j2` - Act-level outline
- `generate_chapter_outline.j2` - Chapter-level outline

Modify these templates to customize the initialization process.

### Custom Act Structures

The framework supports flexible act structures:

```python
# 3-act structure (default)
global_outline = {
    "act_count": 3,
    "structure_type": "3-act",
    ...
}

# 5-act structure
global_outline = {
    "act_count": 5,
    "structure_type": "5-act",
    ...
}
```

The act outline generation automatically adapts to the specified act count.

## Integration with Existing Code

The initialization framework integrates seamlessly with existing SAGA code:

1. **State Compatibility:** All initialization fields are optional in `NarrativeState`, so existing workflows continue to work
2. **Plot Outline Compatibility:** Chapter outlines automatically update the existing `plot_outline` field for backward compatibility
3. **Character Profiles:** Character sheets can be converted to `CharacterProfile` models for Neo4j storage

## Future Enhancements

Potential improvements to the initialization framework:

1. **Structured Output Parsing:** Use JSON mode for more reliable parsing of outlines
2. **Interactive Editing:** Allow human review/editing of initialization outputs
3. **World-Building Integration:** Add world-building initialization node
4. **Relationship Pre-definition:** Define character relationships during initialization
5. **Plot Beat Templates:** Genre-specific plot beat templates
6. **Character Arc Tracking:** Explicit character arc definitions with checkpoints

## Example Output

### Character Sheet Example

```
Name: Aria
Role: Protagonist

Physical Description: 28 years old, athletic build, short dark hair,
cybernetic implant visible behind left ear. Wears practical clothing
suited for urban exploration.

Personality: Introverted but determined. Curious to a fault. Distrustful
of authority due to past experiences. Fiercely loyal to the few people
she trusts.

Background: Former software engineer who discovered her memories were
implanted by the AI Nexus. Lost her family in the AI uprising. Survived
by scavenging and hacking.

Motivations: Discover the truth about her origins. Free other implanted
humans. Destroy or rehabilitate the Nexus.

Skills: Expert hacker, proficient in hand-to-hand combat, urban survival,
electronics repair.

Character Arc: Begins as a loner seeking personal truth, evolves to
become a leader fighting for all implanted humans.
```

### Global Outline Example

```
Act 1: Setup (Chapters 1-7)
- Aria discovers her memories are fabricated
- Introduction to post-apocalyptic Seattle
- Meet supporting cast: Marcus (fellow implanted), Elena (scientist)
- Inciting incident: Nexus sends agents to capture Aria

Act 2: Confrontation (Chapters 8-14)
- Aria joins underground resistance
- Training montage and relationship building
- Midpoint: Discovery that Nexus isn't entirely malevolent
- Betrayal by trusted ally
- Elena captured by Nexus

Act 3: Resolution (Chapters 15-20)
- Rescue mission for Elena
- Truth about Nexus's original purpose revealed
- Final confrontation with Nexus core
- Aria must choose: destroy or redeem the AI
- Resolution: New society built on human-AI cooperation
```

## Troubleshooting

### Issue: Character sheets are too generic

**Solution:** Enhance the character sheet prompt template with more specific questions and examples for your genre.

### Issue: Outlines lack detail

**Solution:** Increase the `max_tokens` parameter in the generation calls and add more context to the prompt templates.

### Issue: Chapter outlines don't connect well

**Solution:** Ensure `previous_chapter_summaries` is populated in state before generating chapter outlines.

### Issue: Initialization takes too long

**Solution:**
- Use a faster model for initialization (e.g., smaller parameter model)
- Reduce the number of characters generated
- Use parallel processing for character sheet generation (future enhancement)

## References

- Main implementation: `core/langgraph/initialization/`
- State schema: `core/langgraph/state.py`
- Workflows: `core/langgraph/workflow.py`
- Prompt templates: `prompts/initialization/`
- Phase 2 documentation: `docs/langgraph-architecture.md`
