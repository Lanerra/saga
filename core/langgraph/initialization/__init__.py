"""
Initialization nodes for LangGraph-based narrative generation.

This package contains nodes for the initialization phase of the narrative
generation workflow, which generates character sheets, outlines, and other
foundational elements before beginning chapter generation.

Initialization Workflow:
    [Initialize State] → [Generate Character Sheets]
       ↓                          ↓
    [Global Outline] → [Act Outlines] → [Chapter Outlines (on-demand)]
       ↓
    [Generation Loop]
"""

from core.langgraph.initialization.act_outlines_node import generate_act_outlines
from core.langgraph.initialization.chapter_outline_node import (
    generate_chapter_outline,
)
from core.langgraph.initialization.character_sheets_node import (
    generate_character_sheets,
)
from core.langgraph.initialization.commit_init_node import (
    commit_initialization_to_graph,
)
from core.langgraph.initialization.global_outline_node import generate_global_outline
from core.langgraph.initialization.persist_files_node import (
    persist_initialization_files,
)
from core.langgraph.initialization.workflow import create_initialization_graph

__all__ = [
    "generate_character_sheets",
    "generate_global_outline",
    "generate_act_outlines",
    "generate_chapter_outline",
    "commit_initialization_to_graph",
    "persist_initialization_files",
    "create_initialization_graph",
]
