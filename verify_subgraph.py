#!/usr/bin/env python3
"""Verify scene_extraction subgraph still works."""
import sys
sys.path.insert(0, '/home/dlewis3/Desktop/AI/saga')

from core.langgraph.subgraphs.scene_extraction import create_scene_extraction_subgraph
print("Subgraph import OK")
graph = create_scene_extraction_subgraph()
print(f"Graph nodes: {graph.nodes}")
