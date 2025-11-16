# init_test_minimal.py
"""
Minimal test - just verify the initialization graph can be created.
"""

from core.langgraph.initialization import create_initialization_graph

# Test 1: Create graph without checkpointing
print("Test 1: Creating initialization graph (no checkpointing)...")
graph = create_initialization_graph(checkpointer=None)
print("✓ Graph created successfully!")

# Test 2: Show graph structure
print("\nGraph structure:")
print(f"  Entry point: {graph.get_graph().nodes}")
print(f"  Nodes: {len(graph.get_graph().nodes)} nodes")

print("\n✓ All tests passed!")
