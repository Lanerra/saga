"""Core package initialization.

NOTE: Some tests patch `core.knowledge_graph_service.knowledge_graph_service`.
`unittest.mock.patch()` resolves dotted names by attribute-walking the package,
so we import the submodule here to ensure `core.knowledge_graph_service` exists.
"""

from __future__ import annotations

# Compatibility import for patch targets / legacy call-sites.
# This makes `core.knowledge_graph_service` resolvable as an attribute.
from . import knowledge_graph_service as knowledge_graph_service  # noqa: F401
