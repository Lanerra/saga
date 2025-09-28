"""
Chapter-scoped narrative state and context snapshot containers.

These lightweight classes centralize deterministic context built once per
chapter and passed by reference through the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

# Import directly from leaf modules to avoid circular imports through models/__init__
from .agent_models import SceneDetail


class ContextSnapshot:
    """
    Immutable snapshot of prompt-context for a chapter/scene.
    """

    def __init__(
        self,
        *,
        chapter_number: int,
        plot_point_focus: str | None,
        chapter_plan: list[SceneDetail] | None,
        hybrid_context: str,
        kg_facts_block: str,
        recent_chapters_map: dict[int, dict[str, Any]],
    ) -> None:
        self.chapter_number = chapter_number
        self.plot_point_focus = plot_point_focus
        self.chapter_plan = chapter_plan
        self.hybrid_context = hybrid_context
        self.kg_facts_block = kg_facts_block
        self.recent_chapters_map = recent_chapters_map

        self.snapshot_fingerprint = self._compute_fingerprint()
        self.created_ts = time.time()

    def _compute_fingerprint(self) -> str:
        """Compute a stable fingerprint of the snapshot contents."""
        def _stable(obj: Any) -> str:
            try:
                return json.dumps(obj, sort_keys=True, ensure_ascii=False)
            except Exception:
                return str(obj)

        parts = [
            str(self.chapter_number),
            self.plot_point_focus or "",
            _stable(self.chapter_plan),
            self.hybrid_context or "",
            self.kg_facts_block or "",
            _stable({k: self.recent_chapters_map.get(k) for k in sorted(self.recent_chapters_map.keys())}),
        ]
        joined = "\n".join(parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class NarrativeState:
    """
    Mutable per-chapter state container, passed by reference.
    """

    def __init__(
        self,
        *,
        neo4j: Any,
        llm: Any,
        embedding: Any | None,
        plot_outline: dict[str, Any],
    ) -> None:
        # Injection handles (no globals)
        self.neo4j = neo4j
        self.llm = llm
        self.embedding = embedding

        # Chapter-scoped fields
        self.plot_outline = plot_outline
        self.context_epoch: int = 0
        self.snapshot: ContextSnapshot | None = None
        self.caches: dict[str, Any] = {}
        self.reads_locked: bool = False

