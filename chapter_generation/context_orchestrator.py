# chapter_generation/context_orchestrator.py
"""Orchestrates context generation from multiple providers."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

import structlog
from config import settings
from core.llm_interface import count_tokens, truncate_text_by_tokens

from models.agent_models import SceneDetail

from .context_providers import ContextChunk, ContextProvider, ContextRequest

logger = structlog.get_logger(__name__)


class TTLCache:
    """Simple TTL-based LRU cache."""

    def __init__(self, maxsize: int, ttl: float) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: dict[tuple, tuple[float, str]] = {}

    def get(self, key: tuple) -> str | None:
        item = self._data.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self.ttl:
            del self._data[key]
            return None
        return value

    def set(self, key: tuple, value: str) -> None:
        if len(self._data) >= self.maxsize:
            oldest = sorted(self._data.items(), key=lambda x: x[1][0])[0][0]
            del self._data[oldest]
        self._data[key] = (time.time(), value)


class ContextOrchestrator:
    """Gather and merge context from configured providers."""

    def __init__(self, providers: Iterable[ContextProvider]) -> None:
        self.providers = list(providers)
        self.cache = TTLCache(settings.CONTEXT_CACHE_SIZE, settings.CONTEXT_CACHE_TTL)

    async def build_context(self, request: ContextRequest) -> str:
        """Return an ordered context string for the request."""
        agent_key = None
        if request.agent_hints:
            try:
                agent_key = json.dumps(request.agent_hints, sort_keys=True, default=str)
            except Exception as exc:  # pragma: no cover - log and continue
                logger.warning(
                    "Failed to serialize agent hints for cache key", error=exc
                )

        cache_key = (
            request.chapter_number,
            request.plot_focus,
            agent_key,
        )
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug("Context cache hit", key=cache_key)
            return cached

        chunks: list[ContextChunk] = []
        for provider in self.providers:
            try:
                res = await provider.get_context(request)
            except Exception as exc:  # pragma: no cover - log and skip
                logger.warning(
                    "Context provider error", provider=provider.source, error=exc
                )
                continue
            if isinstance(res, ContextChunk):
                chunks.append(res)
            else:
                logger.warning(
                    "Invalid context provider result",
                    provider=provider.source,
                    result_type=type(res).__name__,
                )

        merged: list[str] = []
        token_total = 0
        for chunk in chunks:
            if not chunk.text:
                continue
            prefix = f"[{chunk.source}]\n"
            part = prefix + chunk.text
            tokens = count_tokens(part, settings.DRAFTING_MODEL)
            if token_total + tokens > settings.MAX_CONTEXT_TOKENS:
                remaining = settings.MAX_CONTEXT_TOKENS - token_total
                if remaining <= 0:
                    break
                part = truncate_text_by_tokens(part, settings.DRAFTING_MODEL, remaining)
                merged.append(part)
                token_total += remaining
                break
            merged.append(part)
            token_total += tokens

        final_context = "\n---\n".join(merged)
        self.cache.set(cache_key, final_context)
        logger.info("Built context", tokens=token_total)
        return final_context

    async def build_hybrid_context(
        self,
        agent_or_props: Any,
        current_chapter_number: int,
        chapter_plan: list[SceneDetail] | None,
        agent_hints: dict[str, Any] | None = None,
    ) -> str:
        """Backward compatible wrapper for build_context."""
        if isinstance(agent_or_props, dict):
            plot_outline = agent_or_props.get(
                "plot_outline_full", agent_or_props.get("plot_outline", {})
            )
            plot_focus = None
        else:
            plot_outline = getattr(
                agent_or_props, "plot_outline_full", None
            ) or getattr(agent_or_props, "plot_outline", {})
            plot_focus = getattr(agent_or_props, "plot_point_focus", None)

        if hasattr(plot_outline, "model_dump"):
            try:
                plot_outline = plot_outline.model_dump(exclude_none=True)
            except Exception as exc:  # pragma: no cover - log and continue
                logger.warning(
                    "Failed to dump plot outline to dict", error=exc, exc_info=True
                )

        request = ContextRequest(
            chapter_number=current_chapter_number,
            plot_focus=plot_focus,
            plot_outline=plot_outline,
            chapter_plan=chapter_plan,
            agent_hints=agent_hints,
        )
        return await self.build_context(request)
