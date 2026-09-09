"""Run-owned services, scoped to async tasks rather than imported module aliases."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import config
from config.settings import EffectiveSettings

if TYPE_CHECKING:
    import numpy as np

    from core.db_manager import Neo4jManagerSingleton


class LanguageModel(Protocol):
    async def aclose(self) -> None: ...

    async def async_call_llm(
        self, model_name: str, prompt: str, temperature: float | None = None,
        max_tokens: int | None = None, allow_fallback: bool = False,
        auto_clean_response: bool = True, spacy_cleanup: bool = False, *,
        system_prompt: str | None = None, strict: bool = True, **kwargs: Any,
    ) -> tuple[str, dict[str, int] | None]: ...

    async def async_call_llm_json_object(
        self, model_name: str, prompt: str, temperature: float | None = None,
        max_tokens: int | None = None, allow_fallback: bool = False,
        auto_clean_response: bool = True, *, system_prompt: str | None = None,
        strict: bool = True, max_attempts: int = ..., **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, int] | None]: ...

    async def async_call_llm_json_array(
        self, model_name: str, prompt: str, temperature: float | None = None,
        max_tokens: int | None = None, allow_fallback: bool = False,
        auto_clean_response: bool = True, *, system_prompt: str | None = None,
        strict: bool = True, max_attempts: int = ..., **kwargs: Any,
    ) -> tuple[list[Any], dict[str, int] | None]: ...

    async def async_get_embedding(self, text: str) -> np.ndarray | None: ...

    async def async_get_embeddings_batch(self, texts: list[str], batch_size: int | None = None) -> list[np.ndarray | None]: ...

    def count_tokens(self, text: str, model_name: str) -> int: ...

    def truncate_text_by_tokens(self, text: str, model_name: str, max_tokens: int, truncation_marker: str = "\n... (truncated)") -> str: ...

    def get_combined_statistics(self) -> dict[str, Any]: ...


@dataclass
class RunServices:
    language_model: LanguageModel
    database: Neo4jManagerSingleton
    _configuration: EffectiveSettings = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        provider_configuration = getattr(self.language_model, "configuration", None)
        self._configuration = provider_configuration if isinstance(provider_configuration, EffectiveSettings) else config.snapshot_settings()

    @property
    def configuration(self) -> EffectiveSettings:
        return self._configuration


@dataclass
class _ServiceScope:
    services: RunServices
    active: bool = True


_current: ContextVar[_ServiceScope] = ContextVar("saga_run_services")


def get_services() -> RunServices:
    scope = _current.get(None)
    if scope is None or not scope.active or scope.services._closed:
        raise RuntimeError("SAGA services require an active run context")
    return scope.services


@contextmanager
def inject_services(services: RunServices) -> Iterator[RunServices]:
    """Borrow explicit interfaces; the caller retains ownership of their lifetime."""
    if services._closed:
        raise RuntimeError("Cannot borrow closed SAGA run services")
    scope = _ServiceScope(services)
    token = _current.set(scope)
    try:
        with config.bind_settings(services.configuration):
            yield services
    finally:
        scope.active = False
        _current.reset(token)


def _create_language_model() -> LanguageModel:
    from core.llm_interface_refactored import create_llm_service

    effective = config.get_settings()
    if not isinstance(effective, EffectiveSettings):
        raise RuntimeError("Default language model requires a run configuration")
    return create_llm_service(configuration=effective)


@asynccontextmanager
async def managed_services(
    *, language_model_factory: Callable[[], LanguageModel] = _create_language_model,
    database: Neo4jManagerSingleton | None = None,
) -> AsyncIterator[RunServices]:
    """Own one client lifetime, including creation failures and cancellation.

    The default database manager retains its process-pinned project/endpoint identity;
    only its connection is closed and recreated between runs.
    """
    effective = config.snapshot_settings()
    if database is None:
        from core.db_manager import Neo4jManagerSingleton

        database = Neo4jManagerSingleton()
    try:
        with config.bind_settings(effective):
            language_model = language_model_factory()
        services = RunServices(language_model, database)
        try:
            with inject_services(services):
                yield services
        finally:
            services._closed = True
            await language_model.aclose()
    finally:
        await database.close()


@asynccontextmanager
async def service_lifetime(services: RunServices | None = None) -> AsyncIterator[RunServices]:
    """Own a standalone run or borrow an explicitly supplied enclosing run."""
    if services is not None:
        with inject_services(services):
            yield services
    else:
        async with managed_services() as owned:
            yield owned
