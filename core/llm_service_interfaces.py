# core/llm_service_interfaces.py
"""
Abstraction layers and interfaces for LLM services.

This module provides the protocol definitions and abstract interfaces
for LLM-related services, enabling proper dependency injection and
testability.

REFACTORED: Created as part of Phase 3 architectural improvements.
- Defines clear interfaces for all LLM services
- Enables proper dependency injection
- Facilitates testing with mock implementations
- Provides abstraction over concrete implementations
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Protocol, Tuple

import numpy as np


class HTTPClientInterface(Protocol):
    """Protocol defining the interface for HTTP client services."""
    
    async def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        max_retries: int | None = None
    ) -> Any:  # httpx.Response
        """Make a POST request with JSON payload and retry logic."""
        ...
    
    async def aclose(self) -> None:
        """Close the HTTP client and cleanup resources."""
        ...
    
    def get_statistics(self) -> dict[str, Any]:
        """Get HTTP client statistics."""
        ...


class TokenizerInterface(Protocol):
    """Protocol defining the interface for tokenization services."""
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text for a given model."""
        ...
    
    def truncate_text_by_tokens(
        self,
        text: str,
        model_name: str,
        max_tokens: int,
        truncation_marker: str = "\n... (truncated)",
    ) -> str:
        """Truncate text to maximum number of tokens."""
        ...
    
    def get_statistics(self) -> dict[str, Any]:
        """Get tokenizer statistics."""
        ...


class ResponseCleanerInterface(Protocol):
    """Protocol defining the interface for response cleaning services."""
    
    def clean_response(self, text: str) -> str:
        """Clean artifacts from LLM response text."""
        ...
    
    def get_statistics(self) -> dict[str, Any]:
        """Get cleaning statistics."""
        ...


class StreamProcessorInterface(Protocol):
    """Protocol defining the interface for stream processing services."""
    
    def process_streaming_chunks(
        self, 
        chunks: AsyncGenerator[dict[str, Any], None],
        auto_clean: bool = True
    ) -> Tuple[str, dict[str, Any] | None]:
        """Process streaming chunks into accumulated content."""
        ...
    
    def get_statistics(self) -> dict[str, Any]:
        """Get stream processing statistics."""
        ...


class EmbeddingServiceInterface(Protocol):
    """Protocol defining the interface for embedding services."""
    
    async def get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding vector for text."""
        ...
    
    def get_statistics(self) -> dict[str, Any]:
        """Get embedding service statistics."""
        ...


class CompletionServiceInterface(Protocol):
    """Protocol defining the interface for completion services."""
    
    async def get_completion(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> Tuple[str, dict[str, int] | None]:
        """Get completion from LLM."""
        ...
    
    async def get_streaming_completion(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Get streaming completion from LLM."""
        ...
    
    def get_statistics(self) -> dict[str, Any]:
        """Get completion service statistics."""
        ...


class LLMServiceInterface(Protocol):
    """Protocol defining the unified interface for LLM services."""
    
    async def get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding for text."""
        ...
    
    async def call_llm(
        self,
        model_name: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = False,
        stream_to_disk: bool = False,
        auto_clean_response: bool = True,
        **kwargs
    ) -> Tuple[str, dict[str, int] | None]:
        """Call LLM with comprehensive options."""
        ...
    
    def clean_response(self, text: str) -> str:
        """Clean LLM response text."""
        ...
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in text."""
        ...
    
    async def aclose(self) -> None:
        """Close service and cleanup resources."""
        ...


class LLMServiceFactory(ABC):
    """
    Abstract factory for creating LLM service implementations.
    
    This factory enables dependency injection and makes it easy to
    switch between different implementations (e.g., for testing).
    """

    @abstractmethod
    def create_http_client(self, timeout: float | None = None) -> HTTPClientInterface:
        """Create HTTP client service instance."""
        pass

    @abstractmethod
    def create_tokenizer(self) -> TokenizerInterface:
        """Create tokenizer service instance."""
        pass

    @abstractmethod
    def create_response_cleaner(self) -> ResponseCleanerInterface:
        """Create response cleaner service instance."""
        pass

    @abstractmethod
    def create_stream_processor(
        self, response_cleaner: ResponseCleanerInterface
    ) -> StreamProcessorInterface:
        """Create stream processor service instance."""
        pass

    @abstractmethod
    def create_embedding_service(
        self, http_client: HTTPClientInterface
    ) -> EmbeddingServiceInterface:
        """Create embedding service instance."""
        pass

    @abstractmethod
    def create_completion_service(
        self,
        http_client: HTTPClientInterface,
        tokenizer: TokenizerInterface,
        response_cleaner: ResponseCleanerInterface,
        stream_processor: StreamProcessorInterface
    ) -> CompletionServiceInterface:
        """Create completion service instance."""
        pass

    @abstractmethod
    def create_llm_service(
        self,
        completion_service: CompletionServiceInterface,
        embedding_service: EmbeddingServiceInterface,
        tokenizer: TokenizerInterface,
        response_cleaner: ResponseCleanerInterface
    ) -> LLMServiceInterface:
        """Create unified LLM service instance."""
        pass


class ServiceLocator:
    """
    Service locator for managing LLM service dependencies.
    
    This provides centralized service management and dependency injection
    capabilities for the LLM service architecture.
    """

    def __init__(self, factory: LLMServiceFactory):
        """
        Initialize service locator with a factory.
        
        Args:
            factory: Factory for creating service instances
        """
        self._factory = factory
        self._services: dict[str, Any] = {}
        self._initialized = False

    def get_http_client(self) -> HTTPClientInterface:
        """Get HTTP client service instance."""
        if "http_client" not in self._services:
            self._services["http_client"] = self._factory.create_http_client()
        return self._services["http_client"]

    def get_tokenizer(self) -> TokenizerInterface:
        """Get tokenizer service instance."""
        if "tokenizer" not in self._services:
            self._services["tokenizer"] = self._factory.create_tokenizer()
        return self._services["tokenizer"]

    def get_response_cleaner(self) -> ResponseCleanerInterface:
        """Get response cleaner service instance."""
        if "response_cleaner" not in self._services:
            self._services["response_cleaner"] = self._factory.create_response_cleaner()
        return self._services["response_cleaner"]

    def get_stream_processor(self) -> StreamProcessorInterface:
        """Get stream processor service instance."""
        if "stream_processor" not in self._services:
            response_cleaner = self.get_response_cleaner()
            self._services["stream_processor"] = self._factory.create_stream_processor(
                response_cleaner
            )
        return self._services["stream_processor"]

    def get_embedding_service(self) -> EmbeddingServiceInterface:
        """Get embedding service instance."""
        if "embedding_service" not in self._services:
            http_client = self.get_http_client()
            self._services["embedding_service"] = self._factory.create_embedding_service(
                http_client
            )
        return self._services["embedding_service"]

    def get_completion_service(self) -> CompletionServiceInterface:
        """Get completion service instance."""
        if "completion_service" not in self._services:
            http_client = self.get_http_client()
            tokenizer = self.get_tokenizer()
            response_cleaner = self.get_response_cleaner()
            stream_processor = self.get_stream_processor()
            
            self._services["completion_service"] = self._factory.create_completion_service(
                http_client, tokenizer, response_cleaner, stream_processor
            )
        return self._services["completion_service"]

    def get_llm_service(self) -> LLMServiceInterface:
        """Get unified LLM service instance."""
        if "llm_service" not in self._services:
            completion_service = self.get_completion_service()
            embedding_service = self.get_embedding_service()
            tokenizer = self.get_tokenizer()
            response_cleaner = self.get_response_cleaner()
            
            self._services["llm_service"] = self._factory.create_llm_service(
                completion_service, embedding_service, tokenizer, response_cleaner
            )
        return self._services["llm_service"]

    async def cleanup_all_services(self) -> None:
        """Cleanup all created services."""
        # Close HTTP client if it exists
        if "http_client" in self._services:
            await self._services["http_client"].aclose()
        
        # Close LLM service if it exists
        if "llm_service" in self._services:
            await self._services["llm_service"].aclose()
        
        self._services.clear()
        self._initialized = False

    def get_all_statistics(self) -> dict[str, Any]:
        """Get combined statistics from all services."""
        stats = {}
        
        for service_name, service in self._services.items():
            if hasattr(service, 'get_statistics'):
                stats[service_name] = service.get_statistics()
        
        return stats


# Global service locator instance (will be initialized by factory)
_service_locator: ServiceLocator | None = None


def get_service_locator() -> ServiceLocator:
    """
    Get the global service locator instance.
    
    Raises:
        RuntimeError: If service locator has not been initialized
    """
    global _service_locator
    if _service_locator is None:
        raise RuntimeError(
            "Service locator not initialized. Call initialize_service_locator() first."
        )
    return _service_locator


def initialize_service_locator(factory: LLMServiceFactory) -> ServiceLocator:
    """
    Initialize the global service locator with a factory.
    
    Args:
        factory: Factory for creating service instances
        
    Returns:
        Initialized service locator instance
    """
    global _service_locator
    _service_locator = ServiceLocator(factory)
    return _service_locator


def get_llm_service() -> LLMServiceInterface:
    """Convenience function to get the main LLM service."""
    return get_service_locator().get_llm_service()