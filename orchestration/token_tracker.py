from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TokenTracker:
    """Simple helper to accumulate token usage."""

    def __init__(self) -> None:
        self.total = 0

    def add(self, operation_name: str, usage_data: dict[str, int] | None) -> None:
        """Add tokens from an LLM usage response."""
        if isinstance(usage_data, dict) and isinstance(
            usage_data.get("completion_tokens"), int
        ):
            completed_tokens = usage_data["completion_tokens"]
            self.total += completed_tokens
            logger.info(
                "NANA Activity: Tokens from '%s': %s. Total generated this run: %s",
                operation_name,
                completed_tokens,
                self.total,
            )
        elif (
            isinstance(usage_data, dict)
            and usage_data.get("total_tokens") is not None
            and isinstance(usage_data["total_tokens"], int)
            and (
                usage_data.get("completion_tokens") is None
                or not isinstance(usage_data["completion_tokens"], int)
            )
        ):
            logger.info(
                "NANA Activity: Total tokens from '%s': %s. (Completion tokens not specifically available). Total generated this run (completion focused): %s",
                operation_name,
                usage_data["total_tokens"],
                self.total,
            )
        elif usage_data:
            logger.warning(
                "NANA Activity: '%s' - 'completion_tokens' missing or not int in usage_data. Tokens not added. Usage: %s",
                operation_name,
                usage_data,
            )
