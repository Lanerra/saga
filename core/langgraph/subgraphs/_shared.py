# core/langgraph/subgraphs/_shared.py
"""Shared utilities across subgraphs."""

from typing import Literal

from core.langgraph.state import NarrativeState


def _should_continue_or_error(state: NarrativeState) -> Literal["continue", "error"]:
    """Gate on fatal errors and unresolved rollback before proceeding."""
    if state.get("has_fatal_error", False) or state.get("revision_rollback_failure") is not None:
        return "error"
    return "continue"
