# core/langgraph/subgraphs/_shared.py
"""Shared utilities across subgraphs."""

from typing import Literal

from core.langgraph.state import NarrativeState


def _should_continue_or_error(state: NarrativeState) -> Literal["continue", "error"]:
    """Gate on has_fatal_error before proceeding to the next node."""
    if state.get("has_fatal_error", False):
        return "error"
    return "continue"
