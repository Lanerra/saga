"""Validation utilities for model fields (re-exported from utils)."""

from __future__ import annotations

# Delegate to the canonical implementations in utils to avoid drift.
from utils import _normalize_for_id, validate_world_item_fields  # noqa: F401

__all__ = ["_normalize_for_id", "validate_world_item_fields"]
