# config/validator.py
"""
Configuration validation utilities for the SAGA system.

This module provides a single public function `validate_all()` that:
1. Instantiates a fresh `SagaSettings` object (leveraging Pydantic validation).
2. Performs cross‑field sanity checks that cannot be expressed purely with
   Pydantic field validators (e.g., related numeric ranges).
3. Returns a structured health report dictionary that mimics the format used
   elsewhere in the codebase (e.g., `core/schema_initialization.get_schema_health_check`).

The report layout:

{
    "overall_health": "healthy" | "warning" | "error",
    "issues": {
        "errors":   [{ "field": "<field>", "message": "<msg>" }, ...],
        "warnings": [{ "field": "<field>", "message": "<msg>" }, ...],
        "info":     [{ "field": "<field>", "message": "<msg>" }, ...],
    }
}
"""

from __future__ import annotations

from .settings import settings as current_settings


def _add_issue(
    issues: dict[str, list[dict[str, str]]],
    severity: str,
    field: str,
    message: str,
) -> None:
    """Utility to append an issue entry to the report."""
    issues.setdefault(severity, []).append({"field": field, "message": message})


def validate_all() -> dict:
    """
    Validate the current configuration state.

    Returns a health‑report dict with overall status and detailed issue lists.
    """
    issues: dict[str, list[dict[str, str]]] = {"errors": [], "warnings": [], "info": []}

    # 1️⃣ Pydantic field validation – already performed when the settings instance
    #    was created.  If the instance exists, we assume fields are of the correct type.
    #    However, we still guard against a completely missing instance.
    if current_settings is None:
        _add_issue(
            issues, "errors", "settings", "Configuration object not initialized."
        )
        return {
            "overall_health": "error",
            "issues": issues,
        }

    # 2️⃣ Cross‑field sanity checks
    # ------------------------------------------------------------
    # Context vs generation token limits
    if current_settings.MAX_CONTEXT_TOKENS <= current_settings.MAX_GENERATION_TOKENS:
        _add_issue(
            issues,
            "errors",
            "MAX_CONTEXT_TOKENS",
            (
                f"MAX_CONTEXT_TOKENS ({current_settings.MAX_CONTEXT_TOKENS}) must be "
                f"greater than MAX_GENERATION_TOKENS ({current_settings.MAX_GENERATION_TOKENS})."
            ),
        )

    # Embedding dimension consistency
    if (
        current_settings.NEO4J_VECTOR_DIMENSIONS
        != current_settings.EXPECTED_EMBEDDING_DIM
    ):
        _add_issue(
            issues,
            "warnings",
            "NEO4J_VECTOR_DIMENSIONS",
            (
                f"NEO4J_VECTOR_DIMENSIONS ({current_settings.NEO4J_VECTOR_DIMENSIONS}) "
                f"differs from EXPECTED_EMBEDDING_DIM ({current_settings.EXPECTED_EMBEDDING_DIM}). "
                "This may cause indexing errors."
            ),
        )

    # Cache sizes sanity (non‑negative and reasonable upper bounds)
    cache_fields = [
        ("EMBEDDING_CACHE_SIZE", 1, 10_000),
        ("SUMMARY_CACHE_SIZE", 1, 5_000),
        ("KG_TRIPLE_EXTRACTION_CACHE_SIZE", 1, 5_000),
        ("TOKENIZER_CACHE_SIZE", 1, 500),
    ]
    for name, min_val, max_val in cache_fields:
        value = getattr(current_settings, name)
        if value < min_val:
            _add_issue(
                issues, "errors", name, f"{name} must be >= {min_val}; got {value}."
            )
        elif value > max_val:
            _add_issue(
                issues,
                "warnings",
                name,
                f"{name} is very large ({value}); consider lowering to reduce memory usage.",
            )

    # Temperature ranges – all temperatures should be within [0.0, 2.0]
    temp_fields = [
        "TEMPERATURE_INITIAL_SETUP",
        "TEMPERATURE_DRAFTING",
        "TEMPERATURE_REVISION",
        "TEMPERATURE_PLANNING",
        "TEMPERATURE_EVALUATION",
        "TEMPERATURE_CONSISTENCY_CHECK",
        "TEMPERATURE_KG_EXTRACTION",
        "TEMPERATURE_SUMMARY",
        "TEMPERATURE_PATCH",
    ]
    for name in temp_fields:
        value = getattr(current_settings, name)
        if not (0.0 <= value <= 2.0):
            _add_issue(
                issues,
                "warnings",
                name,
                f"{name} = {value} is outside the recommended range 0.0‑2.0.",
            )

    # 3️⃣ Derive overall health
    overall = "healthy"
    if issues["errors"]:
        overall = "error"
    elif issues["warnings"]:
        overall = "warning"

    return {
        "overall_health": overall,
        "issues": issues,
    }
