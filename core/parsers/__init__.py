# core/parsers/__init__.py
"""Parser modules for SAGA initialization.

This package contains parsers for converting initialization artifacts
into knowledge graph entities.
"""

from .character_sheet_parser import CharacterSheetParser

__all__ = ["CharacterSheetParser"]
