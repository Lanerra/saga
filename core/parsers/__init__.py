# core/parsers/__init__.py
"""Parser modules for SAGA initialization.

This package contains parsers for converting initialization artifacts
into knowledge graph entities.
"""

from .act_outline_parser import ActOutlineParser
from .chapter_outline_parser import ChapterOutlineParser
from .character_sheet_parser import CharacterSheetParser
from .global_outline_parser import GlobalOutlineParser, MajorPlotPoint
from .narrative_enrichment_parser import ChapterEmbeddingExtractionResult, NarrativeEnrichmentParser, PhysicalDescriptionExtractionResult

__all__ = [
    "CharacterSheetParser",
    "GlobalOutlineParser",
    "MajorPlotPoint",
    "ActOutlineParser",
    "ChapterOutlineParser",
    "NarrativeEnrichmentParser",
    "PhysicalDescriptionExtractionResult",
    "ChapterEmbeddingExtractionResult",
]
