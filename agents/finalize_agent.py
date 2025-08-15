import asyncio
import json
import re
from typing import Any, Dict, Optional, TypedDict

import numpy as np
import structlog

from agents.knowledge_agent import KnowledgeAgent
from core.llm_interface import llm_service
from data_access import chapter_queries, kg_queries
from models import CharacterProfile, WorldItem
from parsing_utils import parse_rdf_triples_with_rdflib

logger = structlog.get_logger(__name__)


class FinalizationResult(TypedDict, total=False):
    summary: Optional[str]
    embedding: Optional[np.ndarray]
    summary_usage: Optional[Dict[str, int]]
    kg_usage: Optional[Dict[str, int]]


class FinalizeAgent:
    """Handle chapter finalization and KG updates."""

    def __init__(self, kg_agent: Optional[KnowledgeAgent] = None) -> None:
        self.kg_agent = kg_agent or KnowledgeAgent()
        logger.info("FinalizeAgent initialized")

    def _extract_json_block(self, text: str, key: str) -> str:
        """Extract a JSON object associated with a key using brace counting."""
        pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*{{')
        match = pattern.search(text)
        if not match:
            return "{}"
        start = match.end() - 1
        brace_count = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start : idx + 1]
        return "{}"

    def _extract_array_block(self, text: str, key: str) -> str:
        """Extract and join array text for KG triples."""
        pattern = re.compile(rf'"{re.escape(key)}"\s*:\s*\[')
        match = pattern.search(text)
        if not match:
            return ""
        start = match.end() - 1
        bracket_count = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    block = text[start : idx + 1]
                    try:
                        arr = json.loads(block)
                        if isinstance(arr, list):
                            return "\n".join(str(v) for v in arr)
                    except json.JSONDecodeError:
                        pass
                    return block
        return ""

    def _validate_character_updates(self, updates: Dict[str, CharacterProfile]) -> bool:
        for name, profile in updates.items():
            if not name or not profile.name:
                logger.error("Invalid character update", name=name)
                return False
        return True

    def _validate_world_updates(self, updates: Dict[str, Dict[str, WorldItem]]) -> bool:
        for category, items in updates.items():
            if not category:
                logger.error("World update missing category")
                return False
            for item in items.values():
                if not item.name or not item.category:
                    logger.error(
                        "Invalid world item", category=category, item=item.name
                    )
                    return False
        return True

    async def _extract_merge_and_persist(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        chapter_text: str,
        from_flawed_draft: bool,
    ) -> Optional[Dict[str, int]]:
        raw_text, usage_data = await self.kg_agent._llm_extract_updates(
            plot_outline, chapter_text, chapter_number
        )
        if not raw_text.strip():
            logger.warning("LLM extraction returned no text", chapter=chapter_number)
            return usage_data

        char_updates_raw = "{}"
        world_updates_raw = "{}"
        triples_text = ""
        try:
            parsed_json = json.loads(raw_text)
            char_updates_raw = json.dumps(parsed_json.get("character_updates", {}))
            world_updates_raw = json.dumps(parsed_json.get("world_updates", {}))
            triples_list = parsed_json.get("kg_triples", [])
            if isinstance(triples_list, list):
                triples_text = "\n".join([str(t) for t in triples_list])
            else:
                triples_text = str(triples_list)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse extraction JSON, attempting manual extraction",
                chapter=chapter_number,
            )
            char_updates_raw = self._extract_json_block(raw_text, "character_updates")
            world_updates_raw = self._extract_json_block(raw_text, "world_updates")
            triples_text = self._extract_array_block(raw_text, "kg_triples")

        char_updates = self.kg_agent.parse_character_updates(
            char_updates_raw, chapter_number
        )
        world_updates = self.kg_agent.parse_world_updates(
            world_updates_raw, chapter_number
        )
        triples = parse_rdf_triples_with_rdflib(triples_text)

        if not self._validate_character_updates(
            char_updates
        ) or not self._validate_world_updates(world_updates):
            logger.error(
                "Validation failed for extracted updates", chapter=chapter_number
            )
            return usage_data

        self.kg_agent.merge_updates(
            character_profiles,
            world_building,
            char_updates,
            world_updates,
            chapter_number,
            from_flawed_draft,
        )

        if char_updates:
            await self.kg_agent.persist_profiles(char_updates, chapter_number)
        if world_updates:
            await self.kg_agent.persist_world(world_updates, chapter_number)
        if triples:
            try:
                await kg_queries.add_kg_triples_batch_to_db(
                    triples, chapter_number, from_flawed_draft
                )
            except Exception as exc:
                logger.error(
                    "Failed to persist KG triples", chapter=chapter_number, exc_info=exc
                )
        return usage_data

    async def finalize_chapter(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chapter_number: int,
        final_text: str,
        raw_llm_output: Optional[str] = None,
        from_flawed_draft: bool = False,
    ) -> FinalizationResult:
        """Finalize a chapter and persist all related updates.

        Args:
            plot_outline: The current plot outline for the novel.
            character_profiles: Known character profiles before this chapter.
            world_building: Known world elements before this chapter.
            chapter_number: The chapter number being finalized.
            final_text: The approved chapter text.
            raw_llm_output: Optional raw draft from the LLM.
            from_flawed_draft: Whether the text came from a flawed draft.

        Returns:
            A dictionary containing the summary, embedding, and token usage data.
        """
        summary_task = self.kg_agent.summarize_chapter(final_text, chapter_number)
        embedding_task = llm_service.async_get_embedding(final_text)
        kg_task = self._extract_merge_and_persist(
            plot_outline,
            character_profiles,
            world_building,
            chapter_number,
            final_text,
            from_flawed_draft,
        )

        (summary_data, embedding, kg_usage) = await asyncio.gather(
            summary_task, embedding_task, kg_task
        )
        summary, summary_usage = summary_data

        await chapter_queries.save_chapter_data_to_db(
            chapter_number,
            final_text,
            raw_llm_output or "N/A",
            summary,
            embedding,
            from_flawed_draft,
        )

        return {
            "summary": summary,
            "embedding": embedding,
            "summary_usage": summary_usage,
            "kg_usage": kg_usage,
        }

    async def ingest_and_finalize_chunk(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, CharacterProfile],
        world_building: Dict[str, Dict[str, WorldItem]],
        chunk_number: int,
        chunk_text: str,
    ) -> FinalizationResult:
        """Finalize an ingested text chunk using the regular pipeline."""
        return await self.finalize_chapter(
            plot_outline,
            character_profiles,
            world_building,
            chunk_number,
            chunk_text,
            raw_llm_output=None,
            from_flawed_draft=False,
        )
