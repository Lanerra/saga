# agents/pre_flight_check_agent.py
"""Agent for KG pre-flight consistency checks."""

from __future__ import annotations

import json
from typing import Any

import structlog
from config import settings
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from data_access import character_queries, world_queries

logger = structlog.get_logger(__name__)


class PreFlightCheckAgent:
    """Performs core contradiction checks before drafting."""

    def __init__(self, model_name: str = settings.SMALL_MODEL) -> None:
        self.model_name = model_name
        logger.info("PreFlightCheckAgent initialized")

    async def _character_has_conflict(
        self, name: str, trait1: str, trait2: str
    ) -> bool:
        query = (
            "MATCH (c:Character {name: $name})-[:HAS_TRAIT]->(t1:Trait {name: $t1})"
            " MATCH (c)-[:HAS_TRAIT]->(t2:Trait {name: $t2}) RETURN c"
        )
        results = await neo4j_manager.execute_read_query(
            query, {"name": name, "t1": trait1, "t2": trait2}
        )
        return bool(results)

    async def _world_element_has_conflict(
        self, element_id: str, trait1: str, trait2: str
    ) -> bool:
        query = (
            "MATCH (we:WorldElement {id: $we_id})-[:HAS_TRAIT_ASPECT]->"
            "(v1:ValueNode {type: 'traits', value: $t1}) "
            "MATCH (we)-[:HAS_TRAIT_ASPECT]->"
            "(v2:ValueNode {type: 'traits', value: $t2}) RETURN we"
        )
        results = await neo4j_manager.execute_read_query(
            query, {"we_id": element_id, "t1": trait1, "t2": trait2}
        )
        return bool(results)

    async def _identify_contradictory_pairs(
        self,
        plot_outline: dict[str, Any],
        characters: dict[str, Any] | None,
        world: dict[str, dict[str, Any]] | None,
    ) -> list[tuple[str, str]]:
        """Use the LLM to detect contradictory trait pairs."""

        prefix = "/no_think\n" if settings.ENABLE_LLM_NO_THINK_DIRECTIVE else ""
        context_json = json.dumps(
            {
                "plot": plot_outline,
                "characters": characters or {},
                "world": world or {},
            },
            ensure_ascii=False,
        )
        prompt = (
            prefix
            + "Given this story context, list any pairs of character traits that "
            "cannot logically coexist. Respond with JSON like [['A','B'], ...]."
        )
        text, _ = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=context_json + "\n" + prompt,
            temperature=0.0,
            max_tokens=200,
            auto_clean_response=True,
        )
        try:
            data = json.loads(text)
            pairs = [tuple(p) for p in data if isinstance(p, list) and len(p) == 2]
        except json.JSONDecodeError:
            logger.warning(
                "PreFlightCheckAgent: could not parse contradictory trait pairs"
            )
            pairs = []
        return pairs

    async def _resolve_world_trait_conflict(
        self, element_id: str, trait1: str, trait2: str
    ) -> None:
        prefix = "/no_think\n" if settings.ENABLE_LLM_NO_THINK_DIRECTIVE else ""
        prompt = (
            prefix + f"A world element with id {element_id} has both traits "
            f"'{trait1}' and '{trait2}'. Choose the canonical trait. "
            'Respond with JSON {"trait": "chosen"}.'
        )
        text, _ = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.0,
            max_tokens=40,
            auto_clean_response=True,
        )
        chosen = trait1
        try:
            data = json.loads(text)
            if data.get("trait") in {trait1, trait2}:
                chosen = data["trait"]
        except json.JSONDecodeError:
            logger.warning(
                "PreFlightCheckAgent: could not parse world trait resolution for %s",
                element_id,
            )
        to_remove = trait2 if chosen == trait1 else trait1
        await world_queries.remove_world_element_trait_aspect(element_id, to_remove)
        logger.info(
            "PreFlightCheckAgent resolved world trait conflict for %s: kept %s removed %s",
            element_id,
            chosen,
            to_remove,
        )

    async def _resolve_trait_conflict(
        self, name: str, trait1: str, trait2: str
    ) -> None:
        prefix = "/no_think\n" if settings.ENABLE_LLM_NO_THINK_DIRECTIVE else ""
        prompt = (
            prefix
            + f"A character named {name} has both traits '{trait1}' and '{trait2}'. "
            "Choose the single trait that should remain canonical. "
            'Respond with JSON {"trait": "chosen"}.'
        )
        text, _ = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=prompt,
            temperature=0.0,
            max_tokens=40,
            auto_clean_response=True,
        )
        chosen = trait1
        try:
            data = json.loads(text)
            if data.get("trait") in {trait1, trait2}:
                chosen = data["trait"]
        except json.JSONDecodeError:
            logger.warning(
                "PreFlightCheckAgent: could not parse LLM resolution for %s", name
            )
        to_remove = trait2 if chosen == trait1 else trait1
        await character_queries.remove_character_trait(name, to_remove)
        logger.info(
            "PreFlightCheckAgent resolved trait conflict for %s: kept %s removed %s",
            name,
            chosen,
            to_remove,
        )

    async def _gather_canonical_facts(
        self, plot_outline: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Return canonical facts that must remain consistent."""
        query = (
            "MATCH (c:Character:Entity)-[:HAS_TRAIT]->"
            "(t:Trait:Entity {is_canonical_truth: true}) "
            "RETURN c.name AS name, t.name AS trait"
        )

        try:
            records = await neo4j_manager.execute_read_query(query)
        except Exception as exc:  # pragma: no cover - log but continue
            logger.error(
                "PreFlightCheckAgent failed to load canonical facts",
                error=exc,
                exc_info=True,
            )
            return []

        facts: list[dict[str, str]] = []
        for rec in records:
            name = rec.get("name")
            trait = rec.get("trait")
            if name and trait:
                facts.append({"name": str(name), "trait": str(trait)})

        if not facts:
            return []

        prefix = "/no_think\n" if settings.ENABLE_LLM_NO_THINK_DIRECTIVE else ""
        prompt_context = {"plot_outline": plot_outline, "canonical_facts": facts}
        prompt = (
            prefix
            + "For each canonical_facts item, provide a trait that directly contradicts "
            "the listed trait. Respond with JSON as [{'conflicts_with': 'Trait'}, ...] in the same order."
        )

        text, _ = await llm_service.async_call_llm(
            model_name=self.model_name,
            prompt=json.dumps(prompt_context, ensure_ascii=False) + "\n" + prompt,
            temperature=0.0,
            max_tokens=200,
            auto_clean_response=True,
            allow_fallback=True,
        )

        conflicts: list[str] = []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("conflicts_with"):
                        conflicts.append(str(item["conflicts_with"]))
                    elif isinstance(item, str):
                        conflicts.append(item)
        except json.JSONDecodeError:
            logger.warning(
                "PreFlightCheckAgent: could not parse canonical fact conflicts"
            )

        final: list[dict[str, str]] = []
        for fact, conflict in zip(facts, conflicts, strict=False):
            if conflict:
                fact["conflicts_with"] = conflict
                final.append(fact)

        return final

    async def perform_core_checks(
        self,
        plot_outline: dict[str, Any],
        characters: dict[str, Any] | None,
        world: dict[str, dict[str, Any]] | None,
    ) -> None:
        protagonist = plot_outline.get("protagonist_name")
        char_names = set(characters.keys()) if characters else set()
        if protagonist:
            char_names.add(protagonist)

        trait_pairs = await self._identify_contradictory_pairs(
            plot_outline, characters, world
        )

        for char_name in char_names:
            for trait1, trait2 in trait_pairs:
                if await self._character_has_conflict(char_name, trait1, trait2):
                    await self._resolve_trait_conflict(char_name, trait1, trait2)

        for fact in await self._gather_canonical_facts(plot_outline):
            if await self._character_has_conflict(
                fact["name"], fact["trait"], fact["conflicts_with"]
            ):
                logger.warning(
                    "Pre-flight check found canonical conflict for %s.",
                    fact["name"],
                )
                await self._resolve_trait_conflict(
                    fact["name"], fact["trait"], fact["conflicts_with"]
                )

        if not world:
            return

        for category_dict in world.values():
            if not isinstance(category_dict, dict):
                continue
            for item in category_dict.values():
                if not item or not getattr(item, "id", None):
                    continue
                element_id = item.id
                for trait1, trait2 in trait_pairs:
                    if await self._world_element_has_conflict(
                        element_id, trait1, trait2
                    ):
                        await self._resolve_world_trait_conflict(
                            element_id, trait1, trait2
                        )
