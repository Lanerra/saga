# knowledge_agent.py
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from async_lru import alru_cache  # type: ignore
from jinja2 import Template

import config
from core.db_manager import neo4j_manager
from core.llm_interface import llm_service
from data_access import (
    character_queries,
    kg_queries,
    plot_queries,
    world_queries,
)

# Assuming a package structure for kg_maintainer components
from kg_maintainer import merge, models, parsing
from parsing_utils import (
    parse_rdf_triples_with_rdflib,
)
from prompt_renderer import render_prompt

logger = logging.getLogger(__name__)


@alru_cache(maxsize=config.SUMMARY_CACHE_SIZE)
async def _llm_summarize_full_chapter_text(
    chapter_text: str, chapter_number: int
) -> Tuple[str, Optional[Dict[str, int]]]:
    """Summarize full chapter text via the configured LLM."""
    prompt = render_prompt(
        "kg_maintainer_agent/chapter_summary.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "chapter_number": chapter_number,
            "chapter_text": chapter_text,
        },
    )
    summary, usage_data = await llm_service.async_call_llm(
        model_name=config.SMALL_MODEL,  # Using SMALL_MODEL for summarization
        prompt=prompt,
        temperature=config.Temperatures.SUMMARY,
        max_tokens=config.MAX_SUMMARY_TOKENS,  # Should be small for 1-3 sentences
        stream_to_disk=False,
        frequency_penalty=config.FREQUENCY_PENALTY_SUMMARY,
        presence_penalty=config.PRESENCE_PENALTY_SUMMARY,
        auto_clean_response=True,
    )
    summary_text = summary.strip()
    if summary_text:
        try:
            parsed = json.loads(summary_text)
            if isinstance(parsed, dict):
                summary_text = parsed.get("summary", "")
        except json.JSONDecodeError:
            logger.debug(f"Summary for chapter {chapter_number} was not a JSON object.")
    return summary_text, usage_data


# Prompt template for entity resolution, embedded to avoid new file dependency
ENTITY_RESOLUTION_PROMPT_TEMPLATE = """/no_think
You are an expert knowledge graph analyst for a creative writing project. Your task is to determine if two entities from the narrative's knowledge graph are referring to the same canonical thing based on their names, properties, and relationships.

**Entity 1 Details:**
- Name: {{ entity1.name }}
- Labels: {{ entity1.labels }}
- Properties:
{{ entity1.properties | tojson(indent=2) }}
- Key Relationships (up to 10):
{% if entity1.relationships %}
{% for rel in entity1.relationships %}
  - Related to '{{ rel.other_node_name }}' (Labels: {{ rel.other_node_labels }}) via relationship of type '{{ rel.rel_type }}'
{% endfor %}
{% else %}
  - No relationships found.
{% endif %}

**Entity 2 Details:**
- Name: {{ entity2.name }}
- Labels: {{ entity2.labels }}
- Properties:
{{ entity2.properties | tojson(indent=2) }}
- Key Relationships (up to 10):
{% if entity2.relationships %}
{% for rel in entity2.relationships %}
  - Related to '{{ rel.other_node_name }}' (Labels: {{ rel.other_node_labels }}) via relationship of type '{{ rel.rel_type }}'
{% endfor %}
{% else %}
  - No relationships found.
{% endif %}

**Analysis Task:**
Based on all the provided context, including name similarity, properties (like descriptions), and shared relationships, are "Entity 1" and "Entity 2" the same entity within the story's canon? For example, "The Locket" and "The Pendant" might be the same item, or "The Shattered Veil" and "Shattered Veil" are likely the same faction.

**Response Format:**
Respond in JSON format only, with no other text, commentary, or markdown. Your entire response must be a single, valid JSON object with the following structure:
{
  "is_same_entity": boolean,
  "confidence_score": float (from 0.0 to 1.0, representing your certainty),
  "reason": "A brief explanation for your decision."
}
"""

# Prompt template for dynamic relationship resolution
DYNAMIC_REL_RESOLUTION_PROMPT_TEMPLATE = """/no_think
You analyze a relationship from the novel's knowledge graph and provide a
single, canonical predicate name in ALL_CAPS_WITH_UNDERSCORES describing the
relationship between the subject and object.

Subject: {{ subject }} ({{ subject_labels }})
Object: {{ object }} ({{ object_labels }})
Existing Type: {{ type }}
Subject Description: {{ subject_desc }}
Object Description: {{ object_desc }}

Respond with only the predicate string, no extra words.
"""


class KnowledgeAgent:
    """High level interface for KG parsing and persistence."""

    def __init__(self, model_name: str = config.KNOWLEDGE_UPDATE_MODEL):
        self.model_name = model_name
        self.node_labels: List[str] = []
        self.relationship_types: List[str] = []
        logger.info(
            "KnowledgeAgent initialized with model for extraction: %s",
            self.model_name,
        )

    async def load_schema_from_db(self):
        """Loads and caches the defined KG schema from the database."""
        self.node_labels = await kg_queries.get_defined_node_labels()
        self.relationship_types = await kg_queries.get_defined_relationship_types()
        logger.info(
            f"Loaded {len(self.node_labels)} node labels and {len(self.relationship_types)} relationship types from DB."
        )

    def parse_character_updates(
        self, text: str, chapter_number: int
    ) -> Dict[str, models.CharacterProfile]:
        """Parse character update text into structured profiles."""
        return parsing.parse_unified_character_updates(text, chapter_number)

    def parse_world_updates(
        self, text: str, chapter_number: int
    ) -> Dict[str, Dict[str, models.WorldItem]]:
        """Parse world update text into structured items."""
        return parsing.parse_unified_world_updates(text, chapter_number)

    def merge_updates(
        self,
        current_profiles: Dict[str, models.CharacterProfile],
        current_world: Dict[str, Dict[str, models.WorldItem]],
        char_updates_parsed: Dict[str, models.CharacterProfile],
        world_updates_parsed: Dict[str, Dict[str, models.WorldItem]],
        chapter_number: int,
        from_flawed_draft: bool = False,
    ) -> None:
        """Merge parsed updates into existing state (Python objects)."""
        merge.merge_character_profile_updates(
            current_profiles, char_updates_parsed, chapter_number, from_flawed_draft
        )
        merge.merge_world_item_updates(
            current_world, world_updates_parsed, chapter_number, from_flawed_draft
        )

    async def persist_profiles(
        self,
        profiles_to_persist: Dict[str, models.CharacterProfile],
        chapter_number_for_delta: int,
        full_sync: bool = False,
    ) -> None:
        """Persist character profiles to Neo4j."""
        await character_queries.sync_characters(
            profiles_to_persist, chapter_number_for_delta, full_sync=full_sync
        )

    async def persist_world(
        self,
        world_items_to_persist: Dict[str, Dict[str, models.WorldItem]],
        chapter_number_for_delta: int,
        full_sync: bool = False,
    ) -> None:
        """Persist world elements to Neo4j."""
        await world_queries.sync_world_items(
            world_items_to_persist, chapter_number_for_delta, full_sync=full_sync
        )

    async def add_plot_point(self, description: str, prev_plot_point_id: str) -> str:
        """Persist a new plot point and link it in sequence."""
        return await plot_queries.append_plot_point(description, prev_plot_point_id)

    async def summarize_chapter(
        self, chapter_text: Optional[str], chapter_number: int
    ) -> Tuple[Optional[str], Optional[Dict[str, int]]]:
        if (
            not chapter_text
            or len(chapter_text) < config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2
        ):
            logger.warning(
                "Chapter %s text too short for summarization (%d chars, min_req for meaningful summary: %d).",
                chapter_number,
                len(chapter_text or ""),
                config.MIN_ACCEPTABLE_DRAFT_LENGTH // 2,
            )
            return None, None

        try:
            cleaned_summary, usage = await _llm_summarize_full_chapter_text(
                chapter_text, chapter_number
            )
            if cleaned_summary:
                logger.info(
                    "Generated summary for ch %d: '%s...'",
                    chapter_number,
                    cleaned_summary[:100].strip(),
                )
                return cleaned_summary, usage
            logger.warning("LLM returned empty summary for ch %d.", chapter_number)
            return None, usage
        except Exception as e:
            logger.error(
                f"Error during chapter summarization for ch {chapter_number}: {e}",
                exc_info=True,
            )
            return None, None

    async def _llm_extract_updates(
        self,
        plot_outline: Dict[str, Any],
        chapter_text: str,
        chapter_number: int,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        """Call the LLM to extract structured updates from chapter text, including typed entities in triples."""
        protagonist = plot_outline.get(
            "protagonist_name", config.DEFAULT_PROTAGONIST_NAME
        )

        prompt = render_prompt(
            "kg_maintainer_agent/extract_updates.j2",
            {
                "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
                "protagonist": protagonist,
                "chapter_number": chapter_number,
                "novel_title": plot_outline.get("title", "Untitled Novel"),
                "novel_genre": plot_outline.get("genre", "Unknown"),
                "chapter_text": chapter_text,
                "available_node_labels": self.node_labels,
                "available_relationship_types": self.relationship_types,
            },
        )

        try:
            text, usage = await llm_service.async_call_llm(
                model_name=self.model_name,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=config.MAX_KG_TRIPLE_TOKENS,
                allow_fallback=True,
                stream_to_disk=False,
                frequency_penalty=config.FREQUENCY_PENALTY_KG_EXTRACTION,
                presence_penalty=config.PRESENCE_PENALTY_KG_EXTRACTION,
                auto_clean_response=True,
            )
            return text, usage
        except Exception as e:
            logger.error(f"LLM call for KG extraction failed: {e}", exc_info=True)
            return "", None

    async def extract_and_merge_knowledge(
        self,
        plot_outline: Dict[str, Any],
        character_profiles: Dict[str, models.CharacterProfile],
        world_building: Dict[str, Dict[str, models.WorldItem]],
        chapter_number: int,
        chapter_text: str,
        is_from_flawed_draft: bool = False,
    ) -> Optional[Dict[str, int]]:
        if not chapter_text:
            logger.warning(
                "Skipping knowledge extraction for chapter %s: no text provided.",
                chapter_number,
            )
            return None

        logger.info(
            "KnowledgeAgent: Starting knowledge extraction for chapter %d. Flawed draft: %s",
            chapter_number,
            is_from_flawed_draft,
        )

        raw_extracted_text, usage_data = await self._llm_extract_updates(
            plot_outline, chapter_text, chapter_number
        )

        if not raw_extracted_text.strip():
            logger.warning(
                "LLM extraction returned no text for chapter %d.", chapter_number
            )
            return usage_data

        char_updates_raw = "{}"
        world_updates_raw = "{}"
        kg_triples_text = ""

        try:
            parsed_json = json.loads(raw_extracted_text)
            char_updates_raw = json.dumps(parsed_json.get("character_updates", {}))
            world_updates_raw = json.dumps(parsed_json.get("world_updates", {}))
            kg_triples_list = parsed_json.get("kg_triples", [])
            if isinstance(kg_triples_list, list):
                kg_triples_text = "\n".join([str(t) for t in kg_triples_list])
            else:
                kg_triples_text = str(kg_triples_list)

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse full extraction JSON for chapter {chapter_number}: {e}. "
                f"Attempting to extract individual sections with regex."
            )
            # Fallback to regex extraction
            char_match = re.search(
                r'"character_updates"\s*:\s*({.*?})', raw_extracted_text, re.DOTALL
            )
            if char_match:
                char_updates_raw = char_match.group(1)
                logger.info(
                    f"Regex successfully extracted character_updates block for Ch {chapter_number}."
                )
            else:
                logger.warning(
                    f"Could not find character_updates JSON block via regex for Ch {chapter_number}."
                )

            world_match = re.search(
                r'"world_updates"\s*:\s*({.*?})', raw_extracted_text, re.DOTALL
            )
            if world_match:
                world_updates_raw = world_match.group(1)
                logger.info(
                    f"Regex successfully extracted world_updates block for Ch {chapter_number}."
                )
            else:
                logger.warning(
                    f"Could not find world_updates JSON block via regex for Ch {chapter_number}."
                )

            triples_match = re.search(
                r'"kg_triples"\s*:\s*(\[.*?\])', raw_extracted_text, re.DOTALL
            )
            if triples_match:
                try:
                    triples_list_from_regex = json.loads(triples_match.group(1))
                    if isinstance(triples_list_from_regex, list):
                        kg_triples_text = "\n".join(
                            [str(t) for t in triples_list_from_regex]
                        )
                        logger.info(
                            f"Regex successfully extracted and parsed kg_triples block for Ch {chapter_number}."
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Found kg_triples block via regex for Ch {chapter_number}, but it was invalid JSON."
                    )
            else:
                logger.warning(
                    f"Could not find kg_triples JSON array via regex for Ch {chapter_number}."
                )

        char_updates_from_llm = self.parse_character_updates(
            char_updates_raw, chapter_number
        )
        world_updates_from_llm = self.parse_world_updates(
            world_updates_raw, chapter_number
        )

        parsed_triples_structured = parse_rdf_triples_with_rdflib(kg_triples_text)

        logger.info(
            f"Chapter {chapter_number} LLM Extraction: "
            f"{len(char_updates_from_llm)} char updates, "
            f"{sum(len(items) for items in world_updates_from_llm.values())} world item updates, "
            f"{len(parsed_triples_structured)} KG triples."
        )

        current_char_profiles_models = character_profiles
        current_world_models = world_building

        self.merge_updates(
            current_char_profiles_models,  # Pass model instances
            current_world_models,  # Pass model instances
            char_updates_from_llm,  # Already model instances
            world_updates_from_llm,  # Already model instances
            chapter_number,
            is_from_flawed_draft,
        )
        logger.info(
            f"Merged LLM updates into in-memory state for chapter {chapter_number}."
        )

        # Persist the DELTA of updates (char_updates_from_llm, world_updates_from_llm)
        # These functions expect model instances and the chapter number for delta context.
        if char_updates_from_llm:
            await self.persist_profiles(char_updates_from_llm, chapter_number)
        if world_updates_from_llm:
            await self.persist_world(world_updates_from_llm, chapter_number)

        if parsed_triples_structured:
            try:
                await kg_queries.add_kg_triples_batch_to_db(
                    parsed_triples_structured, chapter_number, is_from_flawed_draft
                )
                logger.info(
                    f"Persisted {len(parsed_triples_structured)} KG triples for chapter {chapter_number} to Neo4j."
                )
            except Exception as e:
                logger.error(
                    f"Failed to persist KG triples for chapter {chapter_number}: {e}",
                    exc_info=True,
                )

        logger.info(
            "Knowledge extraction, in-memory merge, and delta persistence complete for chapter %d.",
            chapter_number,
        )
        return usage_data

    async def heal_and_enrich_kg(self):
        """
        Performs maintenance on the Knowledge Graph by enriching thin nodes,
        checking for inconsistencies, and resolving duplicate entities.
        """
        logger.info("KG Healer/Enricher: Starting maintenance cycle.")

        # 1. Enrichment (which includes healing orphans/stubs)
        enrichment_cypher = await self._find_and_enrich_thin_nodes()

        if enrichment_cypher:
            logger.info(
                f"Applying {len(enrichment_cypher)} enrichment updates to the KG."
            )
            try:
                await neo4j_manager.execute_cypher_batch(enrichment_cypher)
            except Exception as e:
                logger.error(
                    f"KG Healer/Enricher: Error applying enrichment batch: {e}",
                    exc_info=True,
                )
        else:
            logger.info(
                "KG Healer/Enricher: No thin nodes found for enrichment in this cycle."
            )

        # 2. Consistency Checks
        await self._run_consistency_checks()

        # 3. Entity Resolution
        await self._run_entity_resolution()

        # 4. Resolve dynamic relationship types using LLM guidance
        await self._resolve_dynamic_relationships()

        # 5. Relationship Healing
        promoted = await kg_queries.promote_dynamic_relationships()
        if promoted:
            logger.info("KG Healer: Promoted %d dynamic relationships.", promoted)
        removed = await kg_queries.deduplicate_relationships()
        if removed:
            logger.info("KG Healer: Deduplicated %d relationships.", removed)

        logger.info("KG Healer/Enricher: Maintenance cycle complete.")

    async def _find_and_enrich_thin_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Finds thin characters and world elements and generates enrichment updates in parallel."""
        statements: List[Tuple[str, Dict[str, Any]]] = []
        enrichment_tasks = []

        # Find all thin nodes first
        thin_chars = await character_queries.find_thin_characters_for_enrichment()
        thin_elements = await world_queries.find_thin_world_elements_for_enrichment()

        # Create tasks for enriching characters
        for char_info in thin_chars:
            enrichment_tasks.append(self._create_character_enrichment_task(char_info))

        # Create tasks for enriching world elements
        for element_info in thin_elements:
            enrichment_tasks.append(
                self._create_world_element_enrichment_task(element_info)
            )

        if not enrichment_tasks:
            return []

        logger.info(
            f"KG Healer: Found {len(enrichment_tasks)} thin nodes to enrich. Running LLM calls in parallel."
        )
        results = await asyncio.gather(*enrichment_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"KG Healer: An enrichment task failed: {result}")
            elif result:
                statements.append(result)

        return statements

    async def _create_character_enrichment_task(
        self, char_info: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        char_name = char_info.get("name")
        if not char_name:
            return None

        logger.info(f"KG Healer: Found thin character '{char_name}' for enrichment.")
        context_chapters = await kg_queries.get_chapter_context_for_entity(
            entity_name=char_name
        )
        prompt = render_prompt(
            "kg_maintainer_agent/enrich_character.j2",
            {"character_name": char_name, "chapter_context": context_chapters},
        )
        enrichment_text, _ = await llm_service.async_call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            auto_clean_response=True,
        )
        if enrichment_text:
            try:
                data = json.loads(enrichment_text)
                new_description = data.get("description")
                if new_description and isinstance(new_description, str):
                    logger.info(
                        f"KG Healer: Generated new description for '{char_name}'."
                    )
                    return (
                        "MATCH (c:Character {name: $name}) SET c.description = $desc",
                        {"name": char_name, "desc": new_description},
                    )
            except json.JSONDecodeError:
                logger.error(
                    f"KG Healer: Failed to parse enrichment JSON for character '{char_name}': {enrichment_text}"
                )
        return None

    async def _create_world_element_enrichment_task(
        self, element_info: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        element_id = element_info.get("id")
        if not element_id:
            return None

        logger.info(
            f"KG Healer: Found thin world element '{element_info.get('name')}' (id: {element_id}) for enrichment."
        )
        context_chapters = await kg_queries.get_chapter_context_for_entity(
            entity_id=element_id
        )
        prompt = render_prompt(
            "kg_maintainer_agent/enrich_world_element.j2",
            {"element": element_info, "chapter_context": context_chapters},
        )
        enrichment_text, _ = await llm_service.async_call_llm(
            model_name=config.KNOWLEDGE_UPDATE_MODEL,
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            auto_clean_response=True,
        )
        if enrichment_text:
            try:
                data = json.loads(enrichment_text)
                new_description = data.get("description")
                if new_description and isinstance(new_description, str):
                    logger.info(
                        f"KG Healer: Generated new description for world element id '{element_id}'."
                    )
                    return (
                        "MATCH (we:WorldElement {id: $id}) SET we.description = $desc",
                        {"id": element_id, "desc": new_description},
                    )
            except json.JSONDecodeError:
                logger.error(
                    f"KG Healer: Failed to parse enrichment JSON for world element id '{element_id}': {enrichment_text}"
                )
        return None

    async def _run_consistency_checks(self) -> None:
        """Runs various consistency checks on the KG and logs findings."""
        logger.info("KG Healer: Running consistency checks...")

        # 1. Check for contradictory traits
        contradictory_pairs = [
            ("Brave", "Cowardly"),
            ("Honest", "Deceitful"),
            ("Kind", "Cruel"),
            ("Generous", "Selfish"),
            ("Loyal", "Treacherous"),
        ]
        trait_findings = await kg_queries.find_contradictory_trait_characters(
            contradictory_pairs
        )
        if trait_findings:
            for finding in trait_findings:
                logger.warning(
                    f"KG Consistency Alert: Character '{finding.get('character_name')}' has contradictory traits: "
                    f"'{finding.get('trait1')}' and '{finding.get('trait2')}'."
                )
        else:
            logger.info("KG Consistency Check: No contradictory traits found.")

        # 2. Check for post-mortem activity
        activity_findings = await kg_queries.find_post_mortem_activity()
        if activity_findings:
            for finding in activity_findings:
                logger.warning(
                    f"KG Consistency Alert: Character '{finding.get('character_name')}' was marked dead in chapter "
                    f"{finding.get('death_chapter')} but has later activities: {finding.get('post_mortem_activities')}."
                )
        else:
            logger.info("KG Consistency Check: No post-mortem activity found.")

    async def _run_entity_resolution(self) -> None:
        """Finds and resolves potential duplicate entities in the KG."""
        logger.info("KG Healer: Running entity resolution...")
        candidate_pairs = await kg_queries.find_candidate_duplicate_entities()

        if not candidate_pairs:
            logger.info("KG Healer: No candidate duplicate entities found.")
            return

        logger.info(
            f"KG Healer: Found {len(candidate_pairs)} candidate pairs for entity resolution."
        )

        jinja_template = Template(ENTITY_RESOLUTION_PROMPT_TEMPLATE)

        for pair in candidate_pairs:
            id1, id2 = pair.get("id1"), pair.get("id2")
            if not id1 or not id2:
                continue

            # Fetch context for both entities in parallel
            context1_task = kg_queries.get_entity_context_for_resolution(id1)
            context2_task = kg_queries.get_entity_context_for_resolution(id2)
            context1, context2 = await asyncio.gather(context1_task, context2_task)

            if not context1 or not context2:
                logger.warning(
                    f"Could not fetch full context for pair ({id1}, {id2}). Skipping."
                )
                continue

            prompt = jinja_template.render(entity1=context1, entity2=context2)
            llm_response, _ = await llm_service.async_call_llm(
                model_name=config.KNOWLEDGE_UPDATE_MODEL,
                prompt=prompt,
                temperature=0.1,
                auto_clean_response=True,
            )

            try:
                decision_data = json.loads(llm_response)
                if (
                    decision_data.get("is_same_entity") is True
                    and decision_data.get("confidence_score", 0.0) > 0.8
                ):
                    logger.info(
                        f"LLM confirmed merge for '{context1.get('name')}' (id: {id1}) and "
                        f"'{context2.get('name')}' (id: {id2}). Reason: {decision_data.get('reason')}"
                    )

                    # Heuristic to decide which node to keep
                    degree1 = context1.get("degree", 0)
                    degree2 = context2.get("degree", 0)

                    # Prefer node with more relationships
                    if degree1 > degree2:
                        target_id, source_id = id1, id2
                    elif degree2 > degree1:
                        target_id, source_id = id2, id1
                    else:
                        # Tie-breaker: prefer the one with a more detailed description
                        desc1_len = len(
                            context1.get("properties", {}).get("description", "")
                        )
                        desc2_len = len(
                            context2.get("properties", {}).get("description", "")
                        )
                        if desc1_len >= desc2_len:
                            target_id, source_id = id1, id2
                        else:
                            target_id, source_id = id2, id1

                    await kg_queries.merge_entities(target_id, source_id)
                else:
                    logger.info(
                        f"LLM decided NOT to merge '{context1.get('name')}' and '{context2.get('name')}'. "
                        f"Reason: {decision_data.get('reason')}"
                    )

            except (json.JSONDecodeError, TypeError) as e:
                logger.error(
                    f"Failed to parse entity resolution response from LLM for pair ({id1}, {id2}): {e}. Response: {llm_response}"
                )

    async def _resolve_dynamic_relationships(self) -> None:
        """Resolve generic DYNAMIC_REL types using a lightweight LLM."""
        logger.info("KG Healer: Resolving dynamic relationship types via LLM...")
        dyn_rels = await kg_queries.fetch_unresolved_dynamic_relationships()
        if not dyn_rels:
            logger.info("KG Healer: No unresolved dynamic relationships found.")
            return
        jinja_template = Template(DYNAMIC_REL_RESOLUTION_PROMPT_TEMPLATE)
        for rel in dyn_rels:
            prompt = jinja_template.render(rel)
            new_type_raw, _ = await llm_service.async_call_llm(
                model_name=config.MEDIUM_MODEL,
                prompt=prompt,
                temperature=config.Temperatures.KG_EXTRACTION,
                max_tokens=10,
                auto_clean_response=True,
            )
            new_type = kg_queries.normalize_relationship_type(new_type_raw)
            if new_type and new_type != "UNKNOWN":
                await kg_queries.update_dynamic_relationship_type(
                    rel["rel_id"], new_type
                )
                logger.info(
                    "KG Healer: Updated relationship %s -> %s",
                    rel["rel_id"],
                    new_type,
                )
            else:
                logger.info(
                    "KG Healer: LLM could not refine relationship %s (response: %s)",
                    rel["rel_id"],
                    new_type_raw,
                )

    async def heal_schema(self) -> None:
        """Ensure all nodes and relationships follow the expected schema."""
        logger.info("KG Healer: Checking base schema conformity...")
        statements = [
            ("MATCH (n) WHERE NOT n:Entity SET n:Entity", {}),
            (
                "MATCH ()-[r:DYNAMIC_REL]-() WHERE r.type IS NULL SET r.type = 'UNKNOWN'",
                {},
            ),
        ]
        try:
            await neo4j_manager.execute_cypher_batch(statements)
            await kg_queries.normalize_existing_relationship_types()
        except Exception as exc:  # pragma: no cover - narrow DB errors
            logger.error("KG Healer: Schema healing failed: %s", exc, exc_info=True)


__all__ = ["KnowledgeAgent"]
