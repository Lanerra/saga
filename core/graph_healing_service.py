# core/graph_healing_service.py
"""Heal and maintain SAGA's Neo4j knowledge graph.

This service focuses on post-ingestion maintenance:
- Enriching provisional entities from accumulated narrative context.
- Graduating provisional entities once confidence crosses a threshold.
- Detecting and merging likely duplicate entities.
- Cleaning up truly orphaned provisional entities.

Notes:
    This module uses Neo4j's internal `elementId()` for write operations. Any cross-module
    calls (for example into `data_access.*`) should use the application-stable `n.id`
    property instead.
"""

from __future__ import annotations

import json
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import structlog

import config
from core.db_manager import neo4j_manager
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


class GraphHealingService:
    """Heal, enrich, and deduplicate knowledge graph entities.

    The primary entrypoint is [`core.graph_healing_service.GraphHealingService.heal_graph()`](core/graph_healing_service.py:690),
    which performs enrichment, graduation, merge execution, and orphan cleanup for a
    given chapter.

    Notes:
        Enrichment uses an LLM and is best-effort; failures generally produce no update
        rather than raising, so the workflow can continue.
    """

    CONFIDENCE_THRESHOLD = 0.6  # Lowered from 0.85 to make graduation achievable
    MERGE_SIMILARITY_THRESHOLD = 0.75
    AUTO_MERGE_THRESHOLD = 0.95
    AGE_GRADUATION_CHAPTERS = 5  # Graduate nodes that survive this many chapters
    ORPHAN_CLEANUP_CHAPTERS = 5  # Remove truly orphaned nodes after this many chapters

    async def identify_provisional_nodes(self) -> list[dict[str, Any]]:
        """List provisional nodes eligible for healing."""
        query = """
            MATCH (n)
            WHERE n.is_provisional = true
            RETURN
                // Neo4j-internal identifier (elementId) is for internal DB operations only.
                elementId(n) AS element_id,
                // Application-stable identifier (n.id) must be used for any cross-module boundary.
                n.id AS id,
                n.name AS name,
                labels(n)[0] AS type,
                n.description AS description,
                n.traits AS traits,
                n.created_chapter AS created_chapter
            ORDER BY n.created_chapter ASC
        """
        return await neo4j_manager.execute_read_query(query)

    async def calculate_node_confidence(self, node: dict[str, Any], current_chapter: int = 0) -> float:
        """Calculate a confidence score for a provisional node.

        Args:
            node: Node properties, including `element_id`, `type`, and optionally
                `description`, `traits`, and `created_chapter`.
            current_chapter: Current chapter number used for age-based scoring.

        Returns:
            Confidence score in `[0.0, 1.0]` (by construction of weighted components).

        Notes:
            The score is a weighted combination of:
            - Relationship connectivity (proxy for mentions/importance).
            - Attribute completeness (presence/quality of description and traits).
            - Age bonus for entities that persist across multiple chapters.
        """
        element_id = node["element_id"]

        # Evidence 1: Relationship connectivity (proxy for importance/mentions)
        # Count both incoming and outgoing relationships
        rel_query = """
            MATCH (n)-[r]-()
            WHERE elementId(n) = $element_id
            RETURN count(r) AS rel_count
        """
        results = await neo4j_manager.execute_read_query(rel_query, {"element_id": element_id})
        record = results[0] if results else None
        rel_count = record["rel_count"] if record else 0

        # Normalize: 3 relationships = max score (lowered from 5)
        connectivity_score = min(rel_count / 3, 1.0) * 0.4

        # Evidence 2: Attribute completeness
        completeness_score = 0.0
        if node.get("description") and node["description"] not in ["Unknown", "", None]:
            desc = node["description"]
            if len(desc) > 20 and "to be developed" not in desc.lower():
                completeness_score += 0.2
            else:
                completeness_score += 0.1

        if node.get("traits") and len(node["traits"]) > 0:
            completeness_score += 0.1

        # Check for additional attributes based on node type
        if node["type"] == "Character":
            status_query = """
                MATCH (n)
                WHERE elementId(n) = $element_id
                RETURN n.status AS status
            """
            results = await neo4j_manager.execute_read_query(status_query, {"element_id": element_id})
            record = results[0] if results else None
            if record and record.get("status") and record["status"] != "Unknown":
                completeness_score += 0.1

        # Evidence 3: Age bonus - nodes that survive multiple chapters are likely important
        age_score = 0.0
        created_chapter = node.get("created_chapter", current_chapter)
        if current_chapter > 0 and created_chapter:
            age = current_chapter - created_chapter
            if age >= self.AGE_GRADUATION_CHAPTERS:
                age_score = 0.2  # Full bonus for surviving 3+ chapters
            elif age >= 1:
                age_score = 0.1  # Partial bonus for surviving 1-2 chapters

        total_confidence = completeness_score + connectivity_score + age_score

        logger.debug(
            "Calculated node confidence",
            name=node["name"],
            completeness_score=completeness_score,
            connectivity_score=connectivity_score,
            age_score=age_score,
            total=total_confidence,
        )

        return total_confidence

    async def enrich_node_from_context(self, node: dict[str, Any], model: str) -> dict[str, Any]:
        """Infer missing attributes for an entity from accumulated chapter context.

        Args:
            node: Provisional node dict. If `node["id"]` is present it is used for
                cross-module lookups; otherwise the entity name is used as a fallback.
            model: Model name to use for enrichment.

        Returns:
            A dict of inferred attributes (for example, inferred description/traits/role)
            and an overall confidence score. Returns an empty dict when no enrichment
            is possible (no mentions found or enrichment fails).

        Notes:
            This function is best-effort and intentionally does not raise for JSON
            parsing failures or LLM call failures.
        """
        # `element_id` is Neo4j-internal. Keep it internal-only.
        # For cross-module calls (data_access.*), use stable application id (`n.id`).
        element_id = node["element_id"]
        entity_id = node.get("id")

        # Get all chapters where this entity appears using shared logic
        from data_access.kg_queries import get_chapter_context_for_entity

        # Prefer stable application id; fall back to name only if id is missing.
        mentions = await get_chapter_context_for_entity(entity_id=entity_id) if entity_id else await get_chapter_context_for_entity(entity_name=node.get("name"))

        if not mentions:
            logger.debug(
                "No chapter mentions found for node",
                name=node["name"],
                entity_id=entity_id,
                element_id=element_id,
            )
            return {}

        current_description = node.get("description") or "Unknown"
        current_traits = node.get("traits") or []

        # Build summaries text (handle different field names if needed)
        summaries_text = ""
        for m in mentions:
            chap_num = m.get("chapter_number") or m.get("chapter")
            summary = m.get("summary")
            if chap_num and summary:
                summaries_text += f"Chapter {chap_num}: {summary}\n"

        prompt = render_prompt(
            "knowledge_agent/enrich_node_from_context.j2",
            {
                "entity_name": node["name"],
                "entity_type": node["type"],
                "current_description": current_description,
                "current_traits": current_traits,
                "summaries_text": summaries_text.strip(),
            },
        )

        try:
            response_text, _ = await llm_service.async_call_llm(
                prompt=prompt,
                model_name=model,
                temperature=0.3,
                max_tokens=16384,
                system_prompt=get_system_prompt("knowledge_agent"),
            )

            enriched = json.loads(response_text)

            logger.info(
                "Enriched node from context",
                name=node["name"],
                confidence=enriched.get("confidence", 0),
            )

            return enriched

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                "Failed to enrich node",
                name=node["name"],
                error=str(e),
            )
            return {}

    async def apply_enrichment(self, element_id: str, enriched: dict[str, Any]) -> bool:
        """Apply validated enrichment fields to a Neo4j node."""
        if not enriched or enriched.get("confidence", 0) < 0.6:
            return False

        updates: list[str] = []
        params: dict[str, Any] = {"element_id": element_id}

        if enriched.get("inferred_description"):
            updates.append("n.description = $description")
            params["description"] = enriched["inferred_description"]

        if enriched.get("inferred_traits"):
            updates.append("n.traits = apoc.coll.toSet(coalesce(n.traits, []) + $new_traits)")
            params["new_traits"] = enriched["inferred_traits"]

        if enriched.get("inferred_role"):
            updates.append("n.role = $role")
            params["role"] = enriched["inferred_role"]

        if not updates:
            return False

        query = f"""
            MATCH (n)
            WHERE elementId(n) = $element_id
            SET {", ".join(updates)},
                n.enriched_at = datetime(),
                n.enrichment_confidence = $confidence
        """
        params["confidence"] = enriched.get("confidence", 0.7)

        await neo4j_manager.execute_write_query(query, params)
        return True

    async def get_node_by_element_id(self, element_id: str) -> dict[str, Any] | None:
        """Reload node properties from Neo4j using the internal element id.

        Notes:
            This is an internal-only helper used to avoid stale in-memory node dicts
            after enrichment writes.
        """
        query = """
            MATCH (n)
            WHERE elementId(n) = $element_id
            RETURN
                elementId(n) AS element_id,
                n.id AS id,
                n.name AS name,
                labels(n)[0] AS type,
                n.description AS description,
                n.traits AS traits,
                n.created_chapter AS created_chapter
        """
        results = await neo4j_manager.execute_read_query(query, {"element_id": element_id})
        return results[0] if results else None

    async def graduate_node(self, element_id: str, confidence: float) -> bool:
        """Mark a provisional node as graduated in Neo4j."""
        query = """
            MATCH (n)
            WHERE elementId(n) = $element_id
            SET n.is_provisional = false,
                n.graduated_at = datetime(),
                n.graduation_confidence = $confidence
            RETURN n.name AS name
        """
        results = await neo4j_manager.execute_write_query(query, {"element_id": element_id, "confidence": confidence})

        if results:
            record = results[0]
            logger.info(
                "Graduated node from provisional status",
                name=record["name"],
                confidence=confidence,
            )
            return True
        return False

    async def find_merge_candidates(self, use_advanced_matching: bool = True) -> list[dict[str, Any]]:
        """Find likely duplicate entity pairs.

        Args:
            use_advanced_matching: If True, delegates candidate discovery to
                `data_access.kg_queries.find_candidate_duplicate_entities` (name-based fuzzy
                matching). If False, computes semantic similarity via embeddings over
                entity descriptions.

        Returns:
            Candidate dicts containing the two entities (as Neo4j element ids), names, type,
            and similarity scores.

        Notes:
            When advanced matching is enabled and entity embedding healing is enabled, this
            method may compute an additional embedding similarity score using stored entity
            embeddings and combine it with name similarity.
        """
        if use_advanced_matching:
            # Use the advanced fuzzy matching from kg_queries
            from data_access.kg_queries import find_candidate_duplicate_entities

            kg_candidates = await find_candidate_duplicate_entities(
                similarity_threshold=self.MERGE_SIMILARITY_THRESHOLD,
                limit=75,
            )

            # Optional: enrich fuzzy candidates with embedding similarity using stored entity embeddings.
            embedding_by_id: dict[str, Any] = {}
            if config.ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING and kg_candidates:
                candidate_ids = sorted({c["id1"] for c in kg_candidates} | {c["id2"] for c in kg_candidates})
                embedding_query = f"""
                    MATCH (n)
                    WHERE n.id IN $ids
                    RETURN
                        n.id AS id,
                        labels(n) AS labels,
                        n.`{config.ENTITY_EMBEDDING_VECTOR_PROPERTY}` AS embedding_vector
                """
                embedding_rows = await neo4j_manager.execute_read_query(embedding_query, {"ids": candidate_ids})
                for row in embedding_rows:
                    node_id = row.get("id")
                    embedding_vector = row.get("embedding_vector")
                    if node_id and embedding_vector:
                        embedding_by_id[str(node_id)] = embedding_vector

            # Convert to our format and map id fields to element IDs
            candidates = []
            for c in kg_candidates:
                # Get element IDs from entity IDs
                get_element_id_query = """
                    MATCH (n)
                    WHERE n.id = $entity_id
                    RETURN elementId(n) AS element_id
                """

                primary_results = await neo4j_manager.execute_read_query(get_element_id_query, {"entity_id": c["id1"]})
                duplicate_results = await neo4j_manager.execute_read_query(get_element_id_query, {"entity_id": c["id2"]})

                if not primary_results or not duplicate_results:
                    continue

                name_similarity = float(c["similarity"])

                embedding_similarity = 0.0
                if config.ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING:
                    emb1 = embedding_by_id.get(str(c["id1"]))
                    emb2 = embedding_by_id.get(str(c["id2"]))
                    if emb1 is not None and emb2 is not None:
                        embedding_similarity = self._cosine_similarity(np.array(emb1, dtype=np.float32), np.array(emb2, dtype=np.float32))

                if config.ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING and embedding_similarity > 0.0:
                    combined_similarity = (0.4 * name_similarity) + (0.6 * embedding_similarity)
                else:
                    combined_similarity = name_similarity

                candidates.append(
                    {
                        "primary_id": primary_results[0]["element_id"],
                        "primary_name": c["name1"],
                        "duplicate_id": duplicate_results[0]["element_id"],
                        "duplicate_name": c["name2"],
                        "type": c["labels1"][0] if c.get("labels1") else "Unknown",
                        "similarity": combined_similarity,
                        "name_similarity": name_similarity,  # kg_queries uses name similarity
                        "embedding_similarity": embedding_similarity,
                        "is_alias": self._is_likely_alias(c["name1"], c["name2"]),
                    }
                )

            candidates.sort(key=lambda x: -x["similarity"])

            logger.info(
                "Found merge candidates via advanced fuzzy matching",
                count=len(candidates),
                embedding_scored=config.ENABLE_ENTITY_EMBEDDING_GRAPH_HEALING,
            )

            return candidates

        # Fallback to embedding-based similarity
        candidates = []

        # Get all entities with descriptions
        query = """
            MATCH (n)
            WHERE n.description IS NOT NULL AND n.description <> "" AND trim(n.description) <> ""
            RETURN
                elementId(n) AS element_id,
                n.name AS name,
                n.description AS description,
                labels(n)[0] AS type
        """
        entities = await neo4j_manager.execute_read_query(query)

        if len(entities) < 2:
            return []

        # Generate embeddings for all descriptions
        descriptions = [e["description"] for e in entities]
        embeddings = await llm_service.async_get_embeddings_batch(descriptions)

        # Compare same-type entities
        for i, e1 in enumerate(entities):
            for j in range(i + 1, len(entities)):
                e2 = entities[j]

                # Only compare same type
                if e1["type"] != e2["type"]:
                    continue

                # Name similarity
                name_sim = SequenceMatcher(None, e1["name"].lower(), e2["name"].lower()).ratio()

                # Embedding similarity
                emb_sim = 0.0
                emb1 = embeddings[i]
                emb2 = embeddings[j]
                if emb1 is not None and emb2 is not None:
                    emb_sim = self._cosine_similarity(emb1, emb2)

                # Combined score
                combined = 0.4 * name_sim + 0.6 * emb_sim

                if combined > self.MERGE_SIMILARITY_THRESHOLD:
                    # Check for alias patterns
                    is_alias = self._is_likely_alias(e1["name"], e2["name"])

                    candidates.append(
                        {
                            "primary_id": e1["element_id"],
                            "primary_name": e1["name"],
                            "duplicate_id": e2["element_id"],
                            "duplicate_name": e2["name"],
                            "type": e1["type"],
                            "similarity": combined,
                            "name_similarity": name_sim,
                            "embedding_similarity": emb_sim,
                            "is_alias": is_alias,
                        }
                    )

        # Sort by similarity descending
        candidates.sort(key=lambda x: -x["similarity"])

        logger.info(
            "Found merge candidates via embedding similarity",
            count=len(candidates),
        )

        return candidates

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity for two embedding vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _is_likely_alias(self, name1: str, name2: str) -> bool:
        """Heuristically detect whether two surface forms are likely aliases."""
        words1 = name1.split()
        words2 = name2.split()

        # Shared surname (last word)
        if len(words1) > 0 and len(words2) > 0:
            if words1[-1].lower() == words2[-1].lower() and len(words1[-1]) > 2:
                return True

        # First name only vs full name
        if len(words1) == 1 and len(words2) > 1:
            if words1[0].lower() == words2[0].lower():
                return True
        if len(words2) == 1 and len(words1) > 1:
            if words2[0].lower() == words1[0].lower():
                return True

        # Title variations (Dr., Mr., etc.)
        titles = {"dr.", "mr.", "mrs.", "ms.", "prof.", "sir", "lady"}
        words1_no_title = [w for w in words1 if w.lower() not in titles]
        words2_no_title = [w for w in words2 if w.lower() not in titles]

        if words1_no_title and words2_no_title:
            if " ".join(words1_no_title).lower() == " ".join(words2_no_title).lower():
                return True

        return False

    async def validate_merge(self, primary_id: str, duplicate_id: str) -> dict[str, Any]:
        """Validate that a candidate merge is unlikely to collapse distinct entities.

        Args:
            primary_id: Neo4j element id of the entity to keep.
            duplicate_id: Neo4j element id of the entity to merge into `primary_id`.

        Returns:
            A dict containing:
            - `cooccurrences`: Number of shared chapter/event neighbors.
            - `relationship_similarity`: Overlap ratio of outgoing relationship types.
            - `is_valid`: Whether the merge is considered safe enough to execute.

        Notes:
            This is a heuristic check. Co-occurrence through shared chapter/event context is
            treated as evidence that the two entities may be distinct.
        """
        # Check for co-occurrence via shared relationships to the same event/chapter
        # Using a more generic path since APPEARS_IN might not exist
        cooccurrence_query = """
            MATCH (n1)-[]-(x)-[]-(n2)
            WHERE elementId(n1) = $primary_id AND elementId(n2) = $duplicate_id
            AND (x:Chapter OR x:Event)
            RETURN count(x) AS cooccurrences
        """
        results = await neo4j_manager.execute_read_query(cooccurrence_query, {"primary_id": primary_id, "duplicate_id": duplicate_id})
        record = results[0] if results else None
        cooccurrences = record["cooccurrences"] if record else 0

        # Get relationship patterns
        rel_query = """
            MATCH (n)-[r]->()
            WHERE elementId(n) IN [$primary_id, $duplicate_id]
            RETURN elementId(n) AS node_id, type(r) AS rel_type, count(*) AS count
        """
        rel_patterns = await neo4j_manager.execute_read_query(rel_query, {"primary_id": primary_id, "duplicate_id": duplicate_id})

        # Build relationship fingerprints
        primary_rels = {r["rel_type"]: r["count"] for r in rel_patterns if r["node_id"] == primary_id}
        dup_rels = {r["rel_type"]: r["count"] for r in rel_patterns if r["node_id"] == duplicate_id}

        # Calculate relationship similarity
        all_types = set(primary_rels.keys()) | set(dup_rels.keys())
        if all_types:
            matching = sum(1 for t in all_types if t in primary_rels and t in dup_rels)
            rel_similarity = matching / len(all_types)
        else:
            rel_similarity = 1.0  # No relationships = compatible

        return {
            "cooccurrences": cooccurrences,
            "relationship_similarity": rel_similarity,
            "is_valid": cooccurrences == 0 and rel_similarity > 0.5,
        }

    async def execute_merge(self, primary_id: str, duplicate_id: str, merge_info: dict[str, Any]) -> bool:
        """Merge two entities by delegating to the canonical `kg_queries.merge_entities` flow.

        Args:
            primary_id: Neo4j element id of the entity to keep.
            duplicate_id: Neo4j element id of the entity to merge into `primary_id`.
            merge_info: Candidate metadata (for example similarity scores) used to build a
                human-readable merge reason.

        Returns:
            True if the merge succeeds.

        Notes:
            The merge implementation lives in `data_access.kg_queries.merge_entities` and
            operates on application-stable `n.id` values. This method resolves element ids
            to stable ids before crossing that module boundary.
        """
        from data_access.kg_queries import get_entity_context_for_resolution, merge_entities

        # Convert element IDs to entity IDs (id property) for any cross-module calls.
        # Element IDs are Neo4j internal; kg_queries APIs are standardized on stable `n.id`.
        get_id_query = """
            MATCH (n)
            WHERE elementId(n) = $element_id
            RETURN n.id AS entity_id
        """

        # Construct merge reason (may be improved using stable-id context below).
        similarity = merge_info.get("similarity", 0)
        reason = f"Merged duplicate entities (similarity: {similarity:.2f})"

        try:
            # Get entity IDs from element IDs
            primary_results = await neo4j_manager.execute_read_query(get_id_query, {"element_id": primary_id})
            duplicate_results = await neo4j_manager.execute_read_query(get_id_query, {"element_id": duplicate_id})

            if not primary_results or not duplicate_results:
                logger.error(
                    "Could not find entity IDs for merge",
                    primary_element_id=primary_id,
                    duplicate_element_id=duplicate_id,
                )
                return False

            primary_entity_id = primary_results[0]["entity_id"]
            duplicate_entity_id = duplicate_results[0]["entity_id"]

            if not primary_entity_id or not duplicate_entity_id:
                logger.error(
                    "Entities missing 'id' property",
                    primary_element_id=primary_id,
                    duplicate_element_id=duplicate_id,
                )
                return False

            # Get context for both entities (boundary call: must use stable `id`)
            primary_context = await get_entity_context_for_resolution(primary_entity_id)
            duplicate_context = await get_entity_context_for_resolution(duplicate_entity_id)

            if primary_context and duplicate_context:
                primary_name = primary_context.get("name", "unknown")
                duplicate_name = duplicate_context.get("name", "unknown")
                reason = f"Merged '{duplicate_name}' into '{primary_name}' (similarity: {similarity:.2f})"

            # Execute the merge using kg_queries implementation (stable ids)
            success = await merge_entities(
                source_id=duplicate_entity_id,
                target_id=primary_entity_id,
                reason=reason,
                max_retries=3,
            )

            if success:
                logger.info(
                    "Successfully executed entity merge via kg_queries.merge_entities",
                    primary_id=primary_entity_id,
                    duplicate_id=duplicate_entity_id,
                    similarity=similarity,
                )

            return success

        except Exception as e:
            logger.error(
                "Failed to execute merge",
                primary_id=primary_id,
                duplicate_id=duplicate_id,
                error=str(e),
                exc_info=True,
            )
            return False

    async def cleanup_orphaned_nodes(self, current_chapter: int) -> dict[str, Any]:
        """Delete provisional nodes that are both orphaned and stale.

        Args:
            current_chapter: Current chapter number used to compute the orphan cutoff window.

        Returns:
            Summary dict including counts for `nodes_checked` and `nodes_removed`.

        Notes:
            This cleanup intentionally targets only provisional nodes with no relationships
            to avoid removing entities that have participated in the graph.
        """
        results = {
            "nodes_removed": 0,
            "nodes_checked": 0,
        }

        # Find orphaned provisional nodes (no relationships, old enough)
        query = """
            MATCH (n)
            WHERE n.is_provisional = true
            AND NOT (n)-[]-()
            AND n.created_chapter IS NOT NULL
            AND n.created_chapter <= $cutoff_chapter
            RETURN
                elementId(n) AS element_id,
                n.name AS name,
                labels(n)[0] AS type,
                n.created_chapter AS created_chapter
        """
        cutoff = current_chapter - self.ORPHAN_CLEANUP_CHAPTERS

        orphaned_nodes = await neo4j_manager.execute_read_query(query, {"cutoff_chapter": cutoff})
        results["nodes_checked"] = len(orphaned_nodes)

        if not orphaned_nodes:
            return results

        # Remove orphaned nodes
        for node in orphaned_nodes:
            delete_query = """
                MATCH (n)
                WHERE elementId(n) = $element_id
                DELETE n
            """
            try:
                await neo4j_manager.execute_write_query(delete_query, {"element_id": node["element_id"]})
                results["nodes_removed"] += 1
                logger.info(
                    "Removed orphaned provisional node",
                    name=node["name"],
                    type=node["type"],
                    created_chapter=node["created_chapter"],
                    current_chapter=current_chapter,
                )
            except Exception as e:
                logger.warning(
                    "Failed to remove orphaned node",
                    name=node["name"],
                    error=str(e),
                )

        return results

    async def heal_graph(self, current_chapter: int, model: str) -> dict[str, Any]:
        """Run one full healing pass for the given chapter.

        Args:
            current_chapter: Chapter number used for age-based scoring and retention windows.
            model: Model name used for enrichment.

        Returns:
            Summary dict describing actions taken (enrichments, graduations, merges, cleanup)
            and any warnings collected.

        Notes:
            - Enrichment is best-effort and failures are recorded via logs rather than
              crashing the workflow.
            - Automatic merges are executed only above a high similarity threshold and
              after a heuristic safety validation step.
        """
        results: dict[str, Any] = {
            "chapter": current_chapter,
            "timestamp": datetime.now().isoformat(),
            "apoc_available": True,
            "nodes_enriched": 0,
            "nodes_graduated": 0,
            "nodes_merged": 0,
            "nodes_removed": 0,
            "merge_candidates_found": 0,
            "actions": [],
            "warnings": [],
        }

        # Step 1: Process provisional nodes
        provisional_nodes = await self.identify_provisional_nodes()
        results["provisional_count"] = len(provisional_nodes)

        logger.info(
            "Starting graph healing",
            chapter=current_chapter,
            provisional_count=len(provisional_nodes),
        )

        for node in provisional_nodes:
            # Calculate confidence with current chapter for age-based scoring
            confidence = await self.calculate_node_confidence(node, current_chapter)

            if confidence >= self.CONFIDENCE_THRESHOLD:
                # Graduate the node
                if await self.graduate_node(node["element_id"], confidence):
                    results["nodes_graduated"] += 1
                    results["actions"].append(
                        {
                            "type": "graduate",
                            "name": node["name"],
                            "confidence": confidence,
                        }
                    )
            else:
                # Try to enrich
                enriched = await self.enrich_node_from_context(node, model)
                if await self.apply_enrichment(node["element_id"], enriched):
                    results["nodes_enriched"] += 1
                    results["actions"].append(
                        {
                            "type": "enrich",
                            "name": node["name"],
                            "new_confidence": enriched.get("confidence", 0),
                        }
                    )

                    # Re-check confidence after enrichment using UPDATED node properties.
                    #
                    # CORE-009: Enrichment writes to Neo4j; recomputing confidence with the
                    # original in-memory `node` dict can be stale. Reload by element_id so
                    # `description`/`traits` reflect the applied enrichment.
                    updated_node = await self.get_node_by_element_id(node["element_id"])
                    if updated_node is None:
                        # Defensive fallback: if reload fails, at least avoid crashing and
                        # proceed with the original node dict.
                        updated_node = node

                    new_confidence = await self.calculate_node_confidence(updated_node, current_chapter)
                    if new_confidence >= self.CONFIDENCE_THRESHOLD:
                        if await self.graduate_node(node["element_id"], new_confidence):
                            results["nodes_graduated"] += 1
                            results["actions"].append(
                                {
                                    "type": "graduate",
                                    "name": node["name"],
                                    "confidence": new_confidence,
                                }
                            )

        # Step 2: Find and process merge candidates
        merge_candidates = await self.find_merge_candidates()
        results["merge_candidates_found"] = len(merge_candidates)

        for candidate in merge_candidates:
            # Validate the merge
            validation = await self.validate_merge(candidate["primary_id"], candidate["duplicate_id"])

            if not validation["is_valid"]:
                continue

            # Auto-approve high confidence merges
            if candidate["similarity"] >= self.AUTO_MERGE_THRESHOLD:
                if await self.execute_merge(candidate["primary_id"], candidate["duplicate_id"], candidate):
                    results["nodes_merged"] += 1
                    results["actions"].append(
                        {
                            "type": "merge",
                            "primary": candidate["primary_name"],
                            "duplicate": candidate["duplicate_name"],
                            "similarity": candidate["similarity"],
                            "auto_approved": True,
                        }
                    )

        # Step 3: Clean up truly orphaned nodes
        cleanup_results = await self.cleanup_orphaned_nodes(current_chapter)
        results["nodes_removed"] = cleanup_results["nodes_removed"]

        if cleanup_results["nodes_removed"] > 0:
            results["actions"].append(
                {
                    "type": "cleanup",
                    "nodes_removed": cleanup_results["nodes_removed"],
                    "nodes_checked": cleanup_results["nodes_checked"],
                }
            )

        logger.info(
            "Graph healing complete",
            chapter=current_chapter,
            graduated=results["nodes_graduated"],
            enriched=results["nodes_enriched"],
            merged=results["nodes_merged"],
            removed=results["nodes_removed"],
        )

        return results


# Singleton instance
graph_healing_service = GraphHealingService()
