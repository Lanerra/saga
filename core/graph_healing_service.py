# core/graph_healing_service.py
"""
Graph Healing Service for SAGA Knowledge Graph.

This service provides functionality for:
1. Healing provisional nodes by enriching them with accumulated evidence
2. Merging semantically similar nodes
3. Graduating nodes from provisional status when confidence is high enough
"""

from __future__ import annotations

import json
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import structlog

from core.db_manager import neo4j_manager
from core.llm_interface_refactored import llm_service
from prompts.grammar_loader import load_grammar

logger = structlog.get_logger(__name__)


class GraphHealingService:
    """Service for healing and enriching the knowledge graph."""

    CONFIDENCE_THRESHOLD = 0.5  # Lowered from 0.85 to make graduation achievable
    MERGE_SIMILARITY_THRESHOLD = 0.75
    AUTO_MERGE_THRESHOLD = 0.95
    AGE_GRADUATION_CHAPTERS = 3  # Graduate nodes that survive this many chapters
    ORPHAN_CLEANUP_CHAPTERS = 5  # Remove truly orphaned nodes after this many chapters

    async def identify_provisional_nodes(self) -> list[dict[str, Any]]:
        """Find all provisional nodes in the graph."""
        query = """
            MATCH (n)
            WHERE n.is_provisional = true
            RETURN
                elementId(n) AS element_id,
                n.name AS name,
                labels(n)[0] AS type,
                n.description AS description,
                n.traits AS traits,
                n.created_chapter AS created_chapter
            ORDER BY n.created_chapter ASC
        """
        return await neo4j_manager.execute_read_query(query)

    async def calculate_node_confidence(
        self, node: dict[str, Any], current_chapter: int = 0
    ) -> float:
        """
        Calculate confidence score for a provisional node.

        Evidence sources:
        1. Relationship connectivity (40%) - lowered weight, needs only 3 relationships for max
        2. Attribute completeness (40%)
        3. Age bonus (20%) - nodes that survive multiple chapters get bonus
        """
        element_id = node["element_id"]

        # Evidence 1: Relationship connectivity (proxy for importance/mentions)
        # Count both incoming and outgoing relationships
        rel_query = """
            MATCH (n)-[r]-()
            WHERE elementId(n) = $element_id
            RETURN count(r) AS rel_count
        """
        results = await neo4j_manager.execute_read_query(
            rel_query, {"element_id": element_id}
        )
        record = results[0] if results else None
        rel_count = record["rel_count"] if record else 0

        # Normalize: 3 relationships = max score (lowered from 5)
        connectivity_score = min(rel_count / 3, 1.0) * 0.4

        # Evidence 2: Attribute completeness
        completeness_score = 0.0
        if node.get("description") and node["description"] not in ["Unknown", "", None]:
            # Check if description is meaningful (not just a stub)
            desc = node["description"]
            if len(desc) > 20 and "to be developed" not in desc.lower():
                completeness_score += 0.2
            else:
                completeness_score += 0.1  # Partial credit for stub descriptions

        if node.get("traits") and len(node["traits"]) > 0:
            completeness_score += 0.1

        # Check for additional attributes based on node type
        if node["type"] == "Character":
            status_query = """
                MATCH (n)
                WHERE elementId(n) = $element_id
                RETURN n.status AS status
            """
            results = await neo4j_manager.execute_read_query(
                status_query, {"element_id": element_id}
            )
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

    async def enrich_node_from_context(
        self, node: dict[str, Any], model: str
    ) -> dict[str, Any]:
        """
        Use LLM to infer missing attributes from all mentions of this entity.

        Returns enriched attributes that can be applied to the node.
        """
        element_id = node["element_id"]

        # Get all chapters where this entity appears using shared logic
        from data_access.kg_queries import get_chapter_context_for_entity

        mentions = await get_chapter_context_for_entity(entity_id=element_id)

        if not mentions:
            logger.debug("No chapter mentions found for node", name=node["name"])
            return {}

        # Build enrichment prompt
        current_description = node.get("description") or "Unknown"
        current_traits = node.get("traits") or []

        # Build summaries text (handle different field names if needed)
        summaries_text = ""
        for m in mentions:
            chap_num = m.get("chapter_number") or m.get("chapter")
            summary = m.get("summary")
            if chap_num and summary:
                summaries_text += f"Chapter {chap_num}: {summary}\n"

        prompt = f"""Based on the following chapter summaries mentioning "{node['name']}",
infer any missing attributes for this {node['type'].lower()}.

Current known attributes:
- Description: {current_description}
- Traits: {current_traits}

Chapter summaries mentioning this entity:
{summaries_text}

Provide the following information (only include fields where you have reasonable confidence):
1. **inferred_description**: string - improved description if current is Unknown or incomplete
2. **inferred_traits**: list of strings - personality traits
3. **inferred_role**: string - character's role in the story
4. **confidence**: float (0.0-1.0) - confidence in these inferences

*Note: The system enforces the output structure. Focus on inferring accurate attributes.*"""

        try:
            # Load healing grammar
            grammar_content = load_grammar("healing")
            # Load the grammar content directly as it already defines the correct root
            grammar = grammar_content

            response_text, _ = await llm_service.async_call_llm(
                prompt=prompt,
                model_name=model,
                temperature=0.3,
                max_tokens=512,
                grammar=grammar,
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
        """Apply enriched attributes to a node."""
        if not enriched or enriched.get("confidence", 0) < 0.6:
            return False

        updates = []
        params = {"element_id": element_id}

        if enriched.get("inferred_description"):
            updates.append("n.description = $description")
            params["description"] = enriched["inferred_description"]

        if enriched.get("inferred_traits"):
            updates.append(
                "n.traits = apoc.coll.toSet(coalesce(n.traits, []) + $new_traits)"
            )
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

    async def graduate_node(self, element_id: str, confidence: float) -> bool:
        """Graduate a node from provisional status."""
        query = """
            MATCH (n)
            WHERE elementId(n) = $element_id
            SET n.is_provisional = false,
                n.graduated_at = datetime(),
                n.graduation_confidence = $confidence
            RETURN n.name AS name
        """
        results = await neo4j_manager.execute_write_query(
            query, {"element_id": element_id, "confidence": confidence}
        )

        if results:
            record = results[0]
            logger.info(
                "Graduated node from provisional status",
                name=record["name"],
                confidence=confidence,
            )
            return True
        return False

    async def find_merge_candidates(self) -> list[dict[str, Any]]:
        """
        Find pairs of entities that may be duplicates.

        Uses multiple similarity measures:
        1. Name similarity (fuzzy matching)
        2. Description embedding similarity
        3. Relationship pattern similarity
        """
        candidates = []

        # Get all entities with descriptions
        # Note: Removed n.is_active check as it's not a standard field
        query = """
            MATCH (n)
            WHERE n.description IS NOT NULL
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
                name_sim = SequenceMatcher(
                    None, e1["name"].lower(), e2["name"].lower()
                ).ratio()

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
            "Found merge candidates",
            count=len(candidates),
        )

        return candidates

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _is_likely_alias(self, name1: str, name2: str) -> bool:
        """Check if two names are likely aliases of the same entity."""
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

    async def validate_merge(
        self, primary_id: str, duplicate_id: str
    ) -> dict[str, Any]:
        """
        Validate a merge by checking for co-occurrence.

        Entities that appear together in the same chapter are less likely
        to be duplicates.
        """
        # Check for co-occurrence via shared relationships to the same event/chapter
        # Using a more generic path since APPEARS_IN might not exist
        cooccurrence_query = """
            MATCH (n1)-[]-(x)-[]-(n2)
            WHERE elementId(n1) = $primary_id AND elementId(n2) = $duplicate_id
            AND (x:Chapter OR x:Event)
            RETURN count(x) AS cooccurrences
        """
        results = await neo4j_manager.execute_read_query(
            cooccurrence_query, {"primary_id": primary_id, "duplicate_id": duplicate_id}
        )
        record = results[0] if results else None
        cooccurrences = record["cooccurrences"] if record else 0

        # Get relationship patterns
        rel_query = """
            MATCH (n)-[r]->()
            WHERE elementId(n) IN [$primary_id, $duplicate_id]
            RETURN elementId(n) AS node_id, type(r) AS rel_type, count(*) AS count
        """
        rel_patterns = await neo4j_manager.execute_read_query(
            rel_query, {"primary_id": primary_id, "duplicate_id": duplicate_id}
        )

        # Build relationship fingerprints
        primary_rels = {
            r["rel_type"]: r["count"]
            for r in rel_patterns
            if r["node_id"] == primary_id
        }
        dup_rels = {
            r["rel_type"]: r["count"]
            for r in rel_patterns
            if r["node_id"] == duplicate_id
        }

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

    async def execute_merge(
        self, primary_id: str, duplicate_id: str, merge_info: dict[str, Any]
    ) -> bool:
        """
        Execute a merge of two entities.

        The primary entity absorbs the duplicate:
        1. Transfer all relationships
        2. Merge attributes (union of traits, keep longer description)
        3. Mark duplicate as merged (preserve history)
        """
        # Step 1: Transfer outgoing relationships
        transfer_out_query = """
            MATCH (dup)-[r]->(target)
            WHERE elementId(dup) = $dup_id
            MATCH (primary)
            WHERE elementId(primary) = $primary_id
            WITH primary, target, type(r) AS rel_type, properties(r) AS props
            MERGE (primary)-[new_rel:TRANSFERRED_REL]->(target)
            SET new_rel = props
            WITH primary, target, rel_type, new_rel
            CALL apoc.refactor.setType(new_rel, rel_type) YIELD output
            RETURN count(*) AS transferred
        """

        # Step 2: Transfer incoming relationships
        transfer_in_query = """
            MATCH (source)-[r]->(dup)
            WHERE elementId(dup) = $dup_id
            MATCH (primary)
            WHERE elementId(primary) = $primary_id
            WITH source, primary, type(r) AS rel_type, properties(r) AS props
            MERGE (source)-[new_rel:TRANSFERRED_REL]->(primary)
            SET new_rel = props
            WITH source, primary, rel_type, new_rel
            CALL apoc.refactor.setType(new_rel, rel_type) YIELD output
            RETURN count(*) AS transferred
        """

        # Step 3: Merge attributes
        merge_attrs_query = """
            MATCH (primary), (dup)
            WHERE elementId(primary) = $primary_id AND elementId(dup) = $dup_id
            SET primary.aliases = coalesce(primary.aliases, []) + [dup.name],
                primary.traits = apoc.coll.toSet(
                    coalesce(primary.traits, []) + coalesce(dup.traits, [])
                ),
                primary.description = CASE
                    WHEN size(coalesce(dup.description, '')) > size(coalesce(primary.description, ''))
                    THEN dup.description
                    ELSE primary.description
                END,
                primary.merged_from = coalesce(primary.merged_from, []) + [elementId(dup)],
                primary.merged_at = datetime()
            RETURN primary.name AS name
        """

        # Step 4: Mark duplicate as merged
        mark_merged_query = """
            MATCH (dup)
            WHERE elementId(dup) = $dup_id
            SET dup.merged_into = $primary_id,
                    dup.is_active = false,
                    dup.merged_at = datetime()
            RETURN dup.name AS dup_name
        """

        try:
            # Execute in sequence
            await neo4j_manager.execute_write_query(
                transfer_out_query, {"dup_id": duplicate_id, "primary_id": primary_id}
            )
            await neo4j_manager.execute_write_query(
                transfer_in_query, {"dup_id": duplicate_id, "primary_id": primary_id}
            )

            primary_results = await neo4j_manager.execute_write_query(
                merge_attrs_query, {"primary_id": primary_id, "dup_id": duplicate_id}
            )
            primary_record = primary_results[0] if primary_results else None

            dup_results = await neo4j_manager.execute_write_query(
                mark_merged_query, {"dup_id": duplicate_id, "primary_id": primary_id}
            )
            dup_record = dup_results[0] if dup_results else None

            if primary_record and dup_record:
                logger.info(
                    "Executed entity merge",
                    primary=primary_record["name"],
                    duplicate=dup_record["dup_name"],
                    similarity=merge_info.get("similarity", 0),
                )
                return True

        except Exception as e:
            logger.error(
                "Failed to execute merge",
                primary_id=primary_id,
                duplicate_id=duplicate_id,
                error=str(e),
            )
            return False

        return False

    async def cleanup_orphaned_nodes(self, current_chapter: int) -> dict[str, Any]:
        """
        Remove truly orphaned provisional nodes that have no relationships
        and have been around for too long.

        Returns summary of cleanup actions.
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

        orphaned_nodes = await neo4j_manager.execute_read_query(
            query, {"cutoff_chapter": cutoff}
        )
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
                await neo4j_manager.execute_write_query(
                    delete_query, {"element_id": node["element_id"]}
                )
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

    async def link_provisional_to_chapter(
        self, element_id: str, chapter_number: int
    ) -> bool:
        """
        Create a MENTIONED_IN relationship from a provisional node to its chapter.

        This helps with context retrieval for enrichment.
        """
        query = """
            MATCH (n), (c:Chapter {number: $chapter_number})
            WHERE elementId(n) = $element_id
            MERGE (n)-[r:MENTIONED_IN]->(c)
            SET r.created_at = timestamp()
            RETURN n.name AS name
        """
        try:
            results = await neo4j_manager.execute_write_query(
                query, {"element_id": element_id, "chapter_number": chapter_number}
            )
            return bool(results)
        except Exception as e:
            logger.warning(
                "Failed to link node to chapter",
                element_id=element_id,
                chapter=chapter_number,
                error=str(e),
            )
            return False

    async def heal_graph(self, current_chapter: int, model: str) -> dict[str, Any]:
        """
        Main entry point for graph healing.

        Performs:
        1. Identify and enrich provisional nodes
        2. Graduate high-confidence nodes
        3. Find and execute merges

        Returns summary of healing actions.
        """
        results: dict[str, Any] = {
            "chapter": current_chapter,
            "timestamp": datetime.now().isoformat(),
            "nodes_enriched": 0,
            "nodes_graduated": 0,
            "nodes_merged": 0,
            "nodes_removed": 0,
            "merge_candidates_found": 0,
            "actions": [],
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

                    # Re-check confidence after enrichment
                    new_confidence = await self.calculate_node_confidence(
                        node, current_chapter
                    )
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
            validation = await self.validate_merge(
                candidate["primary_id"], candidate["duplicate_id"]
            )

            if not validation["is_valid"]:
                continue

            # Auto-approve high confidence merges
            if candidate["similarity"] >= self.AUTO_MERGE_THRESHOLD:
                if await self.execute_merge(
                    candidate["primary_id"], candidate["duplicate_id"], candidate
                ):
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
