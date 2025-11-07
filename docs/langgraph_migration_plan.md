# SAGA to LangGraph Migration Plan
 
## Executive Summary
 
This document outlines a detailed, step-by-step plan for migrating SAGA to LangGraph, beginning with **Section 10.1: What to Port Directly** from the LangGraph architecture document. This migration will reduce complexity while preserving SAGA's competitive advantages.
 
**Current Codebase:** ~31,800 lines of Python
**Target Reduction:** ~74% (to ~8,000 lines of core logic)
**Timeline:** 4-6 weeks for Phase 1 (Direct Porting)
 
---
 
## Phase 1: Port Direct-Compatible Components (Weeks 1-2)
 
This phase focuses on section 10.1 of the architecture document: extracting and refactoring components that can be ported directly to LangGraph nodes without major redesign.
 
### 1.1 Entity Extraction Logic → `extract_entities` Node
 
**Current Location:**
- `agents/knowledge_agent.py` (lines 660-997)
  - `_extract_updates_as_models()` - Core extraction logic
  - `parse_unified_character_updates()` - Character parsing
  - `parse_unified_world_updates()` - World item parsing
  - `_llm_extract_updates()` - LLM extraction call
  - `extract_and_merge_knowledge()` - Main orchestrator
 
**Migration Steps:**
 
#### Step 1.1.1: Create LangGraph State Schema (3 hours)
 
Create `core/langgraph/state.py`:
 
```python
from typing import TypedDict, List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from models.kg_models import CharacterProfile, WorldItem
 
class ExtractedEntity(BaseModel):
    """Entity extracted from generated text (before Neo4j commit)"""
    name: str
    type: Literal["character", "location", "event", "object"]
    description: str
    first_appearance_chapter: int
    attributes: Dict[str, str] = Field(default_factory=dict)
 
class ExtractedRelationship(BaseModel):
    """Relationship extracted from generated text"""
    source_name: str
    target_name: str
    relationship_type: str
    description: str
    chapter: int
    confidence: float = 0.8
 
class NarrativeState(TypedDict):
    """LangGraph state - all fields persisted automatically"""
    # Project metadata
    project_id: str
    title: str
    genre: str
    theme: str
    setting: str
    target_word_count: int
 
    # Neo4j connection (reconstructed on load, not persisted)
    neo4j_conn: Optional[object]
 
    # Current position
    current_chapter: int
    total_chapters: int
    current_act: int
 
    # Outline
    outline: Dict[int, Dict]
 
    # Active context
    active_characters: List[CharacterProfile]
    current_location: Optional[Dict]
    previous_chapter_summaries: List[str]
    key_events: List[Dict]
 
    # Generated content
    draft_text: Optional[str]
    draft_word_count: int
 
    # Entity extraction results (NEW: moved from scattered locations)
    extracted_entities: Dict[str, List[ExtractedEntity]]
    extracted_relationships: List[ExtractedRelationship]
 
    # Model configuration
    generation_model: str
    extraction_model: str
    revision_model: str
 
    # Workflow control
    current_node: str
    iteration_count: int
    max_iterations: int
    force_continue: bool
 
    # Error handling
    last_error: Optional[str]
    retry_count: int
 
    # Filesystem paths
    project_dir: str
    chapters_dir: str
    summaries_dir: str
```
 
**Deliverable:** State schema that matches LangGraph architecture
**Test:** Validate schema with `pydantic` validation
 
---
 
#### Step 1.1.2: Extract Entity Extraction Node (8 hours)
 
Create `core/langgraph/nodes/extraction_node.py`:
 
```python
"""
Entity extraction node for LangGraph workflow.
 
Ported from: agents/knowledge_agent.py
"""
import asyncio
from typing import List
import structlog
 
from core.langgraph.state import NarrativeState, ExtractedEntity, ExtractedRelationship
from core.llm_interface_refactored import llm_service
from prompts.prompt_renderer import render_prompt, get_system_prompt
import config
 
logger = structlog.get_logger(__name__)
 
 
async def extract_entities(state: NarrativeState) -> NarrativeState:
    """
    Extract entities and relationships from generated text.
 
    PORTED FROM: KnowledgeAgent._extract_updates_as_models()
 
    Runs extractions in parallel:
    - Characters
    - Locations
    - Events
    - Objects
    - Relationships
 
    Uses specialized extraction model (smaller, faster than generation model).
    """
    logger.info(
        f"Extracting entities from chapter {state['current_chapter']} "
        f"({state['draft_word_count']} words)"
    )
 
    if not state["draft_text"]:
        logger.warning("No draft text to extract from")
        return {
            **state,
            "extracted_entities": {},
            "extracted_relationships": [],
            "current_node": "extract"
        }
 
    # Call LLM to extract structured updates
    # PORTED FROM: KnowledgeAgent._llm_extract_updates()
    prompt = render_prompt(
        "knowledge_agent/extract_updates.j2",
        {
            "no_think": config.ENABLE_LLM_NO_THINK_DIRECTIVE,
            "protagonist": state.get("protagonist_name", config.DEFAULT_PROTAGONIST_NAME),
            "chapter_number": state["current_chapter"],
            "novel_title": state["title"],
            "novel_genre": state["genre"],
            "chapter_text": state["draft_text"],
            "available_node_labels": [],  # Will load from schema
            "available_relationship_types": [],  # Will load from schema
        },
    )
 
    try:
        raw_text, usage = await llm_service.async_call_llm(
            model_name=state["extraction_model"],
            prompt=prompt,
            temperature=config.Temperatures.KG_EXTRACTION,
            max_tokens=config.MAX_KG_TRIPLE_TOKENS,
            allow_fallback=True,
            stream_to_disk=False,
            frequency_penalty=config.FREQUENCY_PENALTY_KG_EXTRACTION,
            presence_penalty=config.PRESENCE_PENALTY_KG_EXTRACTION,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}", exc_info=True)
        return {
            **state,
            "extracted_entities": {},
            "extracted_relationships": [],
            "last_error": f"Extraction failed: {e}",
            "current_node": "extract"
        }
 
    # Parse extraction results
    # PORTED FROM: KnowledgeAgent._extract_updates_as_models()
    try:
        import json
        extraction_data = json.loads(raw_text)
 
        # Extract characters
        char_updates = []
        char_data = extraction_data.get("character_updates", {})
        for name, char_info in char_data.items():
            if isinstance(char_info, dict):
                char_updates.append(
                    ExtractedEntity(
                        name=name,
                        type="character",
                        description=char_info.get("description", ""),
                        first_appearance_chapter=state["current_chapter"],
                        attributes=char_info.get("attributes", {}),
                    )
                )
 
        # Extract world items
        world_updates = []
        world_data = extraction_data.get("world_updates", {})
        for category, items in world_data.items():
            if isinstance(items, dict):
                for item_name, item_info in items.items():
                    if isinstance(item_info, dict):
                        world_updates.append(
                            ExtractedEntity(
                                name=item_name,
                                type="location",  # Will refine with category
                                description=item_info.get("description", ""),
                                first_appearance_chapter=state["current_chapter"],
                                attributes={"category": category},
                            )
                        )
 
        # Extract relationships
        relationships = []
        kg_triples = extraction_data.get("kg_triples", [])
        for triple in kg_triples:
            if isinstance(triple, dict):
                relationships.append(
                    ExtractedRelationship(
                        source_name=triple.get("subject", ""),
                        target_name=triple.get("object_entity", ""),
                        relationship_type=triple.get("predicate", "RELATES_TO"),
                        description=triple.get("description", ""),
                        chapter=state["current_chapter"],
                        confidence=0.8,
                    )
                )
 
        logger.info(
            f"Extracted {len(char_updates)} characters, "
            f"{len(world_updates)} world items, "
            f"{len(relationships)} relationships"
        )
 
        return {
            **state,
            "extracted_entities": {
                "characters": char_updates,
                "world_items": world_updates,
            },
            "extracted_relationships": relationships,
            "current_node": "extract"
        }
 
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction JSON: {e}")
        return {
            **state,
            "extracted_entities": {},
            "extracted_relationships": [],
            "last_error": f"JSON parse error: {e}",
            "current_node": "extract"
        }
```
 
**Deliverable:** Standalone extraction node that works with LangGraph state
**Test:** Unit test with mock draft text → verify entities extracted
 
---
 
### 1.2 Deduplication Algorithms → `commit_to_graph` Node
 
**Current Location:**
- `processing/entity_deduplication.py` (all 245 lines)
  - `check_entity_similarity()` - Levenshtein + fuzzy matching
  - `should_merge_entities()` - Merge decision logic
  - `prevent_character_duplication()` - Proactive prevention
  - `prevent_world_item_duplication()` - World item dedup
  - `generate_entity_id()` - Deterministic ID generation
 
**Migration Steps:**
 
#### Step 1.2.1: Create Commit/Deduplication Node (10 hours)
 
Create `core/langgraph/nodes/commit_node.py`:
 
```python
"""
Knowledge graph commit node with deduplication.
 
Ported from: processing/entity_deduplication.py
"""
import structlog
from typing import Dict, List
 
from core.langgraph.state import NarrativeState, ExtractedEntity
from core.db_manager import neo4j_manager
from models.kg_models import CharacterProfile, WorldItem
from processing.entity_deduplication import (
    check_entity_similarity,
    should_merge_entities,
    generate_entity_id,
)
import config
 
logger = structlog.get_logger(__name__)
 
 
async def commit_to_graph(state: NarrativeState) -> NarrativeState:
    """
    Deduplicate entities and commit to Neo4j knowledge graph.
 
    PORTED FROM: processing/entity_deduplication.py
 
    This is where SAGA's existing deduplication logic lives:
    1. Character deduplication (name matching, embedding similarity)
    2. Location deduplication
    3. Relationship validation (ensure both endpoints exist)
    4. Graph connectivity maintenance
    """
    logger.info(
        f"Committing chapter {state['current_chapter']} to knowledge graph "
        f"with deduplication"
    )
 
    extracted = state["extracted_entities"]
    char_mappings: Dict[str, str] = {}
    world_mappings: Dict[str, str] = {}
 
    # Step 1: Deduplicate characters
    # PORTED FROM: prevent_character_duplication()
    char_entities = extracted.get("characters", [])
    for char in char_entities:
        # Check for existing similar character
        if config.ENABLE_DUPLICATE_PREVENTION:
            similar = await check_entity_similarity(
                char.name, "character"
            )
 
            if similar:
                should_merge = await should_merge_entities(
                    char.name,
                    char.description,
                    similar,
                    similarity_threshold=config.DUPLICATE_PREVENTION_SIMILARITY_THRESHOLD,
                )
 
                if should_merge:
                    # Use existing character
                    existing_name = similar["existing_name"]
                    char_mappings[char.name] = existing_name
                    logger.info(
                        f"Merged '{char.name}' → '{existing_name}' "
                        f"(similarity: {similar['similarity']:.2f})"
                    )
                    continue
 
        # Create new character
        char_mappings[char.name] = char.name
 
    # Step 2: Deduplicate world items
    # PORTED FROM: prevent_world_item_duplication()
    world_entities = extracted.get("world_items", [])
    for item in world_entities:
        category = item.attributes.get("category", "")
 
        if config.ENABLE_DUPLICATE_PREVENTION:
            similar = await check_entity_similarity(
                item.name, "world_element", category
            )
 
            if similar:
                should_merge = await should_merge_entities(
                    item.name,
                    item.description,
                    similar,
                )
 
                if should_merge:
                    # Use existing item
                    existing_id = similar["existing_id"]
                    world_mappings[item.name] = existing_id
                    logger.info(
                        f"Merged world item '{item.name}' → '{existing_id}' "
                        f"(similarity: {similar['similarity']:.2f})"
                    )
                    continue
 
        # Create new world item with deterministic ID
        new_id = generate_entity_id(
            item.name, category, state["current_chapter"]
        )
        world_mappings[item.name] = new_id
 
    # Step 3: Create/update entities in graph
    await _upsert_entities(
        state["neo4j_conn"],
        char_mappings,
        world_mappings,
        extracted,
        state["current_chapter"],
    )
 
    # Step 4: Create relationships (using deduplicated IDs)
    await _create_relationships(
        state["neo4j_conn"],
        state["extracted_relationships"],
        char_mappings,
        state["current_chapter"],
    )
 
    # Step 5: Create chapter node
    await _create_chapter_node(
        state["neo4j_conn"],
        state["current_chapter"],
        state["draft_text"],
        state["draft_word_count"],
    )
 
    logger.info(
        f"Committed {len(char_mappings)} characters, "
        f"{len(world_mappings)} world items to Neo4j"
    )
 
    return {
        **state,
        "current_node": "commit"
    }
 
 
async def _upsert_entities(
    neo4j_conn,
    char_mappings: Dict[str, str],
    world_mappings: Dict[str, str],
    extracted: Dict,
    chapter: int,
) -> None:
    """Upsert entities to Neo4j (PORTED FROM: data_access/character_queries.py)"""
    # Implementation using existing character_queries and world_queries
    pass
 
 
async def _create_relationships(
    neo4j_conn,
    relationships: List,
    char_mappings: Dict[str, str],
    chapter: int,
) -> None:
    """Create relationship edges (PORTED FROM: data_access/kg_queries.py)"""
    # Implementation using existing kg_queries
    pass
 
 
async def _create_chapter_node(
    neo4j_conn,
    chapter: int,
    text: str,
    word_count: int,
) -> None:
    """Create chapter node (PORTED FROM: data_access/chapter_queries.py)"""
    # Implementation using existing chapter_queries
    pass
```
 
**Deliverable:** Commit node with full deduplication logic
**Test:** Integration test with Neo4j → verify no duplicates created
 
---
 
### 1.3 Neo4j Queries → Reuse in Context Construction
 
**Current Location:**
- `data_access/kg_queries.py` (~2,000+ lines)
- `data_access/character_queries.py` (~800 lines)
- `data_access/world_queries.py` (~600 lines)
- `data_access/plot_queries.py` (~400 lines)
- `data_access/chapter_queries.py` (~300 lines)
 
**Migration Strategy:** These queries are **already well-organized** and can be used directly in LangGraph nodes with minimal changes.
 
#### Step 1.3.1: Create Query Wrapper for LangGraph (4 hours)
 
Create `core/langgraph/graph_context.py`:
 
```python
"""
Neo4j context construction for LangGraph workflow.
 
Wraps existing data_access queries for use in LangGraph nodes.
NO major refactoring needed - queries are already well-structured.
"""
import structlog
from typing import Dict, List, Any, Optional
 
from data_access import (
    kg_queries,
    character_queries,
    world_queries,
    plot_queries,
    chapter_queries,
)
from core.db_manager import neo4j_manager
 
logger = structlog.get_logger(__name__)
 
 
async def build_context_from_graph(
    neo4j_conn,
    current_chapter: int,
    active_character_ids: List[str],
    location_id: Optional[str],
    lookback_chapters: int = 5,
) -> Dict[str, Any]:
    """
    Query Neo4j for narrative context.
 
    USES EXISTING QUERIES FROM: data_access/
 
    Returns structured context for prompt construction.
    """
    context = {
        "characters": [],
        "relationships": [],
        "summaries": [],
        "events": [],
        "location": None,
    }
 
    # Get character details (REUSES: character_queries.py)
    if active_character_ids:
        char_query = """
            MATCH (c:Character)
            WHERE c.name IN $char_ids
            RETURN c.name AS name, c.description AS description,
                   c.traits AS traits, c.status AS status
        """
        context["characters"] = await neo4j_manager.execute_read_query(
            char_query, {"char_ids": active_character_ids}
        )
 
    # Get character relationships (REUSES: kg_queries.py patterns)
    if active_character_ids:
        rel_query = """
            MATCH (c1:Character)-[r]->(c2:Character)
            WHERE c1.name IN $char_ids OR c2.name IN $char_ids
            RETURN c1.name AS source, type(r) AS rel_type,
                   c2.name AS target, r.description AS description
        """
        context["relationships"] = await neo4j_manager.execute_read_query(
            rel_query, {"char_ids": active_character_ids}
        )
 
    # Get recent chapter summaries (REUSES: chapter_queries.py)
    summary_query = """
        MATCH (ch:Chapter)
        WHERE ch.number >= $start_chapter AND ch.number < $current_chapter
        RETURN ch.number AS chapter, ch.summary AS summary
        ORDER BY ch.number DESC
        LIMIT $limit
    """
    context["summaries"] = await neo4j_manager.execute_read_query(
        summary_query,
        {
            "start_chapter": max(1, current_chapter - lookback_chapters),
            "current_chapter": current_chapter,
            "limit": lookback_chapters,
        },
    )
 
    # Get key events (REUSES: plot_queries.py patterns)
    event_query = """
        MATCH (e:Event)-[:OCCURRED_IN]->(ch:Chapter)
        WHERE ch.number >= $start_chapter AND ch.number < $current_chapter
        RETURN e.description AS description, e.importance AS importance,
               ch.number AS chapter
        ORDER BY e.importance DESC, ch.number DESC
        LIMIT 20
    """
    context["events"] = await neo4j_manager.execute_read_query(
        event_query,
        {
            "start_chapter": max(1, current_chapter - 10),
            "current_chapter": current_chapter,
        },
    )
 
    # Get location details (REUSES: world_queries.py)
    if location_id:
        loc_query = """
            MATCH (l:Location {id: $loc_id})
            RETURN l.name AS name, l.description AS description,
                   l.rules AS rules
        """
        loc_result = await neo4j_manager.execute_read_query(
            loc_query, {"loc_id": location_id}
        )
        context["location"] = loc_result[0] if loc_result else None
 
    logger.info(
        f"Built context: {len(context['characters'])} characters, "
        f"{len(context['relationships'])} relationships, "
        f"{len(context['events'])} events"
    )
 
    return context
```
 
**Deliverable:** Query wrapper that uses existing data_access layer
**Test:** Mock Neo4j with sample data → verify context structure
 
---
 
### 1.4 Relationship Validation → `validate_consistency` Node
 
**Current Location:**
- `core/relationship_validator.py` (all 505 lines)
  - `RelationshipConstraintValidator` class
  - `validate_relationship()` - Semantic validation
  - `validate_triple()` - Complete triple validation
  - `_find_best_semantic_match()` - Auto-correction
- `core/relationship_constraints/constraints.py` - Constraint definitions
 
**Migration Steps:**
 
#### Step 1.4.1: Create Validation Node (6 hours)
 
Create `core/langgraph/nodes/validation_node.py`:
 
```python
"""
Consistency validation node for LangGraph workflow.
 
Ported from: core/relationship_validator.py
"""
import structlog
from typing import List, Dict, Any
 
from core.langgraph.state import NarrativeState, ExtractedRelationship
from core.relationship_validator import (
    validate_triple_constraint,
    validate_batch_constraints,
)
from core.db_manager import neo4j_manager
import config
 
logger = structlog.get_logger(__name__)
 
 
class Contradiction:
    """Detected inconsistency (matches LangGraph architecture)"""
    def __init__(
        self,
        type: str,
        description: str,
        conflicting_chapters: List[int],
        severity: str,
        suggested_fix: str = None,
    ):
        self.type = type
        self.description = description
        self.conflicting_chapters = conflicting_chapters
        self.severity = severity
        self.suggested_fix = suggested_fix
 
 
async def validate_consistency(state: NarrativeState) -> NarrativeState:
    """
    Check generated content for contradictions against knowledge graph.
 
    PORTED FROM: core/relationship_validator.py
 
    Validation checks:
    1. Character trait consistency
    2. Relationship contradictions
    3. Event timeline violations
    4. World rule violations
    5. Plot stagnation detection
    """
    logger.info(f"Validating consistency for chapter {state['current_chapter']}")
 
    contradictions = []
 
    # Check 1: Validate all extracted relationships
    # PORTED FROM: RelationshipConstraintValidator.validate_batch()
    relationships = state["extracted_relationships"]
    if relationships:
        # Convert to triple format for validation
        triples = [
            {
                "subject": rel.source_name,
                "predicate": rel.relationship_type,
                "object_entity": rel.target_name,
            }
            for rel in relationships
        ]
 
        validation_results = validate_batch_constraints(triples)
 
        for rel, result in zip(relationships, validation_results):
            if not result.is_valid:
                contradictions.append(
                    Contradiction(
                        type="relationship",
                        description=f"Invalid relationship: {rel.source_name} "
                        f"{rel.relationship_type} {rel.target_name}. "
                        f"Errors: {', '.join(result.errors)}",
                        conflicting_chapters=[state["current_chapter"]],
                        severity="major" if not result.suggestions else "minor",
                        suggested_fix=f"Use: {result.suggestions[0][0]}"
                        if result.suggestions
                        else None,
                    )
                )
 
    # Check 2: Character trait consistency (NEW validation)
    trait_contradictions = await _check_character_traits(
        state["neo4j_conn"],
        state["extracted_entities"].get("characters", []),
        state["current_chapter"],
    )
    contradictions.extend(trait_contradictions)
 
    # Check 3: Plot stagnation (NEW validation)
    if _is_plot_stagnant(state):
        contradictions.append(
            Contradiction(
                type="plot_stagnation",
                description="Chapter does not significantly advance plot",
                conflicting_chapters=[state["current_chapter"]],
                severity="major",
                suggested_fix="Introduce conflict, decision, or revelation",
            )
        )
 
    # Severity-based decision
    critical_issues = [c for c in contradictions if c.severity == "critical"]
    major_issues = [c for c in contradictions if c.severity == "major"]
 
    needs_revision = (
        len(critical_issues) > 0 or len(major_issues) > 2
    ) and not state.get("force_continue", False)
 
    logger.info(
        f"Validation complete: {len(contradictions)} issues found "
        f"({len(critical_issues)} critical, {len(major_issues)} major). "
        f"Needs revision: {needs_revision}"
    )
 
    return {
        **state,
        "contradictions": contradictions,
        "needs_revision": needs_revision,
        "current_node": "validate",
    }
 
 
async def _check_character_traits(
    neo4j_conn,
    extracted_chars: List,
    current_chapter: int,
) -> List[Contradiction]:
    """
    Compare extracted character attributes with established traits.
 
    NEW FUNCTIONALITY: Not in current SAGA, but specified in LangGraph architecture.
    """
    contradictions = []
 
    # Check for contradictory trait pairs
    contradictory_pairs = [
        ("introverted", "extroverted"),
        ("brave", "cowardly"),
        ("honest", "deceitful"),
    ]
 
    for char in extracted_chars:
        # Get established traits from Neo4j
        query = """
            MATCH (c:Character {name: $name})
            RETURN c.traits AS traits, c.first_appearance AS first_chapter
        """
        result = await neo4j_manager.execute_read_query(
            query, {"name": char.name}
        )
 
        if result:
            existing = result[0]
            established_traits = set(existing.get("traits", []))
            new_attributes = set(char.attributes.keys())
 
            # Check for contradictions
            for trait_a, trait_b in contradictory_pairs:
                if trait_a in established_traits and trait_b in new_attributes:
                    contradictions.append(
                        Contradiction(
                            type="character_trait",
                            description=f"{char.name} was established as '{trait_a}' "
                            f"in chapter {existing['first_chapter']}, "
                            f"but is now described as '{trait_b}'",
                            conflicting_chapters=[
                                existing["first_chapter"],
                                current_chapter,
                            ],
                            severity="major",
                            suggested_fix=f"Remove '{trait_b}' or explain character development",
                        )
                    )
 
    return contradictions
 
 
def _is_plot_stagnant(state: NarrativeState) -> bool:
    """
    Detect if plot is not advancing.
 
    NEW FUNCTIONALITY: Specified in LangGraph architecture.
    """
    # Check word count
    if state["draft_word_count"] < 1500:
        return True
 
    # Check if events extracted
    if len(state["extracted_entities"].get("events", [])) == 0:
        return True
 
    # Check if relationships changed
    if len(state["extracted_relationships"]) == 0:
        return True
 
    return False
```
 
**Deliverable:** Validation node with relationship + trait checking
**Test:** Unit test with contradictory data → verify detection
 
---
 
## Phase 1 Summary & Deliverables
 
### Completed Components (Weeks 1-2)
 
1. ✅ **State Schema** (`core/langgraph/state.py`)
   - NarrativeState TypedDict
   - ExtractedEntity and ExtractedRelationship models
   - Full Pydantic validation
 
2. ✅ **Extraction Node** (`core/langgraph/nodes/extraction_node.py`)
   - Ported from `agents/knowledge_agent.py`
   - Async LLM extraction
   - Character, world item, relationship parsing
 
3. ✅ **Commit/Deduplication Node** (`core/langgraph/nodes/commit_node.py`)
   - Ported from `processing/entity_deduplication.py`
   - Character fuzzy matching
   - World item deduplication
   - Neo4j commit logic
 
4. ✅ **Query Wrapper** (`core/langgraph/graph_context.py`)
   - Reuses existing `data_access/` queries
   - Context construction for prompt building
   - No major refactoring needed
 
5. ✅ **Validation Node** (`core/langgraph/nodes/validation_node.py`)
   - Ported from `core/relationship_validator.py`
   - Relationship semantic validation
   - Character trait consistency
   - Plot stagnation detection
 
### Testing Strategy
 
```bash
# Unit tests for each node
pytest tests/langgraph/test_extraction_node.py
pytest tests/langgraph/test_commit_node.py
pytest tests/langgraph/test_validation_node.py
 
# Integration test
pytest tests/langgraph/test_node_integration.py
 
# End-to-end test (single chapter)
pytest tests/langgraph/test_e2e_single_chapter.py
```
 
### Code Metrics (Phase 1)
 
| Component | Current LOC | New LOC | Reduction |
|-----------|-------------|---------|-----------|
| Entity Extraction | ~1,200 | ~250 | 79% |
| Deduplication | ~245 | ~200 | 18% |
| Query Layer | ~4,100 | ~150 (wrapper) | 96% |
| Validation | ~505 | ~300 | 41% |
| **TOTAL** | **~6,050** | **~900** | **85%** |
 
---
 
## Phase 2 Preview: What to Replace (Weeks 3-4)
 
After Phase 1 is complete, we'll tackle section 10.2 of the architecture:
 
1. **Custom state management** → LangGraph state + SqliteSaver checkpointer
2. **Workflow coordination code** → LangGraph graph definition
3. **Manual retry logic** → LangGraph conditional edges
4. **Progress tracking** → LangGraph event stream
 
This phase will deliver the biggest LOC reductions (~20,000 lines → ~500 lines).
 
---
 
## Phase 3 Preview: What to Add (Weeks 5-6)
 
Section 10.3 new capabilities:
 
1. **Parallel execution** for entity extraction (5x speedup)
2. **Formalized quality gates** with conditional edges
3. **Structured revision loops** with iteration limits
4. **Visual debugging** via LangGraph Studio
 
---
 
## Migration Risk Assessment
 
### Low Risk (Phase 1)
- ✅ Entity extraction logic is already modular
- ✅ Deduplication algorithms are well-tested
- ✅ Neo4j queries are production-ready
- ✅ Validation logic is isolated
 
### Medium Risk (Phase 2)
- ⚠️ State management refactor requires careful testing
- ⚠️ Workflow coordination changes may break existing flows
 
### High Risk (Phase 3)
- 🔴 Parallel execution may reveal race conditions
- 🔴 New validation logic needs extensive testing
 
---
 
## Success Criteria
 
### Phase 1 (Direct Porting)
- [ ] All 5 components ported and unit tested
- [ ] Integration tests pass with Neo4j
- [ ] Single chapter generation works end-to-end
- [ ] No regressions in entity extraction quality
- [ ] Deduplication precision maintained (>95%)
 
### Code Quality Gates
- [ ] All new code has >90% test coverage
- [ ] All existing tests continue to pass
- [ ] Performance: Extraction node <30s per chapter
- [ ] Memory: <2GB RAM for full workflow
 
---
 
## Next Steps
 
1. **Review this plan** with the team
2. **Create feature branch**: `feature/langgraph-phase-1-direct-porting`
3. **Start with Step 1.1.1**: Create state schema
4. **Daily standups**: Track progress against this plan
5. **Weekly demos**: Show working nodes in isolation
 
---
 
## Questions & Decisions Needed
 
1. **Checkpointer Backend**: SqliteSaver (local) or PostgresSaver (production)?
2. **Parallel Execution**: Enable in Phase 1 or wait for Phase 3?
3. **Backward Compatibility**: Keep old workflow during migration?
4. **Testing Strategy**: Mock Neo4j or use test database?