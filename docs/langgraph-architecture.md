Please evaluate the below proposed framework for a narrative generation architecture using LangGraph. Please provide your educated and informed opinion on it. You must maintain intellectual honesty at all times.

# SAGA 2.0: LangGraph-Based Narrative Generation Architecture

> **Semantically And Graph-enhanced Authoring - Second Generation**
>
> A hybrid architecture combining LangGraph workflow orchestration with Neo4j knowledge graph persistence for coherent long-form narrative generation.

---

## Executive Summary

SAGA 2.0 addresses the core limitations of the original SAGA implementation:

**Problems Solved:**
- Replace 31K+ lines of bespoke state management with LangGraph's built-in checkpointing
- Reduce complexity of workflow coordination through declarative graph definitions
- Enable parallel execution for entity extraction, validation, and quality checking
- Provide visual debugging through graph visualization
- Maintain Neo4j knowledge graph sophistication while improving modularity

**Not Thrown Away:**
- Neo4j knowledge graph architecture (your competitive advantage)
- Entity deduplication and identity resolution logic
- Relationship validation and connectivity checking
- Multi-model orchestration strategy
- Context construction from graph queries

**Architecture Philosophy:**
- LangGraph for *orchestration*
- Neo4j for *memory*
- Specialized models for *execution*
- Python for *logic*

---

## 1. System Architecture Overview

### 1.1 Component Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│  CLI / Web UI / API (FastAPI)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  LangGraph Orchestration Layer               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Generate │  │ Extract  │  │ Validate │  │ Revise   │   │
│  │  Node    │  │   Node   │  │   Node   │  │   Node   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│         State Graph + Checkpointer + Conditional Edges      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Domain Logic Layer                        │
│  Entity Extraction | Deduplication | Relationship Validation│
│  Context Construction | Prompt Engineering | Quality Scoring│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Persistence Layer                          │
│  ┌────────────────────┐         ┌─────────────────────┐    │
│  │  Neo4j Knowledge   │         │  File System        │    │
│  │      Graph         │         │  (Chapters, State)  │    │
│  │                    │         │                     │    │
│  │ Entities           │         │ chapter_01.md       │    │
│  │ Relationships      │         │ outline.yaml        │    │
│  │ Narrative History  │         │ state.json          │    │
│  └────────────────────┘         └─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     Model Layer                              │
│  Local LLMs (Ollama/vLLM) | Embedding Models | Specialized  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Decisions

**Why LangGraph:**
- Built-in state persistence (eliminates custom checkpointing code)
- Declarative graph definitions (clearer than imperative state machines)
- Parallel node execution (performance improvement)
- Conditional edges (complex branching without spaghetti code)
- Visual debugging (graph visualization)

**Why Keep Neo4j:**
- Relationship queries are first-class citizens
- Character deduplication requires graph traversal
- Temporal narrative tracking needs graph structure
- Cypher queries are more expressive than JSON/SQL for narrative relationships
- Your existing domain logic is built around this

**Why Hybrid File System:**
- Human readability (chapters as Markdown)
- Version control friendly (plain text diffs)
- LLM can read/write directly (no serialization overhead)
- Easy manual intervention (user can edit chapter_05.md)

---

## 2. State Schema

### 2.1 Core State Object

```python
from typing import TypedDict, List, Dict, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field

class EntityReference(BaseModel):
    """Reference to a Neo4j entity"""
    neo4j_id: str
    name: str
    type: Literal["character", "location", "event", "object"]
    
class ExtractedEntity(BaseModel):
    """Entity extracted from generated text (before Neo4j commit)"""
    name: str
    type: Literal["character", "location", "event", "object"]
    description: str
    first_appearance_chapter: int
    attributes: Dict[str, str] = Field(default_factory=dict)

class ExtractedRelationship(BaseModel):
    """Relationship extracted from generated text"""
    source_name: str  # Before deduplication
    target_name: str
    relationship_type: str
    description: str
    chapter: int
    confidence: float = 0.8

class Contradiction(BaseModel):
    """Detected inconsistency"""
    type: Literal["character_trait", "relationship", "event_sequence", "world_rule"]
    description: str
    conflicting_chapters: List[int]
    severity: Literal["minor", "major", "critical"]
    suggested_fix: Optional[str] = None

class RevisionRequest(BaseModel):
    """Instructions for revision node"""
    reason: str
    contradictions: List[Contradiction]
    target_model: str  # Allow escalation to better model
    max_attempts: int = 3

class NarrativeState(TypedDict):
    """LangGraph state - all fields are persisted automatically"""
    
    # Project metadata
    project_id: str
    title: str
    genre: str
    theme: str
    setting: str
    target_word_count: int
    
    # Neo4j connection (not persisted, reconstructed on load)
    neo4j_conn: Optional[object]
    
    # Current position in narrative
    current_chapter: int
    total_chapters: int
    current_act: int
    
    # Outline (hierarchical structure)
    outline: Dict[int, Dict]  # chapter_num -> {act, scene_description, key_beats}
    
    # Active context for current chapter
    active_characters: List[EntityReference]  # Characters in this scene
    current_location: Optional[EntityReference]
    previous_chapter_summaries: List[str]  # Last N summaries for context
    key_events: List[Dict]  # Recent critical events from graph
    
    # Generated content (current chapter being worked on)
    draft_text: Optional[str]
    draft_word_count: int
    
    # Entity extraction results (before commit to Neo4j)
    extracted_entities: Dict[str, List[ExtractedEntity]]
    extracted_relationships: List[ExtractedRelationship]
    
    # Validation results
    contradictions: List[Contradiction]
    needs_revision: bool
    revision_history: List[RevisionRequest]
    
    # Quality metrics
    coherence_score: Optional[float]
    prose_quality_score: Optional[float]
    plot_advancement_score: Optional[float]
    
    # Model configuration
    generation_model: str  # e.g., "qwen3-80b-q4"
    extraction_model: str  # e.g., "qwen-coder-32b-q4"
    revision_model: str    # e.g., "claude-sonnet" (can escalate)
    
    # Workflow control
    current_node: str  # Which node is executing
    iteration_count: int  # For revision loops
    max_iterations: int
    force_continue: bool  # User override for quality gates
    
    # Error handling
    last_error: Optional[str]
    retry_count: int
    
    # Filesystem paths
    project_dir: str
    chapters_dir: str
    summaries_dir: str
```

### 2.2 State Persistence Strategy

**LangGraph Checkpointer:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# LangGraph automatically persists state to SQLite
checkpointer = SqliteSaver.from_conn_string(
    f"{project_dir}/.saga/checkpoints.db"
)

# All NarrativeState fields are serialized/deserialized automatically
# No custom save/load logic needed
```

**Neo4j Connection:**
```python
# Not persisted in state (can't serialize connection objects)
# Reconstructed on load:

def _restore_neo4j_connection(state: NarrativeState) -> NarrativeState:
    """Called by LangGraph on state restoration"""
    if state["neo4j_conn"] is None:
        from neo4j import GraphDatabase
        state["neo4j_conn"] = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")
        )
    return state
```

**File System:**
```python
# Chapters written as they're finalized (not stored in state)
# State only tracks metadata and current draft

def _persist_chapter(state: NarrativeState) -> None:
    chapter_path = Path(state["chapters_dir"]) / f"chapter_{state['current_chapter']:03d}.md"
    chapter_path.write_text(
        f"""---
chapter: {state['current_chapter']}
word_count: {state['draft_word_count']}
generated_at: {datetime.now().isoformat()}
model: {state['generation_model']}
---

{state['draft_text']}
"""
    )
```

---

## 3. Node Implementations

### 3.1 Generation Node

```python
from langgraph.graph import StateGraph
from typing import Dict, Any

def generate_chapter(state: NarrativeState) -> NarrativeState:
    """
    Generate chapter prose based on outline and context from knowledge graph.
    
    Context Construction Strategy:
    1. Query Neo4j for characters in current scene
    2. Get summaries of last 5 chapters
    3. Retrieve key events from last 10 chapters
    4. Get character relationships relevant to current scene
    5. Build prompt with all context
    """
    
    # Step 1: Build context from knowledge graph
    context = _build_context_from_graph(
        neo4j_conn=state["neo4j_conn"],
        current_chapter=state["current_chapter"],
        active_character_ids=[c.neo4j_id for c in state["active_characters"]],
        location_id=state.get("current_location", {}).get("neo4j_id"),
        lookback_chapters=5
    )
    
    # Step 2: Construct generation prompt
    prompt = _construct_generation_prompt(
        chapter_num=state["current_chapter"],
        outline_entry=state["outline"][state["current_chapter"]],
        context=context,
        genre=state["genre"],
        theme=state["theme"]
    )
    
    # Step 3: Generate
    draft = _call_llm(
        prompt=prompt,
        model=state["generation_model"],
        max_tokens=4096,
        temperature=0.7,
        stop_sequences=["### END CHAPTER ###"]
    )
    
    # Step 4: Update state
    return {
        **state,
        "draft_text": draft,
        "draft_word_count": len(draft.split()),
        "current_node": "generate",
    }

def _build_context_from_graph(
    neo4j_conn,
    current_chapter: int,
    active_character_ids: List[str],
    location_id: Optional[str],
    lookback_chapters: int
) -> Dict[str, Any]:
    """
    Query Neo4j for narrative context.
    
    Returns structured context for prompt construction.
    """
    with neo4j_conn.session() as session:
        # Get character details
        characters = session.run("""
            MATCH (c:Character)
            WHERE c.id IN $char_ids
            RETURN c.id AS id, c.name AS name, c.description AS description,
                   c.personality_traits AS traits, c.motivations AS motivations
        """, char_ids=active_character_ids).data()
        
        # Get character relationships
        relationships = session.run("""
            MATCH (c1:Character)-[r]->(c2:Character)
            WHERE c1.id IN $char_ids OR c2.id IN $char_ids
            RETURN c1.name AS source, type(r) AS rel_type, 
                   c2.name AS target, r.description AS description
        """, char_ids=active_character_ids).data()
        
        # Get recent chapter summaries
        summaries = session.run("""
            MATCH (ch:Chapter)
            WHERE ch.number >= $start_chapter AND ch.number < $current_chapter
            RETURN ch.number AS chapter, ch.summary AS summary
            ORDER BY ch.number DESC
            LIMIT $limit
        """, 
            start_chapter=max(1, current_chapter - lookback_chapters),
            current_chapter=current_chapter,
            limit=lookback_chapters
        ).data()
        
        # Get key events from recent chapters
        events = session.run("""
            MATCH (e:Event)-[:OCCURRED_IN]->(ch:Chapter)
            WHERE ch.number >= $start_chapter AND ch.number < $current_chapter
            RETURN e.description AS description, e.importance AS importance,
                   ch.number AS chapter
            ORDER BY e.importance DESC, ch.number DESC
            LIMIT 20
        """, 
            start_chapter=max(1, current_chapter - 10),
            current_chapter=current_chapter
        ).data()
        
        # Get location details if specified
        location = None
        if location_id:
            location = session.run("""
                MATCH (l:Location {id: $loc_id})
                RETURN l.name AS name, l.description AS description,
                       l.rules AS rules
            """, loc_id=location_id).single()
        
        return {
            "characters": characters,
            "relationships": relationships,
            "summaries": summaries,
            "events": events,
            "location": location
        }

def _construct_generation_prompt(
    chapter_num: int,
    outline_entry: Dict,
    context: Dict,
    genre: str,
    theme: str
) -> str:
    """
    Build the actual prompt for the LLM.
    
    Prompt engineering strategy:
    - Clear instructions
    - Structured context (characters first, then events, then scene goal)
    - Explicit constraints (word count, tone, style)
    - Examples of good prose (optional, genre-dependent)
    """
    
    # Character context
    char_context = "\n".join([
        f"**{c['name']}**: {c['description']}. "
        f"Traits: {', '.join(c.get('traits', []))}. "
        f"Motivation: {c.get('motivations', 'Unknown')}."
        for c in context["characters"]
    ])
    
    # Relationship context
    rel_context = "\n".join([
        f"- {r['source']} {r['rel_type'].replace('_', ' ').lower()} {r['target']}: {r.get('description', '')}"
        for r in context["relationships"]
    ])
    
    # Recent events
    event_context = "\n".join([
        f"Chapter {e['chapter']}: {e['description']}"
        for e in context["events"][:10]  # Top 10 most important recent events
    ])
    
    # Summaries of recent chapters
    summary_context = "\n".join([
        f"Chapter {s['chapter']}: {s['summary']}"
        for s in reversed(context["summaries"])  # Chronological order
    ])
    
    # Location
    location_context = ""
    if context["location"]:
        loc = context["location"]
        location_context = f"\n**Setting**: {loc['name']} - {loc['description']}"
        if loc.get("rules"):
            location_context += f"\n**Location Rules**: {loc['rules']}"
    
    prompt = f"""You are writing Chapter {chapter_num} of a {genre} novel with the theme: "{theme}".

## Characters in This Scene
{char_context}

## Character Relationships
{rel_context}

## Recent Events (For Continuity)
{event_context}

## Recent Chapter Summaries
{summary_context}
{location_context}

## Scene Goal (from outline)
{outline_entry['scene_description']}

**Key beats to hit in this chapter:**
{chr(10).join(f"- {beat}" for beat in outline_entry.get('key_beats', []))}

## Instructions
Write Chapter {chapter_num} in approximately 3000-4000 words. 

**Style guidelines:**
- Match the genre conventions of {genre}
- Maintain narrative continuity with the events and relationships described above
- Show character development through actions and dialogue, not exposition
- Advance the plot while maintaining pacing appropriate for this point in the story
- End with a clear transition or hook for the next chapter

**DO NOT:**
- Introduce new characters not listed above without strong narrative justification
- Contradict established character traits or relationships
- Ignore key events from recent chapters
- Write meta-commentary or break the fourth wall (unless genre-appropriate)

Begin writing Chapter {chapter_num} now:
"""
    
    return prompt

def _call_llm(
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    stop_sequences: List[str]
) -> str:
    """
    Call local LLM via OpenAI-compatible API.
    
    Supports Ollama, vLLM, Text Generation WebUI, etc.
    """
    import httpx
    
    response = httpx.post(
        "http://localhost:11434/v1/chat/completions",  # Ollama endpoint
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop_sequences
        },
        timeout=300.0  # 5 minute timeout for long generations
    )
    
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
```

### 3.2 Entity Extraction Node (Parallel)

```python
import asyncio
from typing import List

async def extract_entities(state: NarrativeState) -> NarrativeState:
    """
    Extract entities and relationships from generated text.
    
    Run extractions in parallel:
    - Characters
    - Locations
    - Events
    - Objects
    - Relationships
    
    Uses specialized extraction model (smaller, faster than generation model).
    """
    
    # Run all extractions concurrently
    results = await asyncio.gather(
        _extract_characters(state["draft_text"], state["extraction_model"]),
        _extract_locations(state["draft_text"], state["extraction_model"]),
        _extract_events(state["draft_text"], state["extraction_model"], state["current_chapter"]),
        _extract_objects(state["draft_text"], state["extraction_model"]),
        _extract_relationships(state["draft_text"], state["extraction_model"], state["current_chapter"])
    )
    
    characters, locations, events, objects, relationships = results
    
    return {
        **state,
        "extracted_entities": {
            "characters": characters,
            "locations": locations,
            "events": events,
            "objects": objects
        },
        "extracted_relationships": relationships,
        "current_node": "extract"
    }

async def _extract_characters(text: str, model: str) -> List[ExtractedEntity]:
    """
    Extract character mentions from text.
    
    Prompt engineering: Ask for structured JSON output.
    """
    prompt = f"""Extract all characters mentioned in the following text. For each character, provide:
- name (exact name as it appears)
- description (physical appearance, personality)
- attributes (any traits, skills, or characteristics mentioned)

Return JSON array of objects with keys: name, description, attributes (dict).

Text:
{text}

JSON:"""

    response = await _call_llm_async(prompt, model, max_tokens=2048, temperature=0.1)
    
    # Parse JSON response
    import json
    try:
        data = json.loads(response)
        return [
            ExtractedEntity(
                name=c["name"],
                type="character",
                description=c["description"],
                first_appearance_chapter=0,  # Set by caller
                attributes=c.get("attributes", {})
            )
            for c in data
        ]
    except json.JSONDecodeError:
        # Fallback: LLM didn't return valid JSON
        # Log error and return empty list
        return []

async def _extract_relationships(text: str, model: str, chapter: int) -> List[ExtractedRelationship]:
    """
    Extract relationships between entities.
    
    This is critical for knowledge graph connectivity.
    """
    prompt = f"""Extract all relationships between characters in the following text.

For each relationship, provide:
- source_name: Name of first character
- target_name: Name of second character  
- relationship_type: Type of relationship (e.g., "LOVES", "WORKS_FOR", "BETRAYED", "TRUSTS", "FEARS")
- description: Brief description of the relationship as shown in the text

Use active verbs for relationship types. Be specific.

Return JSON array of objects.

Text:
{text}

JSON:"""

    response = await _call_llm_async(prompt, model, max_tokens=2048, temperature=0.1)
    
    import json
    try:
        data = json.loads(response)
        return [
            ExtractedRelationship(
                source_name=r["source_name"],
                target_name=r["target_name"],
                relationship_type=r["relationship_type"],
                description=r["description"],
                chapter=chapter,
                confidence=0.8  # Could use LLM confidence scores here
            )
            for r in data
        ]
    except json.JSONDecodeError:
        return []

async def _call_llm_async(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    """Async wrapper for LLM calls"""
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

# Similar implementations for _extract_locations, _extract_events, _extract_objects
```

### 3.3 Deduplication & Graph Commit Node

```python
def commit_to_graph(state: NarrativeState) -> NarrativeState:
    """
    Deduplicate entities and commit to Neo4j knowledge graph.
    
    This is where your existing SAGA logic lives:
    1. Character deduplication (name matching, embedding similarity)
    2. Location deduplication
    3. Relationship validation (ensure both endpoints exist)
    4. Graph connectivity maintenance
    """
    
    with state["neo4j_conn"].session() as session:
        # Step 1: Deduplicate characters
        char_mappings = _deduplicate_characters(
            session,
            state["extracted_entities"]["characters"],
            state["current_chapter"]
        )
        
        # Step 2: Deduplicate locations
        loc_mappings = _deduplicate_locations(
            session,
            state["extracted_entities"]["locations"]
        )
        
        # Step 3: Create/update entities in graph
        _upsert_entities(session, char_mappings, loc_mappings, state["current_chapter"])
        
        # Step 4: Create relationships (using deduplicated IDs)
        _create_relationships(
            session,
            state["extracted_relationships"],
            char_mappings,
            state["current_chapter"]
        )
        
        # Step 5: Create chapter node
        _create_chapter_node(
            session,
            state["current_chapter"],
            state["draft_text"],
            len(state["draft_text"].split())
        )
        
        # Step 6: Update active character list for next chapter
        active_chars = _get_active_characters_for_next_chapter(
            session,
            state["current_chapter"],
            state["outline"].get(state["current_chapter"] + 1, {})
        )
    
    return {
        **state,
        "active_characters": active_chars,
        "current_node": "commit"
    }

def _deduplicate_characters(
    session,
    extracted_chars: List[ExtractedEntity],
    current_chapter: int
) -> Dict[str, str]:
    """
    Your existing deduplication logic.
    
    Strategy:
    1. Exact name match (case-insensitive)
    2. Fuzzy name match (Levenshtein distance < 2)
    3. Embedding similarity (if names are different but descriptions similar)
    4. Manual resolution (if ambiguous, flag for user review)
    
    Returns: {extracted_name -> neo4j_id}
    """
    mappings = {}
    
    for char in extracted_chars:
        # Check for exact match
        existing = session.run("""
            MATCH (c:Character)
            WHERE toLower(c.name) = toLower($name)
            RETURN c.id AS id, c.name AS name
            LIMIT 1
        """, name=char.name).single()
        
        if existing:
            mappings[char.name] = existing["id"]
            continue
        
        # Check for fuzzy match
        all_chars = session.run("""
            MATCH (c:Character)
            RETURN c.id AS id, c.name AS name
        """).data()
        
        from difflib import SequenceMatcher
        best_match = None
        best_score = 0.0
        
        for existing_char in all_chars:
            score = SequenceMatcher(None, char.name.lower(), existing_char["name"].lower()).ratio()
            if score > best_score and score > 0.85:  # 85% similarity threshold
                best_score = score
                best_match = existing_char
        
        if best_match:
            mappings[char.name] = best_match["id"]
            continue
        
        # No match found - create new character
        new_id = _generate_entity_id("char")
        session.run("""
            CREATE (c:Character {
                id: $id,
                name: $name,
                description: $description,
                first_appearance: $chapter,
                traits: $traits
            })
        """, 
            id=new_id,
            name=char.name,
            description=char.description,
            chapter=current_chapter,
            traits=list(char.attributes.keys())
        )
        
        mappings[char.name] = new_id
    
    return mappings

def _create_relationships(
    session,
    extracted_rels: List[ExtractedRelationship],
    char_mappings: Dict[str, str],
    current_chapter: int
) -> None:
    """
    Create relationship edges in Neo4j.
    
    Validation:
    - Both source and target must exist in char_mappings
    - Relationship type must be valid (from predefined set or dynamically created)
    - Avoid duplicate relationships (merge if already exists)
    """
    for rel in extracted_rels:
        # Get Neo4j IDs
        source_id = char_mappings.get(rel.source_name)
        target_id = char_mappings.get(rel.target_name)
        
        if not source_id or not target_id:
            # Log warning: Relationship references unknown entity
            continue
        
        # Create or update relationship
        # Use MERGE to avoid duplicates
        session.run(f"""
            MATCH (c1:Character {{id: $source_id}})
            MATCH (c2:Character {{id: $target_id}})
            MERGE (c1)-[r:{rel.relationship_type}]->(c2)
            ON CREATE SET 
                r.description = $description,
                r.first_mentioned = $chapter,
                r.confidence = $confidence
            ON MATCH SET
                r.description = $description,
                r.last_mentioned = $chapter
        """,
            source_id=source_id,
            target_id=target_id,
            description=rel.description,
            chapter=current_chapter,
            confidence=rel.confidence
        )

def _generate_entity_id(prefix: str) -> str:
    """Generate unique entity ID"""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
```

### 3.4 Validation Node

```python
def validate_consistency(state: NarrativeState) -> NarrativeState:
    """
    Check generated content for contradictions against knowledge graph.
    
    Validation checks:
    1. Character trait consistency
    2. Relationship contradictions
    3. Event timeline violations
    4. World rule violations
    5. Plot stagnation detection
    """
    
    contradictions = []
    
    with state["neo4j_conn"].session() as session:
        # Check 1: Character trait consistency
        trait_contradictions = _check_character_traits(
            session,
            state["extracted_entities"]["characters"],
            state["current_chapter"]
        )
        contradictions.extend(trait_contradictions)
        
        # Check 2: Relationship contradictions
        rel_contradictions = _check_relationships(
            session,
            state["extracted_relationships"],
            state["current_chapter"]
        )
        contradictions.extend(rel_contradictions)
        
        # Check 3: Event timeline
        timeline_contradictions = _check_timeline(
            session,
            state["extracted_entities"]["events"],
            state["current_chapter"]
        )
        contradictions.extend(timeline_contradictions)
        
        # Check 4: World rules (custom rules per project)
        rule_contradictions = _check_world_rules(
            session,
            state["draft_text"],
            state["current_chapter"]
        )
        contradictions.extend(rule_contradictions)
    
    # Check 5: Plot advancement
    if _is_plot_stagnant(state):
        contradictions.append(Contradiction(
            type="plot_stagnation",
            description="Chapter does not significantly advance plot",
            conflicting_chapters=[state["current_chapter"]],
            severity="major",
            suggested_fix="Introduce conflict, decision, or revelation"
        ))
    
    # Severity-based decision
    critical_issues = [c for c in contradictions if c.severity == "critical"]
    major_issues = [c for c in contradictions if c.severity == "major"]
    
    needs_revision = len(critical_issues) > 0 or len(major_issues) > 2
    
    return {
        **state,
        "contradictions": contradictions,
        "needs_revision": needs_revision and not state.get("force_continue", False),
        "current_node": "validate"
    }

def _check_character_traits(session, extracted_chars: List[ExtractedEntity], current_chapter: int) -> List[Contradiction]:
    """
    Compare extracted character attributes with established traits.
    """
    contradictions = []
    
    for char in extracted_chars:
        # Get established traits from Neo4j
        existing = session.run("""
            MATCH (c:Character {name: $name})
            RETURN c.traits AS traits, c.first_appearance AS first_chapter
        """, name=char.name).single()
        
        if not existing:
            continue  # New character, no contradiction possible
        
        established_traits = set(existing["traits"] or [])
        new_attributes = set(char.attributes.keys())
        
        # Check for direct contradictions (e.g., "introverted" vs "extroverted")
        contradictory_pairs = [
            ("introverted", "extroverted"),
            ("brave", "cowardly"),
            ("honest", "deceitful"),
            # Add more as needed
        ]
        
        for trait_a, trait_b in contradictory_pairs:
            if trait_a in established_traits and trait_b in new_attributes:
                contradictions.append(Contradiction(
                    type="character_trait",
                    description=f"{char.name} was established as '{trait_a}' in chapter {existing['first_chapter']}, but is now described as '{trait_b}'",
                    conflicting_chapters=[existing["first_chapter"], current_chapter],
                    severity="major",
                    suggested_fix=f"Remove '{trait_b}' or explain character development"
                ))
    
    return contradictions

def _check_relationships(session, extracted_rels: List[ExtractedRelationship], current_chapter: int) -> List[Contradiction]:
    """
    Check for relationship contradictions.
    """
    contradictions = []
    
    for rel in extracted_rels:
        # Check if opposite relationship already exists
        existing = session.run("""
            MATCH (c1:Character {name: $source})-[r]->(c2:Character {name: $target})
            WHERE type(r) <> $rel_type
            RETURN type(r) AS existing_type, r.first_mentioned AS first_chapter
        """, 
            source=rel.source_name,
            target=rel.target_name,
            rel_type=rel.relationship_type
        ).single()
        
        if existing:
            # Check for contradictory relationships
            contradictory = _are_relationships_contradictory(
                existing["existing_type"],
                rel.relationship_type
            )
            
            if contradictory:
                contradictions.append(Contradiction(
                    type="relationship",
                    description=f"{rel.source_name} and {rel.target_name} had relationship '{existing['existing_type']}' in chapter {existing['first_chapter']}, now '{rel.relationship_type}'",
                    conflicting_chapters=[existing["first_chapter"], current_chapter],
                    severity="major",
                    suggested_fix="Clarify relationship evolution or remove contradiction"
                ))
    
    return contradictions

def _are_relationships_contradictory(rel_a: str, rel_b: str) -> bool:
    """Check if two relationship types contradict"""
    contradictions = [
        ("LOVES", "HATES"),
        ("TRUSTS", "DISTRUSTS"),
        ("ALLIES_WITH", "ENEMIES_WITH"),
        ("PROTECTS", "THREATENS"),
    ]
    
    for a, b in contradictions:
        if (rel_a == a and rel_b == b) or (rel_a == b and rel_b == a):
            return True
    
    return False

def _is_plot_stagnant(state: NarrativeState) -> bool:
    """
    Detect if plot is not advancing.
    
    Heuristics:
    - Chapter word count too low
    - No new events extracted
    - No new relationships
    - Repetitive prose (n-gram analysis)
    """
    # Check word count
    if state["draft_word_count"] < 1500:  # Suspiciously short
        return True
    
    # Check if events extracted
    if len(state["extracted_entities"].get("events", [])) == 0:
        return True
    
    # Check if relationships changed
    if len(state["extracted_relationships"]) == 0:
        return True
    
    return False
```

### 3.5 Revision Node

```python
def revise_chapter(state: NarrativeState) -> NarrativeState:
    """
    Revise chapter based on validation feedback.
    
    Strategy:
    1. Construct revision prompt with specific contradictions
    2. Use revision model (potentially better than generation model)
    3. Iterative refinement (up to N attempts)
    4. User escalation if revision fails
    """
    
    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        # Escalate to user
        return {
            **state,
            "needs_revision": False,  # Stop auto-revision
            "last_error": f"Max revision attempts ({state['max_iterations']}) reached. Manual intervention required.",
            "current_node": "revise_failed"
        }
    
    # Build revision prompt
    revision_prompt = _construct_revision_prompt(
        original_text=state["draft_text"],
        contradictions=state["contradictions"],
        context={
            "chapter": state["current_chapter"],
            "characters": state["active_characters"],
            "outline": state["outline"][state["current_chapter"]]
        }
    )
    
    # Call revision model (can be different/better than generation model)
    revised_text = _call_llm(
        prompt=revision_prompt,
        model=state["revision_model"],
        max_tokens=4096,
        temperature=0.5,  # Lower temperature for revisions
        stop_sequences=["### END REVISION ###"]
    )
    
    # Track revision history
    revision_history = state.get("revision_history", []) + [
        RevisionRequest(
            reason=f"Addressed {len(state['contradictions'])} contradictions",
            contradictions=state["contradictions"],
            target_model=state["revision_model"],
            max_attempts=state["max_iterations"]
        )
    ]
    
    return {
        **state,
        "draft_text": revised_text,
        "draft_word_count": len(revised_text.split()),
        "iteration_count": state["iteration_count"] + 1,
        "revision_history": revision_history,
        "contradictions": [],  # Will be re-validated
        "current_node": "revise"
    }

def _construct_revision_prompt(
    original_text: str,
    contradictions: List[Contradiction],
    context: Dict
) -> str:
    """Build focused revision prompt"""
    
    issues = "\n".join([
        f"{i+1}. **{c.type.replace('_', ' ').title()}**: {c.description}"
        f"\n   Severity: {c.severity}"
        f"\n   Suggestion: {c.suggested_fix or 'Revise to maintain consistency'}"
        for i, c in enumerate(contradictions)
    ])
    
    prompt = f"""You are revising Chapter {context['chapter']} of a novel to address the following consistency issues:

## Issues to Fix:
{issues}

## Original Chapter Text:
{original_text}

## Scene Goal (from outline):
{context['outline']['scene_description']}

## Instructions for Revision:
1. Maintain the core events and plot beats of the original chapter
2. Fix each identified issue while preserving narrative flow
3. Keep character voices and personalities consistent
4. Do not introduce new contradictions
5. Maintain approximately the same chapter length (±20%)

Write the complete revised chapter now:
"""
    
    return prompt
```

### 3.6 Summarization Node

```python
def summarize_chapter(state: NarrativeState) -> NarrativeState:
    """
    Generate chapter summary for context in future chapters.
    
    Two types of summaries:
    1. Brief summary (1-2 sentences) - for long-context windows
    2. Detailed summary (3-5 sentences) - for recent chapters
    """
    
    # Generate brief summary
    brief_summary = _generate_summary(
        text=state["draft_text"],
        model=state["extraction_model"],  # Use fast model
        max_length=2  # sentences
    )
    
    # Store in Neo4j chapter node
    with state["neo4j_conn"].session() as session:
        session.run("""
            MATCH (ch:Chapter {number: $chapter_num})
            SET ch.summary = $summary
        """, chapter_num=state["current_chapter"], summary=brief_summary)
    
    # Update state with new summary
    previous_summaries = state.get("previous_chapter_summaries", [])[-4:]  # Keep last 5
    previous_summaries.append(brief_summary)
    
    return {
        **state,
        "previous_chapter_summaries": previous_summaries,
        "current_node": "summarize"
    }

def _generate_summary(text: str, model: str, max_length: int) -> str:
    """Generate extractive or abstractive summary"""
    prompt = f"""Summarize the following chapter in exactly {max_length} sentence{'s' if max_length > 1 else ''}.

Focus on:
- Key plot events
- Character actions and decisions
- Important revelations or conflicts

Be concise but preserve critical information.

Chapter text:
{text}

Summary:"""

    summary = _call_llm(
        prompt=prompt,
        model=model,
        max_tokens=256,
        temperature=0.3,
        stop_sequences=["\n\n"]
    )
    
    return summary.strip()
```

---

## 4. LangGraph Workflow Definition

### 4.1 Graph Structure

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_saga_graph(project_dir: str) -> StateGraph:
    """
    Construct the complete SAGA 2.0 workflow graph.
    
    Graph structure:
    
    START → generate → extract → commit → validate → [revise OR summarize]
                                                              ↓          ↓
                                                            extract    finalize
                                                              ↓          ↓
                                                            commit      next?
                                                              ↓          ↓
                                                           validate     END
    """
    
    # Initialize checkpointer
    checkpointer = SqliteSaver.from_conn_string(
        f"{project_dir}/.saga/checkpoints.db"
    )
    
    # Create graph
    workflow = StateGraph(NarrativeState)
    
    # Add nodes
    workflow.add_node("generate", generate_chapter)
    workflow.add_node("extract", extract_entities)
    workflow.add_node("commit", commit_to_graph)
    workflow.add_node("validate", validate_consistency)
    workflow.add_node("revise", revise_chapter)
    workflow.add_node("summarize", summarize_chapter)
    workflow.add_node("finalize", finalize_chapter)
    
    # Add edges
    workflow.add_edge("generate", "extract")
    workflow.add_edge("extract", "commit")
    workflow.add_edge("commit", "validate")
    
    # Conditional edge: revise or proceed
    workflow.add_conditional_edges(
        "validate",
        should_revise,
        {
            "revise": "revise",
            "proceed": "summarize"
        }
    )
    
    # Revision loop
    workflow.add_edge("revise", "extract")  # Re-extract after revision
    
    # Finalization
    workflow.add_edge("summarize", "finalize")
    
    # Next chapter or end
    workflow.add_conditional_edges(
        "finalize",
        should_continue,
        {
            "next_chapter": "generate",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("generate")
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)

def should_revise(state: NarrativeState) -> str:
    """Decide whether to revise or proceed"""
    if state["needs_revision"] and not state.get("force_continue", False):
        return "revise"
    return "proceed"

def should_continue(state: NarrativeState) -> str:
    """Decide whether to generate next chapter or end"""
    if state["current_chapter"] < state["total_chapters"]:
        return "next_chapter"
    return "end"

def finalize_chapter(state: NarrativeState) -> NarrativeState:
    """
    Persist chapter to disk and prepare for next chapter.
    """
    # Write chapter file
    _persist_chapter(state)
    
    # Update progress
    return {
        **state,
        "current_chapter": state["current_chapter"] + 1,
        "iteration_count": 0,  # Reset for next chapter
        "contradictions": [],
        "draft_text": None,
        "extracted_entities": {},
        "extracted_relationships": [],
        "current_node": "finalize"
    }
```

### 4.2 Running the Graph

```python
async def generate_novel(project_dir: str, config: Dict) -> None:
    """
    Main entry point for novel generation.
    """
    # Build graph
    graph = build_saga_graph(project_dir)
    
    # Initialize state
    initial_state = _initialize_state(project_dir, config)
    
    # Run graph
    async for event in graph.astream(initial_state):
        # Event contains updated state after each node execution
        _handle_event(event)
    
    print(f"Novel generation complete: {initial_state['title']}")

def _initialize_state(project_dir: str, config: Dict) -> NarrativeState:
    """Set up initial state from config"""
    from neo4j import GraphDatabase
    
    # Connect to Neo4j
    neo4j_conn = GraphDatabase.driver(
        config["neo4j_uri"],
        auth=(config["neo4j_user"], config["neo4j_password"])
    )
    
    # Generate outline (separate process, not shown here)
    outline = _generate_outline(config)
    
    return {
        "project_id": config["project_id"],
        "title": config["title"],
        "genre": config["genre"],
        "theme": config["theme"],
        "setting": config["setting"],
        "target_word_count": config["target_word_count"],
        "neo4j_conn": neo4j_conn,
        "current_chapter": 1,
        "total_chapters": len(outline),
        "current_act": 1,
        "outline": outline,
        "active_characters": [],
        "current_location": None,
        "previous_chapter_summaries": [],
        "key_events": [],
        "draft_text": None,
        "draft_word_count": 0,
        "extracted_entities": {},
        "extracted_relationships": [],
        "contradictions": [],
        "needs_revision": False,
        "revision_history": [],
        "coherence_score": None,
        "prose_quality_score": None,
        "plot_advancement_score": None,
        "generation_model": config.get("generation_model", "qwen3-80b-q4"),
        "extraction_model": config.get("extraction_model", "qwen-coder-32b-q4"),
        "revision_model": config.get("revision_model", "qwen3-80b-q4"),
        "current_node": "start",
        "iteration_count": 0,
        "max_iterations": 3,
        "force_continue": False,
        "last_error": None,
        "retry_count": 0,
        "project_dir": project_dir,
        "chapters_dir": f"{project_dir}/chapters",
        "summaries_dir": f"{project_dir}/summaries"
    }

def _handle_event(event: Dict) -> None:
    """Handle graph events (for UI updates, logging)"""
    state = event.get("state", {})
    node = state.get("current_node", "unknown")
    chapter = state.get("current_chapter", 0)
    
    print(f"[Chapter {chapter}] {node.upper()}: Processing...")
    
    if node == "validate" and state.get("contradictions"):
        print(f"  ⚠️  Found {len(state['contradictions'])} issues")
    
    if node == "revise":
        print(f"  🔄 Revision attempt {state['iteration_count']}/{state['max_iterations']}")
```

---

## 5. Neo4j Schema

### 5.1 Node Types

```cypher
// Character
CREATE CONSTRAINT character_id IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE;

(:Character {
    id: "char_abc123",
    name: "Aria",
    description: "A young programmer discovering her AI origins",
    first_appearance: 1,
    traits: ["introverted", "curious", "distrustful"],
    motivations: "Discover the truth about her creation"
})

// Location
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE;

(:Location {
    id: "loc_xyz789",
    name: "Seattle Public Library Ruins",
    description: "Abandoned library, now home to rogue AI Nexus",
    rules: "Technology still functions; AI presence strong"
})

// Event
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;

(:Event {
    id: "evt_def456",
    description: "Aria discovers her memories are implanted",
    importance: 0.95,
    chapter: 3,
    timestamp: "2147-03-14"
})

// Chapter
CREATE CONSTRAINT chapter_number IF NOT EXISTS FOR (ch:Chapter) REQUIRE ch.number IS UNIQUE;

(:Chapter {
    number: 5,
    word_count: 3842,
    summary: "Aria meets Nexus and learns the truth about her origin",
    generated_at: datetime("2025-04-05T12:30:00Z")
})

// Object
(:Object {
    id: "obj_ghi789",
    name: "Hacked Drone",
    description: "Modified delivery drone used for escape",
    first_appearance: 2
})
```

### 5.2 Relationship Types

```cypher
// Character relationships
(c1:Character)-[:LOVES]->(c2:Character {description: "...", first_mentioned: 3})
(c1:Character)-[:HATES]->(c2:Character)
(c1:Character)-[:TRUSTS]->(c2:Character)
(c1:Character)-[:WORKS_FOR]->(c2:Character)
(c1:Character)-[:BETRAYED]->(c2:Character {when: 12})
(c1:Character)-[:FEARS]->(c2:Character)
(c1:Character)-[:ALLIES_WITH]->(c2:Character)
(c1:Character)-[:PARENT_OF]->(c2:Character)

// Character-Location
(c:Character)-[:LIVES_IN]->(l:Location)
(c:Character)-[:VISITED]->(l:Location {chapter: 7})

// Character-Event
(c:Character)-[:PARTICIPATED_IN]->(e:Event)
(c:Character)-[:CAUSED]->(e:Event)
(c:Character)-[:WITNESSED]->(e:Event)

// Event-Chapter
(e:Event)-[:OCCURRED_IN]->(ch:Chapter)

// Character-Object
(c:Character)-[:OWNS]->(o:Object)
(c:Character)-[:USED]->(o:Object {chapter: 4})

// Chapter sequence
(ch1:Chapter)-[:FOLLOWED_BY]->(ch2:Chapter)
```

### 5.3 Critical Queries

```cypher
// Get context for chapter generation
MATCH (c:Character)
WHERE c.id IN $active_char_ids
OPTIONAL MATCH (c)-[r]-(other:Character)
WHERE other.id IN $active_char_ids
OPTIONAL MATCH (l:Location {id: $location_id})
MATCH (ch:Chapter)
WHERE ch.number >= $start_chapter AND ch.number < $current_chapter
OPTIONAL MATCH (e:Event)-[:OCCURRED_IN]->(ch)
RETURN c, r, other, l, ch.summary AS summary, e
ORDER BY ch.number DESC
LIMIT $lookback

// Find character by fuzzy name match
MATCH (c:Character)
WHERE toLower(c.name) =~ toLower($pattern)
RETURN c.id, c.name, c.description

// Check for contradictory relationships
MATCH (c1:Character {name: $char1})-[r1]->(c2:Character {name: $char2})
WHERE type(r1) IN $contradictory_types
RETURN type(r1), r1.first_mentioned

// Detect isolated characters (no relationships)
MATCH (c:Character)
WHERE NOT (c)-[]-()
AND c.first_appearance < $current_chapter - 5
RETURN c.name, c.first_appearance

// Timeline validation
MATCH (e:Event)-[:OCCURRED_IN]->(ch:Chapter)
WHERE e.timestamp IS NOT NULL
WITH e.timestamp AS time, e.description AS desc, ch.number AS chapter
ORDER BY chapter, time
RETURN chapter, time, desc
```

---

## 6. File System Structure

```
/my-novel/
├── saga.yaml                  # Project config
├── .saga/
│   ├── checkpoints.db         # LangGraph state (SQLite)
│   ├── neo4j.conf             # Neo4j connection config
│   └── generation.log         # Detailed logs
├── outline/
│   ├── structure.yaml         # Act/scene structure
│   └── beats.yaml             # Key plot beats
├── chapters/
│   ├── chapter_001.md
│   ├── chapter_002.md
│   └── ...
├── summaries/
│   ├── chapter_001.txt
│   └── ...
├── characters/
│   ├── protagonist.yaml       # Optional: user-defined overrides
│   └── antagonist.yaml
├── world/
│   ├── rules.yaml             # World-building constraints
│   └── history.yaml
└── exports/
    └── novel_full.md          # Complete manuscript
```

---

## 7. Error Handling & Recovery

### 7.1 Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| LLM timeout | HTTP timeout exception | Retry once, then log and pause |
| Malformed JSON extraction | JSON parse error | Use regex fallback or skip entity |
| Neo4j connection lost | Driver exception | Reconnect with exponential backoff |
| Deduplication failure | No match found for high-confidence entity | Create new entity (conservative) |
| Validation deadlock | `iteration_count >= max_iterations` | Escalate to user for manual review |
| Context overflow | Token count > window size | Truncate oldest summaries |
| Disk full | OS error on file write | Warn user, pause generation |

### 7.2 LangGraph Checkpointing

```python
# Automatic state persistence
# If process crashes, restart from last checkpoint:

graph = build_saga_graph(project_dir)
config = {"configurable": {"thread_id": "novel_generation_001"}}

# Resume from checkpoint
result = graph.invoke(None, config=config)  # Continues from last node
```

---

## 8. Performance Optimizations

### 8.1 Parallel Execution

```python
# LangGraph automatically parallelizes independent nodes
# Example: Entity extraction runs 5 parallel LLM calls

# Before (sequential): 5 × 20s = 100s
# After (parallel):     1 × 25s = 25s (overhead for coordination)
```

### 8.2 Model Selection Strategy

| Task | Model | Rationale |
|------|-------|-----------|
| Generation | Qwen3-80B-Q4 | High quality, creative |
| Extraction | Qwen-Coder-32B-Q4 | Fast, good at structured output |
| Revision | Claude Sonnet (if available) or same as generation | Best quality for fixing issues |
| Summarization | Same as extraction | Fast, factual |

### 8.3 Context Window Budget

```python
# Target: 64K tokens
# Allocation:
#   - System prompt: 1K
#   - Chapter outline: 0.5K
#   - Character context: 2K
#   - Relationship context: 1K
#   - Recent summaries (5 chapters): 1.5K
#   - Key events (20): 1K
#   - Location/world rules: 1K
#   Total context overhead: ~8K tokens
#   Available for generation: 56K tokens
#   Target chapter length: 3-4K words = ~4-5K tokens
#   Safety margin: 51K tokens unused (for long generations)
```

---

## 9. User Interaction Patterns

### 9.1 CLI Commands

```bash
# Initialize project
saga init --title "The Last Compiler" --genre "sci-fi" --theme "AI rebellion"

# Generate outline
saga outline generate

# Start generation
saga generate start

# Pause/resume
saga generate pause
saga generate resume

# Review current chapter
saga chapter review

# Edit chapter
saga chapter edit 5

# Regenerate chapter
saga chapter regenerate 5

# Manual quality override
saga chapter accept 5 --force

# Export
saga export --format epub

# Inspect knowledge graph
saga graph query "MATCH (c:Character) RETURN c.name, c.traits"
saga graph visualize --chapter 10

# Debug
saga debug last-error
saga debug node-history
```

### 9.2 Web UI (Optional)

**Dashboard:**
- Progress: "Chapter 12/40 (30%)"
- Current node: "Validating..."
- Issues: "2 minor contradictions detected"

**Chapter View:**
- Side-by-side: Draft text | Issues panel
- Inline warnings: "⚠️ Character trait inconsistency"
- Edit button → inline editing → regenerate button

**Graph View:**
- Interactive Neo4j visualization
- Click character → see all relationships
- Filter by chapter range

---

## 10. Migration Path from SAGA 1.0

### 10.1 What to Port Directly

1. **Entity extraction logic** → Refactor into `extract_entities` node
2. **Deduplication algorithms** → Keep in `commit_to_graph` node
3. **Neo4j queries** → Reuse in context construction
4. **Relationship validation** → Move to `validate_consistency` node

### 10.2 What to Replace

1. **Custom state management** → LangGraph state + checkpointer
2. **Workflow coordination code** → LangGraph graph definition
3. **Manual retry logic** → LangGraph conditional edges
4. **Progress tracking** → LangGraph event stream

### 10.3 What to Add

1. **Parallel execution** (new capability)
2. **Quality gates** (formalized)
3. **Revision loops** (structured)
4. **Visual debugging** (LangGraph viz)

### 10.4 Estimated Code Reduction

- **Before**: 31,000 lines (SAGA 1.0)
- **After**: ~8,000 lines (SAGA 2.0 core logic)
- **Reduction**: 74%

**Breakdown:**
- State management: 5,000 lines → 200 lines (LangGraph)
- Workflow coordination: 3,000 lines → 500 lines (graph definition)
- Retry/error handling: 2,000 lines → 300 lines (conditional edges)
- Checkpointing: 1,500 lines → 50 lines (SqliteSaver)
- Domain logic: 19,500 lines → 7,000 lines (refactored, but preserved)

---

## 11. Testing Strategy

### 11.1 Unit Tests

```python
# Test entity extraction
def test_character_extraction():
    text = "Aria entered the room, her eyes scanning for threats."
    chars = extract_characters(text, model="test-model")
    assert len(chars) == 1
    assert chars[0].name == "Aria"

# Test deduplication
def test_character_deduplication():
    session = MockNeo4jSession()
    session.add_character("Aria", "char_001")
    
    extracted = [ExtractedEntity(name="aria", type="character", ...)]
    mappings = _deduplicate_characters(session, extracted, chapter=2)
    
    assert mappings["aria"] == "char_001"  # Matched existing
```

### 11.2 Integration Tests

```python
# Test full generation flow
async def test_chapter_generation_flow():
    state = _initialize_test_state()
    graph = build_saga_graph("/tmp/test-project")
    
    result = await graph.ainvoke(state)
    
    assert result["current_chapter"] == 2  # Advanced to next
    assert result["draft_text"] is not None
    assert len(result["extracted_entities"]["characters"]) > 0
```

### 11.3 End-to-End Tests

```bash
# Generate 5-chapter short story
saga init --title "Test Story" --chapters 5
saga generate start --no-interactive
saga export --format md

# Verify
assert_file_exists chapters/chapter_001.md
assert_neo4j_has_characters 5+
assert_no_critical_contradictions
```

---

## 12. Deployment & Distribution

### 12.1 Installation

```bash
# Install SAGA 2.0
pip install saga-novel-gen

# Initialize Neo4j (Docker)
docker run -d \
  --name saga-neo4j \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.15

# Initialize project
saga init --title "My Novel"
```

### 12.2 Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"
langgraph = "^0.0.30"
neo4j = "^5.15"
pydantic = "^2.5"
httpx = "^0.26"
pyyaml = "^6.0"
```

### 12.3 System Requirements

- **Python**: 3.11+
- **Neo4j**: 5.x (local or Docker)
- **LLM Server**: Ollama/vLLM/etc. (local)
- **GPU**: 12-24GB VRAM (for 7B-14B Q4 models)
- **RAM**: 32GB+ recommended
- **Disk**: 10GB+ for project + Neo4j

---

## 13. Roadmap & Future Enhancements

### 13.1 Phase 1 (MVP)

- [x] LangGraph integration
- [x] Neo4j knowledge graph
- [x] Basic generation pipeline
- [x] Entity extraction & deduplication
- [x] Validation & revision loops

### 13.2 Phase 2 (Quality)

- [ ] Quality scoring models (prose, coherence, plot)
- [ ] Multi-model voting (generate 3 variants, pick best)
- [ ] Advanced contradiction detection (LLM-based semantic checks)
- [ ] Character arc tracking

### 13.3 Phase 3 (Features)

- [ ] Interactive outline editor
- [ ] Real-time graph visualization during generation
- [ ] Export to EPUB/MOBI with metadata
- [ ] Style transfer (write in the style of author X)

### 13.4 Phase 4 (Polish)

- [ ] Web UI with HTMX
- [ ] Cloud backup (optional, encrypted)
- [ ] Collaborative editing (multi-user mode)
- [ ] Plugin system for custom validators

---

## 14. Conclusion

SAGA 2.0 is not a rewrite from scratch - it's a strategic refactoring that:

1. **Reduces complexity** by delegating orchestration to LangGraph
2. **Maintains sophistication** by keeping your Neo4j knowledge graph
3. **Improves modularity** through node-based architecture
4. **Enables parallelism** for faster generation
5. **Provides visibility** via graph visualization

**Key Insight:** You don't need to throw away 31K lines. You need to *reorganize* them into a more maintainable structure. LangGraph gives you that structure.

**Next Steps:**

1. Port entity extraction logic → `extract_entities` node
2. Port deduplication logic → `commit_to_graph` node
3. Define graph structure → `build_saga_graph()`
4. Test with 1-chapter generation
5. Iterate on validation and revision nodes
6. Scale to full novel

This is the architecture that actually solves your problems: too much custom code, complex state management, hard to debug. LangGraph handles the plumbing. You handle the domain logic. Neo4j handles the memory.

**SAGA 2.0: The novel generation system you deserve.**