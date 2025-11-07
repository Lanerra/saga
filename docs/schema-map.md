# SAGA Schema Map

This document provides a comprehensive and detailed mapping of all schemas used by SAGA (Semantic And Graph-enhanced Authoring).

## Table of Contents
1. [Configuration Schema](#configuration-schema)
2. [Data Models (Pydantic)](#data-models-pydantic)
3. [Neo4j Knowledge Graph Schema](#neo4j-knowledge-graph-schema)
4. [User Input Schema](#user-input-schema)

---

## Configuration Schema

The configuration schema is defined in `config/settings.py` using Pydantic's BaseSettings.

### Main Configuration Class: `SagaSettings`

#### API and Model Configuration
- `EMBEDDING_API_BASE`: str - Base URL for embedding API (default: "http://127.0.0.1:11434")
- `EMBEDDING_API_KEY`: str - API key for embedding service (default: "")
- `OPENAI_API_BASE`: str - Base URL for OpenAI-compatible API (default: "http://127.0.0.1:8080/v1")
- `OPENAI_API_KEY`: str - API key for OpenAI-compatible service (default: "nope")
- `EMBEDDING_MODEL`: str - Name of embedding model (default: "mxbai-embed-large:latest")
- `EXPECTED_EMBEDDING_DIM`: int - Expected embedding dimensions (default: 1024)
- `EMBEDDING_DTYPE`: str - Data type for embeddings (default: "float16")

#### Neo4j Connection Settings
- `NEO4J_URI`: str - Neo4j connection URI (default: "bolt://localhost:7687")
- `NEO4J_USER`: str - Neo4j username (default: "neo4j")
- `NEO4J_PASSWORD`: str - Neo4j password (default: "saga_password")
- `NEO4J_DATABASE`: str | None - Database name (default: "neo4j")

#### Neo4j Vector Index Configuration
- `NEO4J_VECTOR_INDEX_NAME`: str - Vector index name (default: "chapterEmbeddings")
- `NEO4J_VECTOR_NODE_LABEL`: str - Vector index node label (default: "Chapter")
- `NEO4J_VECTOR_PROPERTY_NAME`: str - Vector property name (default: "embedding_vector")
- `NEO4J_VECTOR_DIMENSIONS`: int - Vector dimensions (default: 1024)
- `NEO4J_VECTOR_SIMILARITY_FUNCTION`: str - Similarity function (default: "cosine")

#### Base Model Definitions
- `LARGE_MODEL`: str - Large model name (default: "qwen3-a3b")
- `MEDIUM_MODEL`: str - Medium model name (default: "qwen3-a3b")
- `SMALL_MODEL`: str - Small model name (default: "qwen3-a3b")
- `NARRATIVE_MODEL`: str - Narrative model name (default: "qwen3-a3b")

#### Temperature Settings
- `TEMPERATURE_INITIAL_SETUP`: float - Initial setup temperature (default: 0.8)
- `TEMPERATURE_DRAFTING`: float - Drafting temperature (default: 0.8)
- `TEMPERATURE_REVISION`: float - Revision temperature (default: 0.65)
- `TEMPERATURE_PLANNING`: float - Planning temperature (default: 0.6)
- `TEMPERATURE_EVALUATION`: float - Evaluation temperature (default: 0.3)
- `TEMPERATURE_CONSISTENCY_CHECK`: float - Consistency check temperature (default: 0.2)
- `TEMPERATURE_KG_EXTRACTION`: float - KG extraction temperature (default: 0.4)
- `TEMPERATURE_SUMMARY`: float - Summary temperature (default: 0.5)
- `TEMPERATURE_PATCH`: float - Patch temperature (default: 0.7)

#### LLM Call Settings & Fallbacks
- `LLM_RETRY_ATTEMPTS`: int - Number of retry attempts (default: 3)
- `LLM_RETRY_DELAY_SECONDS`: float - Delay between retries (default: 3.0)
- `HTTPX_TIMEOUT`: float - HTTP timeout (default: 600.0)

#### Concurrency and Rate Limiting
- `MAX_CONCURRENT_LLM_CALLS`: int - Maximum concurrent LLM calls (default: 4)

#### Output and File Paths
- `BASE_OUTPUT_DIR`: str - Base output directory (default: "output")
- `PLOT_OUTLINE_FILE`: str - Plot outline file (default: "plot_outline.json")
- `CHARACTER_PROFILES_FILE`: str - Character profiles file (default: "character_profiles.json")
- `WORLD_BUILDER_FILE`: str - World building file (default: "world_building.json")
- `CHAPTERS_DIR`: str - Chapters directory (default: "chapters")
- `CHAPTER_LOGS_DIR`: str - Chapter logs directory (default: "chapter_logs")
- `DEBUG_OUTPUTS_DIR`: str - Debug outputs directory (default: "debug_outputs")
- `USER_STORY_ELEMENTS_FILE_PATH`: str - User story elements file (default: "user_story_elements.yaml")

#### Generation Parameters
- `MAX_CONTEXT_TOKENS`: int - Max context tokens (default: 40960)
- `MAX_GENERATION_TOKENS`: int - Max generation tokens (default: 16384)
- `CONTEXT_CHAPTER_COUNT`: int - Context chapter count (default: 2)
- `CHAPTERS_PER_RUN`: int - Chapters per run (default: 2)
- `KG_HEALING_INTERVAL`: int - KG healing interval (default: 2)
- `TARGET_PLOT_POINTS_INITIAL_GENERATION`: int - Target plot points for initial gen (default: 20)
- `MAX_CONCURRENT_CHAPTERS`: int - Max concurrent chapters (default: 1)

#### Caching
- `EMBEDDING_CACHE_SIZE`: int - Embedding cache size (default: 128)
- `SUMMARY_CACHE_SIZE`: int - Summary cache size (default: 32)
- `KG_TRIPLE_EXTRACTION_CACHE_SIZE`: int - KG triple extraction cache size (default: 16)
- `TOKENIZER_CACHE_SIZE`: int - Tokenizer cache size (default: 10)

#### Revision and Validation
- `MAX_REVISION_CYCLES_PER_CHAPTER`: int - Max revision cycles per chapter (default: 0)
- `MAX_SUMMARY_TOKENS`: int - Max summary tokens (default: 8192)
- `MIN_ACCEPTABLE_DRAFT_LENGTH`: int - Min acceptable draft length (default: 12000)

#### Logging & UI
- `LOG_LEVEL_STR`: str - Log level (default: "INFO")
- `ENABLE_RICH_PROGRESS`: bool - Enable rich progress (default: True)

---

## Data Models (Pydantic)

### Character Profile Model (`models/kg_models.py`)
```python
class CharacterProfile(BaseModel):
    name: str
    description: str = ""
    traits: list[str] = Field(default_factory=list)
    relationships: dict[str, Any] = Field(default_factory=dict)
    status: str = "Unknown"
    updates: dict[str, Any] = Field(default_factory=dict)
    created_chapter: int = 0
    is_provisional: bool = False
```

### World Item Model (`models/kg_models.py`)
```python
class WorldItem(BaseModel):
    id: str
    category: str
    name: str
    created_chapter: int = 0
    is_provisional: bool = False
    description: str = ""
    goals: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    key_elements: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
    additional_properties: dict[str, Any] = Field(default_factory=dict)
```

### Narrative State Models (`models/narrative_state.py`)

#### ContextSnapshot
```python
class ContextSnapshot:
    chapter_number: int
    plot_point_focus: str | None
    chapter_plan: list[SceneDetail] | None
    hybrid_context: str
    kg_facts_block: str
    recent_chapters_map: dict[int, dict[str, Any]]
```

#### NarrativeState
```python
class NarrativeState:
    neo4j: Any
    llm: Any
    embedding: Any | None
    plot_outline: dict[str, Any]
    context_epoch: int = 0
    snapshot: ContextSnapshot | None = None
    caches: dict[str, Any] = {}
    reads_locked: bool = False
```

### Agent Models (`models/agent_models.py`)

#### SceneDetail (TypedDict)
```python
class SceneDetail(TypedDict, total=False):
    scene_number: int
    summary: str
    characters_involved: list[str]
    key_dialogue_points: list[str]
    setting_details: str
    scene_focus_elements: list[str]
    contribution: str
    scene_type: str
    pacing: str
    character_arc_focus: str | None
    relationship_development: str | None
```

#### ProblemDetail (TypedDict)
```python
class ProblemDetail(TypedDict, total=False):
    issue_category: str
    problem_description: str
    quote_from_original_text: str
    quote_char_start: int | None
    quote_char_end: int | None
    sentence_char_start: int | None
    sentence_char_end: int | None
    suggested_fix_focus: str
```

#### EvaluationResult (TypedDict)
```python
class EvaluationResult(TypedDict, total=False):
    needs_revision: bool
    reasons: list[str]
    problems_found: list[ProblemDetail]
```

#### PatchInstruction (TypedDict)
```python
class PatchInstruction(TypedDict, total=False):
    original_problem_quote_text: str
    target_char_start: int | None
    target_char_end: int | None
    replace_with: str
    reason_for_change: str
```

### User Input Models (`models/user_input_models.py`)

#### NovelConceptModel
```python
class NovelConceptModel(BaseModel):
    title: str = Field(..., min_length=1)
    genre: str | None = None
    setting: str | None = None
    theme: str | None = None
```

#### ProtagonistModel
```python
class ProtagonistModel(BaseModel):
    name: str
    description: str | None = None
    traits: list[str] = Field(default_factory=list)
    motivation: str | None = None
    role: str | None = None
    relationships: dict[str, RelationshipModel] = Field(default_factory=dict)
```

#### SettingModel
```python
class SettingModel(BaseModel):
    primary_setting_overview: str | None = None
    key_locations: list[KeyLocationModel] = Field(default_factory=list)
```

#### KeyLocationModel
```python
class KeyLocationModel(BaseModel):
    name: str
    description: str | None = None
    atmosphere: str | None = None
```

#### PlotElementsModel
```python
class PlotElementsModel(BaseModel):
    inciting_incident: str | None = None
    plot_points: list[str] = Field(default_factory=list)
    central_conflict: str | None = None
    stakes: str | None = None
```

#### UserStoryInputModel
```python
class UserStoryInputModel(BaseModel):
    novel_concept: NovelConceptModel | None = None
    protagonist: ProtagonistModel | None = None
    antagonist: ProtagonistModel | None = None
    characters: CharacterGroupModel | None = None
    plot_elements: PlotElementsModel | None = None
    setting: SettingModel | None = None
    world_details: dict[str, Any] | None = None
    other_key_characters: dict[str, ProtagonistModel] | None = None
    style_and_tone: dict[str, Any] | None = None
```

---

## Neo4j Knowledge Graph Schema

### Node Labels (Defined in `models/kg_constants.py`)

#### Core Existing Types
- `Entity` - Base label for all nodes
- `NovelInfo` - Novel metadata
- `Chapter` - Chapter entities

#### Physical Entities
- Living Beings: `Character`, `Person`, `Creature`, `Spirit`, `Deity`
- Objects & Items: `Object`, `Artifact`, `Document`, `Item`, `Relic`
- Locations & Structures: `Location`, `Structure`, `Region`, `Landmark`, `Territory`, `Path`, `Room`, `Settlement`

#### Abstract Entities
- `Concept`, `Law`, `Tradition`, `Language`, `Symbol`, `Story`, `Song`, `Dream`, `Memory`, `Emotion`, `Skill`

#### Temporal Entities
- `Event`, `Era`, `Timeline`, `DevelopmentEvent`, `WorldElaborationEvent`, `Season`, `Moment`

#### Organizational Entities
- `Faction`, `Organization`, `Role`, `Rank`, `Guild`, `House`, `Order`, `Council`

#### System Entities
- `System`, `Magic`, `Technology`, `Religion`, `Culture`, `Education`, `Government`, `Economy`

#### Resource Entities
- `Resource`, `Currency`, `Trade`, `Food`, `Material`, `Energy`

#### Information Entities
- `Lore`, `Knowledge`, `Secret`, `Rumor`, `News`, `Message`, `Signal`, `ValueNode`, `Record`

#### Action/Event Entities
- `Action`, `Reaction`, `Change`, `Pattern`

#### Physical/Sensory Entities
- `Sound`, `Pollen`

#### Purpose/Intent Entities
- `Purpose`, `Goal`, `Outcome`

#### Relationship/Abstract
- `Relationship`

#### Quality Entities
- `Trait`, `Attribute`, `Quality`, `Reputation`, `Status`

#### Container Entities
- `WorldContainer`, `PlotPoint`, `Collection`, `Archive`, `Treasury`, `Library`

### Relationship Types (Defined in `models/kg_constants.py`)

#### Structural Relationships
- `HAS_PLOT_POINT`, `NEXT_PLOT_POINT`, `HAS_CHARACTER`, `HAS_WORLD_META`, `CONTAINS_ELEMENT`, `DEVELOPED_IN_CHAPTER`, `ELABORATED_INCHAPTER`, `IS_A`

#### Character Social Relationships
- `ALLY_OF`, `ENEMY_OF`, `FRIEND_OF`, `RIVAL_OF`, `FAMILY_OF`, `ROMANTIC_WITH`, `MENTOR_TO`, `STUDENT_OF`, `WORKS_FOR`, `LEADS`, `SERVES`, `KNOWS`, `TRUSTS`, `DISTRUSTS`, `OWES_DEBT_TO`

#### Character Emotional Relationships
- `LOVES`, `HATES`, `FEARS`, `RESPECTS`, `DESPISES`, `ENVIES`, `PITIES`, `OBSESSED_WITH`

#### Plot Causal Relationships
- `CAUSES`, `PREVENTS`, `ENABLES`, `TRIGGERS`, `RESULTS_IN`, `DEPENDS_ON`, `CONFLICTS_WITH`, `SUPPORTS`, `THREATENS`, `PROTECTS`

#### Spatial/Temporal Relationships
- `LOCATED_AT`, `LOCATED_IN`, `NEAR`, `ADJACENT_TO`, `OCCURS_DURING`, `HAPPENS_BEFORE`, `HAPPENS_AFTER`, `ORIGINATES_FROM`, `TRAVELS_TO`

#### Possession/Ownership Relationships
- `OWNS`, `POSSESSES`, `CREATED_BY`, `CREATES`, `INHERITED_FROM`, `STOLEN_FROM`, `GIVEN_BY`, `FOUND_AT`, `LOST_AT`

#### Organizational Relationships
- `MEMBER_OF`, `LEADER_OF`, `FOUNDED`, `BELONGS_TO`, `REPRESENTS`, `OPPOSES`, `ALLIED_WITH`

#### Physical/Structural Relationships
- `PART_OF`, `CONTAINS`, `CONNECTED_TO`, `BUILT_BY`, `DESTROYED_BY`, `DAMAGED_BY`, `REPAIRED_BY`, `OWNED_BY`

#### Abstract/Thematic Relationships
- `SYMBOLIZES`, `REPRESENTS`, `CONTRASTS_WITH`, `PARALLELS`, `FORESHADOWS`, `ECHOES`, `EMBODIES`

#### Ability/Trait Relationships
- `HAS_ABILITY`, `HAS_TRAIT`, `HAS_GOAL`, `HAS_RULE`, `HAS_KEY_ELEMENT`, `HAS_TRAIT_ASPECT`, `SKILLED_IN`, `WEAK_IN`

#### Status/State Relationships
- `HAS_STATUS`, `STATUS_IS`, `IS_DEAD`, `IS_ALIVE`, `IS_MISSING`, `IS_INJURED`, `IS_HEALTHY`, `IS_ACTIVE`, `IS_INACTIVE`

#### Information and Recording Relationships
- `RECORDS`, `PRESERVES`, `HAS_METADATA`

#### Usage and Accessibility Relationships
- `ACCESSIBLE_BY`, `USED_IN`, `TARGETS`

#### Communication and Display Relationships
- `DISPLAYS`, `SPOKEN_BY`, `EMITS`

#### Operational Relationships
- `EMPLOYS`, `CONTROLS`, `REQUIRES`

### Core Property Names (Defined in `models/kg_constants.py`)
- `KG_REL_CHAPTER_ADDED` - Chapter where relationship was added
- `KG_NODE_CREATED_CHAPTER` - Chapter where node was created  
- `KG_NODE_CHAPTER_UPDATED` - Chapter where node was last updated
- `KG_IS_PROVISIONAL` - Whether the node/relationship is provisional

### Neo4j Schema Constraints and Indexes (Defined in `core/db_manager.py`)
- `entity_id_unique` - Unique constraint on Entity.id
- `novelInfo_id_unique` - Unique constraint on NovelInfo.id
- `chapter_number_unique` - Unique constraint on Chapter.number
- `character_name_unique` - Unique constraint on Character.name
- `worldContainer_id_unique` - Unique constraint on WorldContainer.id
- `trait_name_unique` - Unique constraint on Trait.name
- `plotPoint_id_unique` - Unique constraint on PlotPoint.id
- `valueNode_value_type_unique` - Unique constraint on ValueNode.value and ValueNode.type
- `developmentEvent_id_unique` - Unique constraint on DevelopmentEvent.id
- `worldElaborationEvent_id_unique` - Unique constraint on WorldElaborationEvent.id

### Indexes
- `entity_name_property_idx` - Index on Entity.name
- `entity_is_provisional_idx` - Index on Entity.is_provisional
- `entity_is_deleted_idx` - Index on Entity.is_deleted
- `plotPoint_sequence` - Index on PlotPoint.sequence
- `developmentEvent_chapter_updated` - Index on DevelopmentEvent.chapter_updated
- `worldElaborationEvent_chapter_updated` - Index on WorldElaborationEvent.chapter_updated

### Vector Index
- `chapterEmbeddings` - Vector index for Chapter.embedding_vector

---

## User Input Schema

### User Story Elements YAML Format
The schema for `user_story_elements.yaml` is defined by the `UserStoryInputModel` Pydantic model.

#### Top Level Structure
- `novel_concept` - Novel concept information
- `protagonist` - Main protagonist details
- `antagonist` - Main antagonist details
- `characters` - Character groupings
- `plot_elements` - Plot-related elements
- `setting` - Setting information
- `world_details` - World-building details
- `other_key_characters` - Additional key characters
- `style_and_tone` - Style and tone specifications

#### Novel Concept Section
- `title` - Title of the novel
- `genre` - Genre of the novel
- `setting` - Setting description
- `theme` - Central theme

#### Protagonist Section
- `name` - Character name
- `description` - Character description
- `traits` - List of character traits
- `motivation` - Character motivation
- `role` - Character role
- `relationships` - Character relationships

#### Plot Elements Section
- `inciting_incident` - The inciting incident
- `plot_points` - List of key plot points
- `central_conflict` - Central conflict of the story
- `stakes` - Stakes of the story

#### Setting Section
- `primary_setting_overview` - Overview of the primary setting
- `key_locations` - List of key locations

#### World Details Section
- Arbitrary key-value pairs for world-building elements grouped by categories (e.g., factions, technology)

#### Style and Tone Section
- `narrative_style` - Narrative style specification
- `tone` - Tone specification
- `pacing` - Pacing specification
