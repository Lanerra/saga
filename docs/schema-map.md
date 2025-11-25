# SAGA Schema Map

This document provides an exhaustive overview of the **data schemas** used by **SAGA** (Semantic and Graph‑enhanced Authoring) in its current LangGraph‑based architecture.  It explains how configuration values, Pydantic models, state structures and Neo4j metadata interrelate.  The goal of this rewrite is to remove outdated material and align the schema description with the latest codebase.

## Table of Contents

1. [Configuration Schema (`SagaSettings`)](#configuration-schema-sagasettings)
2. [Core Data Models](#core-data-models)
   1. [Entity and Relationship Extraction Models](#entity-and-relationship-extraction-models)
   2. [Contradictions and Revision Models](#contradictions-and-revision-models)
   3. [Narrative Planning Models](#narrative-planning-models)
   4. [User Input Models](#user-input-models)
3. [NarrativeState TypedDict](#narrativestate-typeddict)
4. [Knowledge Graph Schema](#knowledge-graph-schema)
   1. [Node Labels](#node-labels)
   2. [Relationship Types](#relationship-types)
   3. [Property Names, Constraints and Indexes](#property-names-constraints-and-indexes)
5. [User Story YAML Schema](#user-story-yaml-schema)

---

## Configuration Schema (`SagaSettings`)

Configuration values are defined in `config/settings.py` via a Pydantic `BaseSettings` subclass.  These settings control API endpoints, model names, temperature values, token budgets, caching sizes and other runtime parameters.  Below is a categorized summary of the most important fields (defaults are shown where relevant):

### API and Model Configuration

- **Embedding API** – The base URL and key used for the embedding model (`EMBEDDING_API_BASE`, `EMBEDDING_API_KEY`).
- **OpenAI‑compatible API** – Base URL and key for calling an OpenAI‑compatible inference service (`OPENAI_API_BASE`, `OPENAI_API_KEY`).
- **Model names** – Default names for the large, medium, small and narrative models (`LARGE_MODEL`, `MEDIUM_MODEL`, `SMALL_MODEL`, `NARRATIVE_MODEL`).

### Neo4j Connection and Vector Index

- **Connection parameters** – URI, username, password and optional database name for connecting to Neo4j (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`).
- **Vector index configuration** – Name, dimension and similarity function for the chapter embedding vector index (`NEO4J_VECTOR_INDEX_NAME`, `NEO4J_VECTOR_DIMENSIONS`, `NEO4J_VECTOR_SIMILARITY_FUNCTION`).

### Temperature and Sampling Parameters

The system uses different temperature settings for various tasks.  Important parameters include:

- `TEMPERATURE_INITIAL_SETUP`, `TEMPERATURE_DRAFTING`, `TEMPERATURE_REVISION`, `TEMPERATURE_PLANNING`, `TEMPERATURE_EVALUATION`, `TEMPERATURE_CONSISTENCY_CHECK`, `TEMPERATURE_KG_EXTRACTION`, `TEMPERATURE_SUMMARY` and `TEMPERATURE_PATCH`.  Each controls the randomness of the LLM for its respective phase.
- `LLM_TOP_P` specifies nucleus sampling during drafting.
- Frequency and presence penalties for drafting and extraction adjust repetition (`FREQUENCY_PENALTY_*`, `PRESENCE_PENALTY_*`).

### Resource Budgets and Concurrency

- **Token budgets** – Maximum context tokens, generation tokens and planning tokens (`MAX_CONTEXT_TOKENS`, `MAX_GENERATION_TOKENS`, `MAX_PLANNING_TOKENS`).  These control how much text can be processed or generated in each step.
- **Concurrency** – Maximum concurrent LLM calls (`MAX_CONCURRENT_LLM_CALLS`) and maximum concurrent chapters per run (`MAX_CONCURRENT_CHAPTERS`).

### Caching and Deduplication

- **Caches** – Sizes for the embedding, summary, KG triple extraction and tokenizer caches (`EMBEDDING_CACHE_SIZE`, `SUMMARY_CACHE_SIZE`, `KG_TRIPLE_EXTRACTION_CACHE_SIZE`, `TOKENIZER_CACHE_SIZE`).
- **Deduplication** – Flags controlling semantic deduplication, thresholds and minimum segment length (`DEDUPLICATION_USE_SEMANTIC`, `DEDUPLICATION_SEMANTIC_THRESHOLD`, `DEDUPLICATION_MIN_SEGMENT_LENGTH`).

### Output and Miscellaneous Settings

- **Directories and files** – Base output directory and file names for plot outlines, character profiles, world building and logs (`BASE_OUTPUT_DIR`, `PLOT_OUTLINE_FILE`, `CHARACTER_PROFILES_FILE`, `WORLD_BUILDER_FILE`, `CHAPTERS_DIR`, `CHAPTER_LOGS_DIR`).
- **Revision limits and summary tokens** – Maximum revision cycles per chapter and maximum summary or KG triple tokens (`MAX_REVISION_CYCLES_PER_CHAPTER`, `MAX_SUMMARY_TOKENS`, `MAX_KG_TRIPLE_TOKENS`, `MAX_PREPOP_KG_TOKENS`).
- **Entity mention thresholds** – Proper nouns are kept with low mention frequency, while common nouns require more mentions before being persisted (`ENTITY_MENTION_THRESHOLD_PROPER_NOUN`, `ENTITY_MENTION_THRESHOLD_COMMON_NOUN`).
- **Logging and UI** – Log level, file names and Rich progress toggles (`LOG_LEVEL_STR`, `LOG_FILE`, `ENABLE_RICH_PROGRESS`, `SIMPLE_LOGGING_MODE`).
- **Bootstrap settings** – Minimum number of traits required when bootstrapping protagonists, antagonists and supporting characters (`BOOTSTRAP_MIN_TRAITS_PROTAGONIST`, `BOOTSTRAP_MIN_TRAITS_ANTAGONIST`, `BOOTSTRAP_MIN_TRAITS_SUPPORTING`).

These configuration values can be overridden via environment variables thanks to Pydantic's `BaseSettings` mechanism.  The configuration file also contains dynamic logic to reduce budgets when the `FAST_PROFILE` flag is set.

---

## Core Data Models

SAGA defines several Pydantic models and `TypedDict`s to structure data as it flows through the LangGraph workflow.  The models fall into four broad categories: extraction models, contradiction/revision models, planning models and user input models.

### Entity and Relationship Extraction Models

These models represent the structured output of the extraction subgraph.  They live in `core/langgraph/state.py` and are used by the extraction and commit nodes.

- **`ExtractedEntity`** – Represents an entity identified in generated text.  Fields include `name`, `type` (one of `"character"`, `"location"`, `"event"` or `"object"`), a natural‑language `description`, the `first_appearance_chapter` and a dictionary of arbitrary `attributes`.
- **`ExtractedRelationship`** – Represents a relationship between two entities.  It stores the `source_name` and `target_name`, the `relationship_type`, a `description`, the chapter where it was found and a confidence score.

### Contradictions and Revision Models

Validation nodes use specialized models to capture narrative inconsistencies and provide revision instructions.

- **`Contradiction`** – Encapsulates a detected inconsistency.  It has a `type`, human‑readable `description`, the `conflicting_chapters` and a severity (`"minor"`, `"major"` or `"critical"`).
- **`ProblemDetail`** – A `TypedDict` describing a single problem found during evaluation.  It includes the category, a description, a quote from the draft and character offsets to localize the issue.
- **`EvaluationResult`** – A `TypedDict` summarizing evaluator output.  Fields include `needs_revision` (boolean), a list of `reasons` and a list of `problems_found`.
- **`PatchInstruction`** – A `TypedDict` specifying how to modify text.  It contains the original problematic quote, target character ranges and the replacement string.

### Narrative Planning Models

The generation subgraph relies on planning structures defined in `models/agent_models.py`.

- **`SceneDetail`** – A `TypedDict` that plans a single scene.  It captures the scene number, summary, characters involved, key dialogue points, setting details, focus elements and metadata such as scene type, pacing and character arcs.
- **`ContextSnapshot`** – Defined in `models/narrative_state.py`, this class stores a frozen snapshot of context for retrieval: the chapter number, plot point focus, chapter plan (list of `SceneDetail` objects), hybrid context string, KG facts block and a map of recent chapters.

### User Input Models

User input is serialized via the YAML file `user_story_elements.yaml` and validated through models in `models/user_input_models.py`.

- **`NovelConceptModel`** – Contains high‑level novel metadata (`title`, `genre`, `setting`, `theme`).
- **`ProtagonistModel`** – Describes a main character with optional `description`, `traits`, `motivation`, `role` and a mapping of named relationships.
- **`CharacterGroupModel`** – Bundles protagonist, antagonist and supporting characters.  It is used when multiple characters are defined together.
- **`SettingModel`** – Captures a primary setting overview and a list of key locations (`KeyLocationModel` holds `name`, `description` and `atmosphere`).
- **`PlotElementsModel`** – Provides the inciting incident, list of plot points, central conflict and stakes.
- **`UserStoryInputModel`** – The top level structure combining all the above sections along with arbitrary `world_details`, `other_key_characters` and `style_and_tone` preferences.  Extra fields are allowed via the Pydantic config; this model thus functions as a flexible container for user‑provided narrative seeds.

---

## NarrativeState TypedDict

`NarrativeState` is a `TypedDict` defined in `core/langgraph/state.py` and forms the backbone of the LangGraph workflow.  It holds all dynamic data for a project and is automatically persisted to SQLite after each node executes.  The state is deliberately broad to support initialization, generation, extraction, validation, revision and healing.  For clarity, the fields are grouped below by function.  The field names and types are drawn directly from the code.

### Project Metadata and Position

- `project_id`, `title`, `genre`, `theme`, `setting`, `target_word_count` – Immutable descriptors of the novel.
- `current_chapter`, `total_chapters`, `current_act` – Track where the workflow is in the act/chapter hierarchy.

### Connection and Outline

- `neo4j_conn` – A handle to the Neo4j driver; reconstructed on load and not persisted.
- `plot_outline` – A deprecated dictionary of chapter outlines kept for backwards compatibility; canonical outlines live in `chapter_outlines`.

### Context for Prompt Construction

- `active_characters` – List of `CharacterProfile` objects relevant to the current scene.
- `current_location` – A location description for the scene.
- `previous_chapter_summaries` – Summaries of the last few chapters used for context.
- `key_events` – A list of important events from the KG to remind the model of plot obligations.

### Generated Content

- `draft_text` – The full draft for the current chapter (may be `None` initially).
- `draft_word_count` – Word count of the draft so far.
- `generated_embedding` – Embedding vector for the chapter used in similarity search.

### Extraction Results

- `extracted_entities` – Map from entity type to a list of `ExtractedEntity` objects.
- `extracted_relationships` – List of `ExtractedRelationship` objects.
- `character_updates`, `location_updates`, `event_updates`, `relationship_updates` – Temporary lists used by parallel extraction processes.

### Validation and Quality Control

- `contradictions` – List of `Contradiction` objects flagged by the validation subgraph.
- `needs_revision` – Boolean indicating whether the chapter requires rewriting.
- `revision_feedback` – Textual feedback explaining what needs to change.
- `is_from_flawed_draft` – True if the current text stems from a flawed revision or deduplication pass.

### Quality Metrics

The evaluator computes several scores between 0.0 and 1.0:

- `coherence_score`, `prose_quality_score`, `plot_advancement_score`, `pacing_score`, `tone_consistency_score` – Quantify aspects of narrative quality.
- `quality_feedback` – Free‑form feedback summarizing strengths and weaknesses.

### Model Configuration

- `generation_model`, `extraction_model`, `revision_model` – Name of the model used at each stage.
- `large_model`, `medium_model`, `small_model`, `narrative_model` – Tiered model names, mirroring fields in `SagaSettings`.

### Workflow Control and Error Handling

- `current_node` – Name of the last node that updated the state.
- `iteration_count`, `max_iterations` – Counters to cap revision loops.
- `force_continue` – Bypasses validation failures when set.
- `last_error`, `has_fatal_error`, `error_node`, `retry_count` – Track the last error, whether it is fatal, which node it occurred in and how many retries have been attempted.

### Filesystem Paths and Context Management

- `project_dir` – Base directory for output.
- `chapters_dir`, `summaries_dir` – Directories where chapter drafts and summaries are written.
- `context_epoch`, `hybrid_context`, `kg_facts_block` – Counters and blocks that determine how much context is passed to the LLM.

### Chapter Planning and Revision State

- `chapter_plan` – List of `SceneDetail` dictionaries outlining each scene.
- `plot_point_focus` – Focus of the current scene from the outline.
- `current_scene_index` – Index of the scene currently being processed.
- `scene_drafts` – List of text segments for each scene.
- `evaluation_result` – A dictionary matching `EvaluationResult` used by the evaluator.
- `patch_instructions` – List of patch instructions to apply during revision.

### World Building, Characters and Initialization

- `world_items` – A list of `WorldItem` objects representing setting elements.
- `current_world_rules` – List of active world rules that must be respected.
- `protagonist_name`, `protagonist_profile` – Name and optional profile of the main character.
- `character_sheets` – Dictionary of character sheets generated during initialization.
- `global_outline`, `act_outlines`, `chapter_outlines` – High‑level narrative plans produced during initialization and planning.
- `initialization_complete`, `initialization_step` – Flags that control whether initialization should be skipped or resumed.

### Graph Healing Metrics

- `provisional_count`, `last_healing_chapter` – Track the number of provisional nodes and when healing last ran.
- `merge_candidates`, `pending_merges`, `auto_approved_merges` – Lists used to manage proposed merges.
- `healing_history` – Log of healing actions performed.
- `nodes_graduated`, `nodes_merged`, `nodes_enriched` – Counters summarizing healing outcomes.

---

## Knowledge Graph Schema

SAGA persists its world model in a Neo4j knowledge graph.  The schema uses a labelled property graph with a rich set of node labels, relationship types and indexed properties.  All constants below are defined in `models/kg_constants.py`.

### Node Labels

Node labels are grouped by category.  Each group expands the graph with semantic richness:

| Category | Examples |
|---|---|
| **Core** | `Entity` (base label), `NovelInfo`, `Chapter` |
| **Living Beings** | `Character`, `Person`, `Creature`, `Spirit`, `Deity` |
| **Objects & Items** | `Object`, `Artifact`, `Document`, `Item`, `Relic` |
| **Locations & Structures** | `Location`, `Structure`, `Region`, `Landmark`, `Territory`, `Path`, `Room`, `Settlement` |
| **Abstract Concepts** | `Concept`, `Law`, `Tradition`, `Language`, `Symbol`, `Story`, `Song`, `Dream`, `Memory`, `Emotion`, `Skill` |
| **Temporal** | `Event`, `Era`, `Timeline`, `DevelopmentEvent`, `WorldElaborationEvent`, `Season`, `Moment` |
| **Organizations** | `Faction`, `Organization`, `Role`, `Rank`, `Guild`, `House`, `Order`, `Council` |
| **Systems** | `System`, `Magic`, `Technology`, `Religion`, `Culture`, `Education`, `Government`, `Economy` |
| **Resources** | `Resource`, `Currency`, `Trade`, `Food`, `Material`, `Energy` |
| **Information** | `Lore`, `Knowledge`, `Secret`, `Rumor`, `News`, `Message`, `Signal`, `ValueNode`, `Record` |
| **Actions/Events** | `Action`, `Reaction`, `Change`, `Pattern` |
| **Physical/Sensory** | `Sound`, `Pollen` |
| **Intent/Purpose** | `Purpose`, `Goal`, `Outcome` |
| **Qualities** | `Trait`, `Attribute`, `Quality`, `Reputation`, `Status` |
| **Containers** | `WorldContainer`, `PlotPoint`, `Collection`, `Archive`, `Treasury`, `Library` |

### Relationship Types

The graph distinguishes many relationship categories.  For brevity only representative examples are shown; the full set is defined in `RELATIONSHIP_TYPES` in `models/kg_constants.py`.

| Category | Examples |
|---|---|
| **Structural** | `HAS_PLOT_POINT`, `NEXT_PLOT_POINT`, `HAS_CHARACTER`, `CONTAINS_ELEMENT`, `DEVELOPED_IN_CHAPTER` |
| **Character Social** | `ALLY_OF`, `ENEMY_OF`, `FRIEND_OF`, `RIVAL_OF`, `FAMILY_OF`, `ROMANTIC_WITH`, `MENTOR_TO`, `WORKS_FOR` |
| **Character Emotional** | `LOVES`, `HATES`, `FEARS`, `RESPECTS`, `DESPISES`, `ENVIES`, `PITIES`, `OBSESSED_WITH` |
| **Plot Causal** | `CAUSES`, `PREVENTS`, `ENABLES`, `TRIGGERS`, `RESULTS_IN`, `DEPENDS_ON`, `CONFLICTS_WITH`, `SUPPORTS` |
| **Spatial/Temporal** | `LOCATED_AT`, `LOCATED_IN`, `NEAR`, `HAPPENS_BEFORE`, `HAPPENS_AFTER`, `OCCURS_DURING`, `TRAVELS_TO`, `OCCURRED_IN` |
| **Possession** | `OWNS`, `POSSESSES`, `CREATED_BY`, `INHERITED_FROM`, `STOLEN_FROM`, `FOUND_AT`, `LOST_AT` |
| **Organizational** | `MEMBER_OF`, `LEADER_OF`, `FOUNDED`, `BELONGS_TO`, `REPRESENTS`, `OPPOSES`, `ALLIED_WITH` |
| **Physical/Structural** | `PART_OF`, `CONTAINS`, `CONNECTED_TO`, `BUILT_BY`, `DESTROYED_BY`, `OWNED_BY` |
| **Thematic** | `SYMBOLIZES`, `REPRESENTS`, `CONTRASTS_WITH`, `PARALLELS`, `FORESHADOWS`, `ECHOES`, `EMBODIES` |
| **Ability/Trait** | `HAS_ABILITY`, `HAS_TRAIT`, `HAS_GOAL`, `HAS_RULE`, `HAS_KEY_ELEMENT`, `SKILLED_IN`, `WEAK_IN` |
| **Status/State** | `HAS_STATUS`, `IS_DEAD`, `IS_ALIVE`, `IS_MISSING`, `IS_INJURED`, `IS_ACTIVE` |
| **Information & Recording** | `RECORDS`, `PRESERVES`, `HAS_METADATA` |
| **Usage & Accessibility** | `ACCESSIBLE_BY`, `USED_IN`, `TARGETS` |
| **Communication & Display** | `DISPLAYS`, `SPOKEN_BY`, `EMITS` |
| **Operational** | `EMPLOYS`, `CONTROLS`, `REQUIRES` |
| **Enhanced Temporal/Association** | `REPLACED_BY`, `LINKED_TO`, `ASSOCIATED_WITH` |
| **Status Change & Special Action** | `WAS_REPLACED_BY`, `CHARACTERIZED_BY`, `IS_NOW`, `IS_NO_LONGER`, `WHISPERS`, `DEPRECATED` |

All relationships are normalized using the canonical mapping provided in `RELATIONSHIP_NORMALIZATIONS`.  For example, variations like `friends_with` or `befriends` are converted to `FRIEND_OF` to maintain consistency.

### Property Names, Constraints and Indexes

Core property names used on nodes and relationships include `chapter_added`, `created_chapter`, `chapter_updated` and `is_provisional`.  These mark when entities were created or updated and whether they are provisional (subject to merging).

---

## User Story YAML Schema

`user_story_elements.yaml` allows authors to seed SAGA with their own ideas.  The file must conform to the `UserStoryInputModel`.  At the top level the YAML contains the following keys:

- **`novel_concept`** – A dictionary matching `NovelConceptModel` with `title`, `genre`, `setting` and `theme`.
- **`protagonist` / `antagonist`** – Each describes a main character using `ProtagonistModel` fields.  Alternatively, the `characters` section may embed multiple protagonists and antagonists via `CharacterGroupModel`.
- **`characters`** – Groups protagonist, antagonist and a list of supporting characters.  Each character entry may include relationships to other characters.
- **`plot_elements`** – Contains the inciting incident, a list of plot points, the central conflict and the stakes.
- **`setting`** – Holds `primary_setting_overview` plus a list of `key_locations` where each entry specifies a `name`, optional `description` and `atmosphere`.
- **`world_details`** – A free‑form mapping of categories (e.g., factions, technologies) to named world items.  These are converted into `WorldItem` objects during processing.
- **`other_key_characters`** – A mapping from character names to additional `ProtagonistModel` descriptors.
- **`style_and_tone`** – Optional stylistic preferences such as narrative style, tone and pacing.

When the YAML is parsed, it is transformed into internal dictionaries: `plot_outline` (high‑level story plan), `characters` (a mapping of character names to `CharacterProfile`s) and `world_items` (a nested dictionary of `WorldItem`s).  The `user_story_to_objects` function in `models/user_input_models.py` performs this transformation by instantiating Pydantic models and creating `CharacterProfile` or `WorldItem` objects as needed.

---

### Conclusion

The SAGA system uses a layered schema to orchestrate long‑form narrative generation.  Configuration values govern the behaviour of models and hardware; extraction and planning models structure intermediate results; the `NarrativeState` aggregates all runtime data; and the knowledge graph provides a persistent world model.  Understanding these schemas is essential for extending or integrating SAGA with new workflows, ensuring that every piece of data has a well‑defined place in the system.