# Narrative Structure Schema Integration with Incremental Build Approach

## Overview

This document provides a comprehensive mapping of how your narrative structure schema (Chapters, Acts, Scenes, and their relationships) can be integrated with SAGA's incremental build-as-it-goes approach.

## Current Schema Components

Based on your schema design and the existing codebase, the narrative structure includes:

### 1. Core Narrative Structure Nodes

#### Chapter Node (`:Chapter`)
- **Properties**: `id`, `number`, `title`, `summary`, `act_number`, `embedding`, `created_chapter`, `is_provisional`, `created_ts`, `updated_ts`
- **Purpose**: Represents story chapters

#### Scene Node (`:Scene`)
- **Properties**: `id`, `chapter_number`, `scene_index`, `title`, `pov_character`, `setting`, `plot_point`, `conflict`, `outcome`, `beats`, `created_chapter`, `is_provisional`, `created_ts`, `updated_ts`
- **Purpose**: Represents narrative scenes within chapters

#### Event Node (`:Event`)
- **Event Types**:
  - `MajorPlotPoint`: Story-level turning points (Inciting Incident, Midpoint, Climax, Resolution)
  - `ActKeyEvent`: Key events within acts
  - `SceneEvent`: Events within specific scenes
- **Properties**: Varies by event type
- **Purpose**: Represents plot events at different granularity levels

### 2. Relationships

#### Structural Relationships
- `Scene -[PART_OF]-> Chapter`: Scene belongs to Chapter
- `Scene -[FOLLOWS]-> Scene`: Sequential ordering within chapter
- `Event -[PART_OF]-> Event`: Hierarchical event containment (ActKeyEvent → MajorPlotPoint, SceneEvent → ActKeyEvent)
- `Event -[OCCURS_IN_SCENE]-> Scene`: SceneEvent occurs within Scene

#### Narrative Relationships
- `Event -[HAPPENS_BEFORE]-> Event`: Temporal ordering
- `Event -[INVOLVES]-> Character`: Character participation
- `Scene -[FEATURES_CHARACTER]-> Character`: Character presence in scene
- `Scene -[OCCURS_AT]-> Location`: Scene location

## Integration Strategy: Incremental Build Approach

### Phase 1: Initialization (Pre-Generation)

**Goal**: Establish the narrative foundation structure

#### 1.1 Character Sheets Generation
- **Node**: `generate_character_sheets`
- **Output**: Character profiles with personality, traits, and relationships
- **Graph Impact**: Creates `:Character` nodes with `created_chapter=0`

#### 1.2 Global Outline Generation
- **Node**: `generate_global_outline`
- **Output**: High-level story structure (3 or 5 acts, major plot points)
- **Graph Impact**: 
  - Creates `:Event` nodes for MajorPlotPoints
  - `event_type="MajorPlotPoint"`
  - `sequence_order` (1-4): Inciting Incident, Midpoint, Climax, Resolution
  - `created_chapter=0`

#### 1.3 Act Outlines Generation
- **Node**: `generate_act_outlines`
- **Output**: Detailed act structure with key events
- **Graph Impact**:
  - Creates `:Event` nodes for ActKeyEvents
  - `event_type="ActKeyEvent"`
  - Establishes `PART_OF` relationships: ActKeyEvent → MajorPlotPoint
  - `created_chapter=0`

#### 1.4 Chapter Allocation
- **Node**: `generate_all_chapter_outlines`
- **Output**: Chapter-to-act mapping
- **Graph Impact**: None (structural only, no nodes created)

#### 1.5 Initial Commit to Graph
- **Node**: `commit_initialization_to_graph`
- **Purpose**: Persist initialization artifacts to Neo4j
- **Graph Impact**:
  - Commits all initialization nodes (Characters, MajorPlotPoints, ActKeyEvents)
  - Sets `created_chapter=0` for all nodes
  - `is_provisional=false` (canonical nodes)

### Phase 2: Generation Loop (Incremental Build)

**Goal**: Build narrative structure incrementally as chapters are generated

#### 2.1 Chapter Outline Generation (Per Chapter)
- **Node**: `generate_chapter_outline`
- **Timing**: Runs at start of each chapter
- **Output**: Detailed scene-by-scene plan for the chapter
- **Graph Impact**:
  - Creates `:Chapter` node
    - `number` = current chapter number
    - `act_number` = determined from chapter-to-act mapping
    - `created_chapter` = chapter number
    - `is_provisional=true` initially
  - Creates `:Scene` nodes (1-3 per chapter typically)
    - `chapter_number` = current chapter
    - `scene_index` = 0, 1, 2, ...
    - `created_chapter` = chapter number
    - `is_provisional=true` initially
  - Creates `:Event` nodes for SceneEvents
    - `event_type="SceneEvent"`
    - `chapter_number` = current chapter
    - `scene_index` = scene position
    - `created_chapter` = chapter number
    - `is_provisional=true` initially

#### 2.2 Scene Generation
- **Node**: `draft_scene` (within generation subgraph)
- **Timing**: Per scene within chapter
- **Output**: Scene prose
- **Graph Impact**: None (content generation only)

#### 2.3 Context Retrieval
- **Node**: `retrieve_context`
- **Timing**: Before each scene generation
- **Purpose**: Build hybrid context from existing graph
- **Graph Usage**:
  - Queries `:Character` nodes for scene participants
  - Queries `:Event` nodes for relevant plot context
  - Queries `:Location` nodes for scene settings
  - Queries `:Chapter` nodes for act-level context

#### 2.4 Extraction and Consolidation
- **Node**: `extract_from_scenes` and `consolidate`
- **Timing**: After all scenes in chapter are drafted
- **Output**: Extracted entities and relationships
- **Graph Impact**: None (preparation for commit)

#### 2.5 Chapter Assembly
- **Node**: `assemble_chapter`
- **Timing**: After scene extraction
- **Output**: Complete chapter draft
- **Graph Impact**: None

#### 2.6 Relationship Normalization
- **Node**: `normalize_relationships`
- **Timing**: Before graph commit
- **Purpose**: Map extracted relationships to canonical types
- **Graph Impact**: None (preparation for commit)

#### 2.7 Commit to Graph
- **Node**: `commit_to_graph`
- **Timing**: After chapter validation
- **Purpose**: Persist chapter content and structure to Neo4j
- **Graph Impact**:
  - **Updates `:Chapter` node**:
    - Sets `is_provisional=false`
    - Adds `summary` from chapter summary
    - Adds `embedding` from chapter content
    - Updates `updated_ts`
  - **Updates `:Scene` nodes**:
    - Sets `is_provisional=false` for all scenes
    - Updates scene properties from extraction
    - Creates `PART_OF` relationships: Scene → Chapter
    - Creates `FOLLOWS` relationships: Scene → Scene (sequential)
    - Creates `FEATURES_CHARACTER` relationships: Scene → Character
    - Creates `OCCURS_AT` relationships: Scene → Location
  - **Updates `:Event` nodes (SceneEvents)**:
    - Sets `is_provisional=false`
    - Updates event properties
    - Creates `OCCURS_IN_SCENE` relationships: Event → Scene
    - Creates `PART_OF` relationships: Event → ActKeyEvent (where applicable)
    - Creates `HAPPENS_BEFORE` relationships for temporal ordering
    - Creates `INVOLVES` relationships: Event → Character
  - **Creates new entities**:
    - `:Location` nodes for new locations discovered
    - `:Item` nodes for new items mentioned
    - Additional `:Character` nodes for new characters

#### 2.8 Validation
- **Node**: `validate_consistency`, `evaluate_quality`, `detect_contradictions`
- **Timing**: After graph commit
- **Purpose**: Ensure narrative consistency
- **Graph Usage**:
  - Checks relationship evolution (e.g., prevents abrupt HATES → LOVES)
  - Validates character state consistency
  - Verifies temporal ordering of events

#### 2.9 Graph Healing
- **Node**: `heal_graph`
- **Timing**: After validation
- **Purpose**: Maintain graph integrity
- **Graph Impact**:
  - Merges duplicate entities
  - Resolves inconsistencies
  - Updates relationships based on new context

### Phase 3: Revision Loop (If Needed)

**Goal**: Iteratively refine chapters that need revision

#### 3.1 Revision
- **Node**: `revise_chapter`
- **Timing**: If validation detects issues
- **Output**: Revised chapter content
- **Graph Impact**: None initially

#### 3.2 Re-commit
- **Node**: `commit_to_graph` (re-run)
- **Purpose**: Update graph with revised content
- **Graph Impact**:
  - Updates existing nodes with revised information
  - Modifies relationships as needed
  - Updates `updated_ts` timestamps

## Schema Evolution Across Phases

### Stage 1: Initialization (created_chapter=0)
- `:Character` nodes with full profiles
- `:Event` nodes for MajorPlotPoints
- `:Event` nodes for ActKeyEvents
- Relationships between events (PART_OF, HAPPENS_BEFORE)

### Stage 2: Chapter Generation (created_chapter=N)
- `:Chapter` nodes (one per chapter)
- `:Scene` nodes (1-3 per chapter)
- `:Event` nodes for SceneEvents
- Relationships:
  - Scene → Chapter (PART_OF)
  - Scene → Scene (FOLLOWS)
  - Scene → Character (FEATURES_CHARACTER)
  - Scene → Location (OCCURS_AT)
  - Event → Scene (OCCURS_IN_SCENE)
  - Event → Event (PART_OF, HAPPENS_BEFORE)
  - Event → Character (INVOLVES)

### Stage 3: Enrichment (Ongoing)
- Additional `:Location` nodes from scene descriptions
- Additional `:Item` nodes from narrative details
- New `:Character` nodes for introduced characters
- Relationship updates based on narrative development

## Key Benefits of Incremental Approach

1. **Narrative Discovery**: Allows LLM to discover interesting elements during generation that wouldn't exist in a pre-built graph

2. **Flexibility**: Characters can develop, relationships can evolve naturally

3. **Validation**: Easier to detect contradictions when changes are incremental and traceable to specific chapters

4. **Graph Healing**: Maintenance works better with incremental updates than with a static pre-built structure

5. **Context Retrieval**: Scene-specific context queries work naturally with an evolving graph

## Implementation Mapping

### Existing Nodes That Support Your Schema

| Your Schema Element | SAGA Node | Integration Point |
|---------------------|-----------|-------------------|
| Chapter | `generate_chapter_outline` | Creates `:Chapter` node |
| Scene | `generate_chapter_outline` | Creates `:Scene` nodes |
| Act Structure | `generate_global_outline` + `generate_act_outlines` | Creates MajorPlotPoint and ActKeyEvent nodes |
| Scene Events | `commit_to_graph` | Creates SceneEvent nodes and relationships |
| Character Participation | `retrieve_context` + `commit_to_graph` | Creates INVOLVES relationships |
| Scene Sequencing | `commit_to_graph` | Creates FOLLOWS relationships |

### Graph Query Patterns

#### Query: Get Chapter Structure
```cypher
MATCH (c:Chapter {number: $chapter_number})<-[:PART_OF]-(s:Scene)
RETURN s.scene_index, s.title, s.pov_character
ORDER BY s.scene_index
```

#### Query: Get Act Structure
```cypher
MATCH (mpp:Event {event_type: "MajorPlotPoint", sequence_order: $act_number})
<-[:PART_OF]-(ake:Event {event_type: "ActKeyEvent"})
RETURN ake.sequence_in_act, ake.name, ake.description
ORDER BY ake.sequence_in_act
```

#### Query: Get Scene Events with Character Participation
```cypher
MATCH (s:Scene {chapter_number: $chapter, scene_index: $scene})<-[:OCCURS_IN_SCENE]-(e:Event)
-[r:INVOLVES]-(c:Character)
RETURN e.name, c.name, r.role
```

#### Query: Get Temporal Event Flow
```cypher
MATCH path = (e1:Event)-[:HAPPENS_BEFORE*0..10]-(e2:Event)
WHERE e1.chapter_number <= $chapter AND e2.chapter_number <= $chapter
RETURN path
```

## Migration Path

If you've already built some graph structure using the build-first approach:

1. **Audit Existing Nodes**: Identify what narrative structure already exists
2. **Preserve Canonical Nodes**: Keep Characters, MajorPlotPoints, and ActKeyEvents
3. **Update Provisional Flags**: Set `is_provisional=false` for canonical nodes
4. **Switch to Incremental**: Start using the generation loop for chapter-by-chapter building
5. **Validate Consistency**: Run graph healing to resolve any inconsistencies

## Conclusion

Your narrative structure schema integrates beautifully with the incremental build-as-it-goes approach. The key insight is that:

- **Initialization** creates the high-level structure (characters, acts, major plot points)
- **Generation Loop** fills in the details (chapters, scenes, scene events) incrementally
- **Graph relationships** provide the structural connections between all elements
- **Context retrieval** uses the existing structure to inform new content generation
- **Validation and healing** maintain consistency as the graph evolves

This approach gives you the best of both worlds: the structural rigor of your schema with the flexibility of emergent narrative development.