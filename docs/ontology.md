# SAGA Knowledge Graph Ontology

## Overview

This document defines the ontological structure for SAGA's knowledge graph, providing clear guidance on when to use each node type. This ensures that extracted entities are properly classified and the knowledge graph maintains semantic richness.

## Core Principles

1. **Semantic Precision**: Choose the most specific node type that accurately describes the entity
2. **Narrative Relevance**: Only extract entities that have lasting significance to the story
3. **Proper Nouns First**: Prioritize named entities over generic references
4. **Avoid Redundancy**: Don't create nodes for transient atmospheric details

## Node Type Taxonomy

### 1. Living Beings & Sentient Entities

#### Character
**Use when**: Active participants in the narrative with agency and development
**Examples**: Protagonists, antagonists, supporting cast
**Key Properties**: name, description, traits, status, relationships
**Do NOT use for**: Historical figures only mentioned in passing (use Person instead)

#### Person
**Use when**: Inactive individuals, historical figures, or mentioned people who don't directly participate
**Examples**: "Queen Victoria" (historical reference), "John's late father" (deceased backstory figure)
**Key Properties**: name, description, historical_significance

#### Creature
**Use when**: Non-human living beings with some level of consciousness or story presence
**Examples**: Dragons, talking animals, sentient aliens, magical beasts
**Key Properties**: name, description, species_type, behavior_patterns

#### Spirit
**Use when**: Ethereal, ghostly, or supernatural beings without physical form
**Examples**: Ghosts, poltergeists, spectral guardians, ancestral spirits
**Key Properties**: name, description, origin, abilities

#### Deity
**Use when**: Gods, divine beings, or worshipped entities with significant power
**Examples**: Zeus, The Sun God, The Ancient Ones
**Key Properties**: name, description, domain, worship_practices

### 2. Physical Objects & Items

#### Object
**Use when**: General physical items, tools, or mundane objects with narrative significance
**Examples**: "the enchanted sword", "the captain's telescope", "the ancient key"
**Key Properties**: name, description, material, current_location
**Do NOT use for**: Magical/special items (use Artifact instead)

#### Artifact
**Use when**: Special, magical, historical, or culturally significant objects
**Examples**: "The Sword of Destiny", "The Philosopher's Stone", "The Crown of Kings"
**Key Properties**: name, description, powers, historical_significance, provenance

#### Document
**Use when**: Books, scrolls, letters, maps, or any written/recorded material
**Examples**: "The Necronomicon", "Captain's Log", "The Treaty of Westphalia"
**Key Properties**: name, description, author, content_summary, significance

#### Relic
**Use when**: Ancient or sacred objects with deep historical/religious importance
**Examples**: "The Holy Grail", "Buddha's Tooth", "The First Scroll"
**Key Properties**: name, description, age, religious_significance, powers

### 3. Locations & Spatial Entities

#### Location
**Use when**: General places without more specific categorization
**Examples**: "The Dark Forest", "The Wasteland", "The Floating Islands"
**Key Properties**: name, description, geographical_features, inhabitants

#### Settlement
**Use when**: Cities, towns, villages, or any inhabited community
**Examples**: "London", "The Hidden Village", "Port Royal"
**Key Properties**: name, description, population, culture, governance

#### Structure
**Use when**: Buildings, constructions, or man-made locations
**Examples**: "The Tower of London", "The Great Wall", "The Observatory"
**Key Properties**: name, description, purpose, architecture, builder

#### Region
**Use when**: Large geographical areas or territories
**Examples**: "The Northern Wastes", "The Empire of Zhen", "The Deadlands"
**Key Properties**: name, description, climate, borders, rulers

#### Landmark
**Use when**: Notable geographical features or recognizable reference points
**Examples**: "Mount Everest", "The Great Canyon", "The Eternal Waterfall"
**Key Properties**: name, description, geographical_type, significance

#### Room
**Use when**: Interior spaces within structures that have narrative importance
**Examples**: "The Throne Room", "The Secret Laboratory", "The Vault"
**Key Properties**: name, description, location_within, purpose

#### Path
**Use when**: Roads, routes, passages, or connections between locations
**Examples**: "The Silk Road", "The Secret Tunnel", "The King's Highway"
**Key Properties**: name, description, origin, destination, dangers

#### Territory
**Use when**: Claimed, controlled, or politically significant areas
**Examples**: "The Sovereign Lands", "The Disputed Zone", "The King's Domain"
**Key Properties**: name, description, controller, borders, resources

### 4. Organizations & Social Structures

#### Faction
**Use when**: Political groups, military forces, or organizations with specific agendas
**Examples**: "The Rebellion", "The Empire", "The Shadow Council"
**Key Properties**: name, description, goals, leader, ideology

#### Organization
**Use when**: Generic institutions without clear political/military focus
**Examples**: "The Merchants Guild", "The University", "The Hospital"
**Key Properties**: name, description, purpose, structure, members

#### Guild
**Use when**: Professional organizations or trade groups
**Examples**: "The Thieves Guild", "The Artisans Guild", "The Mages Circle"
**Key Properties**: name, description, trade, requirements, benefits

#### House
**Use when**: Noble houses, family organizations, or dynastic structures
**Examples**: "House Stark", "The Medici Family", "Clan MacLeod"
**Key Properties**: name, description, lineage, seat, words_motto

#### Order
**Use when**: Religious or knightly orders with specific missions
**Examples**: "The Knights Templar", "The Order of the Phoenix", "The Jedi Order"
**Key Properties**: name, description, mission, code, hierarchy

#### Council
**Use when**: Governing bodies or decision-making groups
**Examples**: "The Council of Elders", "The High Court", "The War Council"
**Key Properties**: name, description, authority, members, decisions

### 5. Events & Temporal Entities

#### Event
**Use when**: Historical events, battles, ceremonies, or significant occurrences
**Examples**: "The Battle of Waterloo", "The Great Plague", "The Coronation"
**Key Properties**: name, description, date, participants, outcome
**Do NOT use for**: Character development moments (use DevelopmentEvent)

#### DevelopmentEvent
**Use when**: Specific moments that change or develop a character
**Examples**: "Elara's betrayal of the Council", "Marcus learns of his heritage", "The moment Sarah chose revenge"
**Key Properties**: name, description, character_affected, change_type, chapter
**This is CRITICAL for tracking character arcs!**

#### WorldElaborationEvent
**Use when**: Moments that expand or reveal world lore/mechanics
**Examples**: "Discovery of the Third Magic", "Revelation of the Ancient Prophecy", "The World's True Nature Explained"
**Key Properties**: name, description, world_aspect_revealed, implications, chapter
**This is CRITICAL for tracking worldbuilding!**

#### Era
**Use when**: Time periods, ages, or epochs
**Examples**: "The Dark Ages", "The Golden Age of Magic", "The Industrial Revolution"
**Key Properties**: name, description, start_date, end_date, characteristics

#### Moment
**Use when**: Specific, brief points in time with lasting significance
**Examples**: "The moment of first contact", "The instant before the explosion"
**Key Properties**: name, description, significance, timestamp

### 6. Systems & Abstract Frameworks

#### System
**Use when**: General systems or frameworks that govern story elements
**Examples**: "The Class System", "The Honor Code", "The Social Hierarchy"
**Key Properties**: name, description, rules, enforcement

#### Magic
**Use when**: Magical systems, schools of magic, or spell classifications
**Examples**: "Elemental Magic", "Blood Magic", "The Weave"
**Key Properties**: name, description, rules, limitations, practitioners

#### Technology
**Use when**: Technological systems, innovations, or scientific frameworks
**Examples**: "Cybernetics", "The Warp Drive", "Genetic Engineering"
**Key Properties**: name, description, capabilities, limitations, inventors

#### Religion
**Use when**: Religious systems, belief structures, or faith traditions
**Examples**: "Christianity", "The Cult of the Sun", "The Old Ways"
**Key Properties**: name, description, deities, practices, holy_texts

#### Culture
**Use when**: Cultural systems, ways of life, or societal frameworks
**Examples**: "Samurai Culture", "The Nomadic Tradition", "Victorian Society"
**Key Properties**: name, description, values, customs, language

### 7. Information & Knowledge

#### Lore
**Use when**: Myths, legends, or historical knowledge embedded in the world
**Examples**: "The Legend of the First King", "The Prophecy of Five", "Ancient Creation Myths"
**Key Properties**: name, description, origin, truth_value, believers

#### Secret
**Use when**: Hidden information or classified knowledge
**Examples**: "The King's True Identity", "The Location of the Vault", "The Forbidden Technique"
**Key Properties**: name, description, keeper, consequences_if_revealed

#### Knowledge
**Use when**: Specific knowledge, information, or data
**Examples**: "The Formula for Immortality", "Maps of the Underground", "The True History"
**Key Properties**: name, description, source, accuracy, accessibility

#### Rumor
**Use when**: Unconfirmed information or gossip that drives plot
**Examples**: "The King is dying", "There's a traitor among us", "The treasure exists"
**Key Properties**: name, description, source, truth_value, spread

### 8. Character Qualities & Development

#### Trait
**Use when**: Distinct personality aspects, character traits, or defining characteristics
**Examples**: "Courage", "Greed", "Loyalty", "Cunning"
**Key Properties**: name, description, character_associated, manifestations
**IMPORTANT**: Create Trait nodes for traits that are thematically important or have relationships with other entities

#### Attribute
**Use when**: Physical or mental attributes that can be measured
**Examples**: "Superhuman Strength", "Photographic Memory", "Enhanced Speed"
**Key Properties**: name, description, measurement, origin

#### Reputation
**Use when**: Social standing or renown that affects story events
**Examples**: "The Hero of the Realm", "The Feared Assassin", "The Beloved Queen"
**Key Properties**: name, description, character, public_perception, earned_by

### 9. Plot & Narrative Structures

#### PlotPoint
**Use when**: Significant story beats, turning points, or narrative milestones
**Examples**: "The Call to Adventure", "The Betrayal", "The Final Confrontation"
**Key Properties**: name, description, chapter, story_impact, characters_involved
**This is CRITICAL for tracking narrative structure!**

#### Story
**Use when**: Tales, legends, or narratives within the main narrative
**Examples**: "The Tale of the Two Brothers", "The Ballad of the Lost King"
**Key Properties**: name, description, moral, teller, cultural_significance

### 10. Abstract Concepts (Use Sparingly)

#### Concept
**Use when**: Major philosophical ideas central to the plot (NOT emotions)
**Examples**: "Free Will", "The Social Contract", "Manifest Destiny"
**Key Properties**: name, description, proponents, opponents, manifestation
**Do NOT use for**: Emotions, atmospheric elements, or minor themes

#### Law
**Use when**: Important rules, regulations, or natural laws driving story events
**Examples**: "The Three Laws of Robotics", "The Law of Equivalent Exchange", "The Prime Directive"
**Key Properties**: name, description, enforcement, consequences_of_breaking

#### Tradition
**Use when**: Significant cultural practices affecting characters or plot
**Examples**: "The Trial by Combat", "The Coming of Age Ceremony", "The Blood Oath"
**Key Properties**: name, description, culture, purpose, ritual_steps

#### Symbol
**Use when**: Specific named symbols with story significance
**Examples**: "The Mockingjay", "The One Ring", "The Scarlet Letter"
**Key Properties**: name, description, meaning, bearer, power

### 11. Resources & Economy

#### Resource
**Use when**: Materials, natural resources, or commodities driving conflict
**Examples**: "Spice", "Vibranium", "Mana Crystals", "Oil"
**Key Properties**: name, description, scarcity, uses, location

#### Currency
**Use when**: Money or trade systems with narrative importance
**Examples**: "Gold Dragons", "Credits", "Bitcoin"
**Key Properties**: name, description, value, issuer, backing

### 12. Container & Meta Types

#### WorldContainer
**Use when**: Organizational containers for related world elements
**Examples**: "The Magic System", "The Political Landscape", "The Pantheon"
**Key Properties**: name, description, contained_elements, relationships
**Use SPARINGLY**: Only for high-level organizational purposes

#### ValueNode
**Use when**: Literal values or data points that need to be stored
**Examples**: "The number 42", "The coordinates", "The password"
**Key Properties**: value, type, context
**Use SPARINGLY**: Only when the value itself is significant

## Decision Tree for Node Type Selection

1. **Is it a living being?**
   - Active participant with agency? → **Character**
   - Inactive/historical figure? → **Person**
   - Non-human creature? → **Creature**
   - Ethereal/ghostly? → **Spirit**
   - Divine being? → **Deity**

2. **Is it a physical object?**
   - Magical/special? → **Artifact**
   - Written material? → **Document**
   - Ancient/sacred? → **Relic**
   - General item? → **Object**

3. **Is it a place?**
   - Community? → **Settlement**
   - Building? → **Structure**
   - Large area? → **Region**
   - Notable feature? → **Landmark**
   - Interior space? → **Room**
   - Route/connection? → **Path**
   - Controlled area? → **Territory**
   - Other? → **Location**

4. **Is it an organization?**
   - Political/military? → **Faction**
   - Trade group? → **Guild**
   - Noble family? → **House**
   - Religious/knightly? → **Order**
   - Governing body? → **Council**
   - Other? → **Organization**

5. **Is it an event?**
   - Character development moment? → **DevelopmentEvent**
   - World lore revelation? → **WorldElaborationEvent**
   - Historical occurrence? → **Event**
   - Time period? → **Era**

6. **Is it a system?**
   - Magical? → **Magic**
   - Technological? → **Technology**
   - Religious? → **Religion**
   - Cultural? → **Culture**
   - Other? → **System**

7. **Is it information?**
   - Myth/legend? → **Lore**
   - Hidden knowledge? → **Secret**
   - Unconfirmed? → **Rumor**
   - Other? → **Knowledge**

8. **Is it a character quality?**
   - Personality trait? → **Trait**
   - Measurable attribute? → **Attribute**
   - Social standing? → **Reputation**

9. **Is it plot-related?**
   - Story milestone? → **PlotPoint**
   - Story within story? → **Story**

10. **Is it abstract?**
    - Philosophical idea? → **Concept**
    - Rule/regulation? → **Law**
    - Cultural practice? → **Tradition**
    - Symbolic representation? → **Symbol**

11. **Is it a resource?**
    - Material/commodity? → **Resource**
    - Money? → **Currency**

## Common Mistakes to Avoid

### ❌ DON'T Extract These as Entities:
- Emotions (unless using Trait for thematically important traits like "Courage")
- Weather conditions
- Colors, sounds, lights
- Atmospheric descriptions
- Temporary states
- Descriptive adjectives
- Literary devices
- Background crowds
- Generic "the sword" (unless it's named or mentioned 3+ times)

### ✅ DO Extract These:
- Named characters, places, objects
- Significant plot events
- Character development moments (as DevelopmentEvent)
- World lore revelations (as WorldElaborationEvent)
- Important organizations
- Key story concepts
- Narrative turning points (as PlotPoint)

## Special Node Types That Are Often Missed

### DevelopmentEvent
**Critical for character arcs!** Every time a character:
- Makes a significant choice
- Experiences a revelation
- Changes their beliefs or values
- Learns something transformative
- Undergoes trauma or growth

Create a DevelopmentEvent node!

### WorldElaborationEvent
**Critical for worldbuilding!** Every time the narrative:
- Reveals new magic/tech mechanics
- Explains historical context
- Introduces new species/cultures
- Clarifies world rules or physics
- Expands the cosmology

Create a WorldElaborationEvent node!

### PlotPoint
**Critical for structure!** Every time the story:
- Hits a major beat (inciting incident, midpoint, climax)
- Takes a significant turn
- Resolves a major conflict
- Introduces a new complication

Create a PlotPoint node!

### Trait
**Important for thematic tracking!** When a character trait:
- Is central to the theme
- Has relationships with other entities
- Drives multiple plot events
- Undergoes transformation

Create a Trait node (not just a property)!

## Examples of Proper Classification

### Example 1: Character Development
❌ **Wrong**: `Event: "Sarah's Choice"`
✅ **Right**: `DevelopmentEvent: "Sarah chooses vengeance over mercy"`
- **Why**: This is a character development moment, not just a historical event

### Example 2: World Lore
❌ **Wrong**: `Event: "The Magic System Explained"`
✅ **Right**: `WorldElaborationEvent: "Revelation of the Three Sources of Magic"`
- **Why**: This expands world mechanics, not just narrative events

### Example 3: Plot Structure
❌ **Wrong**: `Event: "The Meeting"`
✅ **Right**: `PlotPoint: "The Hero's First Encounter with the Mentor"`
- **Why**: This is a narrative structural element following story patterns

### Example 4: Character Trait
❌ **Wrong**: Just storing "brave" as a character property
✅ **Right**: `Trait: "Courage"` with `HAS_TRAIT` relationship to Character
- **Why**: When the trait has thematic importance and relationships

### Example 5: Magical Object
❌ **Wrong**: `Object: "The Sword of Destiny"`
✅ **Right**: `Artifact: "The Sword of Destiny"`
- **Why**: It's magical/special, not a mundane object

### Example 6: Historical Figure
❌ **Wrong**: `Character: "Alexander the Great"` (only mentioned in passing)
✅ **Right**: `Person: "Alexander the Great"`
- **Why**: Not an active participant, just a reference

## Integration with Extraction

When extracting entities from chapter text:

1. **Read the text carefully** to understand context
2. **Identify entities** using proper noun priority
3. **Classify each entity** using the decision tree
4. **Choose the most specific type** that applies
5. **Create relationships** that make semantic sense
6. **Store in the appropriate format** for the extraction schema

Remember: The goal is semantic richness and narrative clarity, not comprehensive coverage of every word.
