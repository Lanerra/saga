# GBNF Grammar Implementation Plan

This document outlines the plan to implement GBNF (Grammar-Based Normalization Form) grammars for structured LLM outputs in the SAGA codebase. Using GBNF will ensure reliable, strictly formatted JSON responses, eliminating the need for brittle regex parsing and manual fallbacks.

## 1. Initialization Phase (`core/langgraph/initialization/`)

The initialization phase relies heavily on creating structured artifacts (outlines, character sheets) that serve as the foundation for the narrative.

### 1.1 Global Outline Generation
*   **Function:** `generate_global_outline` in `global_outline_node.py`
*   **Prompt:** `prompts/initialization/generate_global_outline.j2`
*   **Current Handling:** Returns JSON, parsed by `_parse_global_outline` using `json.loads` + Pydantic validation (`GlobalOutlineSchema`), with a regex fallback `_fallback_parse_outline`.
*   **Grammar Requirement:**
    *   Strict JSON object matching `GlobalOutlineSchema`.
    *   Fields: `act_count` (int), `acts` (list of objects), `inciting_incident` (string), `midpoint` (string), `climax` (string), `resolution` (string), `character_arcs` (list of objects), `thematic_progression` (string), `pacing_notes` (string).
*   **Benefit:** Guarantees valid JSON structure matching the Pydantic model, eliminating parsing errors and the need for the fallback parser.

### 1.2 Character List Generation
*   **Function:** `_generate_character_list` in `character_sheets_node.py`
*   **Prompt:** `prompts/initialization/generate_character_list.j2`
*   **Current Handling:** Returns a newline-separated list or comma-separated string, parsed by splitting strings and regex cleaning.
*   **Grammar Requirement:**
    *   Strict JSON list of strings: `["Character Name 1", "Character Name 2", ...]`
*   **Benefit:** Removes ambiguity in name separation and formatting (e.g., bullet points, numbering).

### 1.3 Character Sheet Generation
*   **Function:** `_generate_character_sheet` in `character_sheets_node.py`
*   **Prompt:** `prompts/initialization/generate_character_sheet.j2`
*   **Current Handling:** Returns a markdown-formatted structured text (headers like `### DESCRIPTION`), parsed by `_parse_character_sheet_response` which iterates through lines looking for headers.
*   **Grammar Requirement:**
    *   Strict JSON object for `CharacterProfile`.
    *   Fields: `name`, `description`, `traits` (list[str]), `status`, `motivations`, `background`, `skills` (list[str]), `relationships` (map or list of objects), `internal_conflict`.
*   **Benefit:** Replaces brittle custom parsing of markdown headers with reliable JSON parsing.

### 1.4 Chapter Outline Generation
*   **Function:** `_generate_single_chapter_outline` in `chapter_outline_node.py`
*   **Prompt:** `prompts/initialization/generate_chapter_outline.j2`
*   **Current Handling:** Returns semi-structured text, parsed by `_parse_chapter_outline` using keyword matching ("scene", "beat", "plot point").
*   **Grammar Requirement:**
    *   Strict JSON object.
    *   Fields: `scene_description` (string), `key_beats` (list[str]), `plot_point` (string).
*   **Benefit:** Ensures all required sections (scene, beats, plot) are present and correctly identified.

## 2. Graph Healing (`core/graph_healing_service.py`)

### 2.1 Node Enrichment
*   **Function:** `enrich_node_from_context`
*   **Prompt:** Inline string construction (lines 162-180).
*   **Current Handling:** Requests JSON in the prompt, manually parses `response` by stripping code blocks and running `json.loads`.
*   **Grammar Requirement:**
    *   Strict JSON object.
    *   Fields: `inferred_description` (string), `inferred_traits` (list[str]), `inferred_role` (string), `confidence` (float 0.0-1.0).
*   **Benefit:** Eliminates "markdown wrapping" issues (```json ... ```) and guarantees valid JSON syntax for immediate use.

## 3. Extraction Nodes (`core/langgraph/nodes/extraction_nodes.py`)

These nodes are critical for populating the Knowledge Graph and currently rely on a shared `_parse_extraction_json` utility which can be error-prone.

### 3.1 Character Extraction
*   **Function:** `extract_characters`
*   **Prompt:** `prompts/knowledge_agent/extract_characters.j2`
*   **Current Handling:** Expects JSON with `character_updates` key.
*   **Grammar Requirement:**
    *   Strict JSON object: `{ "character_updates": { "Name": { "description": "...", "traits": [...], ... } } }`
    *   Ideally, refactor to a list of objects for safer key handling: `{ "character_updates": [ { "name": "...", "attributes": ... } ] }`
*   **Benefit:** Ensures reliable extraction of complex nested structures (traits, relationships).

### 3.2 Location Extraction
*   **Function:** `extract_locations`
*   **Prompt:** `prompts/knowledge_agent/extract_locations.j2`
*   **Current Handling:** Expects JSON with `world_updates` key.
*   **Grammar Requirement:**
    *   Strict JSON object: `{ "world_updates": { "Category": { "Name": { ... } } } }`
*   **Benefit:** Guarantees proper nesting of categories and entities.

### 3.3 Event Extraction
*   **Function:** `extract_events`
*   **Prompt:** `prompts/knowledge_agent/extract_events.j2`
*   **Current Handling:** Expects JSON with `world_updates` (shared structure with locations).
*   **Grammar Requirement:**
    *   Strict JSON object focusing on Event types.
*   **Benefit:** Prevents hallucination of invalid event types or structures.

### 3.4 Relationship Extraction
*   **Function:** `extract_relationships`
*   **Prompt:** `prompts/knowledge_agent/extract_relationships.j2`
*   **Current Handling:** Expects JSON with `kg_triples` list, then uses `parse_llm_triples` to handle potential string/dict mismatches.
*   **Grammar Requirement:**
    *   Strict JSON object: `{ "kg_triples": [ { "subject": "...", "predicate": "...", "object_entity": "...", "description": "..." } ] }`
*   **Benefit:** Enforces the triple structure directly, removing the need to handle "object_literal" vs "object_entity" ambiguities in post-processing.

## 4. Validation (`core/langgraph/nodes/validation_node.py`)

*   **Analysis:** No LLM calls are currently used for evaluation in this module. It relies on internal logic and database queries.
*   **Action:** No GBNF implementation required at this stage.

## Implementation Priority

1.  **Extraction Nodes:** High volume, critical for data integrity, currently prone to format errors.
2.  **Graph Healing:** "Enrichment" is a specific, self-contained task where GBNF can immediately simplify code.
3.  **Initialization:** High impact on story quality, but run less frequently. Prioritize `Global Outline` and `Character Sheets`.