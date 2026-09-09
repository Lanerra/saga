# SAGA first-party source catalog

Machine inventory of current working-tree bytes; physical lines include documentation and blanks. A catalog row is not a claim of line-by-line semantic review. See the review reports for source-traced coverage. Hashes, imports and full symbol ranges are in the companion JSON files.

| Source | Lines | Git | Module purpose (source docstring) |
|---|---:|---|---|
| `config/__init__.py` | 185 | tracked | Expose SAGA configuration as stable module-level constants. |
| `config/docs_generator.py` | 93 | tracked | Generate configuration reference documentation from the settings model. |
| `config/loader.py` | 91 | tracked | Reload SAGA configuration at runtime. |
| `config/settings.py` | 648 | tracked | Define runtime configuration for SAGA. |
| `config/validator.py` | 151 | tracked | Validate the active SAGA configuration. |
| `core/__init__.py` | 4 | tracked | Core package initialization. |
| `core/db_manager.py` | 782 | tracked | Manage Neo4j connectivity and execute Cypher safely. |
| `core/entity_embedding_service.py` | 268 | tracked | Persist per-entity embedding vectors for semantic operations. |
| `core/exceptions.py` | 108 | tracked | Define standardized exception types for SAGA core. |
| `core/graph_healing_service.py` | 1040 | tracked | Heal and maintain SAGA's Neo4j knowledge graph. |
| `core/http_client_service.py` | 243 | tracked | Perform HTTP I/O for LLM provider integrations. |
| `core/langgraph/__init__.py` | 63 | tracked | Integrate SAGA with LangGraph workflows. |
| `core/langgraph/content_manager.py` | 1315 | tracked | Manage externalized workflow content stored on disk. |
| `core/langgraph/export.py` | 142 | tracked | Export project chapter files into consolidated artifacts. |
| `core/langgraph/graph_context.py` | 332 | tracked | Build Neo4j-derived prompt context for LangGraph nodes. |
| `core/langgraph/initialization/__init__.py` | 49 | tracked | Provide LangGraph initialization node entrypoints. |
| `core/langgraph/initialization/act_outlines_node.py` | 418 | tracked | Generate act-level outlines from the global outline. |
| `core/langgraph/initialization/all_chapter_outlines_node.py` | 134 | tracked | Generate skeleton outlines for all chapters during initialization. |
| `core/langgraph/initialization/chapter_allocation.py` | 187 | tracked | Allocate chapter ranges to acts for initialization workflows. |
| `core/langgraph/initialization/chapter_outline_node.py` | 597 | tracked | Generate chapter outlines on demand. |
| `core/langgraph/initialization/character_sheets_node.py` | 556 | tracked | Generate character sheets during initialization. |
| `core/langgraph/initialization/commit_init_node.py` | 688 | tracked | Commit initialization artifacts to Neo4j. |
| `core/langgraph/initialization/global_outline_node.py` | 334 | tracked | Generate the global story outline during initialization. |
| `core/langgraph/initialization/outline_relationships_node.py` | 277 | tracked | Extract relationships from outline during initialization. |
| `core/langgraph/initialization/persist_files_node.py` | 514 | tracked | Persist initialization artifacts to the project filesystem. |
| `core/langgraph/initialization/run_parsers_node.py` | 96 | tracked | Run parsers to create full graph structure during initialization. |
| `core/langgraph/initialization/validation.py` | 91 | tracked | Validate presence of initialization artifacts on disk. |
| `core/langgraph/nodes/__init__.py` | 57 | tracked | Provide LangGraph node entrypoints for the SAGA workflow. |
| `core/langgraph/nodes/assemble_chapter_node.py` | 92 | tracked | Assemble drafted scenes into a chapter draft. |
| `core/langgraph/nodes/commit_entity_conversion.py` | 144 | **untracked** | Entity conversion helpers: ExtractedEntity -> CharacterProfile/WorldItem. |
| `core/langgraph/nodes/commit_graph_ops.py` | 90 | **untracked** | Graph-operation helpers for chapter node and embedding aggregation. |
| `core/langgraph/nodes/commit_node.py` | 1064 | tracked | Commit extracted entities and relationships to Neo4j. |
| `core/langgraph/nodes/commit_validation.py` | 137 | **untracked** | Validation and filtering helpers for entity/relationship commit operations. |
| `core/langgraph/nodes/context_character_retrieval.py` | 96 | **untracked** | Character profile retrieval for scene-based context. |
| `core/langgraph/nodes/context_plot_retrieval.py` | 218 | **untracked** | Plot and relationship retrieval for scene-based context. |
| `core/langgraph/nodes/context_retrieval_node.py` | 295 | tracked | Orchestrate scene-specific context retrieval from multiple sources. |
| `core/langgraph/nodes/context_scene_retrieval.py` | 386 | **untracked** | Scene-level retrieval for context building. |
| `core/langgraph/nodes/context_world_retrieval.py` | 88 | **untracked** | KG facts retrieval for scene-based context. |
| `core/langgraph/nodes/embedding_node.py` | 95 | tracked | Generate and externalize embeddings for chapter drafts. |
| `core/langgraph/nodes/extraction_nodes.py` | 136 | tracked | Consolidate extracted entities from scene-level extraction. |
| `core/langgraph/nodes/finalize_node.py` | 282 | tracked | Persist the finalized chapter as durable artifacts. |
| `core/langgraph/nodes/graph_healing_node.py` | 128 | tracked | Heal the knowledge graph after chapter persistence. |
| `core/langgraph/nodes/narrative_enrichment_node.py` | 473 | tracked | Narrative enrichment node for Stage 5: Narrative Generation & Enrichment. |
| `core/langgraph/nodes/quality_assurance_node.py` | 182 | tracked | Run periodic quality checks against the knowledge graph. |
| `core/langgraph/nodes/relationship_normalization_node.py` | 244 | tracked | Normalize extracted relationship types to a stable vocabulary. |
| `core/langgraph/nodes/revision_node.py` | 341 | tracked | Revise a chapter draft based on validation feedback. |
| `core/langgraph/nodes/scene_extraction.py` | 728 | tracked | Extract entities from individual scenes instead of full chapters. |
| `core/langgraph/nodes/scene_extraction_normalization.py` | 91 | **untracked** | Entity deduplication and consolidation for scene extraction. |
| `core/langgraph/nodes/scene_extraction_parsing.py` | 186 | **untracked** | LLM output parsing helpers for scene extraction. |
| `core/langgraph/nodes/scene_extraction_validation.py` | 101 | **untracked** | Entity validation and normalization helpers for scene extraction. |
| `core/langgraph/nodes/scene_generation_node.py` | 160 | tracked | Draft individual scenes for a chapter. |
| `core/langgraph/nodes/scene_planning_node.py` | 338 | tracked | Plan scenes for chapter drafting. |
| `core/langgraph/nodes/summary_node.py` | 361 | tracked | Summarize a chapter for use as future drafting context. |
| `core/langgraph/nodes/validation_node.py` | 550 | tracked | Validate generated narrative for internal consistency. |
| `core/langgraph/state.py` | 442 | tracked | Define the LangGraph state schema for SAGA workflows. |
| `core/langgraph/state_helpers.py` | 61 | tracked | Helper functions for managing workflow state field clearing. |
| `core/langgraph/subgraphs/__init__.py` | 6 | tracked | Provide LangGraph subgraph builders for the SAGA workflow. |
| `core/langgraph/subgraphs/_shared.py` | 13 | tracked | Shared utilities across subgraphs. |
| `core/langgraph/subgraphs/generation.py` | 94 | tracked | Build the scene-based generation subgraph for SAGA. |
| `core/langgraph/subgraphs/scene_extraction.py` | 44 | tracked | Build the scene-level extraction subgraph for SAGA. |
| `core/langgraph/subgraphs/validation.py` | 707 | tracked | Build the validation subgraph for SAGA's LangGraph workflow. |
| `core/langgraph/visualization.py` | 222 | tracked | Render LangGraph workflows for debugging and inspection. |
| `core/langgraph/workflow.py` | 641 | tracked | Build LangGraph workflows for SAGA narrative generation. |
| `core/lightweight_cache.py` | 188 | tracked | Ultra-simple per-service in-memory cache. |
| `core/llm_interface_refactored.py` | 906 | tracked | Provide the primary LLM client interface for SAGA. |
| `core/logging_config.py` | 130 | tracked | Configure SAGA logging sinks and formatting. |
| `core/parser_runner.py` | 253 | tracked | CLI command to run parsers independently from SAGA initialization phase. |
| `core/parsers/__init__.py` | 23 | tracked | Parser modules for SAGA initialization. |
| `core/parsers/act_outline_parser.py` | 1081 | tracked | Parse act outlines and create Stage 3 knowledge graph entities. |
| `core/parsers/chapter_outline_parser.py` | 1168 | tracked | Parse chapter outlines and create Stage 4 knowledge graph entities. |
| `core/parsers/character_sheet_parser.py` | 421 | tracked | Parse character sheets and create Character nodes and relationships. |
| `core/parsers/global_outline_parser.py` | 789 | tracked | Parse global outline and create Stage 2 knowledge graph entities. |
| `core/parsers/narrative_enrichment_parser.py` | 602 | tracked | Parse narrative text and extract enrichment data for Stage 5. |
| `core/project_bootstrapper.py` | 222 | tracked | See symbols.json / architecture map |
| `core/project_config.py` | 18 | tracked | See symbols.json / architecture map |
| `core/project_manager.py` | 142 | tracked | See symbols.json / architecture map |
| `core/relationship_normalization_service.py` | 622 | tracked | Normalize extracted relationship types against an evolving vocabulary. |
| `core/relationship_validation.py` | 656 | tracked | Validate relationship semantics for the SAGA workflow. |
| `core/schema_validator.py` | 311 | tracked | Validate and canonicalize knowledge-graph labels and categories. |
| `core/spacy_service.py` | 518 | tracked | SpaCy-based NLP service for entity extraction and text processing. |
| `core/text_processing_service.py` | 484 | tracked | Process and normalize LLM-related text in SAGA. |
| `data_access/__init__.py` | 235 | tracked | Data access layer for Neo4j knowledge graph operations. |
| `data_access/cache_coordinator.py` | 151 | tracked | Coordinate `data_access` cache invalidation. |
| `data_access/chapter_queries.py` | 486 | tracked | See symbols.json / architecture map |
| `data_access/character_queries.py` | 665 | tracked | See symbols.json / architecture map |
| `data_access/cypher_builders/__init__.py` | 11 | tracked | Build parameterized Cypher statements for `data_access`. |
| `data_access/cypher_builders/native_builders.py` | 484 | tracked | Build Cypher statements directly from Pydantic models. |
| `data_access/kg_queries.py` | 1436 | tracked | See symbols.json / architecture map |
| `data_access/plot_queries.py` | 399 | tracked | See symbols.json / architecture map |
| `data_access/scene_queries.py` | 353 | tracked | Query functions for Scene and Event nodes in Neo4j. |
| `data_access/world_queries.py` | 564 | tracked | See symbols.json / architecture map |
| `main.py` | 150 | tracked | See symbols.json / architecture map |
| `models/__init__.py` | 45 | tracked | Export commonly used SAGA model types. |
| `models/agent_models.py` | 79 | tracked | Define inter-agent payload shapes. |
| `models/db_extraction_utils.py` | 97 | tracked | Extract typed values from Neo4j nodes and query results. |
| `models/kg_constants.py` | 440 | tracked | Define constants for the knowledge-graph canonical schema. |
| `models/kg_models.py` | 1016 | tracked | Define core knowledge-graph data models used across SAGA. |
| `models/user_input_models.py` | 213 | tracked | Define user-facing models for providing story input data. |
| `models/validation_utils.py` | 275 | tracked | Validate bootstrap-generated content against runtime configuration. |
| `orchestration/__init__.py` | 2 | tracked | Initialize orchestration package. |
| `orchestration/langgraph_orchestrator.py` | 820 | tracked | Orchestrate LangGraph-based SAGA narrative generation. |
| `processing/__init__.py` | 2 | tracked | Provide text parsing and deduplication utilities for the processing layer. |
| `processing/parsing_utils.py` | 538 | tracked | See symbols.json / architecture map |
| `processing/text_deduplicator.py` | 184 | tracked | Detect and remove duplicate text segments. |
| `prompts/__init__.py` | 13 | tracked | Provide prompt rendering and prompt-context helpers. |
| `prompts/prompt_data_getters.py` | 1048 | tracked | Prepare structured context snippets for prompt templates. |
| `prompts/prompt_renderer.py` | 96 | tracked | Render prompt templates and load per-agent system prompts. |
| `reset_neo4j.py` | 207 | tracked | See symbols.json / architecture map |
| `ui/__init__.py` | 2 | tracked | Initialize UI package. |
| `ui/rich_display.py` | 248 | tracked | Render best-effort terminal progress for SAGA generation runs. |
| `utils/__init__.py` | 139 | tracked | General utility functions for the Saga Novel Generation system. |
| `utils/common.py` | 341 | tracked | Consolidated utilities for SAGA. |
| `utils/file_io.py` | 67 | tracked | See symbols.json / architecture map |
| `utils/similarity.py` | 171 | tracked | See symbols.json / architecture map |
| `utils/text_processing.py` | 593 | tracked | See symbols.json / architecture map |
| `verify_split.py` | 29 | **untracked** | Verify scene_extraction split imports. |
| `verify_subgraph.py` | 9 | **untracked** | Verify scene_extraction subgraph still works. |
| `visualize_workflow.py` | 250 | tracked | CLI tool to visualize LangGraph workflows. |

## Prompts

| Template | Lines |
|---|---:|
| `prompts/initialization/bootstrap_project.j2` | 60 |
| `prompts/initialization/extract_outline_relationships.j2` | 54 |
| `prompts/initialization/generate_act_outline.j2` | 56 |
| `prompts/initialization/generate_chapter_outline.j2` | 43 |
| `prompts/initialization/generate_character_list.j2` | 23 |
| `prompts/initialization/generate_character_sheet.j2` | 51 |
| `prompts/initialization/generate_global_outline.j2` | 57 |
| `prompts/initialization/system.md` | 14 |
| `prompts/initialization/world_building_questions.j2` | 26 |
| `prompts/knowledge_agent/chapter_summary.j2` | 9 |
| `prompts/knowledge_agent/enrich_node_from_context.j2` | 32 |
| `prompts/knowledge_agent/extract_character_structured_lines.j2` | 25 |
| `prompts/knowledge_agent/extract_characters.j2` | 61 |
| `prompts/knowledge_agent/extract_event_characters.j2` | 37 |
| `prompts/knowledge_agent/extract_event_item.j2` | 29 |
| `prompts/knowledge_agent/extract_event_location.j2` | 33 |
| `prompts/knowledge_agent/extract_events.j2` | 44 |
| `prompts/knowledge_agent/extract_item_possession.j2` | 31 |
| `prompts/knowledge_agent/extract_locations.j2` | 44 |
| `prompts/knowledge_agent/extract_relationships.j2` | 49 |
| `prompts/knowledge_agent/extract_world_items_lines.j2` | 45 |
| `prompts/knowledge_agent/relationship_disambiguate_normalize_or_distinct.j2` | 29 |
| `prompts/knowledge_agent/summarize_scene_for_continuity.j2` | 13 |
| `prompts/knowledge_agent/system.md` | 36 |
| `prompts/narrative_agent/draft_scene.j2` | 56 |
| `prompts/narrative_agent/plan_scenes.j2` | 40 |
| `prompts/narrative_agent/system.md` | 21 |
| `prompts/revision_agent/revision_guidance.j2` | 28 |
| `prompts/revision_agent/system.md` | 15 |
| `prompts/validation_agent/evaluate_quality.j2` | 66 |
| `prompts/validation_agent/system.md` | 20 |
