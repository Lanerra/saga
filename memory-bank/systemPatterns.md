# System Architecture Patterns

## Agent Refactoring Pattern
- **Consolidation over Specialization**: Moving from multiple specialized agents (kg_maintainer, knowledge_agent) to a single cohesive knowledge agent
  - Example: All KG update logic now resides in `knowledge_agent.py` instead of distributed across modules
  - Pattern: "When an agent's functionality becomes central to multiple workflows, consolidate into a dedicated agent"

- **Workflow Integration**: Narrative agent now handles scene planning (previously in PlannerAgent)
  - Pattern: "When planning and narrative generation are tightly coupled, integrate the functionality into the primary workflow agent"

## Output Structure Pattern
- **Directory-Based Organization**: All output files now follow `novel_output/` structure
  - Pattern: "For all persistent outputs, use a dedicated directory with consistent substructure (e.g., novel_output/saga_run.log)"
  - Implementation: Log files now stored at `novel_output/saga_run.log`

## Dependency Management Pattern
- **Backward Compatibility**: 
  - All existing workflow paths maintained via adapter patterns in `finalize_agent.py`
  - No changes required to external integration points

## Naming Convention Pattern
- **Consistency Enforcement**: Strict naming standard across agent modules
  - Pattern: "All agent-specific constants and variables follow [agent_name]_MODEL naming pattern"
  - Example: `NARRATOR_MODEL` → `NARRATIVE_MODEL` for consistent terminology

## Data Tracking Pattern
- **Comprehensive Metrics Collection**: 
  - All chapter generation phases track usage data
  - Raw LLM output preserved alongside processed results
  - Usage metrics aggregated across planning, drafting, and revision phases
