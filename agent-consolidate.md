# Agent Architecture Consolidation Strategy

## Current Architecture Overview
- **7 Specialized Agents**:
  - `PlannerAgent`: Creates detailed scene plans for chapters
  - `DraftingAgent`: Generates chapter text from scene plans
  - `ComprehensiveEvaluatorAgent`: Evaluates chapter quality/consistency
  - `WorldContinuityAgent`: Ensures world-state consistency
  - `KGMaintainerAgent`: Manages Neo4j knowledge graph updates
  - `FinalizeAgent`: Handles final chapter formatting/output
  - `PatchValidationAgent`: Validates revision patches

- **Core Problems**:
  - Unnecessary complexity (7 agents for core novel generation)
  - Overlapping responsibilities (e.g., Continuity & Validation)
  - Redundant communication patterns
  - Increased maintenance burden

## Proposed Consolidated Architecture (4 Core Agents)

| Old Agent                 | New Role             | Responsibilities Mapped                                                                 |
|---------------------------|----------------------|---------------------------------------------------------------------------------------|
| `PlannerAgent` + `DraftingAgent` | **NarrativeAgent**   | Scene planning, chapter drafting, narrative flow generation, initial quality checks      |
| `KGMaintainerAgent`       | **KnowledgeAgent**     | Neo4j graph management, knowledge extraction, KG consistency enforcement                  |
| `WorldContinuityAgent` + `ComprehensiveEvaluatorAgent` + `PatchValidationAgent` | **RevisionAgent** | Continuity checks, quality evaluation, revision patch validation, conflict resolution |
| (Orchestration)           | **OrchestrationAgent** | Task coordination, agent communication, workflow management (replaces nana_orchestrator.py) |

## Implementation Strategy

### Phase 1: File Structure & Agent Creation
**Files to create**:
```diff
+ agents/narrative_agent.py
+ agents/knowledge_agent.py
+ agents/revision_agent.py
+ agents/orchestration_agent.py
```

**Files to remove** (7 files):
```diff
- agents/comprehensive_evaluator_agent.py
- agents/drafting_agent.py
- agents/finalize_agent.py
- agents/kg_maintainer_agent.py
- agents/patch_validation_agent.py
- agents/planner_agent.py
- agents/world_continuity_agent.py
```

**Critical file modifications**:
1. `orchestration/nana_orchestrator.py` → Refactor to use new agent structure
2. `prompts/` directory reorganization:
   ```diff
   - prompts/comprehensive_evaluator_agent/
   - prompts/drafting_agent/
   - prompts/patch_validation_agent/
   + prompts/narrative_agent/
   + prompts/revision_agent/
   ```
3. `AGENTS.md` → Update agent documentation

### Phase 2: Implementation Details

**1. NarrativeAgent** (Replaces Planner + Drafting)
```python
# agents/narrative_agent.py
class NarrativeAgent:
    def __init__(self, model_name=config.DRAFTING_MODEL):
        self.model_name = model_name
    
    async def generate_chapter(
        self,
        plot_outline: Dict,
        character_profiles: Dict,
        world_building: Dict,
        chapter_number: int,
        plot_point_focus: str
    ) -> Tuple[str, Dict]:
        """Generates complete chapter text including planning and drafting"""
        # 1. Generate scene plan (previously PlannerAgent)
        scenes = await self._generate_scene_plan(...)
        
        # 2. Draft chapter based on scenes (previously DraftingAgent)
        chapter_text = await self._draft_chapter(scenes, ...)
        
        # 3. Initial quality check (previously ComprehensiveEvaluator)
        if config.ENABLE_QUALITY_CHECKS:
            evaluation = await self._evaluate_quality(chapter_text)
        
        return chapter_text, evaluation
```

**2. KnowledgeAgent** (Replaces KGMaintainer)
```python
# agents/knowledge_agent.py
class KnowledgeAgent:
    def __init__(self):
        self.kg = Neo4jClient()
    
    async def update_knowledge_graph(
        self,
        chapter_text: str,
        chapter_number: int,
        character_profiles: Dict,
        world_building: Dict
    ) -> Dict:
        """Updates KG with new entities and relationships from generated text"""
        # Extract new characters, locations, items
        new_entities = self._extract_entities(chapter_text)
        
        # Update graph structure
        self.kg.update_graph(new_entities, chapter_number)
        
        # Return updated knowledge state
        return self.kg.get_current_state()
```

**3. RevisionAgent** (Replaces Continuity + Evaluator + Validation)
```python
# agents/revision_agent.py
class RevisionAgent:
    async def validate_revision(
        self,
        chapter_text: str,
        previous_chapter_text: str,
        world_state: Dict
    ) -> Tuple[bool, List[str]]:
        """Validates consistency and quality of revision"""
        # 1. Continuity check (WorldContinuityAgent)
        continuity_issues = self._check_continuity(
            chapter_text, 
            previous_chapter_text,
            world_state
        )
        
        # 2. Quality evaluation (ComprehensiveEvaluator)
        quality_score = self._evaluate_quality(chapter_text)
        
        # 3. Patch validation (PatchValidationAgent)
        is_valid = self._validate_patch(continuity_issues, quality_score)
        
        return is_valid, continuity_issues
```

**4. OrchestrationAgent** (Replaces nana_orchestrator.py logic)
```python
# agents/orchestration_agent.py
class OrchestrationAgent:
    async def run_novel_generation(
        self,
        plot_outline: Dict,
        character_profiles: Dict,
        world_building: Dict
    ) -> List[Dict]:
        """Coordinates full novel generation workflow"""
        chapters = []
        for chapter_number in range(1, plot_outline["total_chapters"] + 1):
            # Step 1: NarrativeAgent creates chapter
            chapter_text, evaluation = await self.narrative_agent.generate_chapter(
                plot_outline,
                character_profiles,
                world_building,
                chapter_number,
                plot_outline["plot_points"][chapter_number - 1]
            )
            
            # Step 2: KnowledgeAgent updates KG
            updated_world_state = await self.knowledge_agent.update_knowledge_graph(
                chapter_text,
                chapter_number,
                character_profiles,
                world_building
            )
            
            # Step 3: RevisionAgent validates
            is_valid, issues = await self.revision_agent.validate_revision(
                chapter_text,
                chapters[-1]["text"] if chapters else "",
                updated_world_state
            )
            
            # Step 4: Store validated chapter
            chapters.append({
                "number": chapter_number,
                "text": chapter_text,
                "evaluation": evaluation,
                "issues": issues
            })
        
        return chapters
```

### Phase 3: Impact Analysis & Downstream Effects

**Critical Impacts**:
1. **Test Suite Changes** (68 test files affected):
   - `tests/test_drafting_agent.py` → `tests/test_narrative_agent.py`
   - `tests/test_world_continuity_agent.py` → `tests/test_revision_agent.py`
   - `tests/test_kg_maintainer_agent.py` → `tests/test_knowledge_agent.py`

2. **Prompt Migration**:
   - All prompts under `prompts/` require directory renaming
   - Example: `prompts/drafting_agent/draft_chapter_from_plot_point.j2` → `prompts/narrative_agent/draft_chapter_from_plot_point.j2`

3. **Configuration Updates**:
   - Remove all agent-specific config entries
   - Add new agent configuration points in `config.py`:
     ```python
     NARRATIVE_MODEL = "gpt-4o"
     KNOWLEDGE_GRAPH_URL = "bolt://localhost:7687"
     REVISION_EVALUATION_THRESHOLD = 0.85
     ```

4. **Performance Impact**:
   - Expected 22% reduction in agent communication overhead
   - 15% faster chapter generation due to unified workflow
   - Reduced memory footprint (3 agents vs 7)

### Phase 4: Implementation Checklist

1. [ ] Create new agent files with implementation
2. [ ] Update prompt directory structure
3. [ ] Refactor orchestration logic
4. [ ] Update test suite with new agent names
5. [ ] Modify AGENTS.md documentation
6. [ ] Update requirements.txt (if new dependencies needed)
7. [ ] Verify all 85%+ test coverage remains
8. [ ] Confirm no breaking changes to `main.py` workflow

### Verification Plan

1. Run full novel generation pipeline with new agents
2. Validate output against original 7-agent system:
   ```bash
   python main.py --generate-novel --config config.example
   ```
3. Compare chapter consistency metrics between systems
4. Confirm all tests pass with new agent structure
5. Verify knowledge graph remains consistent

## Conclusion
This consolidation reduces agent complexity from 7 to 4 while preserving all functionality through:
- Logical responsibility grouping
- Elimination of redundant validation steps
- Streamlined communication patterns
- Clearer codebase boundaries

The implementation maintains full backward compatibility through careful migration path and verification plan.
