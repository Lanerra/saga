# Context Management in SAGA

## How Context Is Passed With NarrativeState (Current)

SAGA now uses a chapter‑scoped NarrativeState with an immutable ContextSnapshot to provide deterministic context across all agents during a chapter. The orchestrator owns the lifecycle of this state and explicitly refreshes it at well‑defined boundaries.

### 1. Central Orchestrator Ownership
`NANA_Orchestrator` remains the coordinator. In addition to prior responsibilities, it now manages:
- `NarrativeState` — a per‑chapter, in‑memory container passed by reference
- `ContextSnapshot` — a read‑only, deterministic view of prompt context for the chapter

Key fields:
- NarrativeState: `plot_outline`, `context_epoch`, `snapshot`, `reads_locked`, `caches`
- ContextSnapshot: `chapter_number`, `plot_point_focus`, `chapter_plan`, `hybrid_context`, `kg_facts_block`, `recent_chapters_map`, `snapshot_fingerprint`, `created_ts`

### 2. Deterministic, Snapshot‑Driven Pipeline
The orchestrator constructs and threads state through `run_chapter_pipeline`:

```python
# orchestration/chapter_flow.py
async def run_chapter_pipeline(orchestrator, chapter_num: int) -> str | None:
    orchestrator._update_rich_display(chapter_num=chapter_num, step="Starting Chapter")

    # 1) Chapter-scoped state
    state = None
    if hasattr(orchestrator, "_begin_chapter_state"):
        state = orchestrator._begin_chapter_state(chapter_num)

    # 2) Prerequisites (planning)
    if hasattr(orchestrator, "_prepare_chapter_prerequisites_with_state"):
        prereq = await orchestrator._prepare_chapter_prerequisites_with_state(chapter_num, state)
    else:
        prereq = await orchestrator._prepare_chapter_prerequisites(chapter_num)
    processed = await orchestrator._process_prereq_result(chapter_num, prereq)
    plot_point_focus, plot_point_index, chapter_plan, hybrid_ctx = processed

    # 3) Build initial snapshot once
    if state is not None:
        await orchestrator._refresh_snapshot(state, chapter_plan, chapter_num)
        if state.snapshot and state.snapshot.hybrid_context:
            hybrid_ctx = state.snapshot.hybrid_context

    # 4) Draft → 5) Revise → 6) Finalize
    draft = await orchestrator._draft_initial_chapter_text(chapter_num, plot_point_focus, hybrid_ctx, chapter_plan, state)
    processed_draft = await orchestrator._process_initial_draft(chapter_num, draft)
    revision = await orchestrator._process_and_revise_draft(chapter_num, *processed_draft, plot_point_focus, plot_point_index, hybrid_ctx, chapter_plan, state)
    processed_rev = await orchestrator._process_revision_result(chapter_num, revision)
    return await orchestrator._finalize_and_log(chapter_num, *processed_rev, state)
```

### 3. Source of Truth and Refresh Points
- Neo4j remains the source of truth for persistent state.
- The orchestrator builds a single `ContextSnapshot` after planning and before drafting.
- Snapshot refreshes are explicit and rare (e.g., between phases if required). Most chapters need only the initial snapshot.

### 4. Snapshot Contents and Construction
The snapshot is built once per chapter via `_build_context_snapshot(chapter_number, chapter_plan)`:
- `hybrid_context`: from `ZeroCopyContextGenerator.generate_hybrid_context_native`
- `kg_facts_block`: from `prompts.prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt`
- `recent_chapters_map`: from `data_access.chapter_queries.get_chapter_content_batch_native`
- `plot_point_focus`: resolved from the plot outline
- `snapshot_fingerprint`: SHA256 over the normalized fields for reproducibility

### 5. Prompt Helpers: Dual‑Source Mode
`prompts/prompt_data_getters.get_reliable_kg_facts_for_drafting_prompt` accepts an optional `snapshot`. When provided, it short‑circuits and returns `snapshot.kg_facts_block` instead of re‑querying Neo4j. The orchestrator, not agents, builds these values during snapshot creation.

### 6. Agent Usage
- NarrativeAgent: Accepts an optional `state` and prefers `state.snapshot.hybrid_context` for drafting. It no longer rebuilds hybrid context internally.
- RevisionAgent: May be invoked with a `world_state` whose `previous_chapters_context` comes from the snapshot during scene plan validation.
- KnowledgeAgent: Continues DAO‑based writes; does not perform ad‑hoc context reads during a chapter. The orchestrator decides if/when to refresh the snapshot after writes.

### 7. Caching and Guardrails
- Chapter‑scoped prompt cache is cleared at chapter start (`clear_context_cache`).
- `state.reads_locked` is available as a guardrail to detect accidental live reads during epoch‑locked sections (logging only in the current integration).

### 8. Why This Design
- Determinism: Prompts use a single, immutable snapshot for the chapter.
- Performance: Avoids redundant context rebuilding across agents.
- Local‑first, single‑process: No new services; refresh points are explicit and infrequent.

## Prior Approach (Legacy Overview)

Previously, SAGA rebuilt context on demand and passed ad‑hoc values between steps:
- Neo4j was queried directly by agents during drafting and revision.
- Hybrid context was computed per step using `ZeroCopyContextGenerator`.
- Some context (e.g., chapter plan) was threaded between steps, but determinism depended on live reads.

The current NarrativeState model replaces this with an explicit snapshot lifecycle, improving reproducibility and reducing redundant queries while preserving SAGA’s local‑first, single‑process design.
