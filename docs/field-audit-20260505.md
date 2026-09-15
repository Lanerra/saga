# NarrativeState Field Audit — SAGA Codebase

> Historical field audit from 2026-05-05, not a current dead-field deletion list.
> Use the [current guide](../README.md) and trace current definitions and callers.

**Date:** 2026-05-05
**Auditor:** analyst (kanban worker t_0000001f)
**File audited:** `/home/dlewis3/Desktop/AI/saga/core/langgraph/state.py`

## Executive Summary

The task claimed 30 fields "never read" and 5 "never written". Verification reveals:

- **9 of the "never read" fields ARE read** (original static-analysis missed `.get()` patterns)
- **3 "never written" fields ARE written** (quality_assurance_node writes `qa_results`, `total_qa_issues`, `total_qa_fixes`)
- **quality metric fields** (`coherence_score`, `pacing_score`, etc.) are written as None/placeholder but never read

**Net confirmed truly dead: 21 fields never read, 4 fields never written.**

---

## Category A: Confirmed Dead — Safe to Remove

Neither read nor written anywhere in `core/`.

| Field | Type | Written By (create_initial_state only) |
|-------|------|---------------------------------------|
| `last_apoc_available` | bool \| None | yes |
| `last_healing_chapter` | int | yes |
| `last_healing_warnings` | list[str] | yes |
| `narrative_style` | str | yes |
| `provisional_count` | int | yes |
| `run_start_chapter` | int | yes |
| `medium_model` | str | yes |
| `project_id` | str | yes |
| `revision_model` | str | yes |

---

## Category B: Written as None/Placeholder, Never Consumed

Written to state in `workflow.py` and `validation.py` as `None` or error strings, but **never read** from state anywhere.

| Field | Written (where) | Value written |
|-------|----------------|---------------|
| `coherence_score` | workflow.py:230, validation.py:103,188 | None |
| `pacing_score` | workflow.py:231 | None |
| `plot_advancement_score` | workflow.py:232 | None |
| `prose_quality_score` | workflow.py:233 | None |
| `tone_consistency_score` | workflow.py:234 | None |
| `quality_feedback` | workflow.py:235, validation.py:108,193 | None / error string |

**Note:** These scores ARE parsed locally in `validation.py`'s `_parse_quality_scores()` as local variables, but those values are never stored back to state or used beyond logging/conditional checks.

---

## Category C: Written but Never Read

Written by a node, but the written value is never consumed.

| Field | Written by | Written as |
|-------|-----------|-----------|
| `relationship_vocabulary_size` | relationship_normalization_node.py:211 | `len(vocabulary)` |
| `qa_results` | quality_assurance_node.py:178 | `qa_results` dict |
| `total_qa_issues` | quality_assurance_node.py:180 | cumulative int |
| `total_qa_fixes` | quality_assurance_node.py:181 | cumulative int |

---

## Category D: Original Audit False Negatives — Actually Read

The original audit missed these due to `.get()` accessor patterns.

| Field | Read in | Usage |
|-------|---------|-------|
| `chapter_plan_scene_count` | subgraphs/generation.py:42 | `scene_count = state.get("chapter_plan_scene_count", 0)` |
| `current_scene_index` | generation.py:49, scene_generation_node.py:52, context_retrieval_node.py:74 | Scene loop tracking |
| `current_summary` | nodes/finalize_node.py:155 | Read for summary generation |
| `error_node` | workflow.py:54,80,122 | Error handler for fatal error reporting |
| `healing_history` | nodes/graph_healing_node.py:81 | Spread into new healing call |
| `hybrid_context_ref` | content_manager.py:934 | Context management |
| `last_pruned_chapter` | relationship_normalization_node.py:168 | Tracking pruning progress |
| `outline_relationships_ref` | initialization/commit_init_node.py:61 | Passed to content manager |
| `protagonist_name` | 11 locations | Prompts, character sheets, extraction |
| `revision_guidance_ref` | nodes/scene_generation_node.py:83 | Passed to scene generation |
| `target_word_count` | nodes/scene_generation_node.py:74 | Word count budgeting |

---

## Summary Table

| Category | Count | Action |
|----------|-------|--------|
| A: Dead | 9 | Remove from TypedDict and create_initial_state |
| B: Placeholder (no read) | 6 | Remove from TypedDict and create_initial_state |
| C: Written but unread | 4 | Investigate: either wire up reads or remove |
| D: False negatives | 11 | Keep — these are live fields |

**Total removable after Category C investigation: ~19 fields**
**Current NarrativeState field count: ~70 fields → ~51 fields after cleanup**

---

## Recommended Immediate Actions

1. **Remove Category A fields** — no migration needed, never referenced
2. **Remove Category B fields** — written as None, infrastructure for quality scoring is incomplete (scores are parsed locally but never persisted)
3. **Category C** — requires investigation:
   - `relationship_vocabulary_size`: could be replaced with `len(relationship_vocabulary)` at call sites
   - `qa_results`, `total_qa_issues`, `total_qa_fixes`: either wire into a QA dashboard/summary node or remove
4. **Category D** — update the original audit data to correct false negatives
