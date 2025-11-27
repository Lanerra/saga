## Complexity Hotspots

**1. State bloat trajectory**  
`NarrativeState` accumulates everything—summaries, extractions, drafts, validation results. By chapter 30, you're serializing/deserializing megabytes per node execution. SQLite checkpoint writes become a bottleneck. The state should reference content, not contain it.

**2. Revision loop termination**  
You have three exit mechanisms: `iteration_count`, `force_continue`, and quality thresholds. The validation→revise→extraction cycle creates ambiguous termination conditions. If validation improves scores but finds new contradictions, does it loop again? The control flow is underspecified.

**3. Fuzzy matching on every commit**  
`commit_to_graph` runs Levenshtein distance against the *entire graph* for every extracted entity. This is O(extracted × existing) per chapter. With 500+ entities by chapter 20, you're doing tens of thousands of string comparisons. There's no indexing strategy.

**4. Context retrieval black box**  
"Queries Neo4j for relevant context" is doing heavy lifting here. What's the Cypher query? How many relationship hops? Do you retrieve 5 entities or 50? This could be anywhere from 10ms to 10 seconds per scene, and it's opaque to the rest of the system.

**5. Extraction parallelization failure handling**  
Four simultaneous LLM calls with `asyncio.gather`. If `extract_characters` succeeds but `extract_relationships` fails, what does `consolidate` receive? The doc doesn't specify partial failure behavior. Either you lose all extraction work or you commit incomplete data.

## Single Points of Failure

**1. Neo4j is unrecoverable**  
If the graph becomes corrupted or unavailable, the entire system deadlocks. There's no degraded mode. You can't even generate a chapter using just outlines + previous chapter text because `retrieve_context` assumes graph availability.

**2. Serialization choke point**  
All extraction funnels through one `commit_to_graph` node. If Python crashes between extraction and commit, that chapter's graph updates vanish. The prose exists in state, but the knowledge graph is now desynchronized with the narrative.

**3. Chapter assembly as blind concatenation**  
`assemble_chapter` just joins scene drafts. If scene boundaries overlap (character mid-sentence at end of scene 2, picks up mid-sentence in scene 3), or if two scenes reference "the previous evening" ambiguously, concatenation produces incoherent text. No validation that scenes actually connect.

**4. Embeddings blocking critical path**  
`generate_embeddings` runs *after* generation but *before* extraction. Why? If embedding generation fails (OOM, model crash), does it block extraction? Embeddings seem like a side effect, not a prerequisite for workflow continuation.

**5. Healing as append-only complexity**  
`heal_graph` enriches provisional nodes and merges duplicates. But what if the merge logic is wrong? You've now permanently fused two entities that should be distinct. No undo mechanism. The healing history log doesn't help if the *current* graph state is corrupted.

## Better Patterns

**Within local-first, single-machine constraints:**

**1. Externalize content from state**  
Instead of `state["draft_text"]`, store `state["draft_ref"] = "chapters/chapter_05_draft_v2.md"`. State becomes a pointer DAG. Checkpoints are tiny. Enables easy diffing of revisions.

**2. Staged graph commits**  
Introduce a "shadow graph" pattern:
- Extraction writes to `STAGING` labels
- Validation queries against `STAGING ∪ COMMITTED`  
- On success, `MATCH (n:Character:STAGING) REMOVE n:STAGING SET n:COMMITTED`  
- On failure, `MATCH (n:STAGING) DELETE n`

Enables atomic rollback. Neo4j stays consistent with finalized prose.

**3. Pre-emptive entity registry**  
Maintain `project_root/.saga/entity_registry.json`:
```json
{
  "characters": {"Jonathan_Reeves": ["Jon", "Reeves", "Jonathan"]},
  "locations": {"The_Old_Tower": ["the tower", "old tower", "ancient spire"]}
}
```

Extraction nodes resolve against this registry *first*, then fall back to fuzzy matching only for genuinely new entities. Eliminates 95% of string distance calculations.

**4. Context retrieval budget**  
Make `retrieve_context` explicit:
```python
context = neo4j.query(
    max_entities=15,
    max_depth=2,  # relationship hops
    required_labels=["Character"],
    scene_location=state["current_scene"]["setting"]
)
```

Bounded query complexity. Predictable performance. If you need more context, that's a separate decision point (enlarge budget vs. generate with less context).

**5. Stream generation with micro-validation**  
Instead of:
```
Generate 5 scenes → Validate entire chapter → Maybe revise everything
```

Do:
```
For each scene:
  Generate → Validate continuity with previous scene
  If bad: Revise *this scene only*
  Commit scene to graph immediately
```

Fail-fast. Localizes errors. Graph stays synchronized. If process crashes, resume from last committed scene, not last committed chapter.

**6. Dual-mode context fallback**  
```python
try:
    context = retrieve_from_neo4j()
except Neo4jUnavailable:
    context = retrieve_from_summaries(state["previous_chapter_summaries"])
    state["degraded_mode"] = True
```

System continues in text-only mode. Graph repairs happen async. User gets *a* novel even if graph is broken, just without full consistency checking.

**7. Idempotent commit with content addressing**  
Before creating nodes, hash the entity data:
```python
entity_hash = sha256(json.dumps(entity, sort_keys=True))
MERGE (n:Character {content_hash: $hash})
```

If `commit_to_graph` runs twice, `MERGE` prevents duplicates. Safe to replay.

**8. Explicit dependency markers**  
Add to state:
```python
state["node_execution_log"] = [
  {"node": "plan_scenes", "timestamp": ..., "dependencies_met": ["chapter_outline"]},
  {"node": "draft_scene", "dependencies_met": ["plan_scenes", "retrieve_context"]}
]
```

If state is corrupted or out of order, you can detect it: "draft_scene executed but retrieve_context hasn't?" Signal error explicitly rather than generating garbage.

---

The core architectural tension: **LangGraph wants small, frequent state updates. Neo4j wants batched, transactional commits.** Your system is getting caught in the middle—state is too big, commits are too coarse-grained. The fixes above mostly involve making state smaller (externalize content) and commits more granular (staged, per-scene).
