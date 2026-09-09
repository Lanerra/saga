# SAGA graph/persistence audit — 2026-09-06

## Scope and evidence

**Verdict: not approved for durable multi-project graph persistence.** This was a bounded, static review of the working tree at `4ce8870` (`refactor/kg-schema`, ahead 1). The tree was already broadly dirty; the split commit helpers under `core/langgraph/nodes/` were untracked. This report adds no source finding based on historical audit documents.

Reviewed: `core/db_manager.py`; `data_access/{kg_queries,character_queries,world_queries,chapter_queries,scene_queries}.py`; `core/langgraph/nodes/{commit_node,commit_graph_ops,commit_entity_conversion}.py`; and the directly relevant `core/graph_healing_service.py`, native Cypher builder, embedding service, state definition, and focused tests/fakes.

No tests, Neo4j/Cypher, network, or LLM calls were run. Transaction and query findings are source-level findings; they do **not** claim the shown Cypher was executed.

## Module map

| Module | Persistence/read role and material contract |
|---|---|
| `core/db_manager.py` | Singleton synchronous Neo4j driver wrapped by async APIs; owns one-batch transaction/rollback behavior, schema DDL, vector-index DDL, and NumPy/list embedding conversion. |
| `data_access/kg_queries.py` | Generic extracted-triple persistence, general KG traversal, duplicate discovery, merge/deduplication maintenance, and cache invalidation. |
| `data_access/character_queries.py` | Character name canonicalization cache plus profile reads/sync through `NativeCypherBuilder`; reads are keyed by name or ID. |
| `data_access/world_queries.py` | World name-to-ID cache plus ID/name reads and writes through `NativeCypherBuilder`; reads expect canonical Location/Item/Event labels. |
| `data_access/chapter_queries.py` | Canonical chapter upsert, chapter/vector reads, semantic context search, and its `include_provisional` contract. |
| `data_access/scene_queries.py` | Scene, act, and relationship context reads keyed by chapter/scene or name. No project discriminator is accepted. |
| `commit_entity_conversion.py` | Converts extracted entities into `CharacterProfile`/`WorldItem`; carries character `attributes.relationships` into the profile write channel. |
| `commit_graph_ops.py` | Builds canonical chapter upsert statements and averages scene embeddings. |
| `commit_node.py` | Main commit orchestration: pre-validation, name filtering, entity/relationship/chapter statement construction, one batch execution, cache invalidation, and compensating rollback. |
| `graph_healing_service.py` | Global provisional-node enrichment, duplicate candidate discovery/merge delegation, and orphan cleanup. It crosses from Neo4j `elementId()` to application `n.id`. |

## Prioritized findings

### P0-1 — The graph has no project boundary although workflow state has one

**Evidence:** `core/langgraph/state.py:101-105` declares `NarrativeState.project_id`, but `core/langgraph/nodes/commit_node.py:270-284` reads all entity names globally and `:682-689` deletes all relationships for a chapter globally. `data_access/chapter_queries.py:75-105` merges Chapter solely by `number`; its reads likewise use only chapter number (`:207-221`). The uniqueness constraints are global by ID/name/chapter number in `core/db_manager.py:549-563`. Scene reads take only chapter/scene (`data_access/scene_queries.py:18-61`), and healing selects all provisional nodes (`core/graph_healing_service.py:56-73`) and global duplicate candidates.

**Activation:** Two project directories use the same configured `NEO4J_DATABASE` and both contain chapter 1, a character with the same name, or a same-named world entity.

**Impact:** A second project can merge into, update, read, heal, deduplicate, or delete the first project’s Chapter, entities, relationships, scenes, and vector context. `compute_chapter_id()` namespaces an ID but the actual Chapter merge remains `{number}`.

**Repair:** Make a nonempty immutable graph-project/novel ID required at every persistence and read boundary. Put it on all project-owned nodes and relationship assertions; make it part of merge/lookup keys, cache keys, scene identities, and healing/maintenance predicates. Replace global uniqueness constraints with composite project-local constraints and migrate existing data before enabling the new contract.

**Acceptance test:** Persist two projects with identical chapter numbers, names, and embeddings in one Neo4j test database. Assert separate nodes/edges/vector results; project-local context and healing only; and re-committing, deleting, or healing one project leaves the other graph unchanged.

### P0-2 — A failed commit can destructively remove a previously successful chapter

**Evidence:** `core/db_manager.py:271-339` rolls back the batch transaction when a statement fails. In contrast, `commit_to_graph()` catches every later exception and unconditionally invokes `_rollback_commit()` (`core/langgraph/nodes/commit_node.py:567-588`). That rollback deletes all relationships tagged with the chapter (`:225-231`), orphaned nodes created in the chapter (`:233-241`), and the Chapter by number (`:243-251`).

**Activation:** A retry/revision of an already committed chapter fails before `execute_cypher_batch()` (for example while constructing entity embeddings/statements), or the batch itself fails and has already rolled back.

**Impact:** The compensating delete has no attempt ID, preimage, or proof that this invocation committed anything. It can delete the earlier durable chapter state for the same chapter number. The database transaction protects the failing batch; the subsequent destructive action defeats that protection.

**Repair:** Remove compensating `_rollback_commit()` from failure paths before or during the batch and rely on `execute_cypher_batch()` rollback. Split pre-batch, batch, and post-commit/cache failures. Treat a post-commit cache-invalidation failure as an operational warning/retry unless a versioned, preimage-backed compensation design is introduced.

**Acceptance test:** Seed chapter 4 with entities and relationships. Force (a) statement construction failure before batch execution and (b) a batch statement failure. In both cases, assert the pre-existing Chapter, edges, and entity fields are identical before/after; assert the batch transaction was rolled back exactly once.

### P0-3 — Profile relationships are written and then deleted in the same commit transaction

**Evidence:** `commit_entity_conversion.py:64-77` preserves `attributes["relationships"]` in `CharacterProfile.relationships`. The character builder emits those relationships (`data_access/cypher_builders/native_builders.py:71-114`), and `commit_node.py:441-464` appends entity statements before the relationship-replacement statements. `_build_relationship_statements()` always begins by deleting every edge with the current `chapter_added` (`commit_node.py:682-696`).

**Activation:** A character has nonempty `attributes.relationships`; especially clear when the extracted relationship list is empty.

**Impact:** Builder-created profile edges carry `chapter_added=$chapter_number` and are deleted later in the same batch. The profile relationship channel silently persists no edge; a mixed commit can drop profile-derived relationships while retaining only the separate extracted relationship channel.

**Repair:** Select one relationship-write authority for this commit path. The narrowest change is to pass `relationships={}` to the character entity upsert and transform every relationship into `_build_relationship_statements()` output, including profile-origin metadata if it must remain queryable.

**Acceptance test:** Commit a character whose only relationship is `attributes.relationships={"Bob": {...}}`. Assert the expected edge exists after the first commit, exists exactly once after replay, and is intentionally removed only when the next chapter-set replacement omits that same source assertion.

### P1-1 — Relationship identity cannot represent per-chapter assertions or revision membership

**Evidence:** `commit_node.py:682-689` deletes by `r.chapter_added`, but the relationship ID excludes chapter (`:952-955`). `apoc.merge.relationship` creates the edge with `chapter_added` only in its creation map (`:992-1004`) and does not update it on a match (`:1005-1010`).

**Activation:** The same `(subject, predicate, object)` is asserted in chapters 1 and 2, followed by a chapter-2 retry/revision that omits it.

**Impact:** The chapter-2 assertion is merged into the chapter-1 edge but is never represented as a chapter-2 occurrence. The chapter-2 delete cannot remove it because its retained provenance is chapter 1. Conversely, a temporal query cannot tell that the fact was reasserted in chapter 2. The documented "replace this chapter's relationship set" contract is therefore not implementable for repeated triples.

**Repair:** Separate semantic edge identity from chapter assertion identity. A narrow design creates one idempotent relationship occurrence per `(project_id, subject ID, predicate, object ID/value, chapter)`; a stronger design stores assertion/fact nodes linked to Chapter and canonical entities.

**Acceptance test:** Commit the same triple in chapters 1 and 2, then replace chapter 2 with no relationships. Assert the chapter-1 assertion remains, chapter-2 assertion alone is removed, and temporal retrieval returns the correct occurrence/provenance.

### P1-2 — `include_provisional` is inverted in semantic context retrieval

**Evidence:** Both vector-result and immediate-predecessor predicates in `data_access/chapter_queries.py:348-352,367-372` use:

```cypher
$include_provisional = false OR COALESCE(...is_provisional, false) = false
```

**Activation:** Any call with the default `include_provisional=False` or with `True`.

**Impact:** `False` admits provisional chapters through the first disjunct; `True` excludes them through the second. This reverses the API documentation at `:293-301` and permits provisional context by default.

**Repair:** Use the same direction as the character/world read contracts:

```cypher
$include_provisional = true OR COALESCE(node.is_provisional, false) = false
```

Apply it to both parts of the query.

**Acceptance test:** Seed finalized and provisional previous chapters, including a provisional immediate predecessor. With `False`, no provisional chapter appears. With `True`, both are eligible while the deterministic score ordering and final limit remain unchanged.

### P1-3 — Generic triple persistence discards supplied stable IDs for non-characters

**Evidence:** `data_access/kg_queries.py:337-341,431-434` derives `subject_id_param`/`object_id_param` only when the type is `Character`; it does not read supplied IDs for Locations, Items, or Events. The subsequent candidate query matches by ID *or* label/name and selects an unordered `head(collect(...))` (`:441-469`). World upserts instead merge by label/name (`data_access/cypher_builders/native_builders.py:197-219`).

**Activation:** A triple references an existing non-character by stable ID after rename, or when names are ambiguous within an entity label.

**Impact:** The supplied canonical identity is lost. The write can resolve by stale name, arbitrarily choose one matching candidate, or create a random-ID node. Later world upserts preserve that accidental ID/name identity, producing duplicate or misattached canon.

**Repair:** Carry validated supplied IDs for every canonical entity label and match exclusively by ID when present. Permit label-plus-normalized-name creation only when ID is absent; reject ambiguous name matches rather than using `head(collect(...))`.

**Acceptance test:** Pre-create `(:Location {id: "loc-1", name: "Old Name"})`; submit a triple naming `"Renamed Location"` with `id: "loc-1"`. Assert the edge attaches to `loc-1`, no location is created, and replay is idempotent. Add an ambiguity case that is rejected rather than chosen arbitrarily.

### P1-4 — Embedding writes/searches lack a shared vector validity contract, and a valid fallback can be skipped

**Evidence:** Scene aggregation accepts arbitrary nested vectors and computes `np.mean` without rank, dimension, numeric, or finiteness validation (`commit_graph_ops.py:72-90`). If scene aggregation raises, `commit_node.py:485-505` logs it but skips `embedding_ref` because that fallback is in an `elif` branch. Chapter storage converts arbitrary arrays/lists with `embedding_to_list()` (`data_access/chapter_queries.py:164`; `core/db_manager.py:755-770`), while vector-index DDL specifies an exact configured dimension (`core/db_manager.py:634-640`) and semantic search always calls that index (`chapter_queries.py:344-410`). Entity embedding construction is also included in the commit batch (`commit_node.py:632-641`) without a visible shared shape/finiteness gate.

**Activation:** A stale embedding model changes dimensionality, a scene artifact is ragged/2-D/non-finite, an entity embedding is malformed, or a scene artifact fails while a valid legacy `embedding_ref` exists.

**Impact:** Malformed vectors can reach Neo4j statements or fail late against the index; semantic retrieval is then unreliable. An invalid scene artifact also silently omits a usable chapter embedding instead of attempting the documented fallback.

**Repair:** Add one validator used before every vector write and query: one-dimensional, exact `NEO4J_VECTOR_DIMENSIONS`, float-convertible, and finite. Validate each scene before aggregation and the aggregate before storage. After scene aggregation failure, explicitly attempt a separately validated `embedding_ref`, then `generated_embedding`; record feature-disabled state when no valid vector remains.

**Acceptance test:** Reject rank-2/ragged, dimension-minus-one, dimension-plus-one, `NaN`, and infinite vectors before a DB update statement is emitted. Persist/retrieve one exact-dimension finite vector. With an invalid scene artifact and valid `embedding_ref`, assert the fallback becomes the chapter vector.

### P1-5 — Schema initialization reports success after required constraints/indexes fail

**Evidence:** A failed schema batch falls back to individual DDL (`core/db_manager.py:598-603`), but individual failures are only warnings (`:742-749`). `create_db_schema()` then proceeds and logs completion (`:520-535`). The global unique constraints are relied on by merge/idempotency code (`:549-567`). Vector-index creation failures are likewise warning-only (`:642-657`) even though semantic retrieval unconditionally calls the index.

**Activation:** A legacy duplicate blocks a required uniqueness constraint, one schema statement is unsupported, or the vector index cannot be created while semantic retrieval is enabled.

**Impact:** The process can continue with an unknown partial schema. Identity and idempotency guarantees become configuration-dependent, and later writes or vector context searches fail far from initialization.

**Repair:** Classify schema elements as required versus explicitly optional. After fallback, query Neo4j metadata and verify every required constraint/index before enabling persistence; raise a startup/persistence error on any absence. Treat the Chapter vector index as required whenever semantic context is enabled, or explicitly disable/fallback that feature without issuing `db.index.vector.queryNodes`.

**Acceptance test:** Seed data that makes a required constraint fail and assert initialization fails before commits are accepted. Separately simulate Chapter vector-index DDL failure: semantic retrieval must either be disabled through a tested fallback or startup must fail. A successful initialization must verify the complete required schema, not merely count attempted DDL calls.

### P1-6 — Name-only, type-blind commit deduplication conflicts with canonical identity

**Evidence:** `commit_node.py:270-284` obtains one global lowercased name set across Character/Location/Event/Item. `:373-380` filters both character and world entities against it, while mappings are rebuilt only from the unfiltered remainder (`:393-427`). Relationship endpoints retain their extracted casing (`:860-887`), whereas persistence merges case-sensitively by `name` (`data_access/cypher_builders/native_builders.py:56-66,197-219`).

**Activation:** A graph already contains `Character("Alice")` and extraction uses `"alice"`, or a Character and a Location share one display name.

**Impact:** `"alice"` is suppressed without receiving the existing canonical name/ID mapping; relationship persistence can then create a second lowercase entity. A valid Character can also be suppressed because a world entity has the same name. This does not preserve the type-plus-stable-ID identity contracts exposed by character/world reads.

**Repair:** Replace the set with a project-scoped canonical map keyed by `(canonical label, normalized name)` and returning canonical display name plus stable ID. Use it to resolve relationship endpoints. Deduplicate an extraction batch by label plus normalized name, never name alone.

**Acceptance test:** Seed `Character("Alice")` and `Location("Alex")`. Commit `character "alice"` related to Bob and a distinct `Character("Alex")`. Assert no lowercase duplicate, the first edge resolves to `Alice`, and the Character is not suppressed by the Location.

## Coverage gaps and required follow-up evidence

1. The commit-path tests use a fake manager that records statements rather than executing Neo4j/APOC semantics. They do not prove statement order, `apoc.merge.relationship` on-match behavior, constraints, rollback, or vector-index behavior.
2. No observed test exercises two projects in one Neo4j database. Project checkpoint/thread separation is not graph isolation.
3. No observed test makes a previously committed chapter fail before batch construction, then proves its prior graph state survives. No test proves post-commit cache failure semantics.
4. No observed test covers the profile-relationship channel together with the later chapter-wide delete, or repeated triples across two chapters followed by revision.
5. Semantic-context tests need executable coverage of both `include_provisional` values against vector results and the immediate predecessor, not just mocked records.
6. Vector tests need configured-dimension, rank, numeric, finite-value, aggregation, fallback, vector-index-availability, and real retrieval coverage. Existing small fake vectors are not evidence of the production index contract.
7. Generic triple tests need renamed/ambiguous non-character ID cases and an assertion that every supplied stable ID survives parameter construction.
8. Healing needs project-scoped integration coverage for candidate discovery, merge, orphan cleanup, and cache invalidation. LLM enrichment was intentionally not exercised in this audit.
9. The reviewed commit split helpers are untracked in this snapshot. Before release, verify they are deliberately versioned and run the targeted suite from a clean checkout against a disposable Neo4j database.

## Review limits

This was a source audit of the requested persistence boundary only. It did not review migration/backfill design, every legacy write path, Neo4j/APOC version compatibility, real vector-provider output, concurrent writers, or production data. Those need an integration test database and a source-stable snapshot before any approval decision.
