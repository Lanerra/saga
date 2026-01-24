# Phase 5: Testing - Comprehensive Test Plan

## Overview

This document outlines the comprehensive test plan for Phase 5: Testing based on the SAGA Unified Knowledge Graph Ontology specification in `docs/schema-design.md`.

## Test Categories

### 1. Unit Tests for Parsers

#### 1.1 CharacterSheetParser Tests
**File:** `tests/test_character_sheet_parser.py`
**Status:** ✅ COMPLETE (201 lines, comprehensive coverage)

**Coverage:**
- ✅ Parse character sheets JSON
- ✅ Create character nodes in Neo4j
- ✅ Create character relationships
- ✅ Handle missing/invalid data
- ✅ Error handling for file operations
- ✅ Validation of character properties
- ✅ Bidirectional relationship creation

**Missing Tests:**
- None identified - this test file is comprehensive

---

#### 1.2 GlobalOutlineParser Tests
**File:** `tests/test_global_outline_parser.py`
**Status:** ✅ COMPLETE (375 lines, comprehensive coverage)

**Coverage:**
- ✅ Parse global outline JSON
- ✅ Create MajorPlotPoint nodes (4 required)
- ✅ Parse character arcs
- ✅ Parse locations and items
- ✅ Enrich character arcs with arc_start/arc_end/arc_key_moments
- ✅ Error handling for file operations
- ✅ Validation of major plot points (exactly 4)
- ✅ Sequence order validation (1-4)

**Missing Tests:**
- None identified - this test file is comprehensive

---

#### 1.3 ActOutlineParser Tests
**File:** `tests/test_act_outline_parser.py`
**Status:** ⚠️ PARTIAL (100+ lines, needs verification)

**Coverage:**
- ✅ Parse act outline JSON
- ✅ Create ActKeyEvent nodes
- ✅ Parse locations with names
- ✅ Create relationships: PART_OF, HAPPENS_BEFORE, INVOLVES, OCCURS_AT, FEATURES_ITEM
- ✅ Validation of act structure
- ✅ Error handling

**Missing Tests:**
- Need to verify complete coverage of all ActKeyEvent properties
- Need to verify relationship cardinality tests
- Need to verify event temporal ordering validation

---

#### 1.4 ChapterOutlineParser Tests
**File:** `tests/test_chapter_outline_parser.py`
**Status:** ⚠️ PARTIAL (100+ lines, needs verification)

**Coverage:**
- ✅ Parse chapter outline JSON
- ✅ Create Chapter nodes
- ✅ Create Scene nodes
- ✅ Create SceneEvent nodes
- ✅ Create relationships: PART_OF, FOLLOWS, FEATURES_CHARACTER, OCCURS_AT, OCCURS_IN_SCENE
- ✅ Validation of scene indices (contiguous)
- ✅ Validation of POV character existence

**Missing Tests:**
- Need to verify complete coverage of all Scene properties
- Need to verify complete coverage of all Chapter properties
- Need to verify scene event validation
- Need to verify chapter-scene relationship validation

---

#### 1.5 NarrativeEnrichmentParser Tests
**File:** `tests/test_narrative_enrichment_parser.py`
**Status:** ⚠️ PARTIAL (100+ lines, needs verification)

**Coverage:**
- ✅ Extract physical descriptions from narrative
- ✅ Extract chapter embeddings
- ✅ Enrich character nodes with physical_description
- ✅ Enrich chapter nodes with embeddings
- ✅ Validation of no new structural entities
- ✅ Validation of character name matching

**Missing Tests:**
- Need to verify complete coverage of enrichment logic
- Need to verify validation that enrichments don't contradict existing properties
- Need to verify embedding dimensionality validation

---

### 2. Integration Tests for Full Pipeline

#### 2.1 Stage 1: Character Initialization
**Test File:** `tests/test_langgraph/test_character_sheets_node.py`
**Status:** ✅ COMPLETE

**Coverage:**
- ✅ Full character initialization pipeline
- ✅ Character node creation with all required properties
- ✅ Relationship creation between characters
- ✅ Validation of unique character names
- ✅ Validation of relationship types

**Missing Tests:**
- None identified

---

#### 2.2 Stage 2: Global Outline Processing
**Test File:** `tests/test_langgraph/test_global_outline_node.py`
**Status:** ✅ COMPLETE

**Coverage:**
- ✅ MajorPlotPoint creation (4 required)
- ✅ Location creation without names (Stage 2)
- ✅ Item creation
- ✅ Character arc enrichment
- ✅ Validation of exactly 4 MajorPlotPoints
- ✅ Validation of sequence orders (1-4)

**Missing Tests:**
- None identified

---

#### 2.3 Stage 3: Act Outlines Processing
**Test File:** `tests/test_langgraph/test_act_outlines_node.py`
**Status:** ✅ COMPLETE

**Coverage:**
- ✅ ActKeyEvent creation
- ✅ Location name enrichment
- ✅ Relationship creation: PART_OF, HAPPENS_BEFORE, INVOLVES, OCCURS_AT, FEATURES_ITEM
- ✅ Validation of act structure
- ✅ Validation of event temporal ordering

**Missing Tests:**
- None identified

---

#### 2.4 Stage 4: Chapter Outlines Processing
**Test File:** `tests/test_langgraph/test_chapter_outline_node.py`
**Status:** ✅ COMPLETE

**Coverage:**
- ✅ Chapter node creation
- ✅ Scene node creation
- ✅ SceneEvent node creation
- ✅ Relationship creation: PART_OF, FOLLOWS, FEATURES_CHARACTER, OCCURS_AT, OCCURS_IN_SCENE
- ✅ Validation of scene indices (contiguous)
- ✅ Validation of POV character existence
- ✅ Validation of chapter-scene relationships

**Missing Tests:**
- None identified

---

#### 2.5 Stage 5: Narrative Generation & Enrichment
**Test File:** `tests/test_langgraph/test_narrative_enrichment_node.py`
**Status:** ❌ MISSING

**Coverage Needed:**
- ✅ Character enrichment with physical_description
- ✅ Chapter enrichment with embeddings
- ✅ Validation of no new structural entities
- ✅ Validation of character name matching
- ✅ Validation of no contradictions in enrichment

**Action Required:**
- Create `tests/test_langgraph/test_narrative_enrichment_node.py`
- Add comprehensive tests for Stage 5 pipeline

---

### 3. Validation Tests for Graph Structure

#### 3.1 Node Property Validation Tests
**Test Files:** `tests/test_*_queries.py`
**Status:** ✅ COMPLETE

**Coverage:**
- ✅ Character property validation
- ✅ Event property validation
- ✅ Location property validation
- ✅ Item property validation
- ✅ Scene property validation
- ✅ Chapter property validation

**Missing Tests:**
- None identified

---

#### 3.2 Relationship Validation Tests
**Test Files:** `tests/test_relationship_validation.py`
**Status:** ✅ COMPLETE

**Coverage:**
- ✅ Character-Character relationships (20 types)
- ✅ Event relationships (PART_OF, HAPPENS_BEFORE, INVOLVES, OCCURS_AT, OCCURS_IN_SCENE)
- ✅ Scene relationships (FOLLOWS, FEATURES_CHARACTER, OCCURS_AT, PART_OF)
- ✅ Item relationships (POSSESSES, FEATURES_ITEM)
- ✅ Validation of relationship cardinality
- ✅ Validation of relationship properties

**Missing Tests:**
- None identified

---

#### 3.3 Orphaned Node Detection Tests
**Test Files:** `tests/test_orphaned_nodes.py`
**Status:** ❌ MISSING

**Coverage Needed:**
- ✅ Detect orphaned Character nodes (no relationships)
- ✅ Detect orphaned Event nodes (no relationships)
- ✅ Detect orphaned Location nodes (no relationships)
- ✅ Detect orphaned Item nodes (no relationships)
- ✅ Detect orphaned Scene nodes (no relationships)
- ✅ Detect orphaned Chapter nodes (no relationships)

**Action Required:**
- Create `tests/test_orphaned_nodes.py`
- Add comprehensive tests for orphaned node detection

---

### 4. Query Pattern Tests

#### 4.1 Common Query Tests
**Test Files:** `tests/test_*_queries.py`
**Status:** ✅ COMPLETE

**Coverage:**
- ✅ Get all characters in a chapter
- ✅ Get chapter's scenes in order
- ✅ Get character's POV chapters
- ✅ Get all events in an act
- ✅ Get scene's events
- ✅ Get event hierarchy
- ✅ Get character relationships
- ✅ Find scenes at a location
- ✅ Get character's possessed items
- ✅ Find items featured in events
- ✅ Track item through story

**Missing Tests:**
- None identified

---

### 5. Obsolete Test Removal

#### 5.1 Tests to Remove or Update

**Based on Phase 4: Aggressively Remove Legacy Extraction**

**Tests to Remove:**
1. `tests/test_phase2_deduplication.py` - Deduplication is no longer needed
2. `tests/test_relationship_canonicalization.py` - Relationships are canonical from init
3. `tests/test_extraction_contentref.py` - Heavy post-narrative extraction deprecated
4. `tests/test_kg_relationship_growth.py` - Relationship growth tests (if testing old logic)

**Tests to Update:**
1. `tests/test_character_labeling.py` - Update for new schema
2. `tests/test_character_queries_extended.py` - Update for new schema
3. `tests/test_kg_queries_extended.py` - Update for new schema
4. `tests/test_world_queries_extended.py` - Update for new schema

---

## Test Execution Strategy

### 1. Unit Test Execution

Run all parser unit tests:
```bash
pytest tests/test_*_parser.py -v
```

### 2. Integration Test Execution

Run all integration tests:
```bash
pytest tests/test_langgraph/ -v
```

### 3. Validation Test Execution

Run all validation tests:
```bash
pytest tests/test_*_validation.py -v
pytest tests/test_*_queries.py -v
```

### 4. Full Test Suite Execution

Run complete test suite:
```bash
pytest tests/ -v --tb=short
```

---

## Test Coverage Summary

### ✅ Complete Coverage Areas
1. CharacterSheetParser - Complete
2. GlobalOutlineParser - Complete
3. Character Initialization (Stage 1) - Complete
4. Global Outline Processing (Stage 2) - Complete
5. Act Outlines Processing (Stage 3) - Complete
6. Chapter Outlines Processing (Stage 4) - Complete
7. Node Property Validation - Complete
8. Relationship Validation - Complete
9. Query Patterns - Complete

### ⚠️ Partial Coverage Areas (Need Verification)
1. ActOutlineParser - Needs verification
2. ChapterOutlineParser - Needs verification
3. NarrativeEnrichmentParser - Needs verification
4. Narrative Enrichment (Stage 5) - Missing test file

### ❌ Missing Coverage Areas
1. Orphaned Node Detection - Missing test file
2. Stage 5 Integration Tests - Missing test file

---

## Action Items

### Immediate Actions
1. ✅ Verify ActOutlineParser test coverage
2. ✅ Verify ChapterOutlineParser test coverage  
3. ✅ Verify NarrativeEnrichmentParser test coverage
4. ✅ Create test_narrative_enrichment_node.py for Stage 5 integration
5. ✅ Create test_orphaned_nodes.py for orphaned node detection

### Medium Priority Actions
1. ⚠️ Remove obsolete tests (deduplication, canonicalization, heavy extraction)
2. ⚠️ Update extended query tests for new schema
3. ⚠️ Verify all relationship types are tested
4. ⚠️ Verify all node types have property validation tests

### Low Priority Actions
1. 📋 Add additional edge case tests
2. 📋 Add performance tests for large graphs
3. 📋 Add stress tests for concurrent operations

---

## Test Maintenance Strategy

### 1. Test Organization
- Keep tests organized by feature/parser
- Use clear naming conventions
- Maintain consistent test structure

### 2. Test Documentation
- Add docstrings to all test functions
- Document test purpose and expected behavior
- Use pytest markers for categorization

### 3. Test Execution
- Use pytest.ini for configuration
- Use conftest.py for fixtures
- Use pytest-cov for coverage reporting

### 4. Test Reporting
- Generate HTML coverage reports
- Use pytest-html for visual reports
- Integrate with CI/CD pipeline

---

## Conclusion

Based on the analysis of `docs/schema-design.md` and existing test files, the test coverage is comprehensive but needs verification in several areas and completion in others. The main gaps are:

1. **Stage 5: Narrative Enrichment** - Missing integration test file
2. **Orphaned Node Detection** - Missing test file
3. **Obsolete Test Removal** - Need to clean up legacy tests

The existing test infrastructure is solid and follows best practices. The main task is to verify existing tests and fill in the gaps.

---

## Next Steps

1. Verify test coverage for ActOutlineParser, ChapterOutlineParser, and NarrativeEnrichmentParser
2. Create missing test files for Stage 5 and orphaned node detection
3. Remove obsolete tests based on Phase 4 requirements
4. Run full test suite to verify all tests pass
5. Generate coverage reports to identify any gaps

---

## References

- `docs/schema-design.md` - Main specification document
- `tests/test_*_parser.py` - Parser unit tests
- `tests/test_langgraph/` - Integration tests
- `tests/test_*_queries.py` - Query validation tests
- `tests/test_*_validation.py` - Validation tests
