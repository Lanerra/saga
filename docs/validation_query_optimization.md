# Validation Query Optimization - MAJOR #5

## Problem Statement

The validation subgraph in `core/langgraph/subgraphs/validation.py` was performing three separate Neo4j queries during contradiction detection:
1. Query for events with timestamps (for timeline validation)
2. Query for world rules (for world rule validation)
3. Query for character relationships (for relationship evolution validation)

This resulted in **three round-trips** to the Neo4j database, causing unnecessary latency and resource usage.

## Solution

Combined all three queries into a single optimized query using Cypher's `UNION` clause. The new `_fetch_validation_data()` function retrieves all validation-related data in one database round-trip.

## Implementation Details

### New Function: `_fetch_validation_data()`

```python
async def _fetch_validation_data(current_chapter: int) -> dict[str, Any]:
    """Fetch all validation-related data from Neo4j in a single query."""
```

This function:
- Uses a UNION-based Cypher query to retrieve events, world rules, and relationships
- Returns structured data organized by type
- Handles errors gracefully with best-effort behavior
- Logs performance metrics for debugging

### Updated Helper Functions

Modified three helper functions to accept pre-fetched data:

1. **`_check_timeline()`** - Now accepts `existing_events` parameter
2. **`_check_world_rules()`** - Now accepts `existing_world_rules` parameter  
3. **`_check_relationship_evolution()`** - Now accepts `existing_relationships` parameter

All functions maintain backward compatibility by falling back to individual queries if pre-fetched data is not provided.

### Updated Main Function: `detect_contradictions()`

The main validation function now:
1. Calls `_fetch_validation_data()` once to get all data
2. Passes the pre-fetched data to each helper function
3. Eliminates three separate Neo4j queries

## Performance Impact

**Before:** 3 Neo4j round-trips per validation cycle
**After:** 1 Neo4j round-trip per validation cycle

This represents a **66% reduction** in database queries for the validation phase.

## Testing

Created comprehensive test suite in `tests/test_validation_query_optimization.py` that verifies:
- Combined query returns correct structured data
- Helper functions use pre-fetched data instead of querying
- Validation subgraph makes only one query instead of three
- All existing functionality remains intact

All tests pass successfully.

## Code Quality

- Maintains backward compatibility
- Preserves error handling and logging
- Follows existing code patterns and conventions
- Includes comprehensive documentation
- No breaking changes to public APIs

## Files Modified

1. `core/langgraph/subgraphs/validation.py` - Main implementation
2. `tests/test_validation_query_optimization.py` - Test coverage (new file)

## Verification

Run tests with:
```bash
.venv/bin/python -m pytest tests/test_validation_query_optimization.py -v
```

All 5 tests pass, confirming the optimization works correctly while maintaining all existing functionality.