"""Extended tests for data_access/kg_queries.py to improve coverage."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from data_access import kg_queries
from models.kg_constants import KG_REL_CHAPTER_ADDED, KG_IS_PROVISIONAL

@pytest.mark.asyncio
class TestKGBatchOperationsExtended:
    """Extended tests for batch KG operations."""

    async def test_add_kg_triples_batch_invalid_inputs(self, monkeypatch):
        """Test adding batch with various invalid inputs."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        # 1. Missing subject name
        triples_missing_subj = [{"subject": {}, "predicate": "REL", "object_literal": "Val"}]
        await kg_queries.add_kg_triples_batch_to_db(triples_missing_subj, 1, False)
        assert mock_execute.call_count == 0

        # 2. Missing predicate
        triples_missing_pred = [{"subject": {"name": "S"}, "predicate": "", "object_literal": "Val"}]
        await kg_queries.add_kg_triples_batch_to_db(triples_missing_pred, 1, False)
        assert mock_execute.call_count == 0

        # 3. Literal object None
        triples_none_lit = [{"subject": {"name": "S"}, "predicate": "REL", "object_literal": None, "is_literal_object": True}]
        await kg_queries.add_kg_triples_batch_to_db(triples_none_lit, 1, False)
        assert mock_execute.call_count == 0
        
        # 4. Entity object invalid
        triples_invalid_obj = [{"subject": {"name": "S"}, "predicate": "REL", "object_entity": {"name": ""}}]
        await kg_queries.add_kg_triples_batch_to_db(triples_invalid_obj, 1, False)
        assert mock_execute.call_count == 0

    async def test_add_kg_triples_batch_literal_logic(self, monkeypatch):
        """Test adding batch with literal objects (ValueNode logic)."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        triples = [
            {
                "subject": {"name": "Alice", "type": "Character"},
                "predicate": "HAS_AGE",
                "object_literal": 30,
                "is_literal_object": True
            }
        ]

        await kg_queries.add_kg_triples_batch_to_db(triples, 1, False)
        
        assert mock_execute.call_count == 1
        args, _ = mock_execute.call_args
        statements = args[0]
        assert len(statements) == 1
        query, params = statements[0]
        
        assert "MERGE (o:ValueNode" in query
        assert params["object_literal_value_param"] == "30"
        assert params["subject_name_param"] == "Alice"

    async def test_add_kg_triples_batch_entity_logic(self, monkeypatch):
        """Test adding batch with entity objects (Node merging logic)."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(
            kg_queries.neo4j_manager, "execute_cypher_batch", mock_execute
        )

        triples = [
            {
                "subject": {"name": "Alice", "type": "Character"},
                "predicate": "LIVES_IN",
                "object_entity": {"name": "Wonderland", "type": "Location"}
            }
        ]

        await kg_queries.add_kg_triples_batch_to_db(triples, 1, False)
        
        assert mock_execute.call_count == 1
        args, _ = mock_execute.call_args
        statements = args[0]
        query, params = statements[0]
        
        # Verify constraint-safe MERGE logic used
        assert "MERGE (s:Character" in query or "MERGE (s:Entity" in query 
        assert "MERGE (o:Location" in query or "MERGE (o:Entity" in query
        assert params["object_name_param"] == "Wonderland"

@pytest.mark.asyncio
class TestKGQueriesExtended:
    """Extended tests for KG query functions."""

    async def test_get_most_recent_value_types(self, monkeypatch):
        """Test value type conversion in get_most_recent_value_from_db."""
        
        async def mock_read(query, params):
            if params["subject_param"] == "IntSubj":
                return [{"object": "42", KG_REL_CHAPTER_ADDED: 1}]
            elif params["subject_param"] == "FloatSubj":
                return [{"object": "3.14", KG_REL_CHAPTER_ADDED: 1}]
            elif params["subject_param"] == "BoolTrue":
                return [{"object": "true", KG_REL_CHAPTER_ADDED: 1}]
            elif params["subject_param"] == "BoolFalse":
                return [{"object": "false", KG_REL_CHAPTER_ADDED: 1}]
            return []

        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        # Test Integer
        val = await kg_queries.get_most_recent_value_from_db("IntSubj", "PROP")
        assert val == 42
        assert isinstance(val, int)

        # Test Float
        val = await kg_queries.get_most_recent_value_from_db("FloatSubj", "PROP")
        assert val == 3.14
        assert isinstance(val, float)

        # Test Boolean
        val = await kg_queries.get_most_recent_value_from_db("BoolTrue", "PROP")
        assert val is True
        val = await kg_queries.get_most_recent_value_from_db("BoolFalse", "PROP")
        assert val is False

    async def test_find_candidate_duplicate_entities_logic(self, monkeypatch):
        """Test candidate duplicate finding with specific thresholds."""
        mock_read = AsyncMock(return_value=[
            {"id1": "1", "name1": "Alice", "id2": "2", "name2": "Alicia", "similarity": 0.9}
        ])
        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_read_query", mock_read)

        results = await kg_queries.find_candidate_duplicate_entities(
            similarity_threshold=0.8,
            limit=10,
            desc_threshold=0.5
        )
        
        assert len(results) == 1
        assert mock_read.call_count == 1
        args, kwargs = mock_read.call_args
        params = args[1] if len(args) > 1 else kwargs.get("params", {}) # Handle positional or keyword args for params
        # Note: execute_read_query signature is (query, params=None)
        # Verify params passed correctly
        call_params = args[1]
        assert call_params["name_threshold"] == 0.8
        assert call_params["desc_threshold"] == 0.5

@pytest.mark.asyncio
class TestMergeEntitiesExtended:
    """Tests for entity merging logic."""

    async def test_merge_entities_success(self, monkeypatch):
        """Test successful atomic merge."""
        # Mock successful write queries for all steps
        # Step 1: Copy Props (returns props_copied)
        # Step 2: Move Outgoing (returns outgoing_moved)
        # Step 3: Move Incoming (returns incoming_moved)
        # Step 4: Delete Source (returns deleted)
        
        async def mock_write(query, params):
            if "SET target.description" in query:
                return [{"props_copied": 1}]
            if "MATCH (source)-[r]->(other)" in query:
                return [{"outgoing_moved": 2}]
            if "MATCH (other)-[r]->(source)" in query:
                return [{"incoming_moved": 1}]
            if "DETACH DELETE source" in query:
                return [{"deleted": 1}]
            return []

        monkeypatch.setattr(kg_queries.neo4j_manager, "execute_write_query", mock_write)

        result = await kg_queries.merge_entities("source_id", "target_id", "duplicate")
        assert result is True

    async def test_merge_entities_retry_logic(self, monkeypatch):
        """Test merge retry logic on failure."""
        # Fail twice, succeed on third
        # Error must contain "deadlock", "locked", "transaction", or "entitynotfound" to trigger retry
        mock_execute_atomic = AsyncMock(side_effect=[
            Exception("Deadlock detected"),
            Exception("Database is locked"),
            True
        ])
        
        # We need to patch _execute_atomic_merge directly if we want to test the retry wrapper logic specifically,
        # OR patch neo4j_manager to fail then succeed.
        # Patching the private method is easier to verify retry count.
        with patch("data_access.kg_queries._execute_atomic_merge", side_effect=[
            Exception("Deadlock detected"),
            Exception("Database is locked"),
            True
        ]) as mock_atomic:
             result = await kg_queries.merge_entities("source_id", "target_id", "duplicate", max_retries=3)
             assert result is True
             assert mock_atomic.call_count == 3
