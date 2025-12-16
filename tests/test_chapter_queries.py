"""Tests for data_access/chapter_queries.py"""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from core.exceptions import DatabaseError
from data_access import chapter_queries


@pytest.mark.asyncio
class TestLoadChapterCount:
    """Tests for loading chapter count."""

    async def test_load_chapter_count_zero(self, monkeypatch):
        """Test loading chapter count when zero."""
        mock_read = AsyncMock(return_value=[{"chapter_count": 0}])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.load_chapter_count_from_db()
        assert result == 0

    async def test_load_chapter_count_multiple(self, monkeypatch):
        """Test loading chapter count when multiple chapters exist."""
        mock_read = AsyncMock(return_value=[{"chapter_count": 5}])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.load_chapter_count_from_db()
        assert result >= 0

    async def test_load_chapter_count_no_result(self, monkeypatch):
        """Test loading chapter count when no result."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.load_chapter_count_from_db()
        assert result == 0


@pytest.mark.asyncio
class TestSaveChapterData:
    """Tests for saving chapter data."""

    async def test_save_chapter_data_basic(self, monkeypatch):
        """Test saving basic chapter data (must always set Chapter.id)."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_write_query", mock_execute)

        await chapter_queries.save_chapter_data_to_db(
            chapter_number=1,
            summary="Chapter summary",
            embedding_array=None,
        )
        mock_execute.assert_called_once()

        # Canonical Chapter persistence must always set Chapter.id.
        call_args = mock_execute.call_args
        query = call_args.args[0]
        params = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("parameters", {})

        assert "c.id" in query
        assert params["chapter_number_param"] == 1
        assert params["chapter_id_param"] == chapter_queries.compute_chapter_id(1)
        assert params["summary_param"] == "Chapter summary"
        assert params["embedding_vector_param"] is None
        assert params["is_provisional_param"] is False

    async def test_save_chapter_data_with_embedding(self, monkeypatch):
        """Test saving chapter data with embedding (must always set Chapter.id)."""
        mock_execute = AsyncMock(return_value=None)
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_write_query", mock_execute)

        embedding = np.array([0.1, 0.2, 0.3])
        await chapter_queries.save_chapter_data_to_db(
            chapter_number=1,
            summary="Chapter summary",
            embedding_array=embedding,
        )
        mock_execute.assert_called_once()

        call_args = mock_execute.call_args
        query = call_args.args[0]
        params = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("parameters", {})

        assert "c.id" in query
        assert params["chapter_number_param"] == 1
        assert params["chapter_id_param"] == chapter_queries.compute_chapter_id(1)
        # Neo4j embedding conversion uses float32; allow minor float representation variance.
        assert params["embedding_vector_param"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6, abs=1e-6)


@pytest.mark.asyncio
class TestGetChapterData:
    """Tests for getting chapter data."""

    async def test_get_chapter_data_found(self, monkeypatch):
        """Test getting chapter data when found."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "summary": "Chapter summary",
                    "is_provisional": False,
                }
            ]
        )
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_chapter_data_from_db(1)
        assert result is not None
        assert result["summary"] == "Chapter summary"
        assert result["is_provisional"] is False

    async def test_get_chapter_data_not_found(self, monkeypatch):
        """Test getting chapter data when not found."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_chapter_data_from_db(999)
        assert result is None

    async def test_get_chapter_data_raises_database_error_on_db_failure(self, monkeypatch):
        """P1.9: DB failures should raise standardized DatabaseError (not return None)."""
        mock_read = AsyncMock(side_effect=Exception("connection refused"))
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        with pytest.raises(DatabaseError):
            await chapter_queries.get_chapter_data_from_db(1)


@pytest.mark.asyncio
class TestGetEmbedding:
    """Tests for getting chapter embedding."""

    async def test_get_embedding_found(self, monkeypatch):
        """Test getting embedding when found."""
        embedding_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_read = AsyncMock(return_value=[{"embedding_vector": embedding_list}])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_embedding_from_db(1)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    async def test_get_embedding_not_found(self, monkeypatch):
        """Test getting embedding when not found."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_embedding_from_db(999)
        assert result is None

    async def test_get_embedding_null(self, monkeypatch):
        """Test getting embedding when null."""
        mock_read = AsyncMock(return_value=[{"embedding_vector": None}])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_embedding_from_db(1)
        assert result is None


@pytest.mark.asyncio
class TestFindSemanticContext:
    """Tests for finding semantic context."""

    async def test_find_semantic_context_found(self, monkeypatch):
        """Test finding semantic context when found (correct return shape + deterministic contract)."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "context_chapters": [
                        {
                            "chapter_number": 2,
                            "summary": "Related chapter",
                            "score": 0.9,
                            "context_type": "similarity",
                            "is_provisional": False,
                        }
                    ]
                }
            ]
        )
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        query_embedding = np.array([0.1, 0.2, 0.3])
        result = await chapter_queries.find_semantic_context_native(query_embedding, current_chapter_number=5, limit=3)
        assert isinstance(result, list)
        assert result[0]["chapter_number"] == 2

    async def test_find_semantic_context_empty(self, monkeypatch):
        """Test finding semantic context when empty."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        query_embedding = np.array([0.1, 0.2, 0.3])
        result = await chapter_queries.find_semantic_context_native(query_embedding, current_chapter_number=5, limit=3)
        assert isinstance(result, list)
        assert len(result) == 0

    async def test_find_semantic_context_enforces_limit_and_order(self, monkeypatch):
        """Enforces P1.8: score-desc ordering + strict output limit."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "context_chapters": [
                        {"chapter_number": 10, "summary": "a", "score": 0.1},
                        {"chapter_number": 11, "summary": "b", "score": 0.9},
                        {"chapter_number": 12, "summary": "c", "score": 0.5},
                        {"chapter_number": 13, "summary": "d", "score": 0.8},
                    ]
                }
            ]
        )
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        query_embedding = np.array([0.1, 0.2, 0.3])
        result = await chapter_queries.find_semantic_context_native(query_embedding, current_chapter_number=20, limit=2)

        assert [r["chapter_number"] for r in result] == [11, 13]
        assert len(result) == 2


@pytest.mark.asyncio
class TestGetChapterContentBatch:
    """Tests for getting chapter content in batch."""

    async def test_get_chapter_content_batch_empty(self, monkeypatch):
        """Test getting chapter content batch when empty."""
        mock_read = AsyncMock(return_value=[])
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_chapter_content_batch_native([])
        assert isinstance(result, dict)
        assert len(result) == 0

    async def test_get_chapter_content_batch_single(self, monkeypatch):
        """Test getting chapter content batch with single chapter."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "chapter_number": 1,
                    "summary": "Chapter 1 summary",
                    "is_provisional": False,
                }
            ]
        )
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_chapter_content_batch_native([1])
        assert len(result) == 1
        assert 1 in result
        assert result[1]["summary"] == "Chapter 1 summary"

    async def test_get_chapter_content_batch_multiple(self, monkeypatch):
        """Test getting chapter content batch with multiple chapters."""
        mock_read = AsyncMock(
            return_value=[
                {
                    "chapter_number": 1,
                    "summary": "Chapter 1 summary",
                    "is_provisional": False,
                },
                {
                    "chapter_number": 2,
                    "summary": "Chapter 2 summary",
                    "is_provisional": False,
                },
            ]
        )
        monkeypatch.setattr(chapter_queries.neo4j_manager, "execute_read_query", mock_read)

        result = await chapter_queries.get_chapter_content_batch_native([1, 2])
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
