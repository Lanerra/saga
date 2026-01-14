# tests/core/test_relationship_canonicalization.py
"""Test strict canonical relationship enforcement."""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from core.relationship_normalization_service import RelationshipNormalizationService
from models.kg_constants import (
    PROPERTY_RELATIONSHIPS,
    RELATIONSHIP_TYPES,
    STATIC_RELATIONSHIP_MAP,
)


@pytest.fixture
def service():
    # Reload config to ensure new settings are available
    import config
    config.reload()
    return RelationshipNormalizationService()


@pytest.mark.asyncio
async def test_map_to_canonical_exact_match(service):
    """Test exact match in canonical relationship types."""
    result = await service.map_to_canonical("LOVES")
    
    assert result == ("LOVES", False, 1.0, False)


@pytest.mark.asyncio
async def test_map_to_canonical_case_variant(service):
    """Test case variant normalization."""
    result = await service.map_to_canonical("loves")
    
    assert result == ("LOVES", True, 1.0, False)


@pytest.mark.asyncio
async def test_map_to_canonical_punctuation_variant(service):
    """Test punctuation variant normalization."""
    # LOVES-WITH canonicalizes to LOVES_WITH which doesn't exist in RELATIONSHIP_TYPES
    # and has no static override, so it should be rejected in strict mode
    result = await service.map_to_canonical("LOVES-WITH")
    
    assert result == (None, False, 0.0, False)


@pytest.mark.asyncio
async def test_map_to_canonical_static_override(service):
    """Test static relationship mapping."""
    # HATES should map to CONFLICTS_WITH
    result = await service.map_to_canonical("HATES")
    
    assert result == ("CONFLICTS_WITH", True, 1.0, False)


@pytest.mark.asyncio
async def test_map_to_canonical_property_relationship(service):
    """Test property relationship detection."""
    # HAS_STATUS should be detected as a property
    result = await service.map_to_canonical("HAS_STATUS")
    
    assert result == (None, False, 0.0, True)


@pytest.mark.asyncio
async def test_map_to_canonical_reject_unknown_strict_mode(service):
    """Test rejection of unknown relationship types in strict mode."""
    # UNKNOWNS_REL should be rejected in strict mode
    result = await service.map_to_canonical("UNKNOWNS_REL")
    
    assert result == (None, False, 0.0, False)
    # Should be cached for future rejection
    assert "UNKNOWNS_REL" in service.rejected_cache


@pytest.mark.asyncio
async def test_map_to_canonical_semantic_match_non_strict(service):
    """Test semantic similarity matching when not in strict mode."""
    # Temporarily disable strict mode for this test
    with patch("config.REL_NORM_STRICT_CANONICAL_MODE", False):
        # Pre-populate canonical embeddings
        service.canonical_embeddings["WORKS_WITH"] = np.array([1.0, 0.0, 0.0])
        
        with patch(
            "core.relationship_normalization_service.llm_service.async_get_embedding",
            new_callable=AsyncMock,
        ) as mock_embed:
            # COLLABORATES_WITH should be similar to WORKS_WITH
            mock_embed.return_value = np.array([0.9, 0.1, 0.0])
            
            result = await service.map_to_canonical("COLLABORATES_WITH")
            
            # Should match WORKS_WITH with high similarity
            assert result[0] == "WORKS_WITH"
            assert result[1] is True  # was_normalized
            assert result[2] > 0.75  # similarity threshold
            assert result[3] is False  # not a property


@pytest.mark.asyncio
async def test_map_to_canonical_reject_low_similarity(service):
    """Test rejection when similarity is below threshold."""
    with patch("config.REL_NORM_STRICT_CANONICAL_MODE", False):
        # Pre-populate canonical embeddings
        service.canonical_embeddings["WORKS_WITH"] = np.array([1.0, 0.0, 0.0])
        
        with patch(
            "core.relationship_normalization_service.llm_service.async_get_embedding",
            new_callable=AsyncMock,
        ) as mock_embed:
            # Use a non-canonical relationship type
            # COOPERATES_WITH should be orthogonal to WORKS_WITH
            mock_embed.return_value = np.array([0.0, 1.0, 0.0])
            
            result = await service.map_to_canonical("COOPERATES_WITH")
            
            # Should be rejected due to low similarity
            assert result == (None, False, 0.0, False)
            assert "COOPERATES_WITH" in service.rejected_cache


@pytest.mark.asyncio
async def test_map_to_canonical_category_thresholds(service):
    """Test category-specific similarity thresholds."""
    with patch("config.REL_NORM_STRICT_CANONICAL_MODE", False):
        # Pre-populate canonical embeddings
        service.canonical_embeddings["KNOWS_ABOUT"] = np.array([1.0, 0.0, 0.0])
        
        with patch(
            "core.relationship_normalization_service.llm_service.async_get_embedding",
            new_callable=AsyncMock,
        ) as mock_embed:
            # AWARE_OF should be similar to KNOWS_ABOUT but below default threshold
            # CHARACTER_WORLD threshold is 0.70, which should accept it
            mock_embed.return_value = np.array([0.65, 0.2, 0.1])  # similarity ~0.68
            
            result = await service.map_to_canonical("AWARE_OF", "CHARACTER_WORLD")
            
            # Should match KNOWS_ABOUT with CHARACTER_WORLD threshold
            assert result[0] == "KNOWS_ABOUT"
            assert result[1] is True
            assert result[2] > 0.65


@pytest.mark.asyncio
async def test_map_to_canonical_rejection_cache(service):
    """Test that rejected relationships are cached."""
    # First call should reject and cache
    result1 = await service.map_to_canonical("UNKNOWN_REL")
    assert result1 == (None, False, 0.0, False)
    
    # Second call should use cache
    result2 = await service.map_to_canonical("UNKNOWN_REL")
    assert result2 == (None, False, 0.0, False)
    
    # Should only be in cache once
    assert "UNKNOWN_REL" in service.rejected_cache


@pytest.mark.asyncio
async def test_map_to_canonical_static_overrides_disabled(service):
    """Test behavior when static overrides are disabled."""
    with patch("config.REL_NORM_STATIC_OVERRIDES_ENABLED", False):
        # HATES should not map to CONFLICTS_WITH when static overrides disabled
        result = await service.map_to_canonical("HATES")
        
        # Should be rejected since HATES is not in RELATIONSHIP_TYPES
        assert result == (None, False, 0.0, False)


@pytest.mark.asyncio
async def test_normalize_relationship_type_strict_mode(service):
    """Test normalize_relationship_type uses map_to_canonical in strict mode."""
    with patch("config.REL_NORM_STRICT_CANONICAL_MODE", True):
        # Mock map_to_canonical to return a known result
        with patch.object(service, "map_to_canonical", new_callable=AsyncMock) as mock_map:
            mock_map.return_value = ("FRIENDS_WITH", True, 0.95, False)
            
            result = await service.normalize_relationship_type(
                "FRIENDS_WITH",
                "They are friends",
                {},
                1
            )
            
            assert result == ("FRIENDS_WITH", True, 0.95)
            mock_map.assert_called_once_with("FRIENDS_WITH")


@pytest.mark.asyncio
async def test_normalize_relationship_type_strict_mode_rejection(service):
    """Test normalize_relationship_type handles rejections in strict mode."""
    with patch("config.REL_NORM_STRICT_CANONICAL_MODE", True):
        # Mock map_to_canonical to return None (rejected)
        with patch.object(service, "map_to_canonical", new_callable=AsyncMock) as mock_map:
            mock_map.return_value = (None, False, 0.5, False)
            
            result = await service.normalize_relationship_type(
                "UNKNOWN_REL",
                "Some description",
                {},
                1
            )
            
            # Should return original type but mark as not normalized
            assert result == ("UNKNOWN_REL", False, 0.0)


@pytest.mark.asyncio
async def test_normalize_relationship_type_strict_mode_property(service):
    """Test normalize_relationship_type handles property conversions in strict mode."""
    with patch("config.REL_NORM_STRICT_CANONICAL_MODE", True):
        # Mock map_to_canonical to return property detection
        with patch.object(service, "map_to_canonical", new_callable=AsyncMock) as mock_map:
            mock_map.return_value = (None, False, 0.0, True)
            
            result = await service.normalize_relationship_type(
                "HAS_STATUS",
                "Status description",
                {},
                1
            )
            
            # Should return original type but mark as not normalized
            assert result == ("HAS_STATUS", False, 0.0)


@pytest.mark.asyncio
async def test_normalize_relationship_type_legacy_mode(service):
    """Test that legacy mode still works when strict mode is disabled."""
    with patch("config.REL_NORM_STRICT_CANONICAL_MODE", False):
        vocabulary = {"FRIENDS_WITH": {"canonical_type": "FRIENDS_WITH"}}
        
        result = await service.normalize_relationship_type(
            "FRIENDS_WITH",
            "They are friends",
            vocabulary,
            1
        )
        
        # Should work with legacy logic
        assert result == ("FRIENDS_WITH", False, 1.0)


def test_constants_defined():
    """Test that all required constants are defined."""
    # Test RELATIONSHIP_TYPES is populated
    assert len(RELATIONSHIP_TYPES) > 0
    assert "LOVES" in RELATIONSHIP_TYPES
    
    # Test STATIC_RELATIONSHIP_MAP is populated
    assert len(STATIC_RELATIONSHIP_MAP) > 0
    assert "HATES" in STATIC_RELATIONSHIP_MAP
    assert STATIC_RELATIONSHIP_MAP["HATES"] == "CONFLICTS_WITH"
    
    # Test PROPERTY_RELATIONSHIPS is populated
    assert len(PROPERTY_RELATIONSHIPS) > 0
    assert "HAS_STATUS" in PROPERTY_RELATIONSHIPS


def test_threshold_getter(service):
    """Test the _get_threshold method."""
    # Default threshold
    threshold = service._get_threshold("DEFAULT")
    assert threshold == 0.75
    
    # Character-Character threshold
    threshold = service._get_threshold("CHARACTER_CHARACTER")
    assert threshold == 0.75
    
    # Character-World threshold
    threshold = service._get_threshold("CHARACTER_WORLD")
    assert threshold == 0.70
    
    # Plot structure threshold
    threshold = service._get_threshold("PLOT_STRUCTURE")
    assert threshold == 0.80
    
    # Unknown category should fall back to default
    threshold = service._get_threshold("UNKNOWN_CATEGORY")
    assert threshold == 0.75