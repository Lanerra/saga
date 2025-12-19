from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from core.relationship_normalization_service import RelationshipNormalizationService


@pytest.fixture
def service():
    return RelationshipNormalizationService()


@pytest.mark.asyncio
async def test_normalize_exact_match(service):
    vocabulary = {"FRIENDS_WITH": {"canonical_type": "FRIENDS_WITH"}}

    normalized, was_norm, sim = await service.normalize_relationship_type("FRIENDS_WITH", "They are friends", vocabulary, 1)

    assert normalized == "FRIENDS_WITH"
    assert was_norm is False
    assert sim == 1.0


@pytest.mark.asyncio
async def test_normalize_case_variant(service):
    vocabulary = {"FRIENDS_WITH": {"canonical_type": "FRIENDS_WITH"}}

    normalized, was_norm, sim = await service.normalize_relationship_type("friends_with", "They are friends", vocabulary, 1)

    assert normalized == "FRIENDS_WITH"
    assert was_norm is True
    assert sim == 1.0


@pytest.mark.asyncio
async def test_normalize_punctuation_variant(service):
    vocabulary = {"FRIENDS_WITH": {"canonical_type": "FRIENDS_WITH"}}

    normalized, was_norm, sim = await service.normalize_relationship_type("FRIENDS-WITH", "They are friends", vocabulary, 1)

    assert normalized == "FRIENDS_WITH"
    assert was_norm is True
    assert sim == 1.0


@pytest.mark.asyncio
async def test_normalize_semantic_similarity(service):
    vocabulary = {
        "WORKS_WITH": {
            "canonical_type": "WORKS_WITH",
            "embedding": np.array([1.0, 0.0, 0.0]),
        }
    }

    # Mock embedding for new type to be very similar
    # COLLABORATES_WITH -> [0.9, 0.1, 0.0]

    with patch(
        "core.relationship_normalization_service.llm_service.async_get_embedding",
        new_callable=AsyncMock,
    ) as mock_embed:
        # First call is for "COLLABORATES_WITH", second might be for WORKS_WITH if not in cache (but we put it in cache via dict implicitly? No, service has its own cache)
        # We need to ensure service.embedding_cache has the vocab embedding or mocks return it

        # Pre-populate service cache for vocabulary to simplify
        service.embedding_cache["WORKS_WITH"] = np.array([1.0, 0.0, 0.0])

        mock_embed.return_value = np.array([0.9, 0.1, 0.0])

        normalized, was_norm, sim = await service.normalize_relationship_type("COLLABORATES_WITH", "Working together", vocabulary, 1)

        # 0.9 / (1 * sqrt(0.82)) ~= 0.99
        assert normalized == "WORKS_WITH"
        assert was_norm is True
        assert sim > 0.85  # Default threshold


@pytest.mark.asyncio
async def test_normalize_novel_relationship(service):
    vocabulary = {
        "WORKS_WITH": {
            "canonical_type": "WORKS_WITH",
            "embedding": np.array([1.0, 0.0, 0.0]),
        }
    }

    service.embedding_cache["WORKS_WITH"] = np.array([1.0, 0.0, 0.0])

    # LOVES -> [0.0, 1.0, 0.0] -> Orthogonal, sim = 0

    with patch(
        "core.relationship_normalization_service.llm_service.async_get_embedding",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_embed.return_value = np.array([0.0, 1.0, 0.0])

        normalized, was_norm, sim = await service.normalize_relationship_type("LOVES", "Deep affection", vocabulary, 1)

        assert normalized == "LOVES"
        assert was_norm is False
        assert sim < 0.85


def test_update_vocabulary_usage(service):
    vocabulary = {}

    # First usage
    vocab = service.update_vocabulary_usage(vocabulary, "TEST_REL", "Description 1", 1, False)

    assert "TEST_REL" in vocab
    assert vocab["TEST_REL"]["usage_count"] == 1
    assert vocab["TEST_REL"]["first_used_chapter"] == 1
    assert "Description 1" in vocab["TEST_REL"]["example_descriptions"]

    # Second usage
    vocab = service.update_vocabulary_usage(vocab, "TEST_REL", "Description 2", 2, False)

    assert vocab["TEST_REL"]["usage_count"] == 2
    assert vocab["TEST_REL"]["last_used_chapter"] == 2
    assert len(vocab["TEST_REL"]["example_descriptions"]) == 2


def test_prune_vocabulary(service):
    vocabulary = {
        "KEEP_ME": {"usage_count": 10, "last_used_chapter": 10},
        "KEEP_ME_TOO": {"usage_count": 1, "last_used_chapter": 10},
        "PRUNE_ME": {
            "usage_count": 1,
            "last_used_chapter": 1,  # Old
        },
    }

    # Config defaults: prune single use after 5 chapters
    # Current chapter 10. 10 - 1 = 9 > 5 -> Prune

    pruned = service.prune_vocabulary(vocabulary, 10)

    assert "KEEP_ME" in pruned
    assert "KEEP_ME_TOO" in pruned
    assert "PRUNE_ME" not in pruned


@pytest.mark.asyncio
async def test_llm_disambiguate_json_normalize(service):
    existing_usage = {
        "usage_count": 3,
        "example_descriptions": ["Example one", "Example two"],
    }

    with patch(
        "core.relationship_normalization_service.llm_service.async_call_llm_json_object",
        new_callable=AsyncMock,
    ) as mock_call:
        mock_call.return_value = ({"decision": "NORMALIZE"}, None)

        should_normalize = await service._llm_disambiguate(
            new_type="ALLIES_WITH",
            new_description="They cooperate in secret",
            existing_type="WORKS_WITH",
            existing_usage=existing_usage,
            use_json_mode=True,
        )

    assert should_normalize is True


@pytest.mark.asyncio
async def test_llm_disambiguate_json_distinct(service):
    existing_usage = {
        "usage_count": 3,
        "example_descriptions": ["Example one", "Example two"],
    }

    with patch(
        "core.relationship_normalization_service.llm_service.async_call_llm_json_object",
        new_callable=AsyncMock,
    ) as mock_call:
        mock_call.return_value = ({"decision": "DISTINCT"}, None)

        should_normalize = await service._llm_disambiguate(
            new_type="ALLIES_WITH",
            new_description="They cooperate in secret",
            existing_type="WORKS_WITH",
            existing_usage=existing_usage,
            use_json_mode=True,
        )

    assert should_normalize is False


@pytest.mark.asyncio
async def test_llm_disambiguate_json_rejects_invalid_keyset(service):
    existing_usage = {"usage_count": 1, "example_descriptions": []}

    with patch(
        "core.relationship_normalization_service.llm_service.async_call_llm_json_object",
        new_callable=AsyncMock,
    ) as mock_call:
        mock_call.return_value = ({"decision": "NORMALIZE", "extra": "nope"}, None)

        with pytest.raises(ValueError, match=r"exactly one key.*decision"):
            await service._llm_disambiguate(
                new_type="ALLIES_WITH",
                new_description="They cooperate in secret",
                existing_type="WORKS_WITH",
                existing_usage=existing_usage,
                use_json_mode=True,
            )


@pytest.mark.asyncio
async def test_llm_disambiguate_json_rejects_missing_decision(service):
    existing_usage = {"usage_count": 1, "example_descriptions": []}

    with patch(
        "core.relationship_normalization_service.llm_service.async_call_llm_json_object",
        new_callable=AsyncMock,
    ) as mock_call:
        mock_call.return_value = ({}, None)

        with pytest.raises(ValueError, match=r"exactly one key.*decision"):
            await service._llm_disambiguate(
                new_type="ALLIES_WITH",
                new_description="They cooperate in secret",
                existing_type="WORKS_WITH",
                existing_usage=existing_usage,
                use_json_mode=True,
            )


@pytest.mark.asyncio
async def test_llm_disambiguate_json_rejects_non_string_decision(service):
    existing_usage = {"usage_count": 1, "example_descriptions": []}

    with patch(
        "core.relationship_normalization_service.llm_service.async_call_llm_json_object",
        new_callable=AsyncMock,
    ) as mock_call:
        mock_call.return_value = ({"decision": 123}, None)

        with pytest.raises(ValueError, match=r"must be a string"):
            await service._llm_disambiguate(
                new_type="ALLIES_WITH",
                new_description="They cooperate in secret",
                existing_type="WORKS_WITH",
                existing_usage=existing_usage,
                use_json_mode=True,
            )


@pytest.mark.asyncio
async def test_llm_disambiguate_json_rejects_invalid_enum(service):
    existing_usage = {"usage_count": 1, "example_descriptions": []}

    with patch(
        "core.relationship_normalization_service.llm_service.async_call_llm_json_object",
        new_callable=AsyncMock,
    ) as mock_call:
        mock_call.return_value = ({"decision": "normalize"}, None)

        with pytest.raises(ValueError, match=r"NORMALIZE.*DISTINCT"):
            await service._llm_disambiguate(
                new_type="ALLIES_WITH",
                new_description="They cooperate in secret",
                existing_type="WORKS_WITH",
                existing_usage=existing_usage,
                use_json_mode=True,
            )


@pytest.mark.asyncio
async def test_llm_disambiguate_legacy_substring_behavior(service):
    existing_usage = {"usage_count": 1, "example_descriptions": []}

    with patch(
        "core.relationship_normalization_service.llm_service.async_call_llm",
        new_callable=AsyncMock,
    ) as mock_call:
        mock_call.return_value = ("normalize", None)

        should_normalize = await service._llm_disambiguate(
            new_type="ALLIES_WITH",
            new_description="They cooperate in secret",
            existing_type="WORKS_WITH",
            existing_usage=existing_usage,
            use_json_mode=False,
        )

    assert should_normalize is True
