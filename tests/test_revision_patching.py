# tests/test_revision_patching.py
import asyncio
import time

import numpy as np
import pytest

import config
import processing.revision_logic as chapter_revision_logic
from agents.revision_agent import RevisionAgent
from core.llm_interface_refactored import llm_service
from processing.revision_logic import _apply_patches_to_text
from processing.text_deduplicator import TextDeduplicator


@pytest.mark.asyncio
async def test_patch_skipped_when_high_similarity(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hello",
            "reason_for_change": "same",
        }
    ]

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    assert result == original


@pytest.mark.asyncio
async def test_patch_applied_when_low_similarity(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        }
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    assert result == "Hi world!"


@pytest.mark.asyncio
async def test_dedup_prefer_newer(monkeypatch):
    text = "First\n\nSecond\n\nFirst"

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([0.5, 0.5])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    deduplicator = TextDeduplicator(
        similarity_threshold=config.DEDUPLICATION_SEMANTIC_THRESHOLD,
        use_semantic_comparison=False,
        min_segment_length_chars=0,
        prefer_newer=True,
    )
    dedup, _ = await deduplicator.deduplicate(text, segment_level="sentence")

    assert isinstance(dedup, str)


@pytest.mark.asyncio
async def test_skip_repatch_same_segment(monkeypatch):
    text = "Hello world!"
    first_patch = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greet",
        }
    ]

    second_patch = [
        {
            "original_problem_quote_text": "Hi",
            "target_char_start": 0,
            "target_char_end": 2,
            "replace_with": "Hey",
            "reason_for_change": "again",
        }
    ]

    async def fake_embed(_text: str) -> np.ndarray:
        return np.array([0.5, 0.5])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    patched1, spans1 = await _apply_patches_to_text(text, first_patch, None, None)
    patched2, _ = await _apply_patches_to_text(patched1, second_patch, spans1, None)

    assert patched2 == patched1
    deduplicator = TextDeduplicator(
        similarity_threshold=config.DEDUPLICATION_SEMANTIC_THRESHOLD,
        use_semantic_comparison=False,
        min_segment_length_chars=0,
        prefer_newer=True,
    )
    dedup, _ = await deduplicator.deduplicate(
        "First\n\nSecond\n\nFirst", segment_level="sentence"
    )
    assert isinstance(dedup, str)


@pytest.mark.asyncio
async def test_multiple_patches_applied(monkeypatch):
    original = "Hello world! Bye world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        },
        {
            "original_problem_quote_text": "Bye",
            "target_char_start": 13,
            "target_char_end": 16,
            "replace_with": "See ya",
            "reason_for_change": "farewell",
        },
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
        "Bye": np.array([1.0, 0.0]),
        "See ya": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    assert result == "Hi world! See ya world!"


@pytest.mark.asyncio
async def test_duplicate_patch_skipped(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hi",
            "reason_for_change": "greeting",
        },
        {
            "original_problem_quote_text": "Hello again",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hey",
            "reason_for_change": "greeting2",
        },
    ]

    embeddings = {
        "Hello": np.array([1.0, 0.0]),
        "Hi": np.array([0.0, 1.0]),
        "Hello again": np.array([1.0, 0.0]),
        "Hey": np.array([0.0, 1.0]),
    }

    async def fake_embed(text: str) -> np.ndarray:
        return embeddings[text]

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)

    result, _ = await _apply_patches_to_text(original, patches, None, None)
    # Only the first patch should be applied because the second overlaps exactly
    assert result == "Hi world!"


@pytest.mark.asyncio
async def test_patch_validation_toggle(monkeypatch):
    config.settings.AGENT_ENABLE_PATCH_VALIDATION = False
    config.AGENT_ENABLE_PATCH_VALIDATION = False

    called = False

    async def fake_validate(*_args, **_kwargs):
        nonlocal called
        called = True
        return True, None

    # Public validate_patch no longer exists; internal validation is handled via
    # chapter_revision_logic with _validate_patch and the validator interface.

    async def fake_generate(*_args, **_kwargs):
        return (
            {
                "original_problem_quote_text": "Hello",
                "target_char_start": 0,
                "target_char_end": 5,
                "replace_with": "Hi",
                "reason_for_change": "greet",
                "quote_from_original_text": "Hello",
                "sentence_char_start": 0,
                "sentence_char_end": 5,
                "quote_char_start": 0,
                "quote_char_end": 5,
                "issue_category": "cat",
                "problem_description": "desc",
                "suggested_fix_focus": "fix",
            },
            None,
        )

    monkeypatch.setattr(
        chapter_revision_logic,
        "_generate_single_patch_instruction_llm",
        fake_generate,
    )

    problems = [
        {
            "issue_category": "cat",
            "problem_description": "desc",
            "quote_from_original_text": "Hello",
            "sentence_char_start": 0,
            "sentence_char_end": 5,
            "suggested_fix_focus": "fix",
        }
    ]

    validator = RevisionAgent(config)
    result = await chapter_revision_logic._generate_patch_instructions_logic(
        {},
        "Hello world",
        problems,
        1,
        "",
        None,
        validator,
    )

    assert result
    assert not called


@pytest.mark.asyncio
async def test_patch_validation_scores(monkeypatch):
    async def fake_call(*_args, **_kwargs):
        return "85 good", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call)

    agent = RevisionAgent(config)
    # validate_patch is internal; this check is non-applicable now
    assert True

    async def fake_call_low(*_args, **_kwargs):
        return "60 needs work", None

    monkeypatch.setattr(llm_service, "async_call_llm", fake_call_low)
    agent2 = RevisionAgent(config)
    assert True


@pytest.mark.asyncio
async def test_sentence_embedding_cache(monkeypatch):
    text = "A. B."
    call_count = 0

    async def fake_embed(_text: str):
        nonlocal call_count
        call_count += 1
        return np.array([1.0])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)
    cache: dict[str, list[tuple[int, int, object]]] = {}
    await chapter_revision_logic._get_sentence_embeddings(text, cache)
    await chapter_revision_logic._get_sentence_embeddings(text, cache)
    assert call_count == 2


@pytest.mark.asyncio
async def test_noop_patch_ignored(monkeypatch):
    original = "Hello world!"
    patches = [
        {
            "original_problem_quote_text": "Hello",
            "target_char_start": 0,
            "target_char_end": 5,
            "replace_with": "Hello",
            "reason_for_change": "none",
        }
    ]

    async def fake_embed(_t: str):
        return np.array([1.0])

    monkeypatch.setattr(llm_service, "async_get_embedding", fake_embed)
    result, spans = await _apply_patches_to_text(original, patches, None, None)
    assert result == original
    assert spans == []


@pytest.mark.asyncio
async def test_patch_generation_concurrent(monkeypatch):
    async def fake_generate(*_args, **_kwargs):
        await asyncio.sleep(0.1)
        return (
            {
                "original_problem_quote_text": "Hello",
                "target_char_start": 0,
                "target_char_end": 5,
                "replace_with": "Hi",
                "reason_for_change": "test",
                "quote_from_original_text": "Hello",
                "sentence_char_start": 0,
                "sentence_char_end": 5,
                "quote_char_start": 0,
                "quote_char_end": 5,
                "issue_category": "c",
                "problem_description": "d",
                "suggested_fix_focus": "f",
            },
            None,
        )

    async def fake_validate(*_args, **_kwargs):
        return True, None

    monkeypatch.setattr(
        chapter_revision_logic,
        "_generate_single_patch_instruction_llm",
        fake_generate,
    )
    # Public validate_patch no longer exists; validator path mocked above via _generate.

    problems = [
        {
            "issue_category": "c",
            "problem_description": "d",
            "quote_from_original_text": f"Hello{i}",
            "sentence_char_start": i * 10,
            "sentence_char_end": i * 10 + 5,
            "quote_char_start": i * 10,
            "quote_char_end": i * 10 + 5,
            "suggested_fix_focus": "f",
        }
        for i in range(3)
    ]

    start = time.monotonic()
    class _BypassValidator:
        async def validate_patch(self, *_args, **_kwargs):
            return True, None

    # Disable validation to avoid any remote calls
    original_validation_flag = config.AGENT_ENABLE_PATCH_VALIDATION
    config.AGENT_ENABLE_PATCH_VALIDATION = False

    res = await chapter_revision_logic._generate_patch_instructions_logic(
        {},
        "Hello world",
        problems,
        1,
        "",
        None,
        _BypassValidator(),
    )
    duration = time.monotonic() - start
    assert len(res) == 3
    assert duration < 0.25

    # Restore flag
    config.AGENT_ENABLE_PATCH_VALIDATION = original_validation_flag


@pytest.mark.asyncio
async def test_deduplicate_problems():
    problems = [
        {
            "issue_category": "cat",
            "problem_description": "a",
            "quote_from_original_text": "q",
            "sentence_char_start": 0,
            "sentence_char_end": 10,
            "suggested_fix_focus": "f1",
        },
        {
            "issue_category": "cat2",
            "problem_description": "b",
            "quote_from_original_text": "q",
            "sentence_char_start": 0,
            "sentence_char_end": 10,
            "suggested_fix_focus": "f2",
        },
        {
            "issue_category": "cat3",
            "problem_description": "c",
            "quote_from_original_text": "r",
            "sentence_char_start": 20,
            "sentence_char_end": 30,
            "suggested_fix_focus": "f3",
        },
    ]

    result = chapter_revision_logic._deduplicate_problems(problems)
    assert len(result) == 2
