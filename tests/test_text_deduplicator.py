from unittest.mock import patch

import pytest

from processing.text_deduplicator import TextDeduplicator


@pytest.fixture(autouse=True)
def _configure_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("config.MAX_CONCURRENT_LLM_CALLS", 5)


@pytest.mark.asyncio
class TestEmptyAndWhitespaceInput:
    async def test_empty_string_returns_unchanged_with_zero_removed(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=False,
        )
        result_text, removed_count = await deduplicator.deduplicate("")
        assert result_text == ""
        assert removed_count == 0

    async def test_whitespace_only_returns_unchanged_with_zero_removed(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=False,
        )
        original = "   \n\n\t  "
        result_text, removed_count = await deduplicator.deduplicate(original)
        assert result_text == original
        assert removed_count == 0


@pytest.mark.asyncio
class TestNoDuplicates:
    async def test_unique_segments_returns_original_with_zero_removed(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=False,
        )
        original = "Alpha bravo charlie delta.\n\nEcho foxtrot golf hotel."
        segments = [
            ("Alpha bravo charlie delta.", 0, 26),
            ("Echo foxtrot golf hotel.", 28, 52),
        ]
        with patch("processing.text_deduplicator.utils") as fake_utils:
            fake_utils.get_text_segments.return_value = segments
            fake_utils._normalize_text_for_matching.side_effect = lambda s: s.lower().strip()
            result_text, removed_count = await deduplicator.deduplicate(original)
        assert result_text == original
        assert removed_count == 0

    async def test_multiple_different_segments_all_preserved(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=False,
        )
        original = "Segment one content.\n\nSegment two content.\n\nSegment three content."
        segments = [
            ("Segment one content.", 0, 20),
            ("Segment two content.", 22, 42),
            ("Segment three content.", 44, 66),
        ]
        with patch("processing.text_deduplicator.utils") as fake_utils:
            fake_utils.get_text_segments.return_value = segments
            fake_utils._normalize_text_for_matching.side_effect = lambda s: s.lower().strip()
            result_text, removed_count = await deduplicator.deduplicate(original)
        assert result_text == original
        assert removed_count == 0


@pytest.mark.asyncio
class TestExactDuplicateRemoval:
    async def test_prefer_newer_false_keeps_first_occurrence(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=False,
        )
        original = "Hello world this is a test.\n\nHello world this is a test."
        segments = [
            ("Hello world this is a test.", 0, 27),
            ("Hello world this is a test.", 29, 56),
        ]
        with patch("processing.text_deduplicator.utils") as fake_utils:
            fake_utils.get_text_segments.return_value = segments
            fake_utils._normalize_text_for_matching.side_effect = lambda s: s.lower().strip()
            result_text, removed_count = await deduplicator.deduplicate(original)
        assert result_text == "Hello world this is a test."
        assert removed_count == len(original) - len("Hello world this is a test.")

    async def test_prefer_newer_true_removes_later_occurrence_in_backward_pass(self) -> None:
        """
        With prefer_newer=True the backward iteration encounters the later
        duplicate first.  When the earlier duplicate is reached, the
        previously-stored (later) index is evicted, so the *earlier*
        occurrence survives.
        """
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=True,
        )
        original = "Duplicate segment here.\n\nMiddle unique part.\n\nDuplicate segment here."
        segments = [
            ("Duplicate segment here.", 0, 23),
            ("Middle unique part.", 25, 44),
            ("Duplicate segment here.", 46, 69),
        ]
        with patch("processing.text_deduplicator.utils") as fake_utils:
            fake_utils.get_text_segments.return_value = segments
            fake_utils._normalize_text_for_matching.side_effect = lambda s: s.lower().strip()
            result_text, removed_count = await deduplicator.deduplicate(original)
        assert "Middle unique part." in result_text
        assert result_text.count("Duplicate segment here.") == 1
        assert removed_count > 0
        kept_duplicate_position = result_text.index("Duplicate segment here.")
        middle_position = result_text.index("Middle unique part.")
        assert kept_duplicate_position < middle_position


@pytest.mark.asyncio
class TestMinSegmentLength:
    async def test_short_segments_skip_fingerprinting(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=50,
            prefer_newer=False,
        )
        original = "Short dup.\n\nShort dup."
        segments = [
            ("Short dup.", 0, 10),
            ("Short dup.", 12, 22),
        ]
        with patch("processing.text_deduplicator.utils") as fake_utils:
            fake_utils.get_text_segments.return_value = segments
            fake_utils._normalize_text_for_matching.side_effect = lambda s: s.lower().strip()
            result_text, removed_count = await deduplicator.deduplicate(original)
        assert result_text == original
        assert removed_count == 0


@pytest.mark.asyncio
class TestSemanticComparisonDisabled:
    async def test_no_embedding_calls_when_semantic_disabled(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=False,
        )
        original = "Unique paragraph alpha.\n\nUnique paragraph beta."
        segments = [
            ("Unique paragraph alpha.", 0, 23),
            ("Unique paragraph beta.", 25, 47),
        ]
        with patch("processing.text_deduplicator.utils") as fake_utils, patch("processing.text_deduplicator.llm_service") as fake_llm:
            fake_utils.get_text_segments.return_value = segments
            fake_utils._normalize_text_for_matching.side_effect = lambda s: s.lower().strip()
            await deduplicator.deduplicate(original)
        fake_llm.async_get_embedding.assert_not_called()


@pytest.mark.asyncio
class TestTextReconstruction:
    async def test_duplicate_span_spliced_out_correctly(self) -> None:
        deduplicator = TextDeduplicator(
            similarity_threshold=0.85,
            use_semantic_comparison=False,
            min_segment_length_chars=10,
            prefer_newer=False,
        )
        original = "First unique segment.\n\nDuplicate segment content.\n\nSecond unique segment.\n\nDuplicate segment content."
        segments = [
            ("First unique segment.", 0, 21),
            ("Duplicate segment content.", 23, 48),
            ("Second unique segment.", 50, 72),
            ("Duplicate segment content.", 74, 99),
        ]
        with patch("processing.text_deduplicator.utils") as fake_utils:
            fake_utils.get_text_segments.return_value = segments
            fake_utils._normalize_text_for_matching.side_effect = lambda s: s.lower().strip()
            result_text, removed_count = await deduplicator.deduplicate(original)
        assert "First unique segment." in result_text
        assert "Second unique segment." in result_text
        assert result_text.count("Duplicate segment content.") == 1
        assert removed_count > 0
        first_dup_position = result_text.index("Duplicate segment content.")
        second_unique_position = result_text.index("Second unique segment.")
        assert first_dup_position < second_unique_position
