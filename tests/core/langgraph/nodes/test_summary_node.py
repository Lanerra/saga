import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.summary_node import ChapterSummaryContractError, summarize_chapter
from core.langgraph.state import create_initial_state


def _valid_summary_json(summary: str) -> str:
    return json.dumps({"summary": summary})


@pytest.mark.asyncio
async def test_summarize_chapter_retries_on_non_json_then_succeeds(tmp_path: Path) -> None:
    project_dir = str(tmp_path)

    state = create_initial_state(
        project_id="test",
        title="Test Novel",
        genre="Sci-Fi",
        theme="Testing",
        setting="Lab",
        target_word_count=1000,
        total_chapters=1,
        project_dir=project_dir,
        protagonist_name="Hero",
    )
    state["current_chapter"] = 1

    content_manager = ContentManager(project_dir)
    state["draft_ref"] = content_manager.save_text(
        "Chapter draft text.",
        content_type="draft",
        identifier="chapter_1",
        version=1,
    )

    invalid_first_response = "This is not JSON."
    valid_second_response = _valid_summary_json("A reveal changes everything.")

    with (
        patch(
            "core.langgraph.nodes.summary_node.llm_service.async_call_llm",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "core.langgraph.nodes.summary_node._save_summary_to_neo4j",
            new_callable=AsyncMock,
        ) as mock_save_summary,
        patch(
            "core.langgraph.nodes.summary_node._write_chapter_summary_file",
        ) as mock_write_file,
    ):
        mock_llm.side_effect = [
            (invalid_first_response, {}),
            (valid_second_response, {}),
        ]

        result = await summarize_chapter(state)

        assert result["current_node"] == "summarize"
        assert result["current_summary"] == "A reveal changes everything."

        assert mock_llm.call_count == 2
        second_call_prompt = mock_llm.call_args_list[1].kwargs["prompt"]
        assert "CORRECTION:" in second_call_prompt
        assert 'exactly one key: "summary"' in second_call_prompt

        mock_save_summary.assert_awaited_once()
        mock_write_file.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_chapter_raises_after_retry_exhaustion(tmp_path: Path) -> None:
    project_dir = str(tmp_path)

    state = create_initial_state(
        project_id="test",
        title="Test Novel",
        genre="Sci-Fi",
        theme="Testing",
        setting="Lab",
        target_word_count=1000,
        total_chapters=1,
        project_dir=project_dir,
        protagonist_name="Hero",
    )
    state["current_chapter"] = 1

    content_manager = ContentManager(project_dir)
    state["draft_ref"] = content_manager.save_text(
        "Chapter draft text.",
        content_type="draft",
        identifier="chapter_1",
        version=1,
    )

    invalid_response = "Not JSON."

    with (
        patch(
            "core.langgraph.nodes.summary_node.llm_service.async_call_llm",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "core.langgraph.nodes.summary_node._save_summary_to_neo4j",
            new_callable=AsyncMock,
        ) as mock_save_summary,
        patch(
            "core.langgraph.nodes.summary_node._write_chapter_summary_file",
        ) as mock_write_file,
    ):
        mock_llm.side_effect = [
            (invalid_response, {}),
            (invalid_response, {}),
            (invalid_response, {}),
        ]

        with pytest.raises(ChapterSummaryContractError) as exc:
            await summarize_chapter(state)

        assert str(exc.value) == ("Chapter summary JSON contract violated: could not parse a JSON object from the model response.")

        assert mock_llm.call_count == 3
        mock_save_summary.assert_not_awaited()
        mock_write_file.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_chapter_retries_and_fails_when_json_missing_required_field(tmp_path: Path) -> None:
    project_dir = str(tmp_path)

    state = create_initial_state(
        project_id="test",
        title="Test Novel",
        genre="Sci-Fi",
        theme="Testing",
        setting="Lab",
        target_word_count=1000,
        total_chapters=1,
        project_dir=project_dir,
        protagonist_name="Hero",
    )
    state["current_chapter"] = 1

    content_manager = ContentManager(project_dir)
    state["draft_ref"] = content_manager.save_text(
        "Chapter draft text.",
        content_type="draft",
        identifier="chapter_1",
        version=1,
    )

    invalid_schema_response = json.dumps({"not_summary": "Missing required key."})

    with (
        patch(
            "core.langgraph.nodes.summary_node.llm_service.async_call_llm",
            new_callable=AsyncMock,
        ) as mock_llm,
        patch(
            "core.langgraph.nodes.summary_node._save_summary_to_neo4j",
            new_callable=AsyncMock,
        ) as mock_save_summary,
        patch(
            "core.langgraph.nodes.summary_node._write_chapter_summary_file",
        ) as mock_write_file,
    ):
        mock_llm.side_effect = [
            (invalid_schema_response, {}),
            (invalid_schema_response, {}),
            (invalid_schema_response, {}),
        ]

        with pytest.raises(ChapterSummaryContractError) as exc:
            await summarize_chapter(state)

        assert str(exc.value) == ('Chapter summary JSON contract violated: expected a single JSON object with exactly one key: "summary". ' "Found keys: not_summary")

        assert mock_llm.call_count == 3
        mock_save_summary.assert_not_awaited()
        mock_write_file.assert_not_called()
