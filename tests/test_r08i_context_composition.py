"""Regression controls for source-faithful context and observable CLI fixtures."""
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes import context_world_retrieval
from core.langgraph.state import NarrativeState
from tests.fakes.cli_capture import CLIStderr, capture_cli_stderr
from tests.fakes.generation_boundary import PROFILE_QUERY, GenerationDatabase
from tests.test_r08g_catalog_fixtures import catalog_state


@pytest.mark.run_settings(DEFAULT_PROTAGONIST_NAME="Unselected default")
async def test_world_context_uses_selected_author_protagonist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    async def facts(**arguments: Any) -> str:
        seen.append(arguments)
        return "Selected facts"

    monkeypatch.setattr(context_world_retrieval, "get_reliable_kg_facts_for_drafting_prompt", facts)
    state = NarrativeState(protagonist_name="Hero")
    result = await context_world_retrieval.get_scene_kg_facts(state, {"characters": ["Hero"]}, 1, {}, [], "synthetic", ContentManager(str(tmp_path)))
    assert result == "Selected facts"
    assert seen[0]["protagonist_name"] == "Hero"


def test_cli_capture_preserves_raw_logs_and_unknown_diagnostics() -> None:
    captured = CLIStderr()
    handler = logging.StreamHandler(captured)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    with capture_cli_stderr(captured):
        handler.handle(logging.LogRecord("synthetic", logging.ERROR, __file__, 1, "retained error", (), None))
        print("unexpected plain stderr", file=sys.stderr)
    assert captured.getvalue() == "ERROR:retained error\nunexpected plain stderr\n"
    assert captured.diagnostics == "unexpected plain stderr\n"
    assert len(captured.log_spans) == 1
    assert captured.log_spans[0][2].levelno == logging.ERROR


async def test_generation_profile_boundary_preserves_selected_id_and_unknown_query_failure(tmp_path: Path) -> None:
    database = GenerationDatabase()
    database.select(catalog_state(tmp_path, characters=("Hero",), locations=("Room",), existing=NarrativeState(theme="Discovery")))
    rows = await database.execute_read_query(PROFILE_QUERY, {"name": "Hero", "include_provisional": False})
    assert rows[0]["c"]["id"] == database.profiles["Hero"]["id"]
    assert rows[0]["c"]["name"] == "Hero"
    assert await database.execute_read_query(PROFILE_QUERY, {"name": "hero", "include_provisional": False}) == []
    with pytest.raises(AssertionError, match="Unconfigured synthetic query"):
        await database.execute_read_query("MATCH (unknown) RETURN unknown", {})
