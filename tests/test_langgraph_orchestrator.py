# tests/test_langgraph_orchestrator.py
"""
Tests for LangGraph orchestrator.

Covers orchestration/langgraph_orchestrator.py.
"""

import ast
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TypedDict, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # type: ignore
from langgraph.graph import END, StateGraph  # type: ignore[import-not-found, attr-defined]

from core.exceptions import CheckpointResumeConflictError, WorkflowExecutionError
from core.langgraph.state import Contradiction, NarrativeState, create_initial_state
from core.service_context import RunServices, get_services, inject_services
from data_access.chapter_queries import ChapterProgress
from orchestration.langgraph_orchestrator import (
    LangGraphOrchestrator,
)
from tests.fakes.service_context import patch_service


@pytest.mark.asyncio
async def test_scene_summary_uses_injected_provider_and_restores_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.langgraph.content_manager import ContentManager
    from core.langgraph.nodes import context_scene_retrieval

    managed_service = MagicMock()
    managed_service.async_call_llm = AsyncMock(return_value=("Managed summary", {}))
    previous_service = MagicMock()
    previous_service.async_call_llm = AsyncMock(return_value=("Unmanaged summary", {}))

    monkeypatch.setattr(get_services(), 'language_model', previous_service)
    expected = "\n\n**Previous Scenes in This Chapter:**\n\n--- Opening (Summary) ---\nManaged summary\n"
    monkeypatch.setattr(context_scene_retrieval, "PREVIOUS_SCENES_TOKEN_BUDGET", context_scene_retrieval.count_tokens(expected, "synthetic"))
    monkeypatch.setattr(context_scene_retrieval, "SUMMARY_MAX_TOKENS", 32)

    with inject_services(RunServices(managed_service, get_services().database)):
        result = await context_scene_retrieval.get_previous_scenes_context(
            {}, ["Synthetic scene words " * 100], [{"title": "Opening"}], 1,
            "synthetic", "synthetic", ContentManager(str(tmp_path)),
        )

    assert result == expected
    managed_service.async_call_llm.assert_awaited_once()
    previous_service.async_call_llm.assert_not_awaited()
    assert get_services().language_model is previous_service


@pytest.fixture
def mock_config() -> Iterator[MagicMock]:
    """Mock configuration settings."""
    with patch("orchestration.langgraph_orchestrator.config") as mock_cfg:
        mock_cfg.settings.BASE_OUTPUT_DIR = "/tmp/test_output"
        mock_cfg.DEFAULT_PLOT_OUTLINE_TITLE = "Test Novel"
        mock_cfg.CONFIGURED_GENRE = "fantasy"
        mock_cfg.CONFIGURED_THEME = "courage"
        mock_cfg.CONFIGURED_SETTING_DESCRIPTION = "A magical world"
        mock_cfg.DEFAULT_PROTAGONIST_NAME = "Hero"
        mock_cfg.NARRATIVE_MODEL = "narrative-model"
        mock_cfg.MEDIUM_MODEL = "medium-model"
        mock_cfg.LARGE_MODEL = "large-model"
        mock_cfg.SMALL_MODEL = "small-model"
        mock_cfg.CHAPTERS_PER_RUN = 3
        mock_cfg.TOTAL_CHAPTERS = 20
        mock_cfg.TARGET_WORD_COUNT = 80000
        mock_cfg.DEFAULT_NARRATIVE_STYLE = "Third person limited"
        mock_cfg.MAX_REVISION_CYCLES_PER_CHAPTER = 2
        yield mock_cfg


@pytest.fixture
def orchestrator(mock_config: MagicMock, tmp_path: Path) -> LangGraphOrchestrator:
    """Create orchestrator with mocked dependencies."""
    with patch(
        "orchestration.langgraph_orchestrator.config.settings.BASE_OUTPUT_DIR",
        str(tmp_path),
    ):
        with patch("orchestration.langgraph_orchestrator.RichDisplayManager"):
            orch = LangGraphOrchestrator()
            return orch


def _repo_root_from_test_file(test_file: Path) -> Path:
    return test_file.resolve().parents[1]


def _path_to_module_name(repo_root: Path, file_path: Path) -> str:
    relative_no_suffix = file_path.relative_to(repo_root).with_suffix("")
    parts = list(relative_no_suffix.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _ModuleLevelLLMServiceImportFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self._nesting_depth = 0
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._nesting_depth += 1
        self.generic_visit(node)
        self._nesting_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._nesting_depth += 1
        self.generic_visit(node)
        self._nesting_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._nesting_depth += 1
        self.generic_visit(node)
        self._nesting_depth -= 1

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if self._nesting_depth != 0:
            return

        if node.module != "core.llm_interface_refactored":
            return

        for alias in node.names:
            if alias.name == "llm_service":
                self.found = True
                return


def _module_level_llm_service_importers(repo_root: Path) -> set[str]:
    search_roots = (
        "core",
        "processing",
        "ui",
        "orchestration",
        "data_access",
        "models",
        "prompts",
        "utils",
        "config",
    )

    modules: set[str] = set()
    for root_name in search_roots:
        root_path = repo_root / root_name
        if not root_path.exists():
            continue

        for file_path in root_path.rglob("*.py"):
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
            finder = _ModuleLevelLLMServiceImportFinder()
            finder.visit(tree)
            if finder.found:
                modules.add(_path_to_module_name(repo_root, file_path))

    return modules


class TestLangGraphOrchestratorInit:
    """Tests for LangGraphOrchestrator initialization."""

    def test_init_creates_project_dir_path(self, orchestrator: LangGraphOrchestrator) -> None:
        """Orchestrator initializes with project directory path."""
        assert orchestrator.project_dir is not None
        assert isinstance(orchestrator.project_dir, Path)

    def test_init_creates_checkpointer_path(self, orchestrator: LangGraphOrchestrator) -> None:
        """Orchestrator initializes with checkpointer path."""
        assert orchestrator.checkpointer_path is not None
        assert str(orchestrator.checkpointer_path).endswith("saga.db")

    def test_init_creates_display_manager(self, orchestrator: LangGraphOrchestrator) -> None:
        """Orchestrator initializes with display manager."""
        assert orchestrator.display is not None

    def test_init_sets_run_start_time(self, orchestrator: LangGraphOrchestrator) -> None:
        """Orchestrator initializes with run start time."""
        assert orchestrator.run_start_time == 0.0


@pytest.mark.asyncio
class TestEnsureNeo4jConnection:
    """Tests for _ensure_neo4j_connection method."""

    async def test_ensure_neo4j_connection_connects(self, orchestrator: LangGraphOrchestrator) -> None:
        """Neo4j connection is established and schema created."""
        with patch_service('database') as mock_neo4j:
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.create_db_schema = AsyncMock()

            await orchestrator._ensure_neo4j_connection()

            mock_neo4j.connect.assert_called_once()
            mock_neo4j.create_db_schema.assert_called_once()

    async def test_ensure_neo4j_connection_error_propagates(self, orchestrator: LangGraphOrchestrator) -> None:
        """Neo4j connection errors propagate."""
        with patch_service('database') as mock_neo4j:
            mock_neo4j.connect = AsyncMock(side_effect=Exception("Connection failed"))

            with pytest.raises(Exception, match="Connection failed"):
                await orchestrator._ensure_neo4j_connection()


@pytest.mark.asyncio
class TestLoadOrCreateState:
    """Tests for _load_or_create_state method."""

    async def test_load_or_create_state_no_existing_chapters(self, orchestrator: LangGraphOrchestrator) -> None:
        """State is created with chapter 1 when no chapters exist."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "orchestration.langgraph_orchestrator.validate_initialization_artifacts",
                return_value=(False, ["Missing saga.yaml"]),
            ),
        ):
            mock_load.return_value = ChapterProgress(0, ())

            state = await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)

            assert state["current_chapter"] == 1
            assert state["initialization_complete"] is False

    async def test_load_or_create_state_existing_chapters(self, orchestrator: LangGraphOrchestrator) -> None:
        """State continues from next chapter when chapters exist."""
        with patch(
            "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
            new_callable=AsyncMock,
        ) as mock_load:
            mock_load.return_value = ChapterProgress(5, (1, 2, 3, 4, 5))

            state = await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)

            assert state["current_chapter"] == 6
            assert state["initialization_complete"] is True

    async def test_load_or_create_state_detects_initialization(self, orchestrator: LangGraphOrchestrator) -> None:
        """Chapter-one files cannot replace an initialization receipt."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "orchestration.langgraph_orchestrator.validate_initialization_artifacts",
                return_value=(True, []),
            ),
        ):
            mock_load.return_value = ChapterProgress(0, ())

            with pytest.raises(ValueError, match="Missing initialization receipt"):
                await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)

    async def test_load_or_create_state_fallback_to_file_check(self, orchestrator: LangGraphOrchestrator, tmp_path: Path) -> None:
        """A complete historical file set is preserved but not auto-accepted."""
        with patch(
            "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
            new_callable=AsyncMock,
        ) as mock_load:
            mock_load.return_value = ChapterProgress(0, ())

            orchestrator.project_dir = tmp_path

            (tmp_path / "outline").mkdir(parents=True, exist_ok=True)
            (tmp_path / "characters").mkdir(parents=True, exist_ok=True)
            (tmp_path / "world").mkdir(parents=True, exist_ok=True)

            (tmp_path / "saga.yaml").write_text("project: saga\n")
            (tmp_path / "outline" / "structure.yaml").write_text("acts: []\n")
            (tmp_path / "outline" / "beats.yaml").write_text("beats: []\n")
            (tmp_path / "characters" / "hero.yaml").write_text("name: Hero\n")
            (tmp_path / "world" / "items.yaml").write_text("items: []\n")
            (tmp_path / "world" / "rules.yaml").write_text("rules: []\n")
            (tmp_path / "world" / "history.yaml").write_text("history: []\n")

            before = {str(path): path.read_bytes() for path in tmp_path.rglob("*.yaml")}
            with pytest.raises(ValueError, match="Missing initialization receipt"):
                await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)
            assert {str(path): path.read_bytes() for path in tmp_path.rglob("*.yaml")} == before

    async def test_load_or_create_state_includes_models(self, orchestrator: LangGraphOrchestrator) -> None:
        """State includes model configuration."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "orchestration.langgraph_orchestrator.validate_initialization_artifacts",
                return_value=(False, ["Missing saga.yaml"]),
            ),
        ):
            mock_load.return_value = ChapterProgress(0, ())

            state = await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)

            assert state["extraction_model"] is not None
            assert state["revision_model"] is not None
            assert state["large_model"] is not None
            assert state["medium_model"] is not None
            assert state["small_model"] is not None
            assert state["narrative_model"] is not None

    async def test_load_or_create_state_validates_artifacts(self, orchestrator: LangGraphOrchestrator, tmp_path: Path) -> None:
        """Validation check runs on existing project directory."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch("orchestration.langgraph_orchestrator.validate_initialization_artifacts") as mock_validate,
        ):
            mock_load.return_value = ChapterProgress(0, ())
            mock_validate.return_value = (True, [])

            orchestrator.project_dir = tmp_path
            tmp_path.mkdir(exist_ok=True)

            with pytest.raises(ValueError, match="Missing initialization receipt"):
                await orchestrator._load_or_create_state(project_id="test-project", narrative_config=None)

            mock_validate.assert_called_once_with(tmp_path)


@pytest.mark.asyncio
class TestRunChapterGenerationLoop:
    """Tests for _run_chapter_generation_loop method."""

    async def test_run_chapter_generation_loop_basic(self, orchestrator: LangGraphOrchestrator) -> None:
        """Chapter generation loop runs for configured chapters."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args: object, **kwargs: object) -> AsyncIterator[object]:
            events = [
                {"generate": {"current_node": "generate", "draft_text": "Chapter text"}},
                {"finalize": {"current_node": "finalize", "draft_word_count": 2000}},
            ]
            for event in events:
                yield event

        mock_graph.astream = mock_stream_func

        state: NarrativeState = {
            "project_id": "test_proj",
            "current_chapter": 1,
            "total_chapters": 20,
            "draft_word_count": 2000,
        }

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_multi_chapter(self, orchestrator: LangGraphOrchestrator) -> None:
        """Chapter generation loop handles multiple chapters in a single stream."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args: object, **kwargs: object) -> AsyncIterator[object]:
            events = [
                {"generate": {"current_chapter": 1, "current_node": "generate"}},
                {"finalize": {"current_chapter": 1, "current_node": "finalize", "draft_word_count": 1000}},
                {"advance_chapter": {"current_chapter": 2, "current_node": "advance_chapter"}},
                {"generate": {"current_chapter": 2, "current_node": "generate"}},
                {"finalize": {"current_chapter": 2, "current_node": "finalize", "draft_word_count": 1200}},
            ]
            for event in events:
                yield event

        mock_graph.astream = mock_stream_func

        state: NarrativeState = {
            "project_id": "test_proj",
            "current_chapter": 1,
            "total_chapters": 5,
        }

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock) as mock_handle:
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

            # Check that _handle_workflow_event was called with correct chapter numbers
            assert mock_handle.call_count == 5
            # First 2 calls should be chapter 1
            assert mock_handle.call_args_list[0][0][1] == 1
            assert mock_handle.call_args_list[1][0][1] == 1
            # 3rd call (advance_chapter) should be chapter 2
            assert mock_handle.call_args_list[2][0][1] == 2
            # Remaining calls should be chapter 2
            assert mock_handle.call_args_list[3][0][1] == 2
            assert mock_handle.call_args_list[4][0][1] == 2

    async def test_run_chapter_generation_loop_thread_id(self, orchestrator: LangGraphOrchestrator) -> None:
        """Orchestrator uses project-specific thread ID."""
        mock_graph = MagicMock()

        async def empty_stream(*args: object, **kwargs: object) -> AsyncIterator[object]:
            return
            yield

        mock_graph.astream = MagicMock(side_effect=empty_stream)

        state: NarrativeState = {"project_id": "custom_project"}
        with pytest.raises(WorkflowExecutionError, match="incomplete"):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

        # Check astream call arguments
        # Need to find the call
        found = False
        for call in mock_graph.astream.call_args_list:
            if call.kwargs.get("config", {}).get("configurable", {}).get("thread_id") == "saga_custom_project":
                found = True
                break
        assert found, "Thread ID 'saga_custom_project' not found in astream calls"

    async def test_run_chapter_generation_loop_respects_total_chapters(self, orchestrator: LangGraphOrchestrator) -> None:
        """Loop stops at total chapter count."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args: object, **kwargs: object) -> AsyncIterator[object]:
            events = [{"finalize": {"current_node": "finalize", "draft_word_count": 2000}}]
            for event in events:
                yield event

        mock_graph.astream = mock_stream_func

        state: NarrativeState = {"project_id": "test_proj", "current_chapter": 20, "total_chapters": 20, "draft_word_count": 2000}

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_stops_on_error(self, orchestrator: LangGraphOrchestrator) -> None:
        """Loop re-raises exceptions so caller can handle them."""
        mock_graph = MagicMock()

        async def mock_stream_error(*args: object, **kwargs: object) -> AsyncIterator[object]:
            raise Exception("Generation error")
            yield

        mock_graph.astream = mock_stream_error

        state: NarrativeState = {"project_id": "test_proj", "current_chapter": 1, "total_chapters": 20}

        with pytest.raises(Exception, match="Generation error"):
            with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
                await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_handles_incomplete_generation(self, orchestrator: LangGraphOrchestrator) -> None:
        """Loop stops if generation doesn't reach finalize node."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args: object, **kwargs: object) -> AsyncIterator[object]:
            events = [{"extract": {"current_node": "extract", "extracted_entities": {}}}]
            for event in events:
                yield event

        mock_graph.astream = mock_stream_func

        state: NarrativeState = {
            "project_id": "test_proj",
            "current_chapter": 1,
            "total_chapters": 20,
            "last_error": "Failed at extraction",
        }

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            with pytest.raises(WorkflowExecutionError, match="incomplete"):
                await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_handles_no_events(self, orchestrator: LangGraphOrchestrator) -> None:
        """Loop handles case where no events are received."""
        mock_graph = MagicMock()

        async def empty_stream(*args: object, **kwargs: object) -> AsyncIterator[object]:
            return
            yield

        mock_graph.astream = empty_stream

        state: NarrativeState = {"project_id": "test_proj", "current_chapter": 1, "total_chapters": 20}

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            with pytest.raises(WorkflowExecutionError, match="incomplete"):
                await orchestrator._run_chapter_generation_loop(mock_graph, state)


@pytest.mark.asyncio
class TestHandleWorkflowEvent:
    """Tests for _handle_workflow_event method."""

    async def test_handle_workflow_event_basic(self, orchestrator: LangGraphOrchestrator) -> None:
        """Basic event handling works."""
        event = {"generate": {"current_node": "generate", "draft_text": "Text"}}

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_skips_internal_nodes(self, orchestrator: LangGraphOrchestrator) -> None:
        """Internal nodes are skipped."""
        event = {"__start__": {"current_node": "__start__"}}

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_invalid_event(self, orchestrator: LangGraphOrchestrator) -> None:
        """Invalid events are handled gracefully."""
        assert get_type_hints(LangGraphOrchestrator._handle_workflow_event)["event"] is object
        await orchestrator._handle_workflow_event({}, 1)
        await orchestrator._handle_workflow_event(None, 1)
        await orchestrator._handle_workflow_event("not a dict", 1)

    async def test_handle_workflow_event_validate_with_contradictions(self, orchestrator: LangGraphOrchestrator) -> None:
        """Validation events log contradictions."""
        contradiction = Contradiction(
            type="trait",
            description="Character trait conflict",
            conflicting_chapters=[1, 2],
            severity="major",
            suggested_fix="Resolve trait",
        )

        event = {
            "validate": {
                "current_node": "validate",
                "contradictions": [contradiction],
            }
        }

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_validate_with_dict_contradictions(self, orchestrator: LangGraphOrchestrator) -> None:
        """Validation handles dict-format contradictions."""
        contradiction_dict = {
            "type": "trait",
            "description": "Character trait conflict",
            "severity": "major",
        }

        event = {
            "validate_consistency": {
                "current_node": "validate_consistency",
                "contradictions": [contradiction_dict],
            }
        }

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_revise(self, orchestrator: LangGraphOrchestrator) -> None:
        """Revision events log iteration count."""
        event = {
            "revise": {
                "current_node": "revise",
                "iteration_count": 2,
                "max_iterations": 3,
            }
        }

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_finalize(self, orchestrator: LangGraphOrchestrator) -> None:
        """Finalize events log word count."""
        event = {"finalize": {"current_node": "finalize", "draft_word_count": 2500}}

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_init_complete(self, orchestrator: LangGraphOrchestrator) -> None:
        """Initialization complete events log counts."""
        event = {
            "init_complete": {
                "current_node": "init_complete",
                "character_sheets": {"Alice": {}, "Bob": {}},
                "act_outlines": {"act1": {}, "act2": {}},
            }
        }

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_updates_display(self, orchestrator: LangGraphOrchestrator) -> None:
        """Workflow events update the display."""
        event = {
            "generate": {
                "current_node": "generate",
                "title": "Test Novel",
            }
        }

        orchestrator.run_start_time = 1000.0

        await orchestrator._handle_workflow_event(event, 1)

        assert isinstance(orchestrator.display, MagicMock)
        orchestrator.display.update.assert_called_once()


class TestGetStepDescription:
    """Tests for _get_step_description method."""

    def test_get_step_description_initialization_steps(self, orchestrator: LangGraphOrchestrator) -> None:
        """Initialization steps are mapped correctly."""
        assert "Character Sheets" in orchestrator._get_step_description("node", "character_sheets")
        assert "Global Story Outline" in orchestrator._get_step_description("node", "global_outline")
        assert "Act Structures" in orchestrator._get_step_description("node", "act_outlines")
        assert "Knowledge Graph" in orchestrator._get_step_description("node", "committing")
        assert "Initialization Files" in orchestrator._get_step_description("node", "files_persisted")
        assert "Initialization Complete" in orchestrator._get_step_description("node", "complete")

    def test_get_step_description_generation_nodes(self, orchestrator: LangGraphOrchestrator) -> None:
        """Generation nodes are mapped correctly."""
        assert "Chapter Outline" in orchestrator._get_step_description("chapter_outline")
        assert "Chapter Text" in orchestrator._get_step_description("generate")
        assert "Entities" in orchestrator._get_step_description("extract")
        assert "Normalizing" in orchestrator._get_step_description("normalize_relationships")
        assert "Knowledge Graph" in orchestrator._get_step_description("commit")
        assert "Validating" in orchestrator._get_step_description("validate")
        assert "Revising" in orchestrator._get_step_description("revise")
        assert "Summary" in orchestrator._get_step_description("summarize")
        assert "Finalizing" in orchestrator._get_step_description("finalize")

    def test_get_step_description_ignores_chapter_outline_markers(self, orchestrator: LangGraphOrchestrator) -> None:
        """Chapter outline completion markers are ignored."""
        result = orchestrator._get_step_description("some_node", "chapter_outline_2_complete")

        assert "Processing" in result or "some_node" in result

    def test_get_step_description_unknown_node(self, orchestrator: LangGraphOrchestrator) -> None:
        """Unknown nodes return generic description."""
        result = orchestrator._get_step_description("unknown_node")

        assert "Processing" in result
        assert "unknown_node" in result

    def test_get_step_description_empty_init_step(self, orchestrator: LangGraphOrchestrator) -> None:
        """Empty initialization step falls back to node descriptions."""
        result = orchestrator._get_step_description("generate", "")

        assert "Chapter Text" in result


@pytest.mark.asyncio
class TestRunNovelGenerationLoop:
    """Tests for run_novel_generation_loop method."""

    async def test_run_novel_generation_loop_full_flow(self, orchestrator: LangGraphOrchestrator) -> None:
        """Full generation loop runs successfully."""
        assert isinstance(orchestrator.display, MagicMock)
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        mock_graph = MagicMock()

        async def mock_events(*args: object, **kwargs: object) -> AsyncIterator[object]:
            events = [
                {
                    "finalize": {
                        "current_node": "finalize",
                        "draft_word_count": 2000,
                    }
                }
            ]
            for event in events:
                yield event

        mock_graph.astream = mock_events

        orchestrator.display.stop = AsyncMock()

        with (
            patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock) as mock_neo4j,
            patch.object(orchestrator, "_load_state_for_run", new_callable=AsyncMock) as mock_state_for_run,
            patch("orchestration.langgraph_orchestrator.create_checkpointer") as mock_cp,
            patch("orchestration.langgraph_orchestrator.create_full_workflow_graph") as mock_graph_creator,
        ):
            mock_state_for_run.return_value = {
                "project_id": "test_proj",
                "current_chapter": 1,
                "total_chapters": 20,
            }
            mock_cp.return_value = mock_checkpointer
            mock_graph_creator.return_value = mock_graph

            await orchestrator.run_novel_generation_loop()

            mock_neo4j.assert_called_once()
            mock_state_for_run.assert_called_once()
            orchestrator.display.start.assert_called_once()
            orchestrator.display.stop.assert_called_once()

    async def test_run_novel_generation_loop_closes_llm_http_client_on_success(self, orchestrator: LangGraphOrchestrator) -> None:
        """
        Orchestrator establishes an explicit LLM HTTP client lifecycle boundary and closes it.

        This is the remediation target for CORE-004 / LANGGRAPH-026:
        workflows should not rely on import-time singleton cleanup.
        """
        assert isinstance(orchestrator.display, MagicMock)
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        mock_graph = MagicMock()

        async def mock_events(*args: object, **kwargs: object) -> AsyncIterator[object]:
            # Minimal "successful" run; no node needs to call the LLM for this test.
            yield {"finalize": {"current_node": "finalize", "draft_word_count": 1234}}

        mock_graph.astream = mock_events

        orchestrator.display.stop = AsyncMock()

        # Patch the underlying httpx client constructor used by HTTPClientService so we can assert closure.
        dummy_httpx_client = MagicMock()
        dummy_httpx_client.aclose = AsyncMock()

        with (
            patch("core.http_client_service.httpx.AsyncClient", return_value=dummy_httpx_client) as mock_async_client_ctor,
            patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock),
            patch.object(orchestrator, "_load_state_for_run", new_callable=AsyncMock) as mock_state_for_run,
            patch("orchestration.langgraph_orchestrator.create_checkpointer") as mock_cp,
            patch("orchestration.langgraph_orchestrator.create_full_workflow_graph") as mock_graph_creator,
        ):
            mock_state_for_run.return_value = {"project_id": "test_proj", "current_chapter": 1, "total_chapters": 1}
            mock_cp.return_value = mock_checkpointer
            mock_graph_creator.return_value = mock_graph

            await orchestrator.run_novel_generation_loop()

        # Exactly one managed HTTP client should have been created and closed.
        mock_async_client_ctor.assert_called_once()
        dummy_httpx_client.aclose.assert_awaited_once()

    async def test_run_novel_generation_loop_closes_llm_http_client_on_failure(self, orchestrator: LangGraphOrchestrator) -> None:
        """LLM HTTP client is closed even when workflow creation fails inside the lifecycle boundary."""
        assert isinstance(orchestrator.display, MagicMock)
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        orchestrator.display.stop = AsyncMock()

        dummy_httpx_client = MagicMock()
        dummy_httpx_client.aclose = AsyncMock()

        with (
            patch("core.http_client_service.httpx.AsyncClient", return_value=dummy_httpx_client) as mock_async_client_ctor,
            patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock),
            patch.object(orchestrator, "_load_state_for_run", new_callable=AsyncMock) as mock_state_for_run,
            patch("orchestration.langgraph_orchestrator.create_checkpointer") as mock_cp,
            patch(
                "orchestration.langgraph_orchestrator.create_full_workflow_graph",
                side_effect=RuntimeError("Graph build failed"),
            ),
        ):
            mock_state_for_run.return_value = {"project_id": "test_proj", "current_chapter": 1, "total_chapters": 1}
            mock_cp.return_value = mock_checkpointer

            with pytest.raises(RuntimeError, match="Graph build failed"):
                await orchestrator.run_novel_generation_loop()

        mock_async_client_ctor.assert_called_once()
        dummy_httpx_client.aclose.assert_awaited_once()

    async def test_run_novel_generation_loop_handles_errors(self, orchestrator: LangGraphOrchestrator) -> None:
        """Generation loop handles errors gracefully."""
        assert isinstance(orchestrator.display, MagicMock)
        orchestrator.display.stop = AsyncMock()

        with patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock) as mock_neo4j:
            mock_neo4j.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await orchestrator.run_novel_generation_loop()

            orchestrator.display.stop.assert_called_once()

    async def test_run_novel_generation_loop_stops_display_on_error(self, orchestrator: LangGraphOrchestrator) -> None:
        """Display is stopped even when errors occur."""
        assert isinstance(orchestrator.display, MagicMock)
        orchestrator.display.stop = AsyncMock()

        with patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock) as mock_neo4j:
            mock_neo4j.side_effect = Exception("Test error")

            with pytest.raises(Exception, match="Test error"):
                await orchestrator.run_novel_generation_loop()

            orchestrator.display.stop.assert_called_once()


def test_no_module_level_service_alias_importers() -> None:
    repo_root = _repo_root_from_test_file(Path(__file__))
    expected_modules = _module_level_llm_service_importers(repo_root)
    assert expected_modules == set()


def test_llm_service_import_finder_ignores_non_import_references() -> None:
    source = """
text = "llm_service"
def f():
    value = "llm_service"
"""
    tree = ast.parse(source)
    finder = _ModuleLevelLLMServiceImportFinder()
    finder.visit(tree)
    assert finder.found is False


@pytest.mark.asyncio
async def test_checkpoint_thread_id_is_per_project(tmp_path: Path) -> None:
    """Two different project_ids do not share checkpoints (same DB, different thread_id)."""

    class DemoState(TypedDict, total=False):
        project_id: str
        current_chapter: int

    def bump(state: DemoState) -> DemoState:
        return {**state, "current_chapter": int(state.get("current_chapter", 0)) + 1}

    orchestrator = LangGraphOrchestrator()
    orchestrator.project_dir = tmp_path

    db_path = tmp_path / "checkpoints.db"

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = StateGraph(DemoState)
        graph.add_node("bump", bump)
        graph.set_entry_point("bump")
        graph.add_edge("bump", END)
        compiled = graph.compile(checkpointer=saver)

        thread_a = orchestrator._checkpoint_thread_id("project_a")
        thread_b = orchestrator._checkpoint_thread_id("project_b")

        checkpoint_b_before = await saver.aget({"configurable": {"thread_id": thread_b}})
        assert checkpoint_b_before is None

        await compiled.ainvoke({"project_id": "project_a", "current_chapter": 1}, config={"configurable": {"thread_id": thread_a}})

        checkpoint_a = await saver.aget({"configurable": {"thread_id": thread_a}})
        assert isinstance(checkpoint_a, dict)
        assert checkpoint_a["channel_values"]["project_id"] == "project_a"

        checkpoint_b_after = await saver.aget({"configurable": {"thread_id": thread_b}})
        assert checkpoint_b_after is None


@pytest.mark.asyncio
async def test_resume_uses_checkpoint_state_not_neo4j(orchestrator: LangGraphOrchestrator) -> None:
    requested_project_id = "resume_project"
    thread_id = orchestrator._checkpoint_thread_id(requested_project_id)

    from core.graph_ownership import load_graph_project_id

    fake_checkpointer = MagicMock()
    fake_checkpointer.aget_state = AsyncMock(
        return_value=MagicMock(
            values={**create_initial_state(
                project_id=requested_project_id, project_dir=str(orchestrator.project_dir),
                title="Synthetic", genre="Mystery", theme="Trust", setting="Archive",
                protagonist_name="Mara", target_word_count=1000, total_chapters=10,
            ), "project_id": requested_project_id, "current_chapter": 7, "initialization_complete": True,
                "lifecycle_version": 1, "project_dir": str(orchestrator.project_dir),
                "graph_project_id": load_graph_project_id(orchestrator.project_dir)},
            created_at="2020-01-01T00:00:00Z",
        )
    )

    with (
        patch(
            "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
            new_callable=AsyncMock,
        ) as mock_neo4j_count,
        patch.object(orchestrator, "_load_or_create_state", new_callable=AsyncMock) as mock_seed_state,
    ):
        mock_neo4j_count.return_value = ChapterProgress(6, (1, 2, 3, 4, 5, 6))

        state = await orchestrator._load_state_for_run(
            graph=fake_checkpointer,
            requested_project_id=requested_project_id,
            thread_id=thread_id,
            narrative_config=None,
        )

        assert state["current_chapter"] == 7
        mock_seed_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_conflict_project_id_mismatch_raises(orchestrator: LangGraphOrchestrator) -> None:
    fake_checkpointer = MagicMock()
    fake_checkpointer.aget_state = AsyncMock(
        return_value=MagicMock(values={"project_id": "checkpoint_project", "current_chapter": 1}, created_at="2020-01-01T00:00:00Z")
    )

    with patch(
        "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
        new_callable=AsyncMock,
    ) as mock_neo4j_count:
        mock_neo4j_count.return_value = ChapterProgress(0, ())

        with pytest.raises(
            CheckpointResumeConflictError,
            match="Resume conflict: checkpoint project_id 'checkpoint_project' does not match requested project_id 'requested_project'",
        ):
            await orchestrator._load_state_for_run(
                graph=fake_checkpointer,
                requested_project_id="requested_project",
                thread_id=orchestrator._checkpoint_thread_id("requested_project"),
                narrative_config=None,
            )


@pytest.mark.asyncio
async def test_resume_conflict_missing_artifact_reference_raises(orchestrator: LangGraphOrchestrator, tmp_path: Path) -> None:
    orchestrator.project_dir = tmp_path

    requested_project_id = "artifact_project"

    fake_checkpointer = MagicMock()
    fake_checkpointer.aget_state = AsyncMock(
        return_value=MagicMock(
            values={
                "project_id": requested_project_id,
                "current_chapter": 3,
                "draft_ref": {"path": "does-not-exist.txt"},
            },
            created_at="2020-01-01T00:00:00Z",
        )
    )

    with patch(
        "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
        new_callable=AsyncMock,
    ) as mock_neo4j_count:
        mock_neo4j_count.return_value = ChapterProgress(2, (1, 2))

        with pytest.raises(
            CheckpointResumeConflictError,
            match=r"Resume conflict: checkpoint references missing artifact for field 'draft_ref': path='does-not-exist\.txt'",
        ):
            await orchestrator._load_state_for_run(
                graph=fake_checkpointer,
                requested_project_id=requested_project_id,
                thread_id=orchestrator._checkpoint_thread_id(requested_project_id),
                narrative_config=None,
            )


@pytest.mark.asyncio
async def test_resume_conflict_neo4j_ahead_of_checkpoint_raises(orchestrator: LangGraphOrchestrator) -> None:
    requested_project_id = "neo4j_ahead_project"

    fake_checkpointer = MagicMock()
    fake_checkpointer.aget_state = AsyncMock(
        return_value=MagicMock(values={"project_id": requested_project_id, "current_chapter": 3}, created_at="2020-01-01T00:00:00Z")
    )

    with patch(
        "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_progress_from_db",
        new_callable=AsyncMock,
    ) as mock_neo4j_count:
        mock_neo4j_count.return_value = ChapterProgress(4, (1, 2, 3, 4))

        with pytest.raises(
            CheckpointResumeConflictError,
            match="Resume conflict: Neo4j has finalized chapters at or ahead of checkpoint current_chapter=3",
        ):
            await orchestrator._load_state_for_run(
                graph=fake_checkpointer,
                requested_project_id=requested_project_id,
                thread_id=orchestrator._checkpoint_thread_id(requested_project_id),
                narrative_config=None,
            )


@pytest.mark.asyncio
async def test_load_state_for_run_no_checkpoint_uses_seed_state(orchestrator: LangGraphOrchestrator) -> None:
    requested_project_id = "no_checkpoint_project"

    fake_checkpointer = MagicMock()
    fake_checkpointer.aget_state = AsyncMock(return_value=MagicMock(values={}, created_at=None))

    with patch.object(orchestrator, "_load_or_create_state", new_callable=AsyncMock) as mock_seed_state:
        mock_seed_state.return_value = {"project_id": requested_project_id, "current_chapter": 1}

        state = await orchestrator._load_state_for_run(
            graph=fake_checkpointer,
            requested_project_id=requested_project_id,
            thread_id=orchestrator._checkpoint_thread_id(requested_project_id),
            narrative_config=None,
        )

        assert state["project_id"] == requested_project_id
        assert state["current_chapter"] == 1
        mock_seed_state.assert_awaited_once()
