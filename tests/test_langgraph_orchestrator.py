# tests/test_langgraph_orchestrator.py
"""
Tests for LangGraph orchestrator.

Covers orchestration/langgraph_orchestrator.py.
"""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.langgraph.state import Contradiction
from orchestration.langgraph_orchestrator import (
    LangGraphOrchestrator,
    _LLM_SERVICE_PATCH_MODULES,
)


@pytest.fixture
def mock_config():
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
        yield mock_cfg


@pytest.fixture
def orchestrator(mock_config, tmp_path):
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

    def test_init_creates_project_dir_path(self, orchestrator):
        """Orchestrator initializes with project directory path."""
        assert orchestrator.project_dir is not None
        assert isinstance(orchestrator.project_dir, Path)

    def test_init_creates_checkpointer_path(self, orchestrator):
        """Orchestrator initializes with checkpointer path."""
        assert orchestrator.checkpointer_path is not None
        assert str(orchestrator.checkpointer_path).endswith("checkpoints.db")

    def test_init_creates_display_manager(self, orchestrator):
        """Orchestrator initializes with display manager."""
        assert orchestrator.display is not None

    def test_init_sets_run_start_time(self, orchestrator):
        """Orchestrator initializes with run start time."""
        assert orchestrator.run_start_time == 0.0


@pytest.mark.asyncio
class TestEnsureNeo4jConnection:
    """Tests for _ensure_neo4j_connection method."""

    async def test_ensure_neo4j_connection_connects(self, orchestrator):
        """Neo4j connection is established and schema created."""
        with patch("orchestration.langgraph_orchestrator.neo4j_manager") as mock_neo4j:
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.create_db_schema = AsyncMock()

            await orchestrator._ensure_neo4j_connection()

            mock_neo4j.connect.assert_called_once()
            mock_neo4j.create_db_schema.assert_called_once()

    async def test_ensure_neo4j_connection_error_propagates(self, orchestrator):
        """Neo4j connection errors propagate."""
        with patch("orchestration.langgraph_orchestrator.neo4j_manager") as mock_neo4j:
            mock_neo4j.connect = AsyncMock(side_effect=Exception("Connection failed"))

            with pytest.raises(Exception, match="Connection failed"):
                await orchestrator._ensure_neo4j_connection()


@pytest.mark.asyncio
class TestLoadOrCreateState:
    """Tests for _load_or_create_state method."""

    async def test_load_or_create_state_no_existing_chapters(self, orchestrator):
        """State is created with chapter 1 when no chapters exist."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_count_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "orchestration.langgraph_orchestrator.validate_initialization_artifacts",
                return_value=(False, ["Missing saga.yaml"]),
            ),
        ):
            mock_load.return_value = 0

            state = await orchestrator._load_or_create_state()

            assert state["current_chapter"] == 1
            assert state["initialization_complete"] is False

    async def test_load_or_create_state_existing_chapters(self, orchestrator):
        """State continues from next chapter when chapters exist."""
        with patch(
            "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_count_from_db",
            new_callable=AsyncMock,
        ) as mock_load:
            mock_load.return_value = 5

            state = await orchestrator._load_or_create_state()

            assert state["current_chapter"] == 6
            assert state["initialization_complete"] is True

    async def test_load_or_create_state_detects_initialization(self, orchestrator):
        """Initialization is detected from filesystem artifacts on chapter 1."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_count_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "orchestration.langgraph_orchestrator.validate_initialization_artifacts",
                return_value=(True, []),
            ),
        ):
            mock_load.return_value = 0

            state = await orchestrator._load_or_create_state()

            assert state["initialization_complete"] is True

    async def test_load_or_create_state_fallback_to_file_check(self, orchestrator, tmp_path):
        """Chapter 1 uses filesystem artifacts; a minimal complete set marks init complete."""
        with patch(
            "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_count_from_db",
            new_callable=AsyncMock,
        ) as mock_load:
            mock_load.return_value = 0

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

            state = await orchestrator._load_or_create_state()

            assert state["initialization_complete"] is True

    async def test_load_or_create_state_includes_models(self, orchestrator):
        """State includes model configuration."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_count_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch(
                "orchestration.langgraph_orchestrator.validate_initialization_artifacts",
                return_value=(False, ["Missing saga.yaml"]),
            ),
        ):
            mock_load.return_value = 0

            state = await orchestrator._load_or_create_state()

            assert state["generation_model"] is not None
            assert state["extraction_model"] is not None
            assert state["revision_model"] is not None

    async def test_load_or_create_state_validates_artifacts(self, orchestrator, tmp_path):
        """Validation check runs on existing project directory."""
        with (
            patch(
                "orchestration.langgraph_orchestrator.chapter_queries.load_chapter_count_from_db",
                new_callable=AsyncMock,
            ) as mock_load,
            patch("orchestration.langgraph_orchestrator.validate_initialization_artifacts") as mock_validate,
        ):
            mock_load.return_value = 0
            mock_validate.return_value = (True, [])

            orchestrator.project_dir = tmp_path
            tmp_path.mkdir(exist_ok=True)

            state = await orchestrator._load_or_create_state()

            mock_validate.assert_called_once_with(tmp_path)
            assert state is not None


@pytest.mark.asyncio
class TestRunChapterGenerationLoop:
    """Tests for _run_chapter_generation_loop method."""

    async def test_run_chapter_generation_loop_basic(self, orchestrator):
        """Chapter generation loop runs for configured chapters."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args, **kwargs):
            events = [
                {"generate": {"current_node": "generate", "draft_text": "Chapter text"}},
                {"finalize": {"current_node": "finalize", "draft_word_count": 2000}},
            ]
            for event in events:
                yield event

        mock_graph.astream = mock_stream_func

        state = {
            "current_chapter": 1,
            "total_chapters": 20,
            "draft_word_count": 2000,
        }

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_multi_chapter(self, orchestrator):
        """Chapter generation loop handles multiple chapters in a single stream."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args, **kwargs):
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

        state = {
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

    async def test_run_chapter_generation_loop_thread_id(self, orchestrator):
        """Orchestrator uses project-specific thread ID."""
        mock_graph = MagicMock()

        async def empty_stream(*args, **kwargs):
            if False:
                yield

        mock_graph.astream = MagicMock(side_effect=empty_stream)

        state = {"project_id": "custom_project"}
        await orchestrator._run_chapter_generation_loop(mock_graph, state)

        # Check astream call arguments
        # Need to find the call
        found = False
        for call in mock_graph.astream.call_args_list:
            if call.kwargs.get("config", {}).get("configurable", {}).get("thread_id") == "saga_custom_project":
                found = True
                break
        assert found, "Thread ID 'saga_custom_project' not found in astream calls"

    async def test_run_chapter_generation_loop_respects_total_chapters(self, orchestrator):
        """Loop stops at total chapter count."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args, **kwargs):
            events = [{"finalize": {"current_node": "finalize", "draft_word_count": 2000}}]
            for event in events:
                yield event

        mock_graph.astream = mock_stream_func

        state = {"current_chapter": 20, "total_chapters": 20, "draft_word_count": 2000}

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_stops_on_error(self, orchestrator):
        """Loop stops when chapter generation fails."""
        mock_graph = MagicMock()

        async def mock_stream_error(*args, **kwargs):
            raise Exception("Generation error")
            if False:
                yield

        mock_graph.astream = mock_stream_error

        state = {"current_chapter": 1, "total_chapters": 20}

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_handles_incomplete_generation(self, orchestrator):
        """Loop stops if generation doesn't reach finalize node."""
        mock_graph = MagicMock()

        async def mock_stream_func(*args, **kwargs):
            events = [{"extract": {"current_node": "extract", "extracted_entities": {}}}]
            for event in events:
                yield event

        mock_graph.astream = mock_stream_func

        state = {
            "current_chapter": 1,
            "total_chapters": 20,
            "last_error": "Failed at extraction",
        }

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)

    async def test_run_chapter_generation_loop_handles_no_events(self, orchestrator):
        """Loop handles case where no events are received."""
        mock_graph = MagicMock()

        async def empty_stream(*args, **kwargs):
            if False:
                yield

        mock_graph.astream = empty_stream

        state = {"current_chapter": 1, "total_chapters": 20}

        with patch.object(orchestrator, "_handle_workflow_event", new_callable=AsyncMock):
            await orchestrator._run_chapter_generation_loop(mock_graph, state)


@pytest.mark.asyncio
class TestHandleWorkflowEvent:
    """Tests for _handle_workflow_event method."""

    async def test_handle_workflow_event_basic(self, orchestrator):
        """Basic event handling works."""
        event = {"generate": {"current_node": "generate", "draft_text": "Text"}}

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_skips_internal_nodes(self, orchestrator):
        """Internal nodes are skipped."""
        event = {"__start__": {"current_node": "__start__"}}

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_invalid_event(self, orchestrator):
        """Invalid events are handled gracefully."""
        await orchestrator._handle_workflow_event({}, 1)
        await orchestrator._handle_workflow_event(None, 1)
        await orchestrator._handle_workflow_event("not a dict", 1)

    async def test_handle_workflow_event_validate_with_contradictions(self, orchestrator):
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

    async def test_handle_workflow_event_validate_with_dict_contradictions(self, orchestrator):
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

    async def test_handle_workflow_event_revise(self, orchestrator):
        """Revision events log iteration count."""
        event = {
            "revise": {
                "current_node": "revise",
                "iteration_count": 2,
                "max_iterations": 3,
            }
        }

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_finalize(self, orchestrator):
        """Finalize events log word count."""
        event = {"finalize": {"current_node": "finalize", "draft_word_count": 2500}}

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_init_complete(self, orchestrator):
        """Initialization complete events log counts."""
        event = {
            "init_complete": {
                "current_node": "init_complete",
                "character_sheets": {"Alice": {}, "Bob": {}},
                "act_outlines": {"act1": {}, "act2": {}},
            }
        }

        await orchestrator._handle_workflow_event(event, 1)

    async def test_handle_workflow_event_updates_display(self, orchestrator):
        """Workflow events update the display."""
        event = {
            "generate": {
                "current_node": "generate",
                "title": "Test Novel",
            }
        }

        orchestrator.run_start_time = 1000.0

        await orchestrator._handle_workflow_event(event, 1)

        orchestrator.display.update.assert_called_once()


class TestGetStepDescription:
    """Tests for _get_step_description method."""

    def test_get_step_description_initialization_steps(self, orchestrator):
        """Initialization steps are mapped correctly."""
        assert "Character Sheets" in orchestrator._get_step_description("node", "character_sheets")
        assert "Global Story Outline" in orchestrator._get_step_description("node", "global_outline")
        assert "Act Structures" in orchestrator._get_step_description("node", "act_outlines")
        assert "Knowledge Graph" in orchestrator._get_step_description("node", "committing")
        assert "Initialization Files" in orchestrator._get_step_description("node", "files_persisted")
        assert "Initialization Complete" in orchestrator._get_step_description("node", "complete")

    def test_get_step_description_generation_nodes(self, orchestrator):
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

    def test_get_step_description_ignores_chapter_outline_markers(self, orchestrator):
        """Chapter outline completion markers are ignored."""
        result = orchestrator._get_step_description("some_node", "chapter_outline_2_complete")

        assert "Processing" in result or "some_node" in result

    def test_get_step_description_unknown_node(self, orchestrator):
        """Unknown nodes return generic description."""
        result = orchestrator._get_step_description("unknown_node")

        assert "Processing" in result
        assert "unknown_node" in result

    def test_get_step_description_empty_init_step(self, orchestrator):
        """Empty initialization step falls back to node descriptions."""
        result = orchestrator._get_step_description("generate", "")

        assert "Chapter Text" in result


@pytest.mark.asyncio
class TestRunNovelGenerationLoop:
    """Tests for run_novel_generation_loop method."""

    async def test_run_novel_generation_loop_full_flow(self, orchestrator):
        """Full generation loop runs successfully."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        mock_graph = MagicMock()

        async def mock_events(*args, **kwargs):
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
            patch.object(orchestrator, "_load_or_create_state", new_callable=AsyncMock) as mock_state,
            patch("orchestration.langgraph_orchestrator.create_checkpointer") as mock_cp,
            patch("orchestration.langgraph_orchestrator.create_full_workflow_graph") as mock_graph_creator,
        ):
            mock_state.return_value = {
                "current_chapter": 1,
                "total_chapters": 20,
            }
            mock_cp.return_value = mock_checkpointer
            mock_graph_creator.return_value = mock_graph

            await orchestrator.run_novel_generation_loop()

            mock_neo4j.assert_called_once()
            mock_state.assert_called_once()
            orchestrator.display.start.assert_called_once()
            orchestrator.display.stop.assert_called_once()

    async def test_run_novel_generation_loop_closes_llm_http_client_on_success(self, orchestrator):
        """
        Orchestrator establishes an explicit LLM HTTP client lifecycle boundary and closes it.

        This is the remediation target for CORE-004 / LANGGRAPH-026:
        workflows should not rely on import-time singleton cleanup.
        """
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        mock_graph = MagicMock()

        async def mock_events(*args, **kwargs):
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
            patch.object(orchestrator, "_load_or_create_state", new_callable=AsyncMock) as mock_state,
            patch("orchestration.langgraph_orchestrator.create_checkpointer") as mock_cp,
            patch("orchestration.langgraph_orchestrator.create_full_workflow_graph") as mock_graph_creator,
        ):
            mock_state.return_value = {"current_chapter": 1, "total_chapters": 1}
            mock_cp.return_value = mock_checkpointer
            mock_graph_creator.return_value = mock_graph

            await orchestrator.run_novel_generation_loop()

        # Exactly one managed HTTP client should have been created and closed.
        mock_async_client_ctor.assert_called_once()
        dummy_httpx_client.aclose.assert_awaited_once()

    async def test_run_novel_generation_loop_closes_llm_http_client_on_failure(self, orchestrator):
        """LLM HTTP client is closed even when workflow creation fails inside the lifecycle boundary."""
        mock_checkpointer = MagicMock()
        mock_checkpointer.__aenter__ = AsyncMock(return_value=mock_checkpointer)
        mock_checkpointer.__aexit__ = AsyncMock(return_value=None)

        orchestrator.display.stop = AsyncMock()

        dummy_httpx_client = MagicMock()
        dummy_httpx_client.aclose = AsyncMock()

        with (
            patch("core.http_client_service.httpx.AsyncClient", return_value=dummy_httpx_client) as mock_async_client_ctor,
            patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock),
            patch.object(orchestrator, "_load_or_create_state", new_callable=AsyncMock) as mock_state,
            patch("orchestration.langgraph_orchestrator.create_checkpointer") as mock_cp,
            patch(
                "orchestration.langgraph_orchestrator.create_full_workflow_graph",
                side_effect=RuntimeError("Graph build failed"),
            ),
        ):
            mock_state.return_value = {"current_chapter": 1, "total_chapters": 1}
            mock_cp.return_value = mock_checkpointer

            with pytest.raises(RuntimeError, match="Graph build failed"):
                await orchestrator.run_novel_generation_loop()

        mock_async_client_ctor.assert_called_once()
        dummy_httpx_client.aclose.assert_awaited_once()

    async def test_run_novel_generation_loop_handles_errors(self, orchestrator):
        """Generation loop handles errors gracefully."""
        orchestrator.display.stop = AsyncMock()

        with patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock) as mock_neo4j:
            mock_neo4j.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await orchestrator.run_novel_generation_loop()

            orchestrator.display.stop.assert_called_once()

    async def test_run_novel_generation_loop_stops_display_on_error(self, orchestrator):
        """Display is stopped even when errors occur."""
        orchestrator.display.stop = AsyncMock()

        with patch.object(orchestrator, "_ensure_neo4j_connection", new_callable=AsyncMock) as mock_neo4j:
            mock_neo4j.side_effect = Exception("Test error")

            try:
                await orchestrator.run_novel_generation_loop()
            except Exception:
                pass

            orchestrator.display.stop.assert_called_once()


def test_llm_service_patch_list_covers_all_module_level_importers() -> None:
    repo_root = _repo_root_from_test_file(Path(__file__))
    expected_modules = _module_level_llm_service_importers(repo_root)

    patch_modules = set(_LLM_SERVICE_PATCH_MODULES)
    missing = sorted(expected_modules - patch_modules)
    assert missing == [], f"Missing llm_service patch modules: {missing}"

    # Edge case contract: function-local imports of llm_service should not require patching because
    # they resolve `core.llm_interface_refactored.llm_service` at call time.
    assert "utils.similarity" not in expected_modules


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
