# tests/test_chapter_flow.py
import pytest

from orchestration import chapter_flow


class DummyOrchestrator:
    def __init__(self):
        self.values = {}

    async def _update_rich_display(self, **kwargs):
        pass

    async def _validate_plot_outline(self, chapter):
        return self.values.get("validate", True)

    async def _prepare_chapter_prerequisites(self, chapter):
        return self.values.get("prereq", object())

    async def _process_prereq_result(self, chapter, prereq_result):
        return self.values.get(
            "process_prereqs",
            ("focus", 1, "plan", "ctx"),
        )

    async def _draft_initial_chapter_text(self, *args):
        return self.values.get("draft", "draft_res")

    async def _process_initial_draft(self, chapter, draft_result):
        return self.values.get("process_draft", ("txt", "raw"))

    async def _process_and_revise_draft(self, *args):
        return self.values.get("revise", "revise_res")

    async def _process_revision_result(self, chapter, rev_res):
        return self.values.get(
            "process_revision",
            ("processed", "raw", False),
        )

    async def _finalize_and_log(self, *args):
        return self.values.get("final", "done")


@pytest.mark.asyncio
async def test_run_chapter_pipeline_success():
    orch = DummyOrchestrator()
    result = await chapter_flow.run_chapter_pipeline(orch, 1)
    assert result == "done"


@pytest.mark.asyncio
async def test_run_chapter_pipeline_invalid_outline():
    orch = DummyOrchestrator()
    orch.values["validate"] = False
    result = await chapter_flow.run_chapter_pipeline(orch, 1)
    assert result is None


@pytest.mark.asyncio
async def test_run_chapter_pipeline_prereq_none():
    orch = DummyOrchestrator()
    orch.values["process_prereqs"] = None
    result = await chapter_flow.run_chapter_pipeline(orch, 1)
    assert result is None


@pytest.mark.asyncio
async def test_run_chapter_pipeline_draft_none():
    orch = DummyOrchestrator()
    orch.values["process_draft"] = None
    result = await chapter_flow.run_chapter_pipeline(orch, 1)
    assert result is None


@pytest.mark.asyncio
async def test_run_chapter_pipeline_revision_none():
    orch = DummyOrchestrator()
    orch.values["process_revision"] = None
    result = await chapter_flow.run_chapter_pipeline(orch, 1)
    assert result is None
