import pytest
from unittest.mock import AsyncMock

from models import WorldItem
import processing.revision_logic as revision_logic
from agents.revision_agent import RevisionAgent


@pytest.mark.asyncio
async def test_revision_logic_passes_canonical_world_ids(monkeypatch):
    world_item = WorldItem.from_dict("Places", "City", {"description": "d"})
    world_building = {"Places": {"City": world_item}}
    eval_result = {
        "needs_revision": True,
        "problems_found": [
            {
                "issue_category": "style",
                "problem_description": "d",
                "quote_from_original_text": "Hello",
                "sentence_char_start": 0,
                "sentence_char_end": 5,
                "suggested_fix_focus": "fix",
            }
        ],
    }

    async def fake_generate(*_args, **_kwargs):
        return [
            {"target_char_start": 0, "target_char_end": 5, "replace_with": "Hi"}
        ], None

    async def fake_apply(*_args, **_kwargs):
        return "Hi world", [(0, 5)]

    async def fake_evaluate(*args, **kwargs):
        assert args[2] == {"Places": [world_item.id]}
        return {"problems_found": []}, None

    monkeypatch.setattr(
        revision_logic, "_generate_patch_instructions_logic", fake_generate
    )
    monkeypatch.setattr(revision_logic, "_apply_patches_to_text", fake_apply)
    monkeypatch.setattr(
        RevisionAgent,
        "evaluate_chapter_draft",
        AsyncMock(side_effect=fake_evaluate),
    )

    res, _ = await revision_logic.revise_chapter_draft_logic(
        {},
        {},
        world_building,
        "Hello world",
        1,
        eval_result,
        "",
        None,
    )
    assert res is not None
