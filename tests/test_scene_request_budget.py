"""Scene context must fit the production serialized request without reducing completion allowance."""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import config
from core.http_client_service import CompletionHTTPClient, HTTPClientService
from core.langgraph.content_manager import ContentManager
from core.langgraph.nodes.scene_generation_node import draft_scene
from core.service_context import get_services

SCENE = dict(title='Arrival', pov_character='Ada', setting='Hall', characters=['Ada'],
             plot_point='Find the ledger', conflict='An error', outcome='Read it', beats=['Arrival'])

@pytest.mark.parametrize('oversized', [False, True])
async def test_scene_context_fits_transport_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oversized: bool) -> None:
    manager = ContentManager(str(tmp_path))
    context = 'Retained fact: café 界 🌙 and ledger.\n' * (2000 if oversized else 1)
    ref = manager.save_text(context, 'hybrid_context', 'chapter_1_scene_0')
    llm = AsyncMock(return_value=('A real draft in this offline fixture.', {}))
    monkeypatch.setattr(get_services().language_model, 'async_call_llm', llm)
    result = await draft_scene({'project_dir': str(tmp_path), 'current_chapter': 1, 'current_scene_index': 0,
                                'total_chapters': 1, 'target_word_count': 900, 'narrative_model': 'test-model',
                                'chapter_plan_ref': manager.save_json([SCENE], 'chapter_plan', 'chapter_1'),
                                'hybrid_context_ref': ref})
    assert not result.get('has_fatal_error')
    call = llm.call_args.kwargs
    assert call['max_tokens'] == config.MAX_GENERATION_TOKENS
    assert manager.load_text(ref) == context
    assert ('context truncated for request budget' in call['prompt']) is oversized
    if not oversized:
        assert context in call['prompt']
    client = HTTPClientService()
    monkeypatch.setattr(client, 'post_json', AsyncMock(side_effect=RuntimeError('Reached transport after budget validation')))
    try:
        with pytest.raises(RuntimeError, match='Reached transport'):
            await CompletionHTTPClient(client).get_completion('test-model', [
                {'role': 'system', 'content': call['system_prompt']}, {'role': 'user', 'content': call['prompt']}],
                0.7, config.MAX_GENERATION_TOKENS)
    finally:
        await client.aclose()


async def test_fixed_scene_instructions_cannot_be_silently_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = ContentManager(str(tmp_path))
    llm = AsyncMock(return_value=('Should not be called.', {}))
    monkeypatch.setattr(get_services().language_model, 'async_call_llm', llm)
    result = await draft_scene({'project_dir': str(tmp_path), 'current_chapter': 1, 'current_scene_index': 0,
                                'title': 'Oversized immutable instructions ' * 4000,
                                'chapter_plan_ref': manager.save_json([SCENE], 'chapter_plan', 'chapter_1')})
    assert result['has_fatal_error'] is True
    llm.assert_not_called()
