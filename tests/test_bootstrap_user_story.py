# tests/test_bootstrap_user_story.py
import yaml

import pytest

import config
from initialization.bootstrap_pipeline import run_bootstrap_pipeline


@pytest.mark.asyncio
async def test_run_bootstrap_pipeline_skips_with_user_story(monkeypatch, tmp_path):
    data = {
        "novel_concept": {
            "title": "Clockwork Test",
            "genre": "gaslamp",
            "theme": "balance",
            "setting": "Verdalia",
        },
        "protagonist": {
            "name": "Maris",
            "traits": ["careful"],
        },
        "plot_elements": {
            "inciting_incident": "Engine failure",
            "plot_points": [
                "Maris discovers sabotage",
                "Maris negotiates with Rootbound",
            ],
            "central_conflict": "Control vs growth",
            "stakes": "The city collapses",
        },
        "setting": {
            "primary_setting_overview": "Verdalia",
            "key_locations": [
                {
                    "name": "Heartward Conservatory",
                    "description": "A greenhouse",
                    "atmosphere": "Humid",
                }
            ],
        },
    }

    story_path = tmp_path / "story.yaml"
    story_path.write_text(yaml.dump(data))
    monkeypatch.setattr(config, "USER_STORY_ELEMENTS_FILE_PATH", str(story_path))

    original_overrides = {
        "CONFIGURED_GENRE": config.CONFIGURED_GENRE,
        "CONFIGURED_THEME": config.CONFIGURED_THEME,
        "CONFIGURED_SETTING_DESCRIPTION": config.CONFIGURED_SETTING_DESCRIPTION,
        "DEFAULT_PROTAGONIST_NAME": config.DEFAULT_PROTAGONIST_NAME,
    }

    async def fail_world(*_args, **_kwargs):
        raise AssertionError("bootstrap_world should not run when user story is complete")

    async def fail_characters(*_args, **_kwargs):
        raise AssertionError(
            "bootstrap_characters should not run when user story is complete"
        )

    async def fail_plot(*_args, **_kwargs):
        raise AssertionError("bootstrap_plot_outline should not run when user story is complete")

    monkeypatch.setattr(
        "initialization.bootstrap_pipeline.bootstrap_world",
        fail_world,
    )
    monkeypatch.setattr(
        "initialization.bootstrap_pipeline.bootstrap_characters",
        fail_characters,
    )
    monkeypatch.setattr(
        "initialization.bootstrap_pipeline.bootstrap_plot_outline",
        fail_plot,
    )

    calls: dict[str, int | dict[str, str]] = {}

    class StubKnowledgeAgent:
        async def persist_world(self, *_args, **_kwargs):
            calls["persist_world"] = calls.get("persist_world", 0) + 1

        async def persist_profiles(self, *_args, **_kwargs):
            calls["persist_profiles"] = calls.get("persist_profiles", 0) + 1

        async def heal_and_enrich_kg(self, *_args, **_kwargs):
            calls["heal"] = calls.get("heal", 0) + 1

    async def stub_save_plot(plot_outline):
        calls["save_plot"] = {
            "title": plot_outline.get("title"),
            "source": plot_outline.get("source"),
        }

    monkeypatch.setattr(
        "initialization.bootstrap_pipeline.KnowledgeAgent",
        StubKnowledgeAgent,
    )
    monkeypatch.setattr(
        "initialization.bootstrap_pipeline.plot_queries.save_plot_outline_to_db",
        stub_save_plot,
    )

    try:
        plot_outline, character_profiles, world_building, warnings = (
            await run_bootstrap_pipeline(kg_heal=False)
        )
    finally:
        for key, value in original_overrides.items():
            config.set(key, value)

    assert plot_outline["source"] == "user_story_elements"
    assert calls.get("persist_world") == 1
    assert calls.get("persist_profiles") == 1
    assert isinstance(calls.get("save_plot"), dict)
    assert warnings == []
    assert "Heartward Conservatory" in world_building.get("locations", {})
