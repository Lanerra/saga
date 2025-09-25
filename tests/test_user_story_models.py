# tests/test_user_story_models.py
import yaml

import config
from initialization.data_loader import (
    load_user_supplied_model as _load_user_supplied_data,
)
from models.user_input_models import (
    CharacterGroupModel,
    NovelConceptModel,
    ProtagonistModel,
    UserStoryInputModel,
    user_story_to_objects,
)


def test_load_user_supplied_data_valid(tmp_path, monkeypatch):
    data = {
        "novel_concept": {
            "title": "Test",
            "genre": "test genre",
            "theme": "heroism",
            "setting": "Testland",
        },
        "protagonist": {"name": "Casey"},
    }
    file_path = tmp_path / "story.yaml"
    file_path.write_text(yaml.dump(data))
    monkeypatch.setattr(config, "USER_STORY_ELEMENTS_FILE_PATH", str(file_path))

    original_overrides = {
        "CONFIGURED_GENRE": config.CONFIGURED_GENRE,
        "CONFIGURED_THEME": config.CONFIGURED_THEME,
        "CONFIGURED_SETTING_DESCRIPTION": config.CONFIGURED_SETTING_DESCRIPTION,
        "DEFAULT_PROTAGONIST_NAME": config.DEFAULT_PROTAGONIST_NAME,
    }

    try:
        model = _load_user_supplied_data()
        assert isinstance(model, UserStoryInputModel)
        assert model.novel_concept
        assert model.novel_concept.title == "Test"
        assert config.CONFIGURED_GENRE == "test genre"
        assert config.CONFIGURED_THEME == "heroism"
        assert config.CONFIGURED_SETTING_DESCRIPTION == "Testland"
        assert config.DEFAULT_PROTAGONIST_NAME == "Casey"
    finally:
        for key, value in original_overrides.items():
            config.set(key, value)


def test_load_user_supplied_data_invalid(tmp_path, monkeypatch):
    file_path = tmp_path / "bad.yaml"
    file_path.write_text("- item1\n- item2")
    monkeypatch.setattr(config, "USER_STORY_ELEMENTS_FILE_PATH", str(file_path))

    result = _load_user_supplied_data()
    assert result is None


def test_user_story_to_objects():
    model = UserStoryInputModel(
        novel_concept=NovelConceptModel(title="My Tale"),
        protagonist=ProtagonistModel(name="Hero"),
    )
    plot, characters, world_items = user_story_to_objects(model)
    assert plot["title"] == "My Tale"
    assert "Hero" in characters
    assert world_items == {}


def test_user_story_to_objects_nested_characters():
    model = UserStoryInputModel(
        characters=CharacterGroupModel(
            protagonist=ProtagonistModel(name="Saga"),
            antagonist=ProtagonistModel(name="Collective"),
            supporting_characters=[ProtagonistModel(name="Dr. Larkin", role="mentor")],
        ),
    )
    plot, characters, world_items = user_story_to_objects(model)
    assert plot.get("protagonist_name") == "Saga"
    assert "Saga" in characters
    assert characters["Dr. Larkin"].updates["role"] == "mentor"
    assert world_items == {}
