import yaml

import config
from initial_setup_logic import _load_user_supplied_data
from story_models import (
    NovelConceptModel,
    ProtagonistModel,
    UserStoryInputModel,
    user_story_to_objects,
)


def test_load_user_supplied_data_valid(tmp_path, monkeypatch):
    data = {"novel_concept": {"title": "Test"}}
    file_path = tmp_path / "story.yaml"
    file_path.write_text(yaml.dump(data))
    monkeypatch.setattr(config, "USER_STORY_ELEMENTS_FILE_PATH", str(file_path))

    model = _load_user_supplied_data()
    assert isinstance(model, UserStoryInputModel)
    assert model.novel_concept
    assert model.novel_concept.title == "Test"


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
    assert world_items == []
