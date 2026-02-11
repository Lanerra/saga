"""Tests for models/user_input_models.py - user story input parsing and conversion."""

from typing import Any

import pytest
from pydantic import ValidationError

from models.kg_models import CharacterProfile, WorldItem
from models.user_input_models import (
    CharacterGroupModel,
    KeyLocationModel,
    NovelConceptModel,
    PlotElementsModel,
    ProtagonistModel,
    RelationshipModel,
    SettingModel,
    UserStoryInputModel,
    user_story_to_objects,
)


class TestNovelConceptModel:

    def test_title_is_required(self) -> None:
        with pytest.raises(ValidationError):
            NovelConceptModel()

    def test_empty_title_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NovelConceptModel(title="")

    def test_valid_title_only(self) -> None:
        concept = NovelConceptModel(title="The Great Novel")
        assert concept.title == "The Great Novel"
        assert concept.genre is None
        assert concept.setting is None
        assert concept.theme is None

    def test_all_fields_populated(self) -> None:
        concept = NovelConceptModel(
            title="Epic Tale",
            genre="Fantasy",
            setting="Medieval Kingdom",
            theme="Redemption",
        )
        assert concept.title == "Epic Tale"
        assert concept.genre == "Fantasy"
        assert concept.setting == "Medieval Kingdom"
        assert concept.theme == "Redemption"


class TestProtagonistModel:

    def test_name_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ProtagonistModel()

    def test_name_only(self) -> None:
        protagonist = ProtagonistModel(name="Alice")
        assert protagonist.name == "Alice"
        assert protagonist.description is None
        assert protagonist.traits == []
        assert protagonist.motivation is None
        assert protagonist.role is None
        assert protagonist.relationships == {}

    def test_all_fields_populated(self) -> None:
        protagonist = ProtagonistModel(
            name="Alice",
            description="A brave warrior",
            traits=["courageous", "loyal"],
            motivation="Save the kingdom",
            role="protagonist",
            relationships={
                "Bob": RelationshipModel(
                    name="Bob",
                    status="ally",
                    details="Childhood friend",
                )
            },
        )
        assert protagonist.name == "Alice"
        assert protagonist.description == "A brave warrior"
        assert protagonist.traits == ["courageous", "loyal"]
        assert protagonist.motivation == "Save the kingdom"
        assert protagonist.role == "protagonist"
        assert protagonist.relationships["Bob"].name == "Bob"
        assert protagonist.relationships["Bob"].status == "ally"


class TestUserStoryInputModel:

    def test_allows_extra_fields(self) -> None:
        model = UserStoryInputModel(**{"custom_section": {"key": "value"}})
        assert model.model_extra == {"custom_section": {"key": "value"}}

    def test_all_known_fields_default_to_none(self) -> None:
        model = UserStoryInputModel()
        assert model.novel_concept is None
        assert model.protagonist is None
        assert model.antagonist is None
        assert model.characters is None
        assert model.plot_elements is None
        assert model.setting is None
        assert model.world_details is None
        assert model.other_key_characters is None
        assert model.style_and_tone is None


def _build_full_input() -> UserStoryInputModel:
    return UserStoryInputModel(
        novel_concept=NovelConceptModel(
            title="The Dark Tower",
            genre="Fantasy",
            setting="Post-apocalyptic wasteland",
            theme="Redemption",
        ),
        protagonist=ProtagonistModel(
            name="Roland",
            description="A lone gunslinger",
            traits=["determined", "stoic"],
            motivation="Reach the Dark Tower",
            role="gunslinger",
            relationships={
                "Jake": RelationshipModel(
                    name="Jake",
                    status="companion",
                    details="Reluctant protector",
                )
            },
        ),
        antagonist=ProtagonistModel(
            name="The Man in Black",
            description="A sorcerer",
            traits=["cunning", "manipulative"],
            motivation="Stop Roland",
        ),
        characters=CharacterGroupModel(
            supporting_characters=[
                ProtagonistModel(
                    name="Eddie",
                    description="A recovering addict",
                    traits=["witty", "brave"],
                    role="ka-tet member",
                ),
                ProtagonistModel(
                    name="Susannah",
                    description="A dual personality",
                    traits=["fierce", "intelligent"],
                ),
            ],
        ),
        plot_elements=PlotElementsModel(
            inciting_incident="Roland meets Jake at the way station",
            plot_points=["The drawing of the three", "The waste lands"],
            central_conflict="Roland versus the Crimson King",
            stakes="The fate of all worlds",
        ),
        setting=SettingModel(
            primary_setting_overview="A dying world moved on",
            key_locations=[
                KeyLocationModel(
                    name="Gilead",
                    description="The fallen city of gunslingers",
                    atmosphere="Nostalgic and ruined",
                ),
                KeyLocationModel(
                    name="The Dark Tower",
                    description="The nexus of all realities",
                    atmosphere="Awe-inspiring and terrifying",
                ),
            ],
        ),
        style_and_tone={
            "narrative_style": "Third person limited",
            "tone": "Dark and contemplative",
            "pacing": "Deliberate with action bursts",
        },
        world_details={
            "factions": {
                "Gunslingers": {"description": "Order of knights", "status": "Fallen"},
                "Breakers": {"description": "Psychic prisoners", "status": "Captive"},
            }
        },
    )


class TestUserStoryToObjectsPlotOutline:

    def test_novel_concept_fields_in_plot_outline(self) -> None:
        model = _build_full_input()
        plot_outline, _, _ = user_story_to_objects(model)

        assert plot_outline["title"] == "The Dark Tower"
        assert plot_outline["genre"] == "Fantasy"
        assert plot_outline["setting"] == "Post-apocalyptic wasteland"
        assert plot_outline["theme"] == "Redemption"

    def test_protagonist_name_in_plot_outline(self) -> None:
        model = _build_full_input()
        plot_outline, _, _ = user_story_to_objects(model)

        assert plot_outline["protagonist_name"] == "Roland"

    def test_plot_elements_in_plot_outline(self) -> None:
        model = _build_full_input()
        plot_outline, _, _ = user_story_to_objects(model)

        assert plot_outline["inciting_incident"] == "Roland meets Jake at the way station"
        assert plot_outline["plot_points"] == [
            "The drawing of the three",
            "The waste lands",
        ]
        assert plot_outline["central_conflict"] == "Roland versus the Crimson King"
        assert plot_outline["stakes"] == "The fate of all worlds"

    def test_style_and_tone_in_plot_outline(self) -> None:
        model = _build_full_input()
        plot_outline, _, _ = user_story_to_objects(model)

        assert plot_outline["narrative_style"] == "Third person limited"
        assert plot_outline["tone"] == "Dark and contemplative"
        assert plot_outline["pacing"] == "Deliberate with action bursts"


class TestUserStoryToObjectsProtagonist:

    def test_protagonist_role_in_updates(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        roland = characters["Roland"]
        assert roland.updates["role"] == "gunslinger"

    def test_protagonist_defaults_to_protagonist_role(self) -> None:
        model = UserStoryInputModel(
            protagonist=ProtagonistModel(name="Frodo"),
        )
        _, characters, _ = user_story_to_objects(model)

        assert characters["Frodo"].updates["role"] == "protagonist"

    def test_protagonist_description_and_traits(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        roland = characters["Roland"]
        assert roland.personality_description == "A lone gunslinger"
        assert roland.traits == ["determined", "stoic"]
        assert roland.status == "As described"

    def test_protagonist_motivation_in_updates(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        roland = characters["Roland"]
        assert roland.updates["motivation"] == "Reach the Dark Tower"

    def test_protagonist_relationships_converted(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        roland = characters["Roland"]
        assert "Jake" in roland.relationships
        assert roland.relationships["Jake"] == {
            "name": "Jake",
            "status": "companion",
            "details": "Reluctant protector",
        }


class TestUserStoryToObjectsAntagonist:

    def test_antagonist_role_in_updates(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        antagonist = characters["The Man in Black"]
        assert antagonist.updates["role"] == "antagonist"

    def test_antagonist_description_and_traits(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        antagonist = characters["The Man in Black"]
        assert antagonist.personality_description == "A sorcerer"
        assert antagonist.traits == ["cunning", "manipulative"]
        assert antagonist.status == "As described"

    def test_antagonist_with_explicit_role(self) -> None:
        model = UserStoryInputModel(
            antagonist=ProtagonistModel(
                name="Sauron",
                role="dark lord",
            ),
        )
        _, characters, _ = user_story_to_objects(model)

        assert characters["Sauron"].updates["role"] == "dark lord"


class TestUserStoryToObjectsSupportingCharacters:

    def test_supporting_characters_with_explicit_role(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        eddie = characters["Eddie"]
        assert eddie.updates["role"] == "ka-tet member"

    def test_supporting_characters_default_role(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        susannah = characters["Susannah"]
        assert susannah.updates["role"] == "supporting_character"

    def test_supporting_characters_description_and_traits(self) -> None:
        model = _build_full_input()
        _, characters, _ = user_story_to_objects(model)

        eddie = characters["Eddie"]
        assert eddie.personality_description == "A recovering addict"
        assert eddie.traits == ["witty", "brave"]


class TestUserStoryToObjectsSetting:

    def test_setting_creates_overview_world_item(self) -> None:
        model = _build_full_input()
        _, _, world_items = user_story_to_objects(model)

        overview = world_items["_overview_"]["_overview_"]
        assert isinstance(overview, WorldItem)
        assert overview.description == "A dying world moved on"

    def test_setting_creates_location_world_items(self) -> None:
        model = _build_full_input()
        _, _, world_items = user_story_to_objects(model)

        gilead = world_items["locations"]["Gilead"]
        assert isinstance(gilead, WorldItem)
        assert gilead.description == "The fallen city of gunslingers"
        assert gilead.additional_properties["atmosphere"] == "Nostalgic and ruined"

        tower = world_items["locations"]["The Dark Tower"]
        assert isinstance(tower, WorldItem)
        assert tower.description == "The nexus of all realities"
        assert tower.additional_properties["atmosphere"] == "Awe-inspiring and terrifying"

    def test_no_setting_returns_empty_world_items(self) -> None:
        model = UserStoryInputModel(
            novel_concept=NovelConceptModel(title="Minimal"),
        )
        _, _, world_items = user_story_to_objects(model)

        assert world_items == {}


class TestUserStoryToObjectsWorldDetails:

    def test_world_details_creates_world_items(self) -> None:
        model = _build_full_input()
        _, _, world_items = user_story_to_objects(model)

        gunslingers = world_items["factions"]["Gunslingers"]
        assert isinstance(gunslingers, WorldItem)
        assert gunslingers.description == "Order of knights"
        assert gunslingers.additional_properties["status"] == "Fallen"

        breakers = world_items["factions"]["Breakers"]
        assert isinstance(breakers, WorldItem)
        assert breakers.description == "Psychic prisoners"
        assert breakers.additional_properties["status"] == "Captive"


class TestUserStoryToObjectsCharactersFallback:

    def test_protagonist_from_characters_group(self) -> None:
        model = UserStoryInputModel(
            characters=CharacterGroupModel(
                protagonist=ProtagonistModel(
                    name="Gandalf",
                    description="A wizard",
                    traits=["wise"],
                    motivation="Guide the fellowship",
                ),
            ),
        )
        plot_outline, characters, _ = user_story_to_objects(model)

        assert plot_outline["protagonist_name"] == "Gandalf"
        gandalf = characters["Gandalf"]
        assert gandalf.personality_description == "A wizard"
        assert gandalf.updates["role"] == "protagonist"
        assert gandalf.updates["motivation"] == "Guide the fellowship"

    def test_antagonist_from_characters_group(self) -> None:
        model = UserStoryInputModel(
            characters=CharacterGroupModel(
                antagonist=ProtagonistModel(
                    name="Morgoth",
                    description="The first dark lord",
                ),
            ),
        )
        _, characters, _ = user_story_to_objects(model)

        morgoth = characters["Morgoth"]
        assert morgoth.updates["role"] == "antagonist"
        assert morgoth.personality_description == "The first dark lord"

    def test_top_level_protagonist_takes_precedence(self) -> None:
        model = UserStoryInputModel(
            protagonist=ProtagonistModel(name="TopLevel"),
            characters=CharacterGroupModel(
                protagonist=ProtagonistModel(name="Nested"),
            ),
        )
        plot_outline, characters, _ = user_story_to_objects(model)

        assert plot_outline["protagonist_name"] == "TopLevel"
        assert "TopLevel" in characters
        assert "Nested" not in characters


class TestUserStoryToObjectsEmpty:

    def test_empty_model_returns_empty_outputs(self) -> None:
        model = UserStoryInputModel()
        plot_outline, characters, world_items = user_story_to_objects(model)

        assert plot_outline == {}
        assert characters == {}
        assert world_items == {}
