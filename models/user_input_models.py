# models/user_input_models.py
"""User-facing models for providing story input data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .kg_models import CharacterProfile, WorldItem


class NovelConceptModel(BaseModel):
    """High level concept for the novel."""

    title: str = Field(..., min_length=1)
    genre: str | None = None
    setting: str | None = None
    theme: str | None = None


class RelationshipModel(BaseModel):
    """Relationship details for a character."""

    name: str | None = None
    status: str | None = None
    details: str | None = None


class ProtagonistModel(BaseModel):
    """Primary character information."""

    name: str
    description: str | None = None
    traits: list[str] = []
    motivation: str | None = None
    role: str | None = None
    relationships: dict[str, RelationshipModel] = {}


class CharacterGroupModel(BaseModel):
    """Container for characters provided in the YAML."""

    protagonist: ProtagonistModel | None = None
    antagonist: ProtagonistModel | None = None
    supporting_characters: list[ProtagonistModel] = []


class KeyLocationModel(BaseModel):
    """A single location within the setting."""

    name: str
    description: str | None = None
    atmosphere: str | None = None


class SettingModel(BaseModel):
    """Setting information for the story world."""

    primary_setting_overview: str | None = None
    key_locations: list[KeyLocationModel] = []


class PlotElementsModel(BaseModel):
    """Major plot elements provided by the user."""

    inciting_incident: str | None = None
    plot_points: list[str] = []
    central_conflict: str | None = None
    stakes: str | None = None


class UserStoryInputModel(BaseModel):
    """Top level structure for ``user_story_elements.yaml``."""

    model_config = ConfigDict(extra="allow")

    novel_concept: NovelConceptModel | None = None
    protagonist: ProtagonistModel | None = None
    antagonist: ProtagonistModel | None = None
    characters: CharacterGroupModel | None = None
    plot_elements: PlotElementsModel | None = None
    setting: SettingModel | None = None
    world_details: dict[str, Any] | None = None
    other_key_characters: dict[str, ProtagonistModel] | None = None
    style_and_tone: dict[str, Any] | None = None


def user_story_to_objects(
    model: UserStoryInputModel,
) -> tuple[
    dict[str, Any], dict[str, CharacterProfile], dict[str, dict[str, WorldItem]]
]:
    """Convert ``UserStoryInputModel`` to internal dataclass objects."""

    plot_outline: dict[str, Any] = {}
    characters: dict[str, CharacterProfile] = {}
    world_items: dict[str, dict[str, WorldItem]] = {}

    if model.novel_concept:
        plot_outline.update(model.novel_concept.model_dump(exclude_none=True))
        if model.novel_concept.setting is not None:
            plot_outline["setting"] = model.novel_concept.setting

    main_char_model = model.protagonist
    if not main_char_model and model.characters:
        main_char_model = model.characters.protagonist
    if main_char_model:
        plot_outline["protagonist_name"] = main_char_model.name

        cp = CharacterProfile(name=main_char_model.name)
        cp.description = main_char_model.description or ""
        cp.traits = main_char_model.traits
        cp.relationships = {
            rel_key: rel.model_dump(exclude_none=True)
            for rel_key, rel in main_char_model.relationships.items()
        }
        cp.status = "As described"
        cp.updates["role"] = main_char_model.role or "protagonist"
        cp.updates["motivation"] = main_char_model.motivation or ""
        characters[cp.name] = cp

    antagonist_model = model.antagonist
    if not antagonist_model and model.characters:
        antagonist_model = model.characters.antagonist
    if antagonist_model:
        ant_cp = CharacterProfile(name=antagonist_model.name)
        ant_cp.description = antagonist_model.description or ""
        ant_cp.traits = antagonist_model.traits
        ant_cp.relationships = {
            rel_key: rel.model_dump(exclude_none=True)
            for rel_key, rel in antagonist_model.relationships.items()
        }
        ant_cp.status = "As described"
        ant_cp.updates["role"] = antagonist_model.role or "antagonist"
        characters[ant_cp.name] = ant_cp

    if model.other_key_characters:
        for name, info in model.other_key_characters.items():
            cp = CharacterProfile(name=info.name)
            cp.description = info.description or ""
            cp.traits = info.traits
            cp.updates["role"] = "other_key_character"
            characters[cp.name] = cp

    if model.characters and model.characters.supporting_characters:
        for info in model.characters.supporting_characters:
            cp = CharacterProfile(name=info.name)
            cp.description = info.description or ""
            cp.traits = info.traits
            cp.updates["role"] = info.role or "supporting_character"
            characters[cp.name] = cp

    if model.plot_elements:
        plot_outline["inciting_incident"] = model.plot_elements.inciting_incident
        plot_outline["plot_points"] = model.plot_elements.plot_points
        plot_outline["central_conflict"] = model.plot_elements.central_conflict
        plot_outline["stakes"] = model.plot_elements.stakes

    if model.setting:
        world_items.setdefault("_overview_", {})["_overview_"] = WorldItem.from_dict(
            "_overview_",
            "_overview_",
            {"description": model.setting.primary_setting_overview or ""},
        )
        for loc in model.setting.key_locations:
            world_items.setdefault("locations", {})[loc.name] = WorldItem.from_dict(
                "locations",
                loc.name,
                {
                    "description": loc.description or "",
                    "atmosphere": loc.atmosphere or "",
                },
            )

    if model.style_and_tone:
        if "narrative_style" in model.style_and_tone:
            plot_outline["narrative_style"] = model.style_and_tone["narrative_style"]
        if "tone" in model.style_and_tone:
            plot_outline["tone"] = model.style_and_tone["tone"]
        if "pacing" in model.style_and_tone:
            plot_outline["pacing"] = model.style_and_tone["pacing"]

    if model.world_details:
        for category, items in model.world_details.items():
            world_items.setdefault(category, {})
            if isinstance(items, dict):
                for item_name, item_details in items.items():
                    world_items[category][item_name] = WorldItem.from_dict(
                        category, item_name, item_details
                    )

    return plot_outline, characters, world_items if world_items else {}
