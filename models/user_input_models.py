# models/user_input_models.py
"""Define user-facing models for providing story input data.

These models represent the structure of `user_story_elements.yaml` and other
user-provided inputs. They are intentionally permissive in some areas to support
incremental authoring and partial inputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .kg_models import CharacterProfile, WorldItem


class NovelConceptModel(BaseModel):
    """Describe the high-level concept for a novel."""

    title: str = Field(..., min_length=1)
    genre: str | None = None
    setting: str | None = None
    theme: str | None = None


class RelationshipModel(BaseModel):
    """Describe a relationship between characters."""

    name: str | None = None
    status: str | None = None
    details: str | None = None


class ProtagonistModel(BaseModel):
    """Describe a main character provided by the user."""

    name: str
    description: str | None = None
    traits: list[str] = Field(default_factory=list)
    motivation: str | None = None
    role: str | None = None
    relationships: dict[str, RelationshipModel] = Field(default_factory=dict)


class CharacterGroupModel(BaseModel):
    """Group characters as they appear in user-provided YAML."""

    protagonist: ProtagonistModel | None = None
    antagonist: ProtagonistModel | None = None
    supporting_characters: list[ProtagonistModel] = Field(default_factory=list)


class KeyLocationModel(BaseModel):
    """Describe a named location within the setting."""

    name: str
    description: str | None = None
    atmosphere: str | None = None


class SettingModel(BaseModel):
    """Describe setting information for the story world."""

    primary_setting_overview: str | None = None
    key_locations: list[KeyLocationModel] = Field(default_factory=list)


class PlotElementsModel(BaseModel):
    """Describe major plot elements provided by the user."""

    inciting_incident: str | None = None
    plot_points: list[str] = Field(default_factory=list)
    central_conflict: str | None = None
    stakes: str | None = None


class UserStoryInputModel(BaseModel):
    """Represent the top-level structure for `user_story_elements.yaml`.

    Notes:
        This model allows unknown keys (`extra="allow"`) so callers may include
        additional sections without breaking parsing.
    """

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
) -> tuple[dict[str, Any], dict[str, CharacterProfile], dict[str, dict[str, WorldItem]]]:
    """Convert a user story input model into internal objects.

    The returned dictionaries are used as inputs to downstream prompt building and
    knowledge-graph bootstrap steps.

    Args:
        model: Parsed user story input.

    Returns:
        A tuple of `(plot_outline, characters, world_items)` where:
        - `plot_outline` is a free-form dictionary describing the narrative setup.
        - `characters` maps character name to a [`CharacterProfile`](models/kg_models.py:19).
        - `world_items` maps category name to an item-map of world item name to a
          [`WorldItem`](models/kg_models.py:123). If no world items are produced, this
          value is an empty dictionary.

    Notes:
        - Character names act as the identifiers in the `characters` mapping.
        - World items are grouped by category; a special `_overview_` category and
          `_overview_` item name may be used for a setting overview.
    """

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
        cp.personality_description = main_char_model.description or ""
        cp.traits = main_char_model.traits
        cp.relationships = {rel_key: rel.model_dump(exclude_none=True) for rel_key, rel in main_char_model.relationships.items()}
        cp.status = "As described"
        cp.updates["role"] = main_char_model.role or "protagonist"
        cp.updates["motivation"] = main_char_model.motivation or ""
        characters[cp.name] = cp

    antagonist_model = model.antagonist
    if not antagonist_model and model.characters:
        antagonist_model = model.characters.antagonist
    if antagonist_model:
        ant_cp = CharacterProfile(name=antagonist_model.name)
        ant_cp.personality_description = antagonist_model.description or ""
        ant_cp.traits = antagonist_model.traits
        ant_cp.relationships = {rel_key: rel.model_dump(exclude_none=True) for rel_key, rel in antagonist_model.relationships.items()}
        ant_cp.status = "As described"
        ant_cp.updates["role"] = antagonist_model.role or "antagonist"
        characters[ant_cp.name] = ant_cp

    if model.other_key_characters:
        for _name, info in model.other_key_characters.items():
            cp = CharacterProfile(name=info.name)
            cp.personality_description = info.description or ""
            cp.traits = info.traits
            cp.updates["role"] = "other_key_character"
            characters[cp.name] = cp

    if model.characters and model.characters.supporting_characters:
        for info in model.characters.supporting_characters:
            cp = CharacterProfile(name=info.name)
            cp.personality_description = info.description or ""
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
                    world_items[category][item_name] = WorldItem.from_dict(category, item_name, item_details)

    return plot_outline, characters, world_items if world_items else {}
