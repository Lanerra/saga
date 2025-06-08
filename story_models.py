# story_models.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from kg_maintainer.models import CharacterProfile, WorldItem


class NovelConceptModel(BaseModel):
    """High level concept for the novel."""

    title: str = Field(..., min_length=1)
    genre: Optional[str] = None
    setting: Optional[str] = None
    logline: Optional[str] = None
    theme: Optional[str] = None


class RelationshipModel(BaseModel):
    """Relationship details for a character."""

    name: Optional[str] = None
    status: Optional[str] = None
    details: Optional[str] = None


class ProtagonistModel(BaseModel):
    """Primary character information."""

    name: str
    description: Optional[str] = None
    traits: List[str] = []
    motivation: Optional[str] = None
    relationships: Dict[str, RelationshipModel] = {}


class KeyLocationModel(BaseModel):
    """A single location within the setting."""

    name: str
    description: Optional[str] = None
    atmosphere: Optional[str] = None


class SettingModel(BaseModel):
    """Setting information for the story world."""

    primary_setting_overview: Optional[str] = None
    key_locations: List[KeyLocationModel] = []


class PlotElementsModel(BaseModel):
    """Major plot elements provided by the user."""

    inciting_incident: Optional[str] = None
    key_plot_points: List[str] = []
    central_conflict: Optional[str] = None
    stakes: Optional[str] = None


class UserStoryInputModel(BaseModel):
    """Top level structure for ``user_story_elements.yaml``."""

    model_config = ConfigDict(extra="allow")

    novel_concept: Optional[NovelConceptModel] = None
    protagonist: Optional[ProtagonistModel] = None
    antagonist: Optional[ProtagonistModel] = None
    plot_elements: Optional[PlotElementsModel] = None
    setting: Optional[SettingModel] = None
    world_details: Optional[Dict[str, Any]] = None
    other_key_characters: Optional[Dict[str, ProtagonistModel]] = None
    conflict: Optional[Dict[str, Any]] = None
    style_and_tone: Optional[Dict[str, Any]] = None
    world_specifics: Optional[Dict[str, Any]] = None


def user_story_to_objects(
    model: UserStoryInputModel,
) -> Tuple[
    Dict[str, Any], Dict[str, CharacterProfile], Dict[str, Dict[str, WorldItem]]
]:
    """Convert ``UserStoryInputModel`` to internal dataclass objects."""

    plot_outline: Dict[str, Any] = {}
    characters: Dict[str, CharacterProfile] = {}
    world_items: Dict[str, Dict[str, WorldItem]] = {}

    if model.novel_concept:
        plot_outline.update(model.novel_concept.model_dump(exclude_none=True))

    main_char_model = model.protagonist
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
        cp.updates["role"] = "protagonist"
        cp.updates["motivation"] = main_char_model.motivation or ""
        characters[cp.name] = cp

    antagonist_model = model.antagonist
    if antagonist_model:
        ant_cp = CharacterProfile(name=antagonist_model.name)
        ant_cp.description = antagonist_model.description or ""
        ant_cp.traits = antagonist_model.traits
        ant_cp.relationships = {
            rel_key: rel.model_dump(exclude_none=True)
            for rel_key, rel in antagonist_model.relationships.items()
        }
        ant_cp.status = "As described"
        ant_cp.updates["role"] = "antagonist"
        characters[ant_cp.name] = ant_cp

    if model.other_key_characters:
        for name, info in model.other_key_characters.items():
            cp = CharacterProfile(name=info.name)
            cp.description = info.description or ""
            cp.traits = info.traits
            cp.updates["role"] = "other_key_character"
            characters[cp.name] = cp

    if model.plot_elements:
        plot_outline["inciting_incident"] = model.plot_elements.inciting_incident
        plot_outline["plot_points"] = model.plot_elements.key_plot_points
        plot_outline["conflict_summary"] = model.plot_elements.central_conflict
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

    if model.world_details:
        for category, items in model.world_details.items():
            world_items.setdefault(category, {})
            if isinstance(items, dict):
                for item_name, item_details in items.items():
                    world_items[category][item_name] = WorldItem.from_dict(
                        category, item_name, item_details
                    )

    return plot_outline, characters, world_items
