"""Checksum-selected, immutable initialization admission payloads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.graph_ownership import validate_project_id
from core.langgraph.content_manager import ContentManager, ContentRef
from core.langgraph.initialization.act_outlines_node import ActOutlineSchema
from core.langgraph.initialization.chapter_allocation import determine_act_for_chapter as determine_act_for_chapter_from_outline
from core.langgraph.initialization.persist_files_node import _character_projection_paths
from core.langgraph.state import NarrativeState
from models.kg_constants import RELATIONSHIP_TYPES

ARTIFACTS = ("character_sheets", "global_outline", "act_outlines", "chapter_outlines", "outline_relationships")


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strict_json(text: str) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=unique_pairs)


class FrozenPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Artifact(FrozenPayload):
    path: str
    content_type: str
    version: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    payload: str

    def reference(self) -> ContentRef:
        return cast(ContentRef, self.model_dump(exclude={"payload"}))


class CharacterSheet(FrozenPayload):
    name: str = Field(min_length=1)
    description: str
    traits: tuple[str, ...]
    status: str
    motivations: str
    background: str
    skills: tuple[str, ...]
    internal_conflict: str
    is_protagonist: bool = False
    physical_description: str = ""
    generated_at: str = "initialization"
    raw_response: str = ""
    type: Literal["Character"] = "Character"
    relationships: str


class ChapterOutline(FrozenPayload):
    chapter_number: int = Field(gt=0)
    act_number: int = Field(gt=0)
    scene_description: str = Field(min_length=1)
    key_beats: tuple[str, ...] = Field(min_length=1)
    plot_point: str = Field(min_length=1)
    version: int = Field(ge=0)
    raw_text: str = ""
    generated_at: str = "initialization"
    title: str = ""
    summary: str = ""


class Relationship(FrozenPayload):
    source_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    relationship_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    description: str
    chapter: Literal[0] = 0
    confidence: float = Field(default=0.8, ge=0, le=1)

    @field_validator("relationship_type")
    @classmethod
    def allowlisted_type(cls, value: str) -> str:
        if value not in RELATIONSHIP_TYPES:
            raise ValueError(f"Unknown initialization relationship type: {value}")
        return value


class InitializationSnapshot(FrozenPayload):
    schema_version: Literal[1] = 1
    project_id: str
    total_chapters: int = Field(gt=0)
    total_acts: int = Field(gt=0)
    artifacts: tuple[Artifact, ...]
    characters: tuple[CharacterSheet, ...]
    chapters: tuple[ChapterOutline, ...]
    relationships: tuple[Relationship, ...]
    metadata: str

    def source(self, name: str) -> Any:
        return json.loads(next(artifact.payload for artifact in self.artifacts if artifact.content_type == name))

    @property
    def identity(self) -> str:
        return digest(self.model_dump_json())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class GlobalAct(FrozenPayload):
    act_number: int = Field(gt=0)
    title: str
    summary: str
    key_events: tuple[str, ...] = ()
    chapters_start: int = Field(gt=0)
    chapters_end: int = Field(gt=0)


class CharacterArc(FrozenPayload):
    character_name: str
    starting_state: str
    ending_state: str
    key_moments: tuple[str, ...] = ()


class GlobalOutline(FrozenPayload):
    act_count: int = Field(gt=0)
    acts: tuple[GlobalAct, ...]
    inciting_incident: str = Field(min_length=1)
    midpoint: str = Field(min_length=1)
    climax: str = Field(min_length=1)
    resolution: str = Field(min_length=1)
    character_arcs: tuple[CharacterArc, ...] = ()
    thematic_progression: str
    pacing_notes: str = ""
    total_chapters: int = Field(gt=0)
    structure_type: str = ""
    generated_at: str = "initialization"
    validation_errors: tuple[str, ...] = ()
    raw_text: str


def select_snapshot(state: NarrativeState) -> InitializationSnapshot:
    manager = ContentManager(state["project_dir"])
    artifacts = []
    for name in ARTIFACTS:
        reference = state.get(name + "_ref")
        require(isinstance(reference, dict), f"Initialization requires selected {name}_ref")
        reference = cast(ContentRef, reference)
        manager.load_json_strict(reference)
        artifact = Artifact(**reference, payload=encoded(strict_json(manager.load_text_strict(reference))))
        require(artifact.content_type == name, f"Selected artifact type mismatch: {name}")
        require(bool(re.search(rf"_v{artifact.version}\.json$", artifact.path)), "Selected version/path mismatch")
        artifacts.append(artifact)
    sources = {artifact.content_type: json.loads(artifact.payload) for artifact in artifacts}
    sheets = sources["character_sheets"]
    require(isinstance(sheets, dict) and bool(sheets), "Character sheets must be a nonempty object")
    characters = []
    for name, sheet in sheets.items():
        require(isinstance(sheet, dict) and name == sheet.get("name"), "Character key/name identity mismatch")
        value = dict(sheet)
        value["relationships"] = encoded(value.get("relationships", {}))
        characters.append(CharacterSheet.model_validate_json(encoded(value)))
    require(len({character.name.casefold() for character in characters}) == len(characters), "Ambiguous character identity")
    # Filename collision admission is independent of semantic identity admission.
    _character_projection_paths(Path(state["project_dir"]) / ".saga" / "projection-admission", sheets)
    outline = sources["global_outline"]
    require(isinstance(outline, dict), "Global outline must be an object")
    structured_global = GlobalOutline.model_validate_json(encoded(outline))
    require(not structured_global.validation_errors, "Global producer reports validation errors")
    require(all(arc.character_name in sheets for arc in structured_global.character_arcs), "Unknown character arc identity")
    total = state.get("total_chapters")
    if type(total) is not int or total <= 0:
        raise ValueError("Expected positive total_chapters")
    require(outline.get("total_chapters") == total, "Global/selected chapter count mismatch")
    acts = sources["act_outlines"]
    require(isinstance(acts, dict) and acts.get("format_version") == 2 and set(acts) == {"format_version", "acts"}, "Expected act format_version 2")
    total_acts = outline.get("act_count")
    require(type(total_acts) is int and total_acts > 0, "Expected positive act_count")
    global_acts = outline.get("acts")
    require(isinstance(global_acts, list) and [act.get("act_number") for act in global_acts] == list(range(1, total_acts + 1)), "Incomplete or duplicate global act coverage")
    covered: list[int] = []
    for act in global_acts:
        start, end = act.get("chapters_start"), act.get("chapters_end")
        require(type(start) is int and type(end) is int and 1 <= start <= end <= total, "Malformed global act range")
        covered.extend(range(start, end + 1))
    require(covered == list(range(1, total + 1)), "Global act ranges must partition chapter topology")
    require([act.get("act_number") for act in acts["acts"]] == list(range(1, total_acts + 1)), "Incomplete or duplicate act coverage")
    for act, global_act in zip(acts["acts"], global_acts, strict=True):
        range_fields = ("chapters_start", "chapters_end")
        if any(key in act for key in range_fields):
            require(all(type(act.get(key)) is int and act[key] == global_act[key] for key in range_fields), "Act range metadata mismatch")
        structured = {key: value for key, value in act.items() if key not in {"raw_text", "generated_at", *range_fields}}
        parsed = ActOutlineSchema.model_validate(structured)
        require(parsed.total_acts == total_acts, "Act count mismatch")
        allocated = sum(determine_act_for_chapter_from_outline(global_outline=outline, total_chapters=total, chapter_number=number) == parsed.act_number for number in range(1, total + 1))
        require(parsed.chapters_in_act == allocated, "Act chapter allocation mismatch")
    chapters = sources["chapter_outlines"]
    require(isinstance(chapters, dict) and set(chapters) == {str(number) for number in range(1, total + 1)}, "Incomplete chapter topology")
    selected_version = next(artifact.version for artifact in artifacts if artifact.content_type == "chapter_outlines")
    parsed_chapters = []
    for number in range(1, total + 1):
        chapter = ChapterOutline.model_validate_json(encoded(chapters[str(number)]))
        require(chapter.chapter_number == number, "Chapter key/embedded number mismatch")
        require(chapter.version == selected_version, "Chapter selected version mismatch")
        require(chapter.act_number == determine_act_for_chapter_from_outline(global_outline=outline, total_chapters=total, chapter_number=number), "Chapter act identity mismatch")
        parsed_chapters.append(chapter)
    relationships = sources["outline_relationships"]
    require(isinstance(relationships, list), "Relationships must be an explicit array")
    return InitializationSnapshot(
        project_id=validate_project_id(state["graph_project_id"]), total_chapters=total, total_acts=total_acts,
        artifacts=tuple(artifacts), characters=tuple(characters), chapters=tuple(parsed_chapters),
        relationships=tuple(Relationship.model_validate(item) for item in relationships),
        metadata=encoded({key: state.get(key, "") for key in ("title", "genre", "theme", "setting", "project_id")}),
    )
