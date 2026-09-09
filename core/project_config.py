# core/project_config.py
from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

import config


def allocate_word_target(total_words: int, part_count: int, part_number: int) -> int:
    """Allocate positive word targets, assigning remainders to earlier parts."""
    if any(type(value) is not int for value in (total_words, part_count, part_number)):
        raise ValueError("Word allocation requires integer totals, counts and positions")
    if part_count < 1 or total_words < part_count:
        raise ValueError("Word target must provide at least one word per part")
    if not 1 <= part_number <= part_count:
        raise ValueError("Word allocation position must be within the part count")
    quotient, remainder = divmod(total_words, part_count)
    return quotient + int(part_number <= remainder)


class NarrativeProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: StrictStr = Field(min_length=1)
    genre: StrictStr = Field(min_length=1)
    theme: StrictStr = Field(min_length=1)
    setting: StrictStr = Field(min_length=1)
    protagonist_name: StrictStr = Field(min_length=1)
    narrative_style: StrictStr = Field(min_length=1, pattern=r"\S")
    total_chapters: StrictInt = Field(ge=1)
    target_word_count: StrictInt = Field(default_factory=lambda: config.TARGET_WORD_COUNT, ge=1, validate_default=True)
    created_from: StrictStr = Field(default="")
    original_prompt: StrictStr = Field(default="")

    @model_validator(mode="after")
    def validate_word_allocation(self) -> Self:
        allocate_word_target(self.target_word_count, self.total_chapters, 1)
        return self
