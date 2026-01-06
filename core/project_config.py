from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


class NarrativeProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: StrictStr = Field(min_length=1)
    genre: StrictStr = Field(min_length=1)
    theme: StrictStr = Field(min_length=1)
    setting: StrictStr = Field(min_length=1)
    protagonist_name: StrictStr = Field(min_length=1)
    narrative_style: StrictStr = Field(min_length=1)
    total_chapters: StrictInt = Field(ge=1)
    created_from: StrictStr = Field(default="")
    original_prompt: StrictStr = Field(default="")
