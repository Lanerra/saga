"""Offline lexical regression signals; not semantic judgments or publication gates."""
import re
from collections import Counter
from typing import Annotated, Literal, Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]
DIMENSIONS = ("continuity", "fact_coverage", "repetition", "perspective", "prose_fidelity", "usefulness")


class NarrativeExpectations(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    forbidden_continuity_phrases: list[NonemptyText]
    required_facts: list[NonemptyText] = Field(min_length=1)
    required_perspective_markers: list[NonemptyText] = Field(min_length=1)
    forbidden_perspective_phrases: list[NonemptyText]
    forbidden_style_phrases: list[NonemptyText]
    maximum_sentence_words: int = Field(gt=0)
    required_outcome: NonemptyText
    minimum_words: int = Field(gt=0)
    maximum_words: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_word_range(self) -> Self:
        if self.maximum_words < self.minimum_words:
            raise ValueError("Maximum words must be at least minimum words")
        return self


class NarrativeAssessment(TypedDict):
    scope: Literal["mechanical_only"]
    word_count: int
    issues: dict[str, list[str]]
    human_review_required: list[str]


def assess_narrative(text: str, expectations: NarrativeExpectations) -> NarrativeAssessment:
    """Report explicit phrase, sentence, length and format signals for a fixture.

    Phrase matches cannot establish factual entailment, tense, causal continuity,
    style quality or usefulness. Every dimension still requires human review.
    """
    if not isinstance(text, str):
        raise TypeError("Narrative candidate must be text")
    normalized = " ".join(text.casefold().split())

    def contains(phrase: str) -> bool:
        return re.search(r"(?<!\w)" + re.escape(" ".join(phrase.casefold().split())) + r"(?!\w)", normalized) is not None

    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+(?:\s+|$)", normalized) if sentence.strip()]
    counts = Counter(sentences)
    issues = {
        "continuity": [phrase for phrase in expectations.forbidden_continuity_phrases if contains(phrase)],
        "fact_coverage": [phrase for phrase in expectations.required_facts if not contains(phrase)],
        "repetition": [sentence for sentence, count in counts.items() if count > 1],
        "perspective": [phrase for phrase in expectations.required_perspective_markers if not contains(phrase)]
        + [phrase for phrase in expectations.forbidden_perspective_phrases if contains(phrase)],
        "prose_fidelity": [phrase for phrase in expectations.forbidden_style_phrases if contains(phrase)],
        "usefulness": [],
    }
    if any(len(sentence.split()) > expectations.maximum_sentence_words for sentence in sentences):
        issues["prose_fidelity"].append("sentence_word_limit_exceeded")
    word_count = len(text.split())
    if not expectations.minimum_words <= word_count <= expectations.maximum_words:
        issues["usefulness"].append("word_count_outside_range")
    if not contains(expectations.required_outcome):
        issues["usefulness"].append("missing_outcome")
    if "```" in text or re.search(r"^\s*#", text, re.MULTILINE):
        issues["usefulness"].append("non_prose_format")
    return {
        "scope": "mechanical_only", "word_count": word_count, "issues": issues,
        "human_review_required": list(DIMENSIONS),
    }
