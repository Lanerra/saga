"""Extract relationships from outline during initialization."""

from __future__ import annotations

import json
from typing import Any

import structlog

from config import settings as config
from core.langgraph.content_manager import (
    ContentManager,
    get_act_outlines,
    get_global_outline,
)
from core.langgraph.state import NarrativeState
from core.llm_interface_refactored import llm_service
from models.kg_constants import RELATIONSHIP_TYPES
from prompts.prompt_renderer import get_system_prompt, render_prompt

logger = structlog.get_logger(__name__)


async def extract_outline_relationships(state: NarrativeState) -> NarrativeState:
    """Extract relationships from the global outline for initialization.

    This node runs after act outlines are generated and before commit to graph.
    It extracts relationships between entities mentioned in the outline to seed
    the knowledge graph with foundational relationships (origins, locations, etc.)

    Args:
        state: Current workflow state.

    Returns:
        State update with outline_relationships_ref.
    """
    from core.langgraph.content_manager import require_project_dir

    logger.info("extract_outline_relationships: starting")

    content_manager = ContentManager(require_project_dir(state))
    global_outline = get_global_outline(state, content_manager)
    act_outlines = get_act_outlines(state, content_manager)

    if not global_outline and not act_outlines:
        logger.warning("extract_outline_relationships: no outlines found, skipping")
        return {
            **state,
            "outline_relationships_ref": None,
            "current_node": "outline_relationships",
        }  # type: ignore[return-value]

    global_text = global_outline.get("raw_text", "") if global_outline else ""
    act_texts = []
    if act_outlines:
        for act_num in sorted(act_outlines.keys()):
            act_data = act_outlines[act_num]
            act_text = act_data.get("raw_text", "")
            if act_text:
                act_texts.append(f"Act {act_num}: {act_text}")

    combined_outline_text = global_text
    if act_texts:
        combined_outline_text += "\n\n" + "\n\n".join(act_texts)

    logger.info(
        "extract_outline_relationships: combined outline text",
        global_length=len(global_text),
        act_count=len(act_texts),
        combined_length=len(combined_outline_text),
    )

    story_metadata = state.get("story_metadata", {})
    novel_title = story_metadata.get("title", "Unknown")
    novel_genre = story_metadata.get("genre", "Unknown")
    protagonist = story_metadata.get("protagonist", "Unknown")
    setting = story_metadata.get("setting", "")

    relationships = await _extract_relationships_from_outline(
        outline_text=combined_outline_text,
        novel_title=novel_title,
        novel_genre=novel_genre,
        protagonist=protagonist,
        setting=setting,
    )

    if not relationships:
        logger.info("extract_outline_relationships: no relationships extracted")
        return {
            **state,
            "outline_relationships_ref": None,
            "current_node": "outline_relationships",
        }

    logger.info(
        "extract_outline_relationships: extracted relationships",
        count=len(relationships),
    )

    outline_relationships_ref = content_manager.save_json(
        relationships,
        "outline_relationships",
        "main",
        version=1,
    )

    logger.info(
        "extract_outline_relationships: saving state update",
        ref=outline_relationships_ref,
        ref_type=type(outline_relationships_ref).__name__,
    )

    result: NarrativeState = {
        **state,
        "outline_relationships_ref": outline_relationships_ref,
        "current_node": "outline_relationships",
        "initialization_step": "outline_relationships_extracted",
    }

    logger.info(
        "extract_outline_relationships: returning state",
        has_ref=("outline_relationships_ref" in result),
        ref_value=result.get("outline_relationships_ref"),
    )

    return result


async def _extract_relationships_from_outline(
    outline_text: str,
    novel_title: str,
    novel_genre: str,
    protagonist: str,
    setting: str,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Extract relationships from outline text (global + act outlines combined).

    Args:
        outline_text: Combined outline text from global and act outlines.
        novel_title: Story title for context.
        novel_genre: Story genre for context.
        protagonist: Protagonist name for context.
        setting: Story setting description.
        model_name: LLM model name override.

    Returns:
        List of relationship dictionaries with keys: source_name, target_name, relationship_type, description.
    """
    if not outline_text:
        logger.warning("_extract_relationships_from_outline: no outline text provided")
        return []

    prompt = render_prompt(
        "initialization/extract_outline_relationships.j2",
        {
            "novel_title": novel_title,
            "novel_genre": novel_genre,
            "protagonist": protagonist,
            "setting": setting,
            "outline_text": outline_text,
            "canonical_relationship_types": sorted(RELATIONSHIP_TYPES),
        },
    )

    model = model_name or config.NARRATIVE_MODEL

    for attempt in range(1, 3):
        logger.info(
            "_extract_relationships_from_outline: calling LLM",
            attempt=attempt,
            model=model,
        )

        response, _ = await llm_service.async_call_llm(
            model_name=model,
            prompt=prompt,
            temperature=0.5,
            max_tokens=config.MAX_GENERATION_TOKENS,
            allow_fallback=True,
            auto_clean_response=True,
            system_prompt=get_system_prompt("knowledge_agent"),
        )

        try:
            relationships = _parse_relationships_extraction(response)
            logger.info(
                "_extract_relationships_from_outline: successfully parsed relationships",
                count=len(relationships),
            )
            return relationships
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "_extract_relationships_from_outline: failed to parse",
                attempt=attempt,
                error=str(e),
            )
            if attempt == 2:
                logger.error(
                    "_extract_relationships_from_outline: max attempts exceeded",
                    response_preview=response[:500] if response else None,
                )
                return []

    return []


def _parse_relationships_extraction(response: str) -> list[dict[str, Any]]:
    """Parse LLM response into relationship dictionaries.

    Args:
        response: LLM response (JSON with kg_triples key).

    Returns:
        List of relationship dictionaries.

    Raises:
        ValueError: If the output violates the JSON/schema contract.
        json.JSONDecodeError: If response is not valid JSON.
    """
    raw_text = response.strip()

    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]

    raw_text = raw_text.strip()

    data = json.loads(raw_text)

    if not isinstance(data, dict):
        raise ValueError("Expected JSON object with kg_triples key")

    kg_triples_list = data.get("kg_triples", [])
    if not isinstance(kg_triples_list, list):
        raise ValueError("kg_triples must be a JSON array")

    relationships: list[dict[str, Any]] = []

    for triple in kg_triples_list:
        if not isinstance(triple, dict):
            continue

        subject = triple.get("subject", "")
        predicate = triple.get("predicate", "")
        object_entity = triple.get("object_entity", "")
        description = triple.get("description", "")

        if isinstance(subject, dict):
            subject = subject.get("name", str(subject))
        if isinstance(object_entity, dict):
            object_entity = object_entity.get("name", str(object_entity))

        subject_text = str(subject).strip() if subject else ""
        target_text = str(object_entity).strip() if object_entity else ""
        predicate_text = str(predicate).strip() if predicate else ""

        if not subject_text or not target_text or not predicate_text:
            logger.warning(
                "_parse_relationships_extraction: skipping incomplete relationship",
                subject=subject_text,
                predicate=predicate_text,
                target=target_text,
            )
            continue

        relationships.append(
            {
                "source_name": subject_text,
                "target_name": target_text,
                "relationship_type": predicate_text,
                "description": str(description).strip() if description else "",
                "chapter": 0,
                "confidence": 0.8,
            }
        )

    return relationships
