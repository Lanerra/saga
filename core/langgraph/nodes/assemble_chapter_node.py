# core/langgraph/nodes/assemble_chapter_node.py
import structlog

from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def assemble_chapter(state: NarrativeState) -> NarrativeState:
    """
    Assemble scenes into a full chapter.
    """
    logger.info("assemble_chapter: finalizing chapter draft")

    scene_drafts = state.get("scene_drafts", [])

    if not scene_drafts:
        logger.warning("assemble_chapter: no scene drafts found")
        return {
            **state,
            "draft_text": "",
            "draft_word_count": 0,
            "current_node": "assemble_chapter",
        }

    # Join scenes with a separator
    full_text = "\n\n# ***\n\n".join(scene_drafts)
    word_count = len(full_text.split())

    logger.info("assemble_chapter: assembled chapter", word_count=word_count)

    return {
        **state,
        "draft_text": full_text,
        "draft_word_count": word_count,
        "current_node": "assemble_chapter",
    }
