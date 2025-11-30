# core/langgraph/nodes/assemble_chapter_node.py
import structlog

from core.langgraph.content_manager import ContentManager, get_scene_drafts
from core.langgraph.state import NarrativeState

logger = structlog.get_logger(__name__)


def assemble_chapter(state: NarrativeState) -> NarrativeState:
    """
    Assemble scenes into a full chapter.
    """
    logger.info("assemble_chapter: finalizing chapter draft")

    # Initialize content manager for external storage
    content_manager = ContentManager(state.get("project_dir", ""))

    scene_drafts = get_scene_drafts(state, content_manager)

    if not scene_drafts:
        logger.warning("assemble_chapter: no scene drafts found")
        return {
            **state,
            "draft_ref": None,
            "draft_word_count": 0,
            "current_node": "assemble_chapter",
        }

    # Join scenes with a separator
    full_text = "\n\n# ***\n\n".join(scene_drafts)
    word_count = len(full_text.split())

    logger.info("assemble_chapter: assembled chapter", word_count=word_count)

    chapter_number = state.get("current_chapter", 1)

    # Get current version (for revision tracking)
    current_version = (
        content_manager.get_latest_version("draft", f"chapter_{chapter_number}") + 1
    )

    # Externalize scene_drafts to reduce state bloat
    scene_drafts_ref = content_manager.save_list_of_texts(
        scene_drafts,
        "scenes",
        f"chapter_{chapter_number}",
        current_version,
    )

    # Externalize draft_text
    draft_ref = content_manager.save_text(
        full_text,
        "draft",
        f"chapter_{chapter_number}",
        current_version,
    )

    logger.info(
        "assemble_chapter: content externalized",
        chapter=chapter_number,
        version=current_version,
        draft_size=draft_ref["size_bytes"],
        scene_count=len(scene_drafts),
    )

    return {
        "draft_ref": draft_ref,
        "scene_drafts_ref": scene_drafts_ref,
        "draft_word_count": word_count,
        "current_node": "assemble_chapter",
    }
