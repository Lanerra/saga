# utils/ingestion_utils.py
import config


def split_text_into_chapters(
    text: str, max_chars: int = config.MIN_ACCEPTABLE_DRAFT_LENGTH
) -> list[str]:
    """Split text into pseudo-chapters by paragraph boundaries."""
    separator = "\n\n"
    sep_len = len(separator)
    paragraphs = text.split(separator)
    chapters: list[str] = []
    current: list[str] = []
    current_length = 0
    for para in paragraphs:
        if current:
            para_len = sep_len + len(para)
        else:
            para_len = len(para)
        if current_length + para_len > max_chars and current:
            chapters.append(separator.join(current).strip())
            current = [para]
            current_length = len(para)
        else:
            current.append(para)
            current_length += para_len
    if current:
        chapters.append(separator.join(current).strip())
    return [c for c in chapters if c.strip()]
