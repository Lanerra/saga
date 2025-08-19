import config


def split_text_into_chapters(
    text: str, max_chars: int = config.MIN_ACCEPTABLE_DRAFT_LENGTH
) -> list[str]:
    """Split text into pseudo-chapters by paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chapters: list[str] = []
    current: list[str] = []
    current_length = 0
    for para in paragraphs:
        para_len = len(para) + 2
        if current_length + para_len > max_chars and current:
            chapters.append("\n\n".join(current).strip())
            current = [para]
            current_length = para_len
        else:
            current.append(para)
            current_length += para_len
    if current:
        chapters.append("\n\n".join(current).strip())
    return [c for c in chapters if c.strip()]
