# prompts/grammar_loader.py
"""
Utility for loading and combining GBNF grammar files.
"""

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

GRAMMARS_DIR = Path(__file__).parent / "grammars"


def load_grammar(grammar_name: str) -> str:
    """
    Load a GBNF grammar file and resolve dependencies (specifically common.gbnf).

    This function reads the requested grammar file and appends the contents of
    common.gbnf (excluding its root definition) to ensure all common definitions
    like json_value, ws, etc. are available.

    Args:
        grammar_name: Name of the grammar file (e.g., 'initialization.gbnf' or 'initialization')

    Returns:
        The complete, concatenated GBNF grammar string.

    Raises:
        FileNotFoundError: If the grammar file does not exist.
    """
    if not grammar_name.endswith(".gbnf"):
        grammar_name += ".gbnf"

    grammar_path = GRAMMARS_DIR / grammar_name
    if not grammar_path.exists():
        raise FileNotFoundError(f"Grammar file not found: {grammar_path}")

    logger.debug(f"Loading grammar: {grammar_name}")
    grammar_content = grammar_path.read_text(encoding="utf-8")

    # If asking for common.gbnf explicitly, just return it
    if grammar_name == "common.gbnf":
        return grammar_content

    # Load common.gbnf to append
    common_path = GRAMMARS_DIR / "common.gbnf"
    if common_path.exists():
        common_content = common_path.read_text(encoding="utf-8")

        # Filter out the root definition from common.gbnf to avoid conflicts
        # We assume the specific grammar defines its own root
        common_lines = []
        for line in common_content.splitlines():
            if not line.strip().startswith("root ::="):
                common_lines.append(line)

        common_content_no_root = "\n".join(common_lines)

        # Concatenate: Specific grammar first (with its root), then common definitions
        return f"{grammar_content}\n\n# --- Included from common.gbnf ---\n{common_content_no_root}"
    else:
        logger.warning("common.gbnf not found in grammars directory")
        return grammar_content
