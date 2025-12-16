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

    IMPORTANT:
    - Many SAGA grammars historically duplicated common JSON rules inline.
    - Naively concatenating common.gbnf can therefore introduce duplicate rule
      definitions (same `rule ::= ...` name multiple times), which can lead to
      ambiguous or implementation-defined behavior in downstream grammar parsers.
    - To avoid this, we append ONLY common rules that are not already defined in
      the requested grammar (and we still remove `root ::= ...` from common).

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

    def _extract_rule_names(text: str) -> set[str]:
        """
        Extract rule names from a GBNF grammar string.

        Rule lines are assumed to start with: <name> ::= ...
        Comments and blank lines are ignored.
        """
        rule_names: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Very small/robust parser: take token before '::='
            if "::=" in line:
                name = line.split("::=", 1)[0].strip()
                if name:
                    rule_names.add(name)
        return rule_names

    existing_rules = _extract_rule_names(grammar_content)

    # Load common.gbnf to append
    common_path = GRAMMARS_DIR / "common.gbnf"
    if not common_path.exists():
        logger.warning("common.gbnf not found in grammars directory")
        return grammar_content

    common_content = common_path.read_text(encoding="utf-8")

    # Filter out `root ::= ...` and any rule whose name already exists in the target grammar.
    common_lines: list[str] = []
    for raw in common_content.splitlines():
        line = raw.strip()

        # Always drop common root to avoid conflicts.
        if line.startswith("root ::="):
            continue

        # Preserve comments/blank lines as-is (keeps readability).
        if not line or line.startswith("#"):
            common_lines.append(raw)
            continue

        # If this is a rule line, skip if already defined by the specific grammar.
        if "::=" in line:
            rule_name = line.split("::=", 1)[0].strip()
            if rule_name in existing_rules:
                continue

        common_lines.append(raw)

    common_content_deduped = "\n".join(common_lines).strip()

    if not common_content_deduped:
        # No missing rules to append; return specific grammar as-is.
        return grammar_content

    # Concatenate: Specific grammar first (with its root), then only missing common definitions.
    return f"{grammar_content}\n\n# --- Included from common.gbnf (deduped) ---\n{common_content_deduped}\n"
