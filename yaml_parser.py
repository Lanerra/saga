# yaml_parser.py
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def normalize_keys_recursive(data: Any) -> Any:
    """
    Recursively normalizes keys in a dictionary to lowercase and replaces spaces with underscores.
    This is to maintain some compatibility with the output of the previous Markdown parser if needed,
    assuming YAML keys might be more human-readable (e.g., "Novel Concept" vs "novel_concept").
    Users of the YAML should ideally use the normalized key format directly in their YAML files
    for clarity.
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            normalized_key = str(key).lower().replace(" ", "_")
            new_dict[normalized_key] = normalize_keys_recursive(value)
        return new_dict
    elif isinstance(data, list):
        return [normalize_keys_recursive(item) for item in data]
    else:
        return data


def load_yaml_file(filepath: str, normalize_keys: bool = True) -> dict[str, Any] | None:
    """
    Loads and parses a YAML file.

    Args:
        filepath: Path to the YAML file.
        normalize_keys: Whether to recursively normalize dictionary keys
                        (lowercase, spaces to underscores). Defaults to True.

    Returns:
        A dictionary representing the YAML content, or None if an error occurs.
    """
    if not filepath.endswith((".yaml", ".yml")):
        logger.error(f"File specified is not a YAML file: {filepath}")
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            content = yaml.safe_load(f)

        if not isinstance(content, dict):
            logger.warning(
                f"YAML file {filepath} did not parse into a dictionary at the root level. Parsed type: {type(content)}"
            )
            # Depending on requirements, you might want to return `content` if it's a list,
            # or enforce dictionary structure. For now, allowing non-dict if safe_load returns one.
            # However, for consistency with previous parser, usually a Dict is expected.
            # Let's assume for now we want a dictionary, makes `normalize_keys` simpler.
            if content is None:  # Empty file
                return {}
            if not isinstance(content, dict):  # If not None and not dict, log error.
                logger.error(
                    f"YAML file {filepath} must have a dictionary as its root element for this application."
                )
                return None

        if normalize_keys and content:  # only normalize if content is not None
            return normalize_keys_recursive(content)  # type: ignore
        return content
    except FileNotFoundError:
        logger.warning(f"YAML file '{filepath}' not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {filepath}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading YAML file {filepath}: {e}",
            exc_info=True,
        )
        return None
