# parsing_utils.py
# utils/parsing_utils.py
import logging
from typing import Any

# from rdflib import Graph, URIRef, Literal, BNode # No longer needed for triples
# from rdflib.namespace import RDF, RDFS # No longer needed for triples

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Custom exception for parsing errors."""


# --- New RDF Triple Parsing using rdflib ---
# Modified to be a custom plain-text triple parser


def _get_entity_type_and_name_from_text(entity_text: str) -> dict[str, str | None]:
    """
    Parses 'EntityType:EntityName' or just 'EntityName' string.
    If EntityType is missing, it's set to None.
    """
    name_part = entity_text
    type_part = None
    if ":" in entity_text:
        parts = entity_text.split(":", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            type_part = parts[0].strip()
            name_part = parts[1].strip()
        elif (
            parts[0].strip()
        ):  # Only one part before colon, might be a type or a name with an odd colon
            # Heuristic: if it starts with uppercase and has no spaces, assume it's a type and name is missing/error.
            # Or if it's a common entity type. For now, simpler: if only one part before ':', it's the type.
            # This logic might need refinement if LLM is inconsistent.
            # Let's assume if one part before ':', it's the type and the rest is name.
            # If no part after ':', then name is effectively empty.
            type_part = parts[0].strip()
            name_part = parts[1].strip() if len(parts) > 1 else ""

    return {
        "type": type_part if type_part else None,
        "name": name_part.strip() if name_part else None,
    }


def parse_rdf_triples_with_rdflib(
    text_block: str,
    rdf_format: str = "turtle",
) -> list[dict[str, Any]]:
    """
    Custom parser for LLM-generated plain text triples.
    Expected format: 'SubjectEntityType:SubjectName | Predicate | ObjectEntityType:ObjectName'
                 OR 'SubjectEntityType:SubjectName | Predicate | LiteralValue'
    """
    logger_func = logging.getLogger(__name__)
    triples_list: list[dict[str, Any]] = []
    if not text_block or not text_block.strip():
        return triples_list

    lines = text_block.strip().splitlines()

    for line_num, line in enumerate(lines):
        line = line.strip()
        if (
            not line or line.startswith("#") or line.startswith("//")
        ):  # Skip empty or comment lines
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            logger_func.warning(
                f"Line {line_num + 1}: Malformed triple (expected at least 3 parts separated by '|'): '{line}'"
            )
            continue

        subject_text = parts[0]
        predicate_text = parts[1]
        object_text = parts[2]

        if len(parts) > 3:
            logger_func.debug(
                f"Line {line_num + 1}: Triple had extra parts which are being ignored: '{' | '.join(parts[3:])}' from original line '{line}'"
            )

        subject_details = _get_entity_type_and_name_from_text(subject_text)
        predicate_str = (
            predicate_text.strip().upper().replace(" ", "_")
        )  # Normalize predicate

        if not subject_details.get("name") or not predicate_str:
            logger_func.warning(
                f"Line {line_num + 1}: Missing subject name or predicate: S='{subject_text}', P='{predicate_text}'"
            )
            continue

        # Determine if object is an entity or a literal
        # If object_text contains 'EntityType:', assume it's an entity.
        # Otherwise, treat as a literal value.
        object_entity_payload: dict[str, str | None] | None = None
        object_literal_payload: str | None = None
        is_literal_object = True  # Default to literal

        if ":" in object_text:
            obj_parts_check = object_text.split(":", 1)
            # Heuristic: if part before colon is a known type or capitalized, assume entity
            # This can be made more robust by checking against a list of known types.
            potential_obj_type = obj_parts_check[0].strip()
            # A simple check: if it's capitalized and has no spaces, maybe it's a type.
            # Or if it matches any of the example types.
            # For now, if a colon is present and there's content on both sides, assume it's Type:Name
            if (
                len(obj_parts_check) == 2
                and obj_parts_check[0].strip()
                and obj_parts_check[1].strip()
            ):
                # Check if potential_obj_type is likely an entity type (e.g. starts with uppercase)
                if potential_obj_type[0].isupper() and " " not in potential_obj_type:
                    object_entity_payload = _get_entity_type_and_name_from_text(
                        object_text
                    )
                    is_literal_object = False

        if is_literal_object:
            object_literal_payload = object_text.strip()
            # Further clean if it's a string literal that might have quotes (LLM sometimes adds them)
            if object_literal_payload.startswith(
                '"'
            ) and object_literal_payload.endswith('"'):
                object_literal_payload = object_literal_payload[1:-1]
            if object_literal_payload.startswith(
                "'"
            ) and object_literal_payload.endswith("'"):
                object_literal_payload = object_literal_payload[1:-1]

        if not is_literal_object and (
            not object_entity_payload or not object_entity_payload.get("name")
        ):
            # This means we thought it was an entity due to ':', but parsing failed to get a name.
            # So, revert to treating it as a literal.
            logger_func.debug(
                f"Line {line_num + 1}: Object '{object_text}' looked like entity but parsed no name. Reverting to literal."
            )
            object_literal_payload = object_text.strip()
            is_literal_object = True
            object_entity_payload = None

        triples_list.append(
            {
                "subject": subject_details,
                "predicate": predicate_str,
                "object_entity": object_entity_payload,
                "object_literal": object_literal_payload,
                "is_literal_object": is_literal_object,
            }
        )

    return triples_list
