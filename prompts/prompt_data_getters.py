# prompts/prompt_data_getters.py
"""Prepare structured context snippets for prompt templates.

This module formats state and knowledge-graph data into plain-text snippets that
are intended to be interpolated into Jinja templates rendered via
`prompts.prompt_renderer.render_prompt()`. The returned strings are
human-readable context blocks (often with lightweight Markdown formatting) and
are not guaranteed to be valid JSON/YAML unless a specific function explicitly
states otherwise.

Strictness and failure modes:

- Many functions are best-effort: missing entities or failed read operations are
  typically skipped and replaced with a short "no data available" message.
- Some helpers may allow exceptions from underlying `data_access` queries to
  propagate (notably when they do not wrap calls in `try/except`). Docstrings
  below indicate when behavior is best-effort vs. exception-propagating.

Caching:

- This module maintains a small, module-level in-memory cache (`_context_cache`)
  for expensive lookups.
- Cache entries are scoped to a chapter number using
  `_ensure_cache_is_scoped_to_chapter()`. The cache is cleared when the chapter
  changes (or when explicitly cleared via `clear_context_cache()`).
- This cache is process-local and not safe to share across processes. It is also
  not designed for concurrent use across independent chapter generations.

User-provided content and escaping:

- Returned snippets may include user-supplied or model-supplied values stored in
  the knowledge graph (names, descriptions, etc.).
- No escaping or sanitization is performed. Callers/templates must handle any
  necessary escaping conventions for the target model.
"""

import asyncio
import copy
import re
from typing import Any

import structlog

import config
import utils  # For _is_fill_in
from data_access import character_queries, kg_queries, world_queries
from models import CharacterProfile, SceneDetail, WorldItem

logger = structlog.get_logger(__name__)

# Context cache for expensive operations within a single chapter generation
_context_cache: dict[str, Any] = {}
# Track which chapter the current cache contents belong to so we can
# automatically clear the cache once the pipeline advances to a new chapter.
_current_cache_chapter: int | None = None


def clear_context_cache() -> None:
    """Clear module-level context cache for the current process.

    This resets both `_context_cache` and the chapter scoping marker
    (`_current_cache_chapter`). Callers typically invoke this at the start of a
    new chapter generation to avoid reusing stale context.

    Notes:
        This is a process-local cache. Clearing it does not affect other
        processes or machines.
    """
    global _context_cache
    global _current_cache_chapter
    _context_cache.clear()
    _current_cache_chapter = None


def _ensure_cache_is_scoped_to_chapter(chapter_number: int | None) -> None:
    """Scope module-level cache entries to a single chapter number.

    Args:
        chapter_number: Chapter number that the caller's request pertains to. If
            `None`, cache scoping is not enforced.

    Notes:
        When `chapter_number` differs from the stored `_current_cache_chapter`,
        this function clears the cache via `clear_context_cache()` before
        updating the chapter marker. This prevents cached query results from
        being reused across chapter boundaries.
    """
    global _current_cache_chapter
    if chapter_number is None:
        return
    if _current_cache_chapter == chapter_number:
        return
    # Moving to a new chapter (or first call) – drop any cached state first.
    clear_context_cache()
    _current_cache_chapter = chapter_number


def _deterministic_character_order(characters: set[str], protagonist_name: str | None) -> list[str]:
    """Return a reproducible ordering for character-focused facts."""

    protagonist_norm = utils._normalize_for_id(protagonist_name) if protagonist_name and isinstance(protagonist_name, str) else None

    def _sort_key(name: str) -> tuple[int, str]:
        normalized = utils._normalize_for_id(name)
        is_protagonist = 0 if normalized == protagonist_norm else 1
        return (is_protagonist, normalized)

    return sorted(characters, key=_sort_key)


async def _cached_character_info(char_name: str, chapter_limit: int | None) -> dict[str, Any] | None:
    """Fetch and cache a character info payload for prompt snippets.

    Args:
        char_name: Character name to query.
        chapter_limit: Chapter cutoff used to constrain returned facts. When
            `None`, a large default is used.

    Returns:
        A dictionary payload from the underlying query, or `None` if the
        character is not found.

    Raises:
        Exception: Propagates exceptions raised by the underlying
            `data_access.character_queries` call.

    Notes:
        Results are cached in `_context_cache` using a key derived from
        `char_name` and `chapter_limit`.
    """
    cache_key = f"char_info_{char_name}_{chapter_limit}"
    if cache_key not in _context_cache:
        limit_val = chapter_limit if chapter_limit is not None else 1000
        result = await character_queries.get_character_info_for_snippet_from_db(char_name, limit_val)
        _context_cache[cache_key] = result
    return _context_cache[cache_key]


async def _cached_world_item_by_id(item_id: str) -> WorldItem | None:
    """Fetch and cache a world item by id.

    Args:
        item_id: World item identifier.

    Returns:
        The corresponding `WorldItem` if found; otherwise `None`.

    Raises:
        Exception: Propagates exceptions raised by the underlying
            `data_access.world_queries` call.

    Notes:
        Results are cached in `_context_cache` using a key derived from `item_id`.
    """
    cache_key = f"world_item_{item_id}"
    if cache_key not in _context_cache:
        result = await world_queries.get_world_item_by_id(item_id)
        _context_cache[cache_key] = result
    return _context_cache[cache_key]

    # Note: cache key uses item_id to avoid repeated lookups per chapter run


def _format_dict_for_plain_text_prompt(data: dict[str, Any], indent_level: int = 0, name_override: str | None = None) -> list[str]:
    """Format a nested mapping into a deterministic, plain-text outline.

    Args:
        data: Mapping to format. Nested dictionaries and lists are formatted with
            indentation.
        indent_level: Base indentation level (two spaces per level).
        name_override: Optional heading label to emit before formatting `data`.

    Returns:
        A list of lines suitable for joining with newlines.

    Notes:
        Some internal bookkeeping keys are omitted (for example,
        `source_quality_chapter_*` and `updated_in_chapter_*`) to keep snippets
        concise for prompting.
    """
    lines = []
    indent = "  " * indent_level
    if name_override:
        lines.append(f"{indent}{name_override}:")
        indent_level += 1
        indent = "  " * indent_level

    priority_keys = [
        "description",
        "status",
        "traits",
        "relationships",
        "character_arc_summary",
        "role",
        "role_in_story",
        "goals",
        "rules",
        "key_elements",
        "atmosphere",
        "motivations",
        "prompt_notes",
    ]

    data_keys = list(data.keys())
    sorted_keys = [pk for pk in priority_keys if pk in data_keys]
    remaining_keys = sorted([k for k in data_keys if k not in priority_keys])
    sorted_keys.extend(remaining_keys)

    for key in sorted_keys:
        value = data[key]

        if key.startswith(("source_quality_chapter_", "updated_in_chapter_", "added_in_chapter_")) and key not in ["prompt_notes"]:
            continue
        if key == "is_provisional_hint":
            continue

        key_str_display = str(key).replace("_", " ").capitalize()

        if isinstance(value, dict):
            if value:
                lines.append(f"{indent}{key_str_display}:")
                lines.extend(_format_dict_for_plain_text_prompt(value, indent_level + 1))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{indent}{key_str_display}: (empty list or N/A)")
            else:
                lines.append(f"{indent}{key_str_display}:")
                try:
                    display_items = sorted([str(x) for x in value if not isinstance(x, dict)])
                    dict_items = [x for x in value if isinstance(x, dict)]
                except TypeError:
                    display_items = [str(x) for x in value if not isinstance(x, dict)]
                    dict_items = [x for x in value if isinstance(x, dict)]

                for item_str in display_items:
                    lines.append(f"{indent}  - {item_str}")
                for item_dict in dict_items:
                    lines.extend(_format_dict_for_plain_text_prompt(item_dict, indent_level + 1, name_override="- Item"))
        elif value is not None and (isinstance(value, bool) or str(value).strip()):
            lines.append(f"{indent}{key_str_display}: {str(value)}")

    return lines


def _add_provisional_notes_and_filter_developments(
    item_data_original: dict[str, Any],
    up_to_chapter_inclusive: int | None,
    is_character: bool = True,
) -> dict[str, Any]:
    """Filter future chapter developments and add provisional-data notes.

    Args:
        item_data_original: Source dictionary, typically produced by
            `CharacterProfile.to_dict()` or `WorldItem.to_dict()`.
        up_to_chapter_inclusive: Chapter cutoff. Keys representing developments
            beyond this cutoff are removed from the returned mapping. When
            `None`, no chapter-based filtering is applied.
        is_character: Whether `item_data_original` represents a character (affects
            which key prefixes are considered "developments").

    Returns:
        A deep-copied mapping with:
        - development/elaboration keys beyond the chapter cutoff removed, and
        - optional `prompt_notes` and `is_provisional_hint` fields added when
          the item includes provisional data relevant to the cutoff.

    Notes:
        This function is purely a formatter/filter. It does not fetch additional
        data and does not mutate the original input mapping.
    """
    item_data = copy.deepcopy(item_data_original)
    prompt_notes_list = []
    effective_filter_chapter = config.KG_PREPOPULATION_CHAPTER_NUM if up_to_chapter_inclusive == config.KG_PREPOPULATION_CHAPTER_NUM else up_to_chapter_inclusive

    dev_elaboration_prefix = "development_in_chapter_" if is_character else "elaboration_in_chapter_"
    added_prefix = "added_in_chapter_"

    keys_to_remove = []
    has_provisional_data_relevant_to_filter = False

    for key in list(item_data.keys()):
        if key.startswith((dev_elaboration_prefix, added_prefix)):
            try:
                chap_num_of_key_str = key.split("_")[-1]
                chapter_match = re.match(r"\d+", chap_num_of_key_str)
                chap_num_of_key = int(chapter_match.group(0)) if chapter_match else -1
                if effective_filter_chapter is not None and chap_num_of_key > effective_filter_chapter:
                    keys_to_remove.append(key)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse chapter from key '{key}' during filtering: {e}")

        if key.startswith("source_quality_chapter_"):
            try:
                chap_num_of_source_str = key.split("_")[-1]
                source_chapter_match = re.match(r"\d+", chap_num_of_source_str)
                chap_num_of_source = int(source_chapter_match.group(0)) if source_chapter_match else -1

                if (effective_filter_chapter is None or chap_num_of_source <= effective_filter_chapter) and item_data[key] == "provisional_from_unrevised_draft":
                    has_provisional_data_relevant_to_filter = True
                    note = f"Data from Chapter {chap_num_of_source} may be provisional (from unrevised draft)."
                    if note not in prompt_notes_list:
                        prompt_notes_list.append(note)
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse chapter from source_quality key '{key}': {e}")

    for k_rem in keys_to_remove:
        item_data.pop(k_rem, None)

    if has_provisional_data_relevant_to_filter:
        item_data["is_provisional_hint"] = True

    if prompt_notes_list:
        item_data["prompt_notes"] = sorted(list(set(prompt_notes_list)))

    return item_data


async def _get_character_profiles_dict_with_notes(
    character_names: list[str],
    up_to_chapter_inclusive: int | None,
) -> dict[str, Any]:
    """Fetch character profiles and apply chapter filtering and prompt notes.

    Args:
        character_names: Character names to fetch.
        up_to_chapter_inclusive: Chapter cutoff for filtering developments.

    Returns:
        Mapping of character name to a filtered dictionary representation of that
        character. Characters that cannot be fetched are omitted.

    Notes:
        This function is best-effort: individual characters that fail to fetch
        are skipped after logging.
    """
    logger.debug(
        "Internal: Getting character profiles dict with notes up to chapter %s.",
        up_to_chapter_inclusive,
    )
    processed_profiles: dict[str, Any] = {}
    filter_chapter = up_to_chapter_inclusive

    if not character_names:
        logger.warning("Character name list is empty.")
        return {}

    for char_name in character_names:
        try:
            profile_obj = await character_queries.get_character_profile_by_name(char_name)
        except Exception as exc:
            logger.error(
                "Error fetching character '%s' from Neo4j: %s",
                char_name,
                exc,
            )
            continue

        if not profile_obj:
            logger.warning("Character '%s' not found in Neo4j.", char_name)
            continue

        profile_dict = profile_obj.to_dict()
        processed_profiles[char_name] = _add_provisional_notes_and_filter_developments(
            profile_dict,
            filter_chapter,
            is_character=True,
        )

    return processed_profiles


async def get_filtered_character_profiles_for_prompt_plain_text(
    character_names: list[str],
    up_to_chapter_inclusive: int | None = None,
) -> str:
    """Build a plain-text "Key Character Profiles" block for prompt templates.

    Args:
        character_names: Character names to fetch from the knowledge graph.
        up_to_chapter_inclusive: Chapter cutoff for filtering. When `0`, this is
            mapped to `config.KG_PREPOPULATION_CHAPTER_NUM` to represent
            initial/prepopulated state.

    Returns:
        A plain-text block headed by "Key Character Profiles:" followed by
        per-character outlines. Returns a short "No character profiles available."
        message when no profiles can be fetched.

    Notes:
        This function is best-effort: missing characters and per-character fetch
        errors are skipped after logging.
    """
    logger.info(f"Fetching and formatting filtered character profiles as PLAIN TEXT up to chapter {up_to_chapter_inclusive}.")
    filter_chapter_for_profiles = config.KG_PREPOPULATION_CHAPTER_NUM if up_to_chapter_inclusive == 0 else up_to_chapter_inclusive

    profiles_dict_with_notes = await _get_character_profiles_dict_with_notes(
        character_names,
        filter_chapter_for_profiles,
    )
    if not profiles_dict_with_notes:
        return "No character profiles available."

    output_lines_list = ["Key Character Profiles:"]
    sorted_char_names = sorted(profiles_dict_with_notes.keys())

    for char_name in sorted_char_names:
        profile_data = profiles_dict_with_notes[char_name]
        if not isinstance(profile_data, dict) or not profile_data:
            continue

        output_lines_list.append("")
        formatted_profile_lines = _format_dict_for_plain_text_prompt(profile_data, indent_level=0, name_override=char_name)
        output_lines_list.extend(formatted_profile_lines)

    return "\n".join(output_lines_list).strip()


async def _get_world_data_dict_with_notes(
    world_item_ids_by_category: dict[str, list[str]],
    up_to_chapter_inclusive: int | None,
) -> dict[str, Any]:
    """Fetch world items and apply chapter filtering and prompt notes.

    Args:
        world_item_ids_by_category: Mapping of category name to world item ids.
        up_to_chapter_inclusive: Chapter cutoff for filtering developments.

    Returns:
        Mapping of category name to a mapping of item name to filtered item data.
        Missing items and failed reads are omitted.

    Notes:
        This function is best-effort: individual world items that fail to fetch
        are skipped after logging.
    """
    logger.debug(
        "Internal: Getting world data dict with notes up to chapter %s.",
        up_to_chapter_inclusive,
    )
    processed_world_data: dict[str, Any] = {}
    filter_chapter = up_to_chapter_inclusive

    if not world_item_ids_by_category:
        logger.warning("World item ID mapping is empty.")
        return {}

    for category_name, item_id_list in world_item_ids_by_category.items():
        processed_category: dict[str, Any] = {}

        for item_id in item_id_list:
            try:
                item_obj = await _cached_world_item_by_id(item_id)
            except Exception as exc:
                logger.error(
                    "Error fetching world item '%s' from Neo4j: %s",
                    item_id,
                    exc,
                )
                continue

            if not item_obj:
                logger.warning(
                    "World item id '%s' not found in Neo4j.",
                    item_id,
                )
                continue

            item_dict = item_obj.to_dict()
            processed_category[item_obj.name] = _add_provisional_notes_and_filter_developments(
                item_dict,
                filter_chapter,
                is_character=False,
            )

        processed_world_data[category_name] = processed_category

    return processed_world_data


async def get_filtered_world_data_for_prompt_plain_text(
    world_item_ids_by_category: dict[str, list[str]],
    up_to_chapter_inclusive: int | None = None,
) -> str:
    """Build a plain-text world-building block for prompt templates.

    Args:
        world_item_ids_by_category: Mapping of category name to world item ids.
        up_to_chapter_inclusive: Chapter cutoff for filtering. When `0`, this is
            mapped to `config.KG_PREPOPULATION_CHAPTER_NUM` to represent
            initial/prepopulated state.

    Returns:
        A plain-text block containing an optional "World-Building Overview:"
        section and then per-category item outlines. Returns a short "No ... data
        available" message when the graph contains no relevant items.

    Notes:
        This function is best-effort: missing items and per-item fetch errors are
        skipped after logging.
    """
    logger.info(f"Fetching and formatting filtered world data as PLAIN TEXT up to chapter {up_to_chapter_inclusive}.")
    filter_chapter_for_world = config.KG_PREPOPULATION_CHAPTER_NUM if up_to_chapter_inclusive == 0 else up_to_chapter_inclusive

    world_data_dict_with_notes = await _get_world_data_dict_with_notes(
        world_item_ids_by_category,
        filter_chapter_for_world,
    )
    if not world_data_dict_with_notes:
        return "No world-building data available."

    output_lines_list = []
    overview_data = world_data_dict_with_notes.get("_overview_")
    if overview_data and isinstance(overview_data, dict) and overview_data.get("description"):
        output_lines_list.append("World-Building Overview:")
        output_lines_list.extend(_format_dict_for_plain_text_prompt(overview_data, indent_level=0))
        output_lines_list.append("")

    sorted_categories = sorted([cat for cat in world_data_dict_with_notes.keys() if cat not in ["_overview_", "is_default", "source", "user_supplied_data"]])

    for category_name in sorted_categories:
        category_items = world_data_dict_with_notes[category_name]
        if not isinstance(category_items, dict) or not category_items:
            continue

        category_header_added = False
        sorted_item_names = sorted(category_items.keys())

        for item_name in sorted_item_names:
            item_data = category_items[item_name]
            if not isinstance(item_data, dict) or not item_data:
                continue

            if not category_header_added:
                output_lines_list.append(f"{category_name.replace('_', ' ').capitalize()}:")
                category_header_added = True

            item_lines = _format_dict_for_plain_text_prompt(item_data, indent_level=0, name_override=item_name)
            output_lines_list.extend(item_lines)
            if item_lines:
                output_lines_list.append("")

        if output_lines_list and output_lines_list[-1] == "" and category_header_added:
            pass
        elif category_header_added:
            output_lines_list.append("")

    if not output_lines_list:
        return "No significant world-building data available after filtering."
    while output_lines_list and not output_lines_list[-1].strip():
        output_lines_list.pop()
    return "\n".join(output_lines_list).strip()


async def get_reliable_kg_facts_for_drafting_prompt(
    chapter_outlines: dict[int, dict[str, Any]] | None = None,
    chapter_number: int = 1,
    chapter_plan: list[SceneDetail] | None = None,
    max_facts_per_char: int = 2,
    max_total_facts: int = 7,
    snapshot: Any | None = None,
    protagonist_name: str | None = None,
) -> str:
    """Assemble a "reliable KG facts" block for drafting prompts.

    This function combines a small set of novel-level facts (theme, central
    conflict) with character-specific facts (status, location, key relationships)
    from Neo4j. The output is intended to be embedded directly into prompt
    templates as a context section.

    Args:
        chapter_outlines: Unused; accepted for backward compatibility.
        chapter_number: Chapter number being drafted. When `<= 0`, a short
            "no facts applicable" message is returned.
        chapter_plan: Optional scene plan used to discover characters of
            interest.
        max_facts_per_char: Maximum facts to include per character.
        max_total_facts: Maximum facts to include overall.
        snapshot: Optional snapshot object. If it exposes a non-empty
            `kg_facts_block` attribute, that value is returned directly.
        protagonist_name: Protagonist name used for prioritization and proximity
            filtering. When omitted, falls back to
            `config.DEFAULT_PROTAGONIST_NAME`.

    Returns:
        A plain-text block beginning with a header line:
        `"**Key Reliable KG Facts (from Neo4j - up to previous chapter/initial state):**"`
        followed by bullet-like fact lines (prefixed with `- `). If no facts can
        be gathered, returns a short explanatory message.

    Notes:
        - This function is largely best-effort. Many Neo4j query failures are
          logged and skipped so prompt generation can proceed with partial
          context.
        - It calls `_ensure_cache_is_scoped_to_chapter()` to prevent cache reuse
          across chapters, even though most KG fact queries are not cached in
          this module.
    """
    _ensure_cache_is_scoped_to_chapter(chapter_number)

    # Snapshot fast-path: when a ContextSnapshot is provided and contains the
    # precomputed KG facts block, prefer returning it to avoid redundant reads.
    try:
        if snapshot is not None and getattr(snapshot, "kg_facts_block", None):
            return snapshot.kg_facts_block
    except Exception:
        # Non-fatal; fall back to normal path
        pass

    if chapter_number <= 0:
        return "No KG facts applicable for pre-first chapter."

    kg_chapter_limit = config.KG_PREPOPULATION_CHAPTER_NUM if chapter_number == 1 else chapter_number - 1

    facts_for_prompt_list: list[str] = []

    # Use provided protagonist_name or fall back to config default
    if protagonist_name is None:
        protagonist_name = config.DEFAULT_PROTAGONIST_NAME

    # Step 1: Discover characters of interest
    characters_of_interest = await _discover_characters_of_interest(protagonist_name, chapter_plan, chapter_number)

    # Step 2: Apply protagonist-proximity filtering
    characters_of_interest = await _apply_protagonist_proximity_filtering(characters_of_interest, protagonist_name)

    # Step 3: Gather novel-level facts
    await _gather_novel_info_facts(facts_for_prompt_list, max_total_facts)

    # Step 4: Gather character-specific facts
    await _gather_character_facts(
        characters_of_interest,
        kg_chapter_limit,
        facts_for_prompt_list,
        max_facts_per_char,
        max_total_facts,
        protagonist_name,
    )

    # Step 5: Assemble final result
    if not facts_for_prompt_list:
        return "No specific reliable KG facts identified as highly relevant for this chapter's current focus from Neo4j."

    unique_facts = sorted(list(set(facts_for_prompt_list)))
    final_prompt_parts = ["**Key Reliable KG Facts (from Neo4j - up to previous chapter/initial state):**"]
    final_prompt_parts.extend(unique_facts[:max_total_facts])
    return "\n".join(final_prompt_parts)


# Native list-based prompt data getters for improved performance
async def get_character_state_snippet_for_prompt(
    character_profiles: list[CharacterProfile],
    protagonist_name: str | None = None,
    current_chapter_num_for_filtering: int | None = None,
) -> str:
    """Build a concise character-state snippet for prompt templates.

    Args:
        character_profiles: Character profiles to summarize.
        protagonist_name: Protagonist name to prioritize in the output. When
            omitted, falls back to `config.DEFAULT_PROTAGONIST_NAME`.
        current_chapter_num_for_filtering: Chapter number used to scope
            `_context_cache` and constrain Neo4j-backed details.

    Returns:
        A plain-text block with one section per included character, using
        lightweight Markdown formatting (for example `**Name:**` headings and
        `- ` bullet lines). Characters may include data sourced from Neo4j
        (current state summary, relationships).

    Raises:
        Exception: Propagates exceptions raised by underlying Neo4j reads (for
            example from `_cached_character_info()`).

    Notes:
        This function does not escape or sanitize values. Names/descriptions may
        originate from user- or model-supplied data in the knowledge graph.
    """
    _ensure_cache_is_scoped_to_chapter(current_chapter_num_for_filtering)
    text_output_lines_list: list[str] = []
    char_names_to_process: list[str] = []

    # Use provided protagonist_name or fall back to config default
    if protagonist_name is None:
        protagonist_name = config.DEFAULT_PROTAGONIST_NAME

    # Use protagonist priority
    if protagonist_name:
        char_names_to_process.append(protagonist_name)

    # Add other characters up to limit
    for char_profile in character_profiles:
        if char_profile.name != protagonist_name:
            char_names_to_process.append(char_profile.name)
        max_chars = getattr(config, "PLANNING_CONTEXT_MAX_CHARACTERS_IN_SNIPPET", 5)
        if len(char_names_to_process) >= max_chars:
            break

    # Build character profiles from the list
    characters_to_include = {}
    for char_profile in character_profiles:
        if char_profile.name in char_names_to_process:
            characters_to_include[char_profile.name] = char_profile

    # Use existing logic for building the snippet
    for char_name in char_names_to_process:
        if char_name in characters_to_include:
            char_profile = characters_to_include[char_name]

            # Get character data from Neo4j (using cached query)
            neo4j_char_data = await _cached_character_info(char_name, current_chapter_num_for_filtering)

            profile_lines = []
            if char_profile.personality_description:
                profile_lines.append(f"Description: {char_profile.personality_description}")
            if char_profile.traits:
                traits_str = ", ".join(char_profile.traits[:3])  # Limit to 3 traits
                profile_lines.append(f"Traits: {traits_str}")
            if char_profile.status and char_profile.status != "Unknown":
                profile_lines.append(f"Status: {char_profile.status}")

            # Add additional data from updates field
            if char_profile.updates:
                if "personality" in char_profile.updates and char_profile.updates["personality"]:
                    profile_lines.append(f"Personality: {char_profile.updates['personality']}")
                if "background" in char_profile.updates and char_profile.updates["background"]:
                    profile_lines.append(f"Background: {char_profile.updates['background']}")

            # Add Neo4j data if available
            if neo4j_char_data:
                summary = neo4j_char_data.get("summary", "").strip()
                if summary:
                    profile_lines.append(f"Current State: {summary}")

                # Check for personality and background in Neo4j data if not already added
                if not any("Personality:" in line for line in profile_lines):
                    personality = neo4j_char_data.get("personality", "").strip()
                    if personality:
                        profile_lines.append(f"Personality: {personality}")

                if not any("Background:" in line for line in profile_lines):
                    background = neo4j_char_data.get("background", "").strip()
                    if background:
                        profile_lines.append(f"Background: {background}")

                key_relationships = neo4j_char_data.get("key_relationships", [])
                if key_relationships:
                    relationships_str = ", ".join(key_relationships[:3])  # Limit to 3
                    profile_lines.append(f"Key Relationships: {relationships_str}")

            if profile_lines:
                text_output_lines_list.append(f"**{char_name}:**")
                for line in profile_lines:
                    text_output_lines_list.append(f"  - {line}")
                text_output_lines_list.append("")

    return "\n".join(text_output_lines_list)


# Helper functions for KG facts gathering - extracted from get_reliable_kg_facts_for_drafting_prompt
async def _discover_characters_of_interest(
    protagonist_name: str,
    chapter_plan: list[SceneDetail] | None,
    chapter_number: int,
) -> set[str]:
    """Discover character names that are relevant to the current chapter plan.

    Args:
        protagonist_name: Protagonist name to include (if present and not a
            fill-in placeholder).
        chapter_plan: Optional scene plan. If present, character names are read
            from each scene's `characters_involved` field.
        chapter_number: Chapter number used for logging.

    Returns:
        A set of character names that may be used for follow-on KG lookups.

    Notes:
        This function performs lightweight extraction from the provided plan and
        does not query Neo4j.
    """
    # Start with protagonist-centric filtering
    characters_of_interest: set[str] = {protagonist_name} if protagonist_name and not utils._is_fill_in(protagonist_name) else set()

    # Add characters from chapter plan
    if chapter_plan and isinstance(chapter_plan, list):
        for scene_detail in chapter_plan:
            if isinstance(scene_detail, dict) and "characters_involved" in scene_detail and isinstance(scene_detail["characters_involved"], list):
                for char_name_in_plan in scene_detail["characters_involved"]:
                    if isinstance(char_name_in_plan, str) and char_name_in_plan.strip() and not utils._is_fill_in(char_name_in_plan):
                        characters_of_interest.add(char_name_in_plan.strip())

    logger.debug(f"KG fact gathering for Ch {chapter_number} draft: Chars of interest: {characters_of_interest}")
    return characters_of_interest


async def _apply_protagonist_proximity_filtering(
    characters_of_interest: set[str],
    protagonist_name: str,
) -> set[str]:
    """Filter characters by proximity to the protagonist in the knowledge graph.

    Args:
        characters_of_interest: Candidate character names.
        protagonist_name: Protagonist name used as the reference node.

    Returns:
        A set containing the protagonist and any characters whose shortest-path
        distance to the protagonist is within a small threshold.

    Notes:
        This function queries Neo4j for shortest-path distance. When run in the
        parallel path, individual query failures are treated as non-matches.
    """
    if not protagonist_name or not characters_of_interest:
        return characters_of_interest

    pruned: set[str] = set()
    # Small N expected, but parallelize when > 2 to reduce latency.
    others = [c for c in characters_of_interest if c != protagonist_name]
    if not others:
        return {protagonist_name}
    if len(others) <= 2:
        for c in others:
            path_len = await kg_queries.get_shortest_path_length_between_entities(protagonist_name, c)
            if path_len is not None and path_len <= 3:
                pruned.add(c)
    else:
        tasks = [kg_queries.get_shortest_path_length_between_entities(protagonist_name, c) for c in others]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for c, result in zip(others, results, strict=False):
            if isinstance(result, BaseException):
                continue
            if isinstance(result, int) and result <= 3:
                pruned.add(c)
    pruned.add(protagonist_name)

    logger.debug("After protagonist-proximity filtering: %s", pruned)
    return pruned


async def _gather_novel_info_facts(
    facts_list: list[str],
    max_total_facts: int,
) -> None:
    """Gather novel-level facts (theme and central conflict) for prompting.

    Args:
        facts_list: Output list to append formatted fact lines to.
        max_total_facts: Maximum total fact lines allowed in `facts_list`.

    Notes:
        This function is best-effort: query failures are logged and skipped.
    """
    if len(facts_list) >= max_total_facts:
        return

    # Parallel execution of novel info property queries
    novel_info_tasks = [
        kg_queries.get_novel_info_property_from_db("theme"),
        kg_queries.get_novel_info_property_from_db("central_conflict"),
    ]
    novel_info_results = await asyncio.gather(*novel_info_tasks, return_exceptions=True)

    # Process theme
    if len(facts_list) < max_total_facts and not isinstance(novel_info_results[0], Exception):
        value = novel_info_results[0]
        if value:
            fact_text = f"- The novel's central theme is: {value}."
            if fact_text not in facts_list:
                facts_list.append(fact_text)
    elif isinstance(novel_info_results[0], Exception):
        logger.warning(f"KG Query for novel context 'theme' failed: {novel_info_results[0]}")

    # Process central conflict
    if len(facts_list) < max_total_facts and not isinstance(novel_info_results[1], Exception):
        value = novel_info_results[1]
        if value:
            fact_text = f"- The main conflict summary: {value}."
            if fact_text not in facts_list:
                facts_list.append(fact_text)
    elif isinstance(novel_info_results[1], Exception):
        logger.warning(f"KG Query for novel context 'central_conflict' failed: {novel_info_results[1]}")


async def _gather_character_facts(
    characters_of_interest: set[str],
    kg_chapter_limit: int,
    facts_list: list[str],
    max_facts_per_char: int,
    max_total_facts: int,
    protagonist_name: str | None,
) -> None:
    """Gather character-specific facts (status, location, relationships).

    Args:
        characters_of_interest: Candidate character names.
        kg_chapter_limit: Chapter cutoff used when querying the KG.
        facts_list: Output list to append formatted fact lines to.
        max_facts_per_char: Per-character cap on added facts.
        max_total_facts: Overall cap on added facts.
        protagonist_name: Protagonist name used to prioritize character ordering.

    Notes:
        This function is best-effort for the parallel query path: individual
        failures are logged and skipped.
    """
    character_names_list = _deterministic_character_order(characters_of_interest, protagonist_name)[:3]
    if not character_names_list:
        return

    # Prepare character-related queries for parallel execution
    character_tasks = []
    for char_name in character_names_list:
        character_tasks.extend(
            [
                character_queries.get_character_info_for_snippet_from_db(char_name, kg_chapter_limit),
                kg_queries.get_most_recent_value_from_db(char_name, "LOCATED_AT", kg_chapter_limit),
                kg_queries.query_kg_from_db(subject=char_name, chapter_limit=kg_chapter_limit),
            ]
        )

    # Execute all character-related queries in parallel
    character_results = await asyncio.gather(*character_tasks, return_exceptions=True)

    # Process results
    interesting_rel_types = [
        "ally_of",
        "enemy_of",
        "mentor_of",
        "protege_of",
        "works_for",
        "related_to",
    ]

    for i, char_name in enumerate(character_names_list):
        if len(facts_list) >= max_total_facts:
            break

        facts_for_this_char = 0
        # Get results for this character (3 results per character)
        char_info_result = character_results[i * 3]
        location_result = character_results[i * 3 + 1]
        relationships_result = character_results[i * 3 + 2]

        # Process status from character info
        if not isinstance(char_info_result, Exception) and char_info_result and facts_for_this_char < max_facts_per_char and len(facts_list) < max_total_facts:
            status_value = char_info_result.get("current_status")
            if status_value:
                fact_text = f"- {char_name}'s status is: {status_value}."
                if fact_text not in facts_list:
                    facts_list.append(fact_text)
                    facts_for_this_char += 1
        elif isinstance(char_info_result, Exception):
            logger.warning("KG Query for %s's character info failed: %s", char_name, char_info_result)

        # Process location
        if not isinstance(location_result, Exception) and location_result and facts_for_this_char < max_facts_per_char and len(facts_list) < max_total_facts:
            fact_text = f"- {char_name} is located in: {location_result}."
            if fact_text not in facts_list:
                facts_list.append(fact_text)
                facts_for_this_char += 1
        elif isinstance(location_result, Exception):
            logger.warning(f"KG Query for {char_name}'s location failed: {location_result}")

        # Process relationships
        if not isinstance(relationships_result, Exception) and isinstance(relationships_result, list) and facts_for_this_char < max_facts_per_char and len(facts_list) < max_total_facts:
            for rel_res in relationships_result:
                if facts_for_this_char >= max_facts_per_char or len(facts_list) >= max_total_facts:
                    break
                if rel_res.get("predicate") in interesting_rel_types:
                    rel_type_display = rel_res["predicate"].replace("_", " ")
                    fact_text = f"- {char_name} has a key relationship ({rel_type_display}) with: {rel_res.get('object')}."
                    if fact_text not in facts_list:
                        facts_list.append(fact_text)
                        facts_for_this_char += 1
        elif isinstance(relationships_result, Exception):
            logger.warning(f"KG Query for {char_name}'s relationships failed: {relationships_result}")


async def get_world_state_snippet_for_prompt(
    world_building: list[WorldItem],
    current_chapter_num_for_filtering: int | None = None,
) -> str:
    """Build a concise world-state snippet for prompt templates.

    Args:
        world_building: World items to summarize.
        current_chapter_num_for_filtering: Chapter number used to scope
            `_context_cache`.

    Returns:
        A plain-text block grouped by category and using lightweight Markdown
        formatting (for example `**Category:**` headings and indented item
        bullets).

    Notes:
        This function does not escape or sanitize values. Item names and
        descriptions may originate from user- or model-supplied data in the
        knowledge graph.
    """
    _ensure_cache_is_scoped_to_chapter(current_chapter_num_for_filtering)
    text_output_lines_list: list[str] = []

    # Group world items by category
    world_by_category: dict[str, list[WorldItem]] = {}
    for item in world_building:
        category = item.category or "Miscellaneous"
        if category not in world_by_category:
            world_by_category[category] = []
        world_by_category[category].append(item)

    # Limit items per category for prompt efficiency
    max_items_per_category = config.PLANNING_CONTEXT_MAX_WORLD_ITEMS_PER_CATEGORY if hasattr(config, "PLANNING_CONTEXT_MAX_WORLD_ITEMS_PER_CATEGORY") else 3

    for category, items in world_by_category.items():
        if not items:
            continue

        text_output_lines_list.append(f"**{category}:**")

        # Sort by importance/relevance (items with descriptions first)
        sorted_items = sorted(items, key=lambda x: (not bool(x.description), x.name))

        for item in sorted_items[:max_items_per_category]:
            item_lines = []
            if item.description:
                item_lines.append(f"Description: {item.description}")

            # Add additional data from the model (goals, rules, key_elements if available)
            if item.goals:
                goals_str = ", ".join(item.goals[:2])  # Limit to 2 goals
                item_lines.append(f"Goals: {goals_str}")

            if item.rules:
                rules_str = ", ".join(item.rules[:2])  # Limit to 2 rules
                item_lines.append(f"Rules: {rules_str}")

            if item.key_elements:
                elements_str = ", ".join(item.key_elements[:3])  # Limit to 3 elements
                item_lines.append(f"Key Elements: {elements_str}")

            if item_lines:
                text_output_lines_list.append(f"  - **{item.name}:**")
                for line in item_lines:
                    text_output_lines_list.append(f"    {line}")
            else:
                text_output_lines_list.append(f"  - **{item.name}**")

        text_output_lines_list.append("")

    return "\n".join(text_output_lines_list)


# Helper functions for KG facts gathering - extracted from get_reliable_kg_facts_for_drafting_prompt
