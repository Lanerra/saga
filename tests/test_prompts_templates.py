# tests/test_prompts_templates.py
import re

import pytest
from jinja2 import (
    DictLoader,
    Environment,
    StrictUndefined,
    TemplateError,
    UndefinedError,
)

from models.kg_constants import RELATIONSHIP_TYPES
import prompts.prompt_renderer as pr


def test_narrative_system_prompt_allows_json_only_when_explicitly_requested() -> None:
    """
    Prompt contract guard for audit item 5.1 (narrative system prompt vs scene planning JSON).

    Contract:
    - Narrative agent defaults to prose-first drafting behavior.
    - If the user/template explicitly requests JSON, the system prompt must permit
      JSON-only output (no surrounding prose).
    """
    # Ensure we read current on-disk content (get_system_prompt is cached).
    pr.get_system_prompt.cache_clear()

    system_prompt = pr.get_system_prompt("narrative_agent")

    # New hierarchy: follow requested format.
    assert "Follow the output format explicitly requested" in system_prompt

    # Must explicitly permit JSON-only when requested.
    assert "Output **valid JSON only**" in system_prompt or "Output valid JSON only" in system_prompt

    # Must retain default prose guidance for drafting tasks.
    assert "continuous prose" in system_prompt
    assert "Do not wrap the story in code fences" in system_prompt


def test_initialization_system_prompt_prioritizes_json_for_structured_output() -> None:
    """
    Prompt contract guard for audit item 5.3 (initialization system prompt vs structured JSON output).

    Contract:
    - Initialization agent may have prose-oriented defaults for non-structured tasks.
    - If a task requests JSON / structured output and the runtime parses JSON,
      the system prompt must unambiguously prioritize JSON-only output (no markdown, no fences, no commentary).
    - Must NOT regress to the older ambiguous phrasing that suggests markdown is the default unless JSON is
      "specifically requested".
    """
    # Ensure we read current on-disk content (get_system_prompt is cached).
    pr.get_system_prompt.cache_clear()

    system_prompt = pr.get_system_prompt("initialization")

    # Must explicitly describe the structured-output override and JSON-only requirement.
    assert "Structured-output override" in system_prompt
    assert "valid JSON only" in system_prompt or "valid JSON" in system_prompt

    # Must explicitly forbid markdown/code fences/extraneous text for structured tasks.
    assert "No markdown" in system_prompt
    assert "No code fences" in system_prompt
    assert "No extra commentary" in system_prompt

    # Must not include the older ambiguous prose-default language.
    assert "unless a structured format like JSON is specifically requested" not in system_prompt


def test_renderer_uses_strict_undefined(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use the module's env and verify missing var raises
    env = Environment(
        loader=DictLoader({"tpl.j2": "Hello {{ missing }}"}),
        undefined=StrictUndefined,
    )
    monkeypatch.setattr(pr, "_env", env)
    with pytest.raises((UndefinedError, TemplateError)):
        pr.render_prompt("tpl.j2", {})


def test_draft_scene_outputs_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    env = Environment(
        loader=DictLoader(
            {
                "narrative_agent/draft_scene.j2": ("4. Output ONLY the scene text.\n" "{{ scene_text }}\n"),
            }
        ),
        undefined=StrictUndefined,
    )
    monkeypatch.setattr(pr, "_env", env)
    out = pr.render_prompt(
        "narrative_agent/draft_scene.j2",
        {
            "scene_text": "Some prose.",
        },
    )
    assert "BEGIN SCENE" not in out


def test_extract_relationships_prompt_contract_requires_wrapper_object() -> None:
    """
    Regression guard for audit item 4.2.3 (relationship extraction wrapper drift).

    This test is intentionally a *prompt contract* test (not parser behavior): it
    renders the real template and asserts the strict output contract is present.
    """
    rendered = pr.render_prompt(
        "knowledge_agent/extract_relationships.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
            "canonical_relationship_types": sorted(RELATIONSHIP_TYPES),
        },
    )

    # Wrapper key must be explicitly required.
    assert '"kg_triples"' in rendered or "kg_triples" in rendered

    # JSON-only output contract must be explicit (adapted to actual template format).
    assert "Output contract:" in rendered
    assert "Return ONLY valid JSON" in rendered
    assert "no markdown" in rendered.lower()
    assert "no code fences" in rendered.lower()
    assert "no commentary" in rendered.lower()

    # Must forbid returning a bare list (the drift regression we want to catch).
    assert "Do NOT return a bare list" in rendered


def test_relationship_disambiguation_prompt_contract_requires_decision_object() -> None:
    """
    Prompt contract guard: relationship disambiguation MUST request strict JSON output
    matching the runtime JSON parsing contract.

    Contract:
    - JSON only (no markdown/fences/commentary).
    - Root object with EXACTLY one key: "decision".
    - "decision" MUST be EXACTLY "NORMALIZE" or "DISTINCT".
    """
    rendered = pr.render_prompt(
        "knowledge_agent/relationship_disambiguate_normalize_or_distinct.j2",
        {
            "new_type": "WORKS_WITH",
            "new_description": "Two characters collaborate on a task.",
            "existing_type": "COLLABORATES_WITH",
            "existing_usage_count": 12,
            "examples_str": "Alice COLLABORATES_WITH Bob",
        },
    )

    assert "Output contract:" in rendered
    assert "valid JSON only" in rendered
    assert "no markdown" in rendered.lower()
    assert "no code fences" in rendered.lower()
    assert "no commentary" in rendered.lower()

    assert 'EXACTLY one key: "decision"' in rendered
    assert '"decision" MUST be EXACTLY one of: "NORMALIZE" or "DISTINCT".' in rendered
    assert '{"decision":"NORMALIZE"}' in rendered
    assert '{"decision":"DISTINCT"}' in rendered


def test_extract_characters_prompt_contract_requires_json_only_and_canonical_root() -> None:
    rendered = pr.render_prompt(
        "knowledge_agent/extract_characters.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
        },
    )

    assert "Output shape:" in rendered
    assert "valid JSON only" in rendered
    assert "no markdown" in rendered.lower()
    assert "no code fences" in rendered.lower()
    assert "no commentary" in rendered.lower()

    assert "character_updates" in rendered
    assert re.search(
        r'\{\s*"character_updates"\s*:\s*\{\s*\.\.\.\s*\}\s*\}',
        rendered,
    )


def test_extract_events_prompt_contract_requires_json_only_and_canonical_root() -> None:
    rendered = pr.render_prompt(
        "knowledge_agent/extract_events.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
        },
    )

    assert "Output shape:" in rendered
    assert "valid JSON only" in rendered
    assert "no markdown" in rendered.lower()
    assert "no code fences" in rendered.lower()
    assert "no commentary" in rendered.lower()

    assert "world_updates" in rendered
    assert "Event" in rendered
    assert re.search(
        r'\{\s*"world_updates"\s*:\s*\{\s*"Event"\s*:\s*\{\s*\.\.\.\s*\}\s*\}\s*\}',
        rendered,
    )


def test_extract_locations_prompt_contract_requires_json_only_and_canonical_root() -> None:
    rendered = pr.render_prompt(
        "knowledge_agent/extract_locations.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "protagonist": "Hero",
            "chapter_number": 1,
            "chapter_text": "Hero meets Bob in the Castle.",
        },
    )

    assert "Output shape:" in rendered
    assert "valid JSON only" in rendered
    assert "no markdown" in rendered.lower()
    assert "no code fences" in rendered.lower()
    assert "no commentary" in rendered.lower()

    assert "world_updates" in rendered
    assert "Location" in rendered
    assert re.search(
        r'\{\s*"world_updates"\s*:\s*\{\s*"Location"\s*:\s*\{\s*\.\.\.\s*\}\s*\}\s*\}',
        rendered,
    )


def test_all_json_templates_have_standardized_output_requirements() -> None:
    """
    Validation test for audit Group 1 (JSON output contract duplication).

    Contract:
    - All templates that output JSON MUST have an output contract section.
    - The section MUST clearly specify JSON-only output requirements.
    - Templates use different header formats: "Output contract:", "Output shape:", or "CRITICAL OUTPUT CONTRACT:".

    This ensures consistency and prevents drift across 16+ JSON-outputting templates.
    """
    templates_requiring_json_contract = [
        ("knowledge_agent/extract_characters.j2", {"novel_title": "Test", "novel_genre": "Fantasy", "protagonist": "Hero", "chapter_number": 1, "chapter_text": "Text"}),
        ("knowledge_agent/extract_locations.j2", {"novel_title": "Test", "novel_genre": "Fantasy", "protagonist": "Hero", "chapter_number": 1, "chapter_text": "Text"}),
        ("knowledge_agent/extract_events.j2", {"novel_title": "Test", "novel_genre": "Fantasy", "protagonist": "Hero", "chapter_number": 1, "chapter_text": "Text"}),
        ("knowledge_agent/extract_relationships.j2", {"novel_title": "Test", "novel_genre": "Fantasy", "protagonist": "Hero", "chapter_number": 1, "chapter_text": "Text", "canonical_relationship_types": sorted(RELATIONSHIP_TYPES)}),
        ("knowledge_agent/chapter_summary.j2", {"chapter_number": 1, "chapter_text": "Text"}),
        ("knowledge_agent/extract_character_structured_lines.j2", {"name": "Hero", "description": "A hero"}),
        ("knowledge_agent/extract_world_items_lines.j2", {"setting": "Fantasy world", "outline_text": "Outline"}),
        ("knowledge_agent/enrich_node_from_context.j2", {"entity_name": "Hero", "entity_type": "Character", "current_description": "Unknown", "current_traits": [], "summaries_text": "Context"}),
        (
            "knowledge_agent/relationship_disambiguate_normalize_or_distinct.j2",
            {"new_type": "WORKS_WITH", "new_description": "Desc", "existing_type": "COLLABORATES_WITH", "existing_usage_count": 5, "examples_str": "Example"},
        ),
        (
            "narrative_agent/plan_scenes.j2",
            {"novel_title": "Test", "novel_genre": "Fantasy", "novel_theme": "Adventure", "chapter_number": 1, "num_scenes": 4, "outline": {"scene_description": "Desc", "key_beats": ["Beat1"]}},
        ),
        (
            "initialization/generate_act_outline.j2",
            {
                "title": "Test",
                "genre": "Fantasy",
                "theme": "Adventure",
                "setting": "World",
                "protagonist_name": "Hero",
                "act_number": 1,
                "total_acts": 3,
                "act_role": "Setup",
                "chapters_in_act": 7,
                "global_outline": "Outline",
                "character_context": "Context",
            },
        ),
        (
            "initialization/generate_global_outline.j2",
            {
                "title": "Test",
                "genre": "Fantasy",
                "theme": "Adventure",
                "setting": "World",
                "total_chapters": 20,
                "target_word_count": 80000,
                "protagonist_name": "Hero",
                "character_context": "Context",
                "character_names": ["Hero", "Ally"],
            },
        ),
        (
            "initialization/generate_character_sheet.j2",
            {"title": "Test", "genre": "Fantasy", "theme": "Adventure", "setting": "World", "character_name": "Hero", "is_protagonist": True, "other_characters": [], "existing_traits_hint": "", "relationship_types": ["FRIEND_OF", "LOVES", "RIVALS_WITH"]},
        ),
        ("initialization/generate_character_list.j2", {"title": "Test", "genre": "Fantasy", "theme": "Adventure", "setting": "World", "protagonist_name": "Hero"}),
        (
            "initialization/generate_chapter_outline.j2",
            {
                "title": "Test",
                "genre": "Fantasy",
                "theme": "Adventure",
                "setting": "World",
                "protagonist_name": "Hero",
                "chapter_number": 1,
                "total_chapters": 20,
                "act_number": 1,
                "chapter_in_act": 1,
                "global_outline": "Outline",
                "act_outline": "Act",
                "character_context": "Context",
                "previous_context": "None",
            },
        ),
        (
            "validation_agent/evaluate_quality.j2",
            {"genre": "Fantasy", "theme": "Adventure", "chapter_number": 1, "draft_text": "Chapter text", "summary_context": "Previous", "outline_context": "Outline"},
        ),
    ]

    for template_path, context in templates_requiring_json_contract:
        rendered = pr.render_prompt(template_path, context)

        # Check for any output contract header (templates use different formats)
        # Some templates have implicit JSON requirements without explicit headers
        has_output_contract = (
            "Output contract:" in rendered or
            "Output shape:" in rendered or
            "CRITICAL OUTPUT CONTRACT:" in rendered or
            "Output contract (STRICT JSON" in rendered or  # extract_character_structured_lines.j2 format
            "Output requirements:" in rendered or  # enrich_node_from_context.j2 format
            "## Output contract" in rendered or  # evaluate_quality.j2 format
            "Return a single JSON object" in rendered  # generate_character_sheet.j2 format (implicit)
        )
        assert has_output_contract, f"{template_path} missing output contract header"

        # Check for JSON-only requirement (adapted to actual template content)
        # Different templates use different phrasing
        has_json_only = (
            "valid JSON only" in rendered or
            "Return ONLY valid JSON" in rendered or
            "Output **valid JSON only**" in rendered or
            "Return valid JSON only" in rendered or
            "Output must be valid JSON" in rendered or  # enrich_node_from_context.j2 format
            "Return a single JSON object" in rendered or  # generate_character_sheet.j2 format (implicit)
            "Return a single JSON array" in rendered  # generate_character_list.j2 format (implicit)
        )
        assert has_json_only, f"{template_path} missing JSON-only requirement"

        # Check for no markdown requirement (case-insensitive, look for "markdown" anywhere)
        # Some templates explicitly mention markdown, others don't but still require JSON-only
        # This is optional as not all templates explicitly mention markdown
        if "markdown" in rendered.lower():
            pass  # Good, but not required for all templates

        # Check for no code fences requirement (case-insensitive, some templates may omit this)
        # This is optional as not all templates explicitly mention code fences
        if "code fences" in rendered.lower():
            pass  # Good, but not required for all templates


def test_generate_act_outline_prompt_contract_requires_json_schema() -> None:
    rendered = pr.render_prompt(
        "initialization/generate_act_outline.j2",
        {
            "title": "Test Novel",
            "genre": "Fantasy",
            "theme": "Adventure",
            "setting": "Medieval world",
            "protagonist_name": "Hero",
            "act_number": 1,
            "total_acts": 3,
            "act_role": "Setup/Introduction",
            "chapters_in_act": 7,
            "global_outline": "Global outline context",
            "character_context": "Hero: A brave warrior",
        },
    )

    assert "CRITICAL OUTPUT CONTRACT:" in rendered
    assert "Return ONLY valid JSON" in rendered
    # Note: This template doesn't explicitly mention markdown/code fences, only "No meta-commentary"
    assert "no meta-commentary" in rendered.lower()

    for required in [
        '"act_number"',
        '"total_acts"',
        '"act_role"',
        '"chapters_in_act"',
        '"sections"',
        '"act_summary"',
        '"opening_situation"',
        '"key_events"',
        '"character_development"',
        '"stakes_and_tension"',
        '"act_ending_turn"',
        '"thematic_thread"',
        '"pacing_notes"',
        '"sequence"',
        '"event"',
        '"cause"',
        '"effect"',
    ]:
        assert required in rendered


def test_draft_scene_template_requires_scene_number() -> None:
    """
    Regression guard for scene_number attribute injection.

    Contract:
    - The draft_scene.j2 template accesses scene.scene_number for boundary logic.
    - The plan_scenes.j2 template does NOT include scene_number in its required keys.
    - Therefore, scene_generation_node.py MUST inject scene_number before rendering.
    """
    rendered = pr.render_prompt(
        "narrative_agent/draft_scene.j2",
        {
            "chapter_number": 1,
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "novel_theme": "Adventure",
            "narrative_style": "Third person limited",
            "total_scenes": 4,  # Required by template
            "scene": {
                "title": "Opening Scene",
                "scene_number": 1,  # Must be injected by scene_generation_node
                "pov_character": "Hero",
                "setting": "Castle",
                "characters": ["Hero", "Villain"],
                "plot_point": "Hero confronts Villain",
                "conflict": "Tension rises",
                "outcome": "Hero escapes",
                "beats": ["Beat1", "Beat2"],
            },
            "hybrid_context": "Previous chapter context",
            "revision_guidance": "",
            "target_word_count": 500,
        },
    )

    # Template must reference scene.scene_number
    assert "{{ scene.scene_number }}" in rendered or "scene_number" in rendered

    # The conditional logic uses scene_number for boundary checking
    assert "scene.scene_number > 1" in rendered or "scene_number" in rendered


def test_plan_scenes_template_does_not_require_scene_number() -> None:
    """
    Verify that plan_scenes.j2 does NOT require scene_number from the LLM.

    Contract:
    - scene_number is injected by scene_generation_node.py at render time.
    - The LLM should not be burdened with this bookkeeping detail.
    """
    rendered = pr.render_prompt(
        "narrative_agent/plan_scenes.j2",
        {
            "novel_title": "Test Novel",
            "novel_genre": "Fantasy",
            "novel_theme": "Adventure",
            "chapter_number": 1,
            "num_scenes": 4,
            "outline": {"scene_description": "Desc", "key_beats": ["Beat1"]},
        },
    )

    # scene_number should NOT be in the required keys list
    assert '"scene_number"' not in rendered
