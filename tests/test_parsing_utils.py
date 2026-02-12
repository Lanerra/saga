# tests/test_parsing_utils.py
from processing.parsing_utils import (
    _get_entity_type_and_name_from_text,
    _is_proper_noun,
    _should_filter_entity,
    parse_llm_triples,
)


class TestRdfTripleParsing:
    def test_parse_simple_turtle(self) -> None:
        triple_input = """
        Jax | hasAlias | J.X.
        Jax | livesIn | Hourglass Curios
        Jax | type | Character
        Hourglass Curios | type | Location
        Hourglass Curios | description | A dusty shop
        SomeEvent | involvedCharacter | Jax
        SomeEvent | involvedCharacter | Lila
        Lila | type | Character
        Lila | label | Lila
        """

        parsed_triples = parse_llm_triples(triple_input)

        assert len(parsed_triples) == 9

        jax_alias_triple = next(
            (t for t in parsed_triples if t["subject"]["name"] == "Jax" and t["predicate"] == "HASALIAS"),
            None,
        )
        assert jax_alias_triple is not None
        assert jax_alias_triple["object_literal"] == "J.X."
        assert jax_alias_triple["is_literal_object"]

        jax_livesin_triple = next(
            (t for t in parsed_triples if t["subject"]["name"] == "Jax" and t["predicate"] == "LIVESIN"),
            None,
        )
        assert jax_livesin_triple is not None
        assert jax_livesin_triple["is_literal_object"]
        assert jax_livesin_triple["object_literal"] == "Hourglass Curios"

        jax_type_triple = next(
            (t for t in parsed_triples if t["subject"]["name"] == "Jax" and t["predicate"] == "TYPE"),
            None,
        )
        assert jax_type_triple is not None
        assert jax_type_triple["object_literal"] == "Character"

        lila_type_triple = next(
            (t for t in parsed_triples if t["subject"]["name"] == "Lila" and t["predicate"] == "TYPE"),
            None,
        )
        assert lila_type_triple is not None
        assert lila_type_triple["object_literal"] == "Character"

    def test_empty_input(self) -> None:
        parsed_triples = parse_llm_triples("")
        assert len(parsed_triples) == 0

    def test_invalid_turtle(self) -> None:
        invalid_turtle = r"char:Jax prop:hasAlias 'J.X.'"
        parsed_triples = parse_llm_triples(invalid_turtle)
        assert len(parsed_triples) == 0


class TestProperNounDetection:
    """Proper noun detection for entity filtering."""

    def test_clear_proper_nouns(self) -> None:
        assert _is_proper_noun("Alice")
        assert _is_proper_noun("Elara Moonwhisper")
        assert _is_proper_noun("Sunken Library")
        assert _is_proper_noun("Order of the Flame")
        assert _is_proper_noun("Starfall Map")

    def test_clear_common_nouns(self) -> None:
        assert not _is_proper_noun("the girl")
        assert not _is_proper_noun("the artifact")
        assert not _is_proper_noun("the sword")
        assert not _is_proper_noun("the rebellion")
        assert not _is_proper_noun("a mysterious stranger")

    def test_mixed_case_titles(self) -> None:
        assert _is_proper_noun("Council of Elders")
        assert _is_proper_noun("Library of Alexandria")
        assert _is_proper_noun("Sword of Truth")
        assert not _is_proper_noun("the Council")

    def test_edge_cases(self) -> None:
        assert not _is_proper_noun("")
        assert not _is_proper_noun("   ")
        assert not _is_proper_noun("the")

    def test_all_lowercase(self) -> None:
        assert not _is_proper_noun("ancient artifact")
        assert not _is_proper_noun("old hermit")

    def test_mixed_capitalization(self) -> None:
        assert _is_proper_noun("Alice in Wonderland")
        assert _is_proper_noun("Old Man of the Mountain")


class TestEntityFiltering:
    """Entity filtering with proper noun preference."""

    def test_proper_noun_single_mention(self) -> None:
        assert not _should_filter_entity("Alice", entity_type="Character", mention_count=1)
        assert not _should_filter_entity("Sunken Library", entity_type="Location", mention_count=1)

    def test_common_noun_single_mention(self) -> None:
        assert _should_filter_entity("the girl", entity_type="Character", mention_count=1)
        assert _should_filter_entity("the artifact", entity_type="Object", mention_count=1)

    def test_common_noun_three_mentions(self) -> None:
        assert not _should_filter_entity("the rebellion", entity_type="Faction", mention_count=3)
        assert not _should_filter_entity("the mysterious visitor", entity_type="Character", mention_count=5)

    def test_common_noun_two_mentions(self) -> None:
        assert _should_filter_entity("the rebellion", entity_type="Faction", mention_count=2)

    def test_blacklisted_entities(self) -> None:
        assert _should_filter_entity("violet glow", entity_type="Concept", mention_count=10)
        assert _should_filter_entity("fear", entity_type="Emotion", mention_count=5)
        assert _should_filter_entity("the moment", entity_type="Moment", mention_count=3)

    def test_very_short_names(self) -> None:
        assert _should_filter_entity("he", mention_count=10)
        assert _should_filter_entity("it", mention_count=10)
        assert _should_filter_entity("a", mention_count=10)

    def test_descriptive_adjectives(self) -> None:
        assert _should_filter_entity("beautiful", mention_count=5)
        assert _should_filter_entity("dark", mention_count=3)

    def test_empty_names(self) -> None:
        assert _should_filter_entity("")
        assert _should_filter_entity(None)
        assert _should_filter_entity("   ")


class TestProperNounPreferenceInTriples:
    """Proper noun preference in actual triple parsing."""

    def test_proper_noun_entities_extracted(self) -> None:
        triple_input = """
        Character:Alice | LOVES | Character:Bob
        Character:Alice | LOCATED_IN | Location:Wonderland
        """
        parsed = parse_llm_triples(triple_input)

        assert len(parsed) == 2

        alice_loves = next(
            (t for t in parsed if t["subject"]["name"] == "Alice" and t["predicate"] == "LOVES"),
            None,
        )
        assert alice_loves is not None
        assert alice_loves["object_entity"]["name"] == "Bob"

    def test_blacklisted_entities_filtered_from_triples(self) -> None:
        triple_input = """
        Character:Alice | FEELS | Emotion:fear
        Location:Library | EMITS | violet glow
        Character:Alice | HAS_STATUS | healthy
        """
        parsed = parse_llm_triples(triple_input)

        alice_triples = [t for t in parsed if t["subject"]["name"] == "Alice"]

        assert len(alice_triples) == 1
        assert alice_triples[0]["predicate"] == "HAS_STATUS"


class TestEntityParsingHeuristics:
    def test_standard_colon_format(self) -> None:
        result = _get_entity_type_and_name_from_text("Character: Alice")
        assert result["type"] == "Character"
        assert result["name"] == "Alice"

    def test_missing_colon_known_type(self) -> None:
        result = _get_entity_type_and_name_from_text("Character Alice")
        assert result["type"] == "Character"
        assert result["name"] == "Alice"

    def test_missing_colon_unknown_type(self) -> None:
        result = _get_entity_type_and_name_from_text("Alice Bob")
        assert result["type"] is None
        assert result["name"] == "Alice Bob"

    def test_colon_with_empty_name(self) -> None:
        result = _get_entity_type_and_name_from_text("Location: ")
        assert result["type"] == "Location"
        assert result["name"] is None

    def test_just_type_name(self) -> None:
        result = _get_entity_type_and_name_from_text("Character")
        assert result["type"] == "Character"
        assert result["name"] is None

    def test_whitespace_handling(self) -> None:
        result = _get_entity_type_and_name_from_text("  Location  :  The Void  ")
        assert result["type"] == "Location"
        assert result["name"] == "The Void"

    def test_case_insensitive_type_match(self) -> None:
        result = _get_entity_type_and_name_from_text("character Bob")
        assert result["type"] is not None
        assert result["type"].lower() == "character"
        assert result["name"] == "Bob"


class TestObjectDetection:
    """Object entity vs literal detection in parse_llm_triples."""

    def test_standard_colon_entity(self) -> None:
        input_text = "Character:Alice | LOVES | Character:Bob"
        parsed = parse_llm_triples(input_text)
        assert len(parsed) == 1
        assert not parsed[0]["is_literal_object"]
        assert parsed[0]["object_entity"] is not None
        assert parsed[0]["object_entity"]["type"] == "Character"
        assert parsed[0]["object_entity"]["name"] == "Bob"

    def test_missing_colon_known_type_entity(self) -> None:
        input_text = "Character:Alice | LOVES | Character Bob"
        parsed = parse_llm_triples(input_text)
        assert len(parsed) == 1
        assert not parsed[0]["is_literal_object"]
        assert parsed[0]["object_entity"] is not None
        assert parsed[0]["object_entity"]["type"] == "Character"
        assert parsed[0]["object_entity"]["name"] == "Bob"

    def test_literal_string(self) -> None:
        input_text = "Character:Alice | HAS_STATUS | Happy and healthy"
        parsed = parse_llm_triples(input_text)
        assert len(parsed) == 1
        assert parsed[0]["is_literal_object"]
        assert parsed[0]["object_literal"] == "Happy and healthy"

    def test_unknown_type_prefix_is_literal(self) -> None:
        input_text = "Character:Alice | HAS_TRAIT | Strange Behavior"
        parsed = parse_llm_triples(input_text)
        assert len(parsed) == 1
        assert parsed[0]["is_literal_object"]
        assert parsed[0]["object_literal"] == "Strange Behavior"

    def test_single_word_name_is_literal(self) -> None:
        input_text = "Character:Alice | KNOWS | Bob"
        parsed = parse_llm_triples(input_text)
        assert len(parsed) == 1
        assert parsed[0]["is_literal_object"]
        assert parsed[0]["object_literal"] == "Bob"

    def test_single_known_type_is_literal(self) -> None:
        input_text = "Character:Alice | TYPE | Character"
        parsed = parse_llm_triples(input_text)
        assert len(parsed) == 1
        assert parsed[0]["is_literal_object"]
        assert parsed[0]["object_literal"] == "Character"
