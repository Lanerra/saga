# tests/test_parsing_utils.py
import logging
import sys
import unittest

from processing.parsing_utils import (
    _get_entity_type_and_name_from_text,
    _is_proper_noun,
    _should_filter_entity,
    parse_llm_triples,
)


class TestRdfTripleParsing(unittest.TestCase):
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

        expected_triples_count = 9

        parsed_triples = parse_llm_triples(triple_input)

        self.assertEqual(len(parsed_triples), expected_triples_count)

        jax_alias_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Jax" and t["predicate"] == "HASALIAS"
            ),
            None,
        )
        self.assertIsNotNone(jax_alias_triple, "Jax hasAlias triple not found")
        if jax_alias_triple:
            self.assertEqual(jax_alias_triple["object_literal"], "J.X.")
            self.assertTrue(jax_alias_triple["is_literal_object"])

        jax_livesin_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Jax" and t["predicate"] == "LIVESIN"
            ),
            None,
        )
        self.assertIsNotNone(jax_livesin_triple, "Jax livesIn triple not found")
        if jax_livesin_triple:
            self.assertTrue(jax_livesin_triple["is_literal_object"])
            self.assertEqual(jax_livesin_triple["object_literal"], "Hourglass Curios")

        jax_type_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Jax" and t["predicate"] == "TYPE"
            ),
            None,
        )
        self.assertIsNotNone(jax_type_triple, "Jax rdf:type Character triple not found")
        if jax_type_triple:
            self.assertEqual(jax_type_triple["object_literal"], "Character")

        lila_type_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Lila" and t["predicate"] == "TYPE"
            ),
            None,
        )
        self.assertIsNotNone(
            lila_type_triple, "Lila rdf:type Character triple not found"
        )
        if lila_type_triple:
            self.assertEqual(lila_type_triple["object_literal"], "Character")

    def test_empty_input(self) -> None:
        parsed_triples = parse_llm_triples("")
        self.assertEqual(len(parsed_triples), 0)

    def test_invalid_turtle(self) -> None:
        invalid_turtle = r"char:Jax prop:hasAlias 'J.X.'"
        parsed_triples = parse_llm_triples(invalid_turtle)
        self.assertEqual(len(parsed_triples), 0)


class TestProperNounDetection(unittest.TestCase):
    """Test proper noun detection for entity filtering."""

    def test_clear_proper_nouns(self) -> None:
        """Test obviously proper noun cases."""
        self.assertTrue(_is_proper_noun("Alice"))
        self.assertTrue(_is_proper_noun("Elara Moonwhisper"))
        self.assertTrue(_is_proper_noun("Sunken Library"))
        self.assertTrue(_is_proper_noun("Order of the Flame"))
        self.assertTrue(_is_proper_noun("Starfall Map"))

    def test_clear_common_nouns(self) -> None:
        """Test obviously common noun cases."""
        self.assertFalse(_is_proper_noun("the girl"))
        self.assertFalse(_is_proper_noun("the artifact"))
        self.assertFalse(_is_proper_noun("the sword"))
        self.assertFalse(_is_proper_noun("the rebellion"))
        self.assertFalse(_is_proper_noun("a mysterious stranger"))

    def test_mixed_case_titles(self) -> None:
        """Test titles with articles and prepositions."""
        self.assertTrue(_is_proper_noun("Council of Elders"))
        self.assertTrue(_is_proper_noun("Library of Alexandria"))
        self.assertTrue(_is_proper_noun("Sword of Truth"))
        # "the" at start with only 1-2 words is filtered
        self.assertFalse(_is_proper_noun("the Council"))

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        self.assertFalse(_is_proper_noun(""))
        self.assertFalse(_is_proper_noun("   "))
        self.assertFalse(_is_proper_noun("the"))

    def test_all_lowercase(self) -> None:
        """Test all lowercase strings."""
        self.assertFalse(_is_proper_noun("ancient artifact"))
        self.assertFalse(_is_proper_noun("old hermit"))

    def test_mixed_capitalization(self) -> None:
        """Test partially capitalized strings."""
        # 60% threshold: "Alice in Wonderland" has 2/3 significant words capitalized
        self.assertTrue(_is_proper_noun("Alice in Wonderland"))
        # "the Old Man" has 2/2 significant words capitalized
        self.assertTrue(_is_proper_noun("Old Man of the Mountain"))


class TestEntityFiltering(unittest.TestCase):
    """Test entity filtering with proper noun preference."""

    def test_proper_noun_single_mention(self) -> None:
        """Proper nouns should pass with just 1 mention."""
        self.assertFalse(
            _should_filter_entity("Alice", entity_type="Character", mention_count=1)
        )
        self.assertFalse(
            _should_filter_entity(
                "Sunken Library", entity_type="Location", mention_count=1
            )
        )

    def test_common_noun_single_mention(self) -> None:
        """Common nouns should be filtered with only 1 mention."""
        self.assertTrue(
            _should_filter_entity("the girl", entity_type="Character", mention_count=1)
        )
        self.assertTrue(
            _should_filter_entity("the artifact", entity_type="Object", mention_count=1)
        )

    def test_common_noun_three_mentions(self) -> None:
        """Common nouns should pass with 3+ mentions."""
        self.assertFalse(
            _should_filter_entity(
                "the rebellion", entity_type="Faction", mention_count=3
            )
        )
        self.assertFalse(
            _should_filter_entity(
                "the mysterious visitor", entity_type="Character", mention_count=5
            )
        )

    def test_common_noun_two_mentions(self) -> None:
        """Common nouns should still be filtered with only 2 mentions."""
        self.assertTrue(
            _should_filter_entity(
                "the rebellion", entity_type="Faction", mention_count=2
            )
        )

    def test_blacklisted_entities(self) -> None:
        """Blacklisted patterns should always be filtered."""
        # Even with high mention count and as "proper noun"
        self.assertTrue(
            _should_filter_entity(
                "violet glow", entity_type="Concept", mention_count=10
            )
        )
        self.assertTrue(
            _should_filter_entity("fear", entity_type="Emotion", mention_count=5)
        )
        self.assertTrue(
            _should_filter_entity("the moment", entity_type="Moment", mention_count=3)
        )

    def test_very_short_names(self) -> None:
        """Very short names should be filtered."""
        self.assertTrue(_should_filter_entity("he", mention_count=10))
        self.assertTrue(_should_filter_entity("it", mention_count=10))
        self.assertTrue(_should_filter_entity("a", mention_count=10))

    def test_descriptive_adjectives(self) -> None:
        """Pure adjectives should be filtered."""
        self.assertTrue(_should_filter_entity("beautiful", mention_count=5))
        self.assertTrue(_should_filter_entity("dark", mention_count=3))

    def test_empty_names(self) -> None:
        """Empty or None names should be filtered."""
        self.assertTrue(_should_filter_entity(""))
        self.assertTrue(_should_filter_entity(None))
        self.assertTrue(_should_filter_entity("   "))


class TestProperNounPreferenceInTriples(unittest.TestCase):
    """Test proper noun preference in actual triple parsing."""

    def test_proper_noun_entities_extracted(self) -> None:
        """Proper noun entities should be extracted from triples."""
        triple_input = """
        Character:Alice | LOVES | Character:Bob
        Character:Alice | LOCATED_IN | Location:Wonderland
        """
        parsed = parse_llm_triples(triple_input)

        # Should have 2 triples
        self.assertEqual(len(parsed), 2)

        # Check Alice -> Bob triple
        alice_loves = next(
            (
                t
                for t in parsed
                if t["subject"]["name"] == "Alice" and t["predicate"] == "LOVES"
            ),
            None,
        )
        # mypy check
        assert alice_loves is not None
        self.assertIsNotNone(alice_loves)
        self.assertEqual(alice_loves["object_entity"]["name"], "Bob")

    def test_blacklisted_entities_filtered_from_triples(self) -> None:
        """Blacklisted entities should be filtered even in triples."""
        triple_input = """
        Character:Alice | FEELS | Emotion:fear
        Location:Library | EMITS | violet glow
        Character:Alice | HAS_STATUS | healthy
        """
        parsed = parse_llm_triples(triple_input)

        # First triple should be filtered (fear is blacklisted)
        # Second triple should be filtered (violet glow is blacklisted)
        # Third triple should pass (healthy is a literal, Alice is proper noun)
        alice_triples = [t for t in parsed if t["subject"]["name"] == "Alice"]

        # Only the HAS_STATUS triple should remain
        self.assertEqual(len(alice_triples), 1)
        self.assertEqual(alice_triples[0]["predicate"], "HAS_STATUS")


class TestEntityParsingHeuristics(unittest.TestCase):
    def test_standard_colon_format(self) -> None:
        """Test 'Type: Name' format."""
        result = _get_entity_type_and_name_from_text("Character: Alice")
        self.assertEqual(result["type"], "Character")
        self.assertEqual(result["name"], "Alice")

    def test_missing_colon_known_type(self) -> None:
        """Test 'Type Name' format where Type is a known entity type."""
        result = _get_entity_type_and_name_from_text("Character Alice")
        self.assertEqual(result["type"], "Character")
        self.assertEqual(result["name"], "Alice")

    def test_missing_colon_unknown_type(self) -> None:
        """Test 'Word Name' where Word is NOT a known type."""
        result = _get_entity_type_and_name_from_text("Alice Bob")
        self.assertIsNone(result["type"])
        self.assertEqual(result["name"], "Alice Bob")

    def test_colon_with_empty_name(self) -> None:
        """Test 'Type: ' format."""
        result = _get_entity_type_and_name_from_text("Location: ")
        self.assertEqual(result["type"], "Location")
        self.assertIsNone(result["name"])

    def test_just_type_name(self) -> None:
        """Test 'Type' single word where it is a known entity type."""
        result = _get_entity_type_and_name_from_text("Character")
        self.assertEqual(result["type"], "Character")
        self.assertIsNone(result["name"])

    def test_whitespace_handling(self) -> None:
        """Test robust whitespace handling."""
        result = _get_entity_type_and_name_from_text("  Location  :  The Void  ")
        self.assertEqual(result["type"], "Location")
        self.assertEqual(result["name"], "The Void")

    def test_case_insensitive_type_match(self) -> None:
        """Test that we detect types even if lowercase in 'Type Name' format."""
        result = _get_entity_type_and_name_from_text("character Bob")
        # mypy check
        assert result["type"] is not None
        self.assertIsNotNone(result["type"])
        self.assertEqual(result["type"].lower(), "character")
        self.assertEqual(result["name"], "Bob")


class TestObjectDetection(unittest.TestCase):
    """Test object entity vs literal detection in parse_llm_triples."""

    def test_standard_colon_entity(self) -> None:
        """Test 'Type: Name' format in object position."""
        input_text = "Character:Alice | LOVES | Character:Bob"
        parsed = parse_llm_triples(input_text)
        self.assertEqual(len(parsed), 1)
        self.assertFalse(parsed[0]["is_literal_object"])
        self.assertIsNotNone(parsed[0]["object_entity"])
        self.assertEqual(parsed[0]["object_entity"]["type"], "Character")
        self.assertEqual(parsed[0]["object_entity"]["name"], "Bob")

    def test_missing_colon_known_type_entity(self) -> None:
        """Test 'Type Name' format in object position with known type."""
        # Character is a known type
        input_text = "Character:Alice | LOVES | Character Bob"
        parsed = parse_llm_triples(input_text)
        self.assertEqual(len(parsed), 1)
        self.assertFalse(parsed[0]["is_literal_object"])
        self.assertIsNotNone(parsed[0]["object_entity"])
        self.assertEqual(parsed[0]["object_entity"]["type"], "Character")
        self.assertEqual(parsed[0]["object_entity"]["name"], "Bob")

    def test_literal_string(self) -> None:
        """Test literal string that shouldn't be parsed as entity."""
        input_text = "Character:Alice | HAS_STATUS | Happy and healthy"
        parsed = parse_llm_triples(input_text)
        self.assertEqual(len(parsed), 1)
        self.assertTrue(parsed[0]["is_literal_object"])
        self.assertEqual(parsed[0]["object_literal"], "Happy and healthy")

    def test_unknown_type_prefix_is_literal(self) -> None:
        """Test 'UnknownType Name' should be treated as literal if type is unknown."""
        # 'Strange' is not a known node label
        input_text = "Character:Alice | HAS_TRAIT | Strange Behavior"
        parsed = parse_llm_triples(input_text)
        self.assertEqual(len(parsed), 1)
        self.assertTrue(parsed[0]["is_literal_object"])
        self.assertEqual(parsed[0]["object_literal"], "Strange Behavior")

    def test_single_word_name_is_literal(self) -> None:
        """Test single word in object position defaults to literal."""
        # "Bob" alone should be a literal unless specified as Character:Bob
        input_text = "Character:Alice | KNOWS | Bob"
        parsed = parse_llm_triples(input_text)
        self.assertEqual(len(parsed), 1)
        self.assertTrue(parsed[0]["is_literal_object"])
        self.assertEqual(parsed[0]["object_literal"], "Bob")

    def test_single_known_type_is_literal(self) -> None:
        """Test single known type word in object position is literal."""
        # "Character" alone in object position is ambiguous but likely a value/literal
        # e.g. "Alice | TYPE | Character" -> Character is a literal value for TYPE
        input_text = "Character:Alice | TYPE | Character"
        parsed = parse_llm_triples(input_text)
        self.assertEqual(len(parsed), 1)
        self.assertTrue(parsed[0]["is_literal_object"])
        self.assertEqual(parsed[0]["object_literal"], "Character")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
