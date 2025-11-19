# tests/test_parsing_utils.py
import logging
import sys
import unittest

from processing.parsing_utils import (
    _is_proper_noun,
    _should_filter_entity,
    parse_llm_triples,
)


class TestRdfTripleParsing(unittest.TestCase):
    def test_parse_simple_turtle(self):
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

    def test_empty_input(self):
        parsed_triples = parse_llm_triples("")
        self.assertEqual(len(parsed_triples), 0)

    def test_invalid_turtle(self):
        invalid_turtle = r"char:Jax prop:hasAlias 'J.X.'"
        parsed_triples = parse_llm_triples(invalid_turtle)
        self.assertEqual(len(parsed_triples), 0)


class TestProperNounDetection(unittest.TestCase):
    """Test proper noun detection for entity filtering."""

    def test_clear_proper_nouns(self):
        """Test obviously proper noun cases."""
        self.assertTrue(_is_proper_noun("Alice"))
        self.assertTrue(_is_proper_noun("Elara Moonwhisper"))
        self.assertTrue(_is_proper_noun("Sunken Library"))
        self.assertTrue(_is_proper_noun("Order of the Flame"))
        self.assertTrue(_is_proper_noun("Starfall Map"))

    def test_clear_common_nouns(self):
        """Test obviously common noun cases."""
        self.assertFalse(_is_proper_noun("the girl"))
        self.assertFalse(_is_proper_noun("the artifact"))
        self.assertFalse(_is_proper_noun("the sword"))
        self.assertFalse(_is_proper_noun("the rebellion"))
        self.assertFalse(_is_proper_noun("a mysterious stranger"))

    def test_mixed_case_titles(self):
        """Test titles with articles and prepositions."""
        self.assertTrue(_is_proper_noun("Council of Elders"))
        self.assertTrue(_is_proper_noun("Library of Alexandria"))
        self.assertTrue(_is_proper_noun("Sword of Truth"))
        # "the" at start with only 1-2 words is filtered
        self.assertFalse(_is_proper_noun("the Council"))

    def test_edge_cases(self):
        """Test edge cases."""
        self.assertFalse(_is_proper_noun(""))
        self.assertFalse(_is_proper_noun("   "))
        self.assertFalse(_is_proper_noun("the"))

    def test_all_lowercase(self):
        """Test all lowercase strings."""
        self.assertFalse(_is_proper_noun("ancient artifact"))
        self.assertFalse(_is_proper_noun("old hermit"))

    def test_mixed_capitalization(self):
        """Test partially capitalized strings."""
        # 60% threshold: "Alice in Wonderland" has 2/3 significant words capitalized
        self.assertTrue(_is_proper_noun("Alice in Wonderland"))
        # "the Old Man" has 2/2 significant words capitalized
        self.assertTrue(_is_proper_noun("Old Man of the Mountain"))


class TestEntityFiltering(unittest.TestCase):
    """Test entity filtering with proper noun preference."""

    def test_proper_noun_single_mention(self):
        """Proper nouns should pass with just 1 mention."""
        self.assertFalse(
            _should_filter_entity("Alice", entity_type="Character", mention_count=1)
        )
        self.assertFalse(
            _should_filter_entity(
                "Sunken Library", entity_type="Location", mention_count=1
            )
        )

    def test_common_noun_single_mention(self):
        """Common nouns should be filtered with only 1 mention."""
        self.assertTrue(
            _should_filter_entity(
                "the girl", entity_type="Character", mention_count=1
            )
        )
        self.assertTrue(
            _should_filter_entity(
                "the artifact", entity_type="Object", mention_count=1
            )
        )

    def test_common_noun_three_mentions(self):
        """Common nouns should pass with 3+ mentions."""
        self.assertFalse(
            _should_filter_entity(
                "the rebellion", entity_type="Faction", mention_count=3
            )
        )
        self.assertFalse(
            _should_filter_entity(
                "the mysterious stranger", entity_type="Character", mention_count=5
            )
        )

    def test_common_noun_two_mentions(self):
        """Common nouns should still be filtered with only 2 mentions."""
        self.assertTrue(
            _should_filter_entity(
                "the rebellion", entity_type="Faction", mention_count=2
            )
        )

    def test_blacklisted_entities(self):
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

    def test_very_short_names(self):
        """Very short names should be filtered."""
        self.assertTrue(_should_filter_entity("he", mention_count=10))
        self.assertTrue(_should_filter_entity("it", mention_count=10))
        self.assertTrue(_should_filter_entity("a", mention_count=10))

    def test_descriptive_adjectives(self):
        """Pure adjectives should be filtered."""
        self.assertTrue(_should_filter_entity("beautiful", mention_count=5))
        self.assertTrue(_should_filter_entity("dark", mention_count=3))

    def test_empty_names(self):
        """Empty or None names should be filtered."""
        self.assertTrue(_should_filter_entity(""))
        self.assertTrue(_should_filter_entity(None))
        self.assertTrue(_should_filter_entity("   "))


class TestProperNounPreferenceInTriples(unittest.TestCase):
    """Test proper noun preference in actual triple parsing."""

    def test_proper_noun_entities_extracted(self):
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
        self.assertIsNotNone(alice_loves)
        self.assertEqual(alice_loves["object_entity"]["name"], "Bob")

    def test_blacklisted_entities_filtered_from_triples(self):
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
        alice_triples = [
            t for t in parsed if t["subject"]["name"] == "Alice"
        ]

        # Only the HAS_STATUS triple should remain
        self.assertEqual(len(alice_triples), 1)
        self.assertEqual(alice_triples[0]["predicate"], "HAS_STATUS")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
