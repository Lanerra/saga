# tests/test_parsing_utils.py
import logging
import sys
import unittest

from parsing_utils import parse_rdf_triples_with_rdflib


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

        parsed_triples = parse_rdf_triples_with_rdflib(triple_input)

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
        parsed_triples = parse_rdf_triples_with_rdflib("", rdf_format="turtle")
        self.assertEqual(len(parsed_triples), 0)

    def test_invalid_turtle(self):
        invalid_turtle = r"char:Jax prop:hasAlias 'J.X.'"
        parsed_triples = parse_rdf_triples_with_rdflib(
            invalid_turtle, rdf_format="turtle"
        )
        self.assertEqual(len(parsed_triples), 0)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
