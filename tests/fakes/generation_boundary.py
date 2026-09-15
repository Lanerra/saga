"""Selected-identity profile responses plus real owner-check test driver."""
import json
from typing import Any

import config
from core.langgraph.initialization.catalog import select_catalog
from core.langgraph.state import NarrativeState
from tests.fakes.fake_neo4j_manager import FakeNeo4jManager
from tests.test_langgraph.test_chapter_lifecycle import DriverExample, Rows, TransactionExample

PROFILE_QUERY = """
MATCH (c:Character {name: $name})
// Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
OPTIONAL MATCH (c)-[r]->(target)
WITH c, collect(DISTINCT CASE
WHEN coalesce(r.source_profile_managed, false) = true
AND ($include_provisional = TRUE OR coalesce(r.is_provisional, FALSE) = FALSE)
THEN {target_name: target.name, rel_type: type(r), rel_props: properties(r)}
END) AS relationships_raw
RETURN c, coalesce(c.traits, []) AS traits,
[rel IN relationships_raw WHERE rel IS NOT NULL] AS relationships
"""
SNIPPET_QUERY = """
MATCH (c:Character {name: $char_name_param})
// Do NOT add a WHERE clause after OPTIONAL MATCH; it will null-drop the row.
OPTIONAL MATCH (c)-[r]-()
WITH c, count(DISTINCT CASE
WHEN coalesce(r.is_provisional, FALSE) = TRUE
AND coalesce(r.chapter_added, -1) <= $chapter_limit_param THEN r END) AS provisional_rel_count
RETURN c.personality_description AS description, c.status AS current_status,
c.is_provisional AS char_is_provisional, provisional_rel_count
"""
LOCATION_QUERY = """
MATCH (s)-[r:`LOCATED_AT`]->(o)
WHERE s.name = $subject_param AND coalesce(r.chapter_added, -1) <= $chapter_limit_param
AND (r.is_provisional = FALSE OR r.is_provisional IS NULL)
RETURN CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
coalesce(r.chapter_added, -1) AS chapter_added, coalesce(r.confidence, 0.0) AS confidence,
coalesce(r.is_provisional, FALSE) AS is_provisional
ORDER BY coalesce(r.chapter_added, -1) DESC, coalesce(r.confidence, 0.0) DESC LIMIT 1
"""
RELATIONSHIPS_QUERY = """
MATCH (s)-[r]->(o)
WHERE s.name = $subject_param AND coalesce(r.chapter_added, -1) <= $chapter_limit_param
AND coalesce(r.is_provisional, FALSE) = FALSE
RETURN s.name AS subject, type(r) AS predicate,
CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
CASE WHEN o:ValueNode THEN 'Literal' ELSE labels(o)[0] END AS object_type,
coalesce(r.chapter_added, -1) AS chapter_added, coalesce(r.confidence, 0.0) AS confidence,
coalesce(r.is_provisional, FALSE) AS is_provisional
ORDER BY coalesce(r.chapter_added, -1) DESC, coalesce(r.confidence, 0.0) DESC
"""


def tokens(query: str) -> str:
    return "".join(query.split())


class GenerationDatabase(FakeNeo4jManager):
    def __init__(self) -> None:
        super().__init__()
        self.driver = GenerationDriver("11111111-1111-4111-8111-111111111111")
        self.profiles: dict[str, dict[str, Any]] = {}
        self.novel: dict[str, Any] = {}

    def select(self, state: NarrativeState) -> None:
        self.driver.project_id = state["graph_project_id"]
        self.novel = {"theme": state["theme"], "central_conflict": None}
        self.profiles = {
            payload["name"]: payload
            for entity in select_catalog(state).entities
            if entity.label == "Character"
            for payload in [json.loads(entity.payload)]
        }

    async def execute_read_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        for key, value in self.novel.items():
            if tokens(query) == tokens(f"MATCH (ni:NovelInfo {{id: $novel_id_param}}) RETURN ni.{key} AS value"):
                assert parameters == {"novel_id_param": config.MAIN_NOVEL_INFO_NODE_ID}
                self.executed_queries.append((query, parameters))
                return [{"value": value}]
        if tokens(query) == tokens(SNIPPET_QUERY):
            assert parameters is not None and set(parameters) == {"char_name_param", "chapter_limit_param"}
            self.executed_queries.append((query, parameters))
            profile = self.profiles[parameters["char_name_param"]]
            assert parameters["chapter_limit_param"] == 0 and not profile["relationships"]
            return [{"description": profile["personality_description"], "current_status": profile.get("status"), "char_is_provisional": False, "provisional_rel_count": 0}]
        if tokens(query) in {tokens(LOCATION_QUERY), tokens(RELATIONSHIPS_QUERY)}:
            assert parameters is not None and set(parameters) == {"subject_param", "chapter_limit_param"}
            assert parameters["chapter_limit_param"] == 0
            assert not self.profiles[parameters["subject_param"]]["relationships"]
            self.executed_queries.append((query, parameters))
            return []  # Explicitly selected fixture has no relationship assertions.
        if tokens(query) == tokens(PROFILE_QUERY):
            self.executed_queries.append((query, parameters))
            assert parameters is not None and set(parameters) == {"name", "include_provisional"}
            assert type(parameters["include_provisional"]) is bool
            selected_profile = self.profiles.get(parameters["name"])
            if selected_profile is None:
                return []
            assert not selected_profile["relationships"], "Fixture must project declared relationships before use"
            return [{"c": selected_profile.copy(), "traits": list(selected_profile["traits"]), "relationships": []}]
        return await super().execute_read_query(query, parameters)


class GenerationTransaction(TransactionExample):
    def run(self, query: str, parameters: Any = None, **keywords: Any) -> Rows:
        if query == "RETURN 1 AS ownership_verified":
            assert not parameters and not keywords
            return Rows([{"ownership_verified": 1}])
        return super().run(query, parameters, **keywords)


class GenerationDriver(DriverExample):
    def execute_read(self, callback: Any, *arguments: Any) -> Any:
        return callback(GenerationTransaction(self), *arguments)
