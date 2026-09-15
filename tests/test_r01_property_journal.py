"""Graph timestamps must survive durable compensation without string coercion."""
from typing import Any

import pytest
from neo4j._codec.hydration.v2.temporal import hydrate_datetime
from neo4j.time import Date, DateTime, Duration, Time

from data_access.cypher_builders.graph_compensation import apply_compensation, capture_graph, encode_compensation
from tests.test_r01_graph_compensation import Graph


@pytest.mark.parametrize("value", [DateTime.from_iso_format("2026-09-13T04:00:00.123456789+00:00"), hydrate_datetime(0, 123456789, "America/Los_Angeles"), Date(2026, 9, 13), Time.from_iso_format("04:00:00.123456789"), Duration(months=1, days=2, seconds=3, nanoseconds=4), b"synthetic-bytes"])
def test_native_graph_temporal_properties_roundtrip(value: Any) -> None:
    graph = Graph()
    graph.nodes["character"]["properties"]["enriched_at"] = value
    before = capture_graph(graph)
    graph.nodes["character"]["properties"]["description"] = "New provisional description"
    journal = encode_compensation(before, capture_graph(graph))
    apply_compensation(graph, journal)
    assert capture_graph(graph) == before
    assert type(graph.nodes["character"]["properties"]["enriched_at"]) is type(value)
    if isinstance(value, DateTime):
        from neo4j._codec.hydration.v2.temporal import dehydrate_datetime

        assert dehydrate_datetime(graph.nodes["character"]["properties"]["enriched_at"]) == dehydrate_datetime(value)
