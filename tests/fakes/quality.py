"""Explicit completed evidence for storage-only synthetic fixtures, not real evaluation."""
from typing import Any, cast

from core.langgraph.quality_policy import SCORE_FIELDS, CheckEvidence, configured_policy, quality_source, record_check
from core.langgraph.state import NarrativeState


def example_quality_state(state: NarrativeState) -> NarrativeState:
    scores = {name: state.get(name) if state.get(name) is not None else 0.9 for name in SCORE_FIELDS}
    result = cast(NarrativeState, {**state, "quality_policy": state.get("quality_policy") or configured_policy(), "quality_checks": {}, **scores})
    result["quality_checks"] = record_check(result, "consistency", "completed")
    result["quality_checks"] = record_check(result, "evaluation", "completed", details={"scores": scores})
    result["quality_checks"] = record_check(result, "contradictions", "completed", details={"findings": [item.model_dump(mode="json") for item in result.get("contradictions", [])]})
    result["graph_quality_check"] = CheckEvidence(source=quality_source(result), status="completed", reason="", details={"issues_found": 0}).model_dump()
    return result


async def finalize_example(state: NarrativeState | dict[str, Any]) -> NarrativeState:
    """Exercise the production finalizer with synthetic storage-fixture evidence."""
    from core.langgraph.nodes.finalize_node import finalize_chapter

    evidence = example_quality_state(cast(NarrativeState, state))
    evidence["quality_policy"] = {**evidence["quality_policy"], "graph_quality_enabled": False}
    return await finalize_chapter(evidence)
