def test_context_snapshot_determinism():
    from models.narrative_state import ContextSnapshot

    recent_map = {1: {"summary": "S1", "text": "T1", "is_provisional": False}}
    snap1 = ContextSnapshot(
        chapter_number=2,
        plot_point_focus="PP2",
        chapter_plan=[{"scene_number": 1, "summary": "A"}],
        hybrid_context="HYBRID_STABLE",
        kg_facts_block="KG_STABLE",
        recent_chapters_map=recent_map,
    )
    snap2 = ContextSnapshot(
        chapter_number=2,
        plot_point_focus="PP2",
        chapter_plan=[{"scene_number": 1, "summary": "A"}],
        hybrid_context="HYBRID_STABLE",
        kg_facts_block="KG_STABLE",
        recent_chapters_map=recent_map,
    )

    assert snap1.snapshot_fingerprint == snap2.snapshot_fingerprint
    assert isinstance(snap1.created_ts, float)


def test_context_snapshot_changes_after_write():
    from models.narrative_state import ContextSnapshot

    recent_map = {1: {"summary": "S1", "text": "T1", "is_provisional": False}}
    before = ContextSnapshot(
        chapter_number=3,
        plot_point_focus="PP3",
        chapter_plan=[{"scene_number": 1, "summary": "B"}],
        hybrid_context="HYBRID_STABLE",
        kg_facts_block="KG_A",
        recent_chapters_map=recent_map,
    )
    after = ContextSnapshot(
        chapter_number=3,
        plot_point_focus="PP3",
        chapter_plan=[{"scene_number": 1, "summary": "B"}],
        hybrid_context="HYBRID_STABLE",
        kg_facts_block="KG_B",
        recent_chapters_map=recent_map,
    )
    assert before.snapshot_fingerprint != after.snapshot_fingerprint
