# tests/test_models_recommendations.py
from typing import Any

import pytest

from models.db_extraction_utils import Neo4jExtractor
from models.kg_constants import RELATIONSHIP_NORMALIZATIONS
from models.kg_models import CharacterProfile, WorldItem
from models.user_input_models import (
    CharacterGroupModel,
    PlotElementsModel,
    ProtagonistModel,
    SettingModel,
)


def test_relationship_normalizations_conflicts_resolved() -> None:
    assert RELATIONSHIP_NORMALIZATIONS.get("related_to") == "FAMILY_OF"
    assert RELATIONSHIP_NORMALIZATIONS.get("commands") == "LEADS"
    assert RELATIONSHIP_NORMALIZATIONS.get("manages") == "LEADS"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("123", 123),
        ("[123]", 123),
        ("123,456", 123),
        ("", 0),
        (None, 0),
        ("abc", 0),
        (["5"], 5),
    ],
)
def test_safe_int_extract_tolerant(value: Any, expected: int) -> None:
    assert Neo4jExtractor.safe_int_extract(value) == expected


def test_pydantic_default_factories_not_shared() -> None:
    a = ProtagonistModel(name="A")
    b = ProtagonistModel(name="B")
    a.traits.append("x")
    assert b.traits == []

    cg1 = CharacterGroupModel()
    cg2 = CharacterGroupModel()
    cg1.supporting_characters.append(ProtagonistModel(name="C"))
    assert cg2.supporting_characters == []

    s1 = SettingModel()
    s2 = SettingModel()
    # Mocking KeyLocationModel which isn't imported here but appended
    # Just skip append or use ignore if types mismatch,
    # but SettingModel key_locations expects KeyLocationModel.
    # We can try to rely on Pydantic or just ignore the append type check for this test.
    s1.key_locations.append(Any)  # type: ignore
    assert s2.key_locations == []

    pe1 = PlotElementsModel()
    pe2 = PlotElementsModel()
    pe1.plot_points.append("pp")
    assert pe2.plot_points == []


def test_worlditem_from_db_node_uses_dict_access() -> None:
    # Fake neo4j.Node-like object supporting dict() but not .get
    class FakeNode(dict):  # type: ignore
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError(".get should not be used on FakeNode in test")

    node = FakeNode(
        id="id1",
        category="cat",
        name="nm",
        description="desc",
        goals=["g"],
        rules=["r"],
        key_elements=["k"],
        traits=["t"],
        created_chapter=1,
        is_provisional=True,
        extra="x",
    )
    item = WorldItem.from_db_node(node)  # type: ignore
    assert item.id == "id1" and item.additional_properties.get("extra") == "x"


def test_characterprofile_from_db_record_robust_record_access() -> None:
    # Fake record with .get semantics
    class FakeRecord(dict):  # type: ignore
        def __getitem__(self, key: Any) -> Any:
            return super().__getitem__(key)

        def get(self, key: Any, default: Any = None) -> Any:
            return super().get(key, default)

    class FakeNode(dict):  # type: ignore
        pass

    rec = FakeRecord(
        c=FakeNode(name="Alice", traits=["brave"], status="Unknown"),
        relationships=[
            {"target_name": "Bob", "type": "FRIEND_OF", "description": "pals"}
        ],
    )
    cp = CharacterProfile.from_db_record(rec)  # type: ignore
    assert (
        cp.name == "Alice"
        and cp.relationships.get("Bob", {}).get("type") == "FRIEND_OF"
    )
