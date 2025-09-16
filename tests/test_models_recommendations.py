import types

import pytest

from models.kg_constants import RELATIONSHIP_NORMALIZATIONS
from models.db_extraction_utils import Neo4jExtractor
from models.user_input_models import (
    CharacterGroupModel,
    PlotElementsModel,
    ProtagonistModel,
    SettingModel,
)
from models.kg_models import CharacterProfile, WorldItem


def test_relationship_normalizations_conflicts_resolved():
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
def test_safe_int_extract_tolerant(value, expected):
    assert Neo4jExtractor.safe_int_extract(value) == expected


def test_pydantic_default_factories_not_shared():
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
    s1.key_locations.append(type("KL", (), {"name": "X"})())
    assert s2.key_locations == []

    pe1 = PlotElementsModel()
    pe2 = PlotElementsModel()
    pe1.plot_points.append("pp")
    assert pe2.plot_points == []


def test_worlditem_from_db_node_uses_dict_access():
    # Fake neo4j.Node-like object supporting dict() but not .get
    class FakeNode(dict):
        def get(self, *args, **kwargs):
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
    item = WorldItem.from_db_node(node)
    assert item.id == "id1" and item.additional_properties.get("extra") == "x"


def test_characterprofile_from_db_record_robust_record_access():
    # Fake record with .get semantics
    class FakeRecord(dict):
        def __getitem__(self, key):
            return super().__getitem__(key)

        def get(self, key, default=None):
            return super().get(key, default)

    class FakeNode(dict):
        pass

    rec = FakeRecord(
        c=FakeNode(name="Alice", traits=["brave"], status="Unknown"),
        relationships=[{"target_name": "Bob", "type": "FRIEND_OF", "description": "pals"}],
    )
    cp = CharacterProfile.from_db_record(rec)
    assert cp.name == "Alice" and cp.relationships.get("Bob", {}).get("type") == "FRIEND_OF"

