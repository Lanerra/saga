"""Consistency and integrity checks for models/kg_constants.py."""

from models.kg_constants import (
    CHARACTER_EMOTIONAL_RELATIONSHIPS,
    CHARACTER_EVENT_RELATIONSHIPS,
    CHARACTER_ITEM_RELATIONSHIPS,
    CHARACTER_LOCATION_RELATIONSHIPS,
    CHARACTER_SOCIAL_RELATIONSHIPS,
    CONTRADICTORY_TRAIT_PAIRS,
    EVENT_ITEM_RELATIONSHIPS,
    EVENT_LOCATION_RELATIONSHIPS,
    EVENT_TEMPORAL_RELATIONSHIPS,
    LABEL_NORMALIZATION_MAP,
    LOCATION_ITEM_RELATIONSHIPS,
    LOCATION_SPATIAL_RELATIONSHIPS,
    NOVEL_INFO_ALLOWED_PROPERTY_KEYS,
    RELATIONSHIP_CATEGORIES,
    RELATIONSHIP_TYPES,
    SCENE_CHAPTER_RELATIONSHIPS,
    SCENE_EVENT_RELATIONSHIPS,
    SCENE_LOCATION_RELATIONSHIPS,
    SCENE_SEQUENTIAL_RELATIONSHIPS,
    STATIC_RELATIONSHIP_MAP,
    SUGGESTED_CATEGORIES,
    VALID_NODE_LABELS,
    WORLD_ITEM_CANONICAL_LABELS,
    WORLD_ITEM_LEGACY_LABELS,
)

ALL_CATEGORY_SETS = [
    CHARACTER_SOCIAL_RELATIONSHIPS,
    CHARACTER_EMOTIONAL_RELATIONSHIPS,
    CHARACTER_EVENT_RELATIONSHIPS,
    CHARACTER_ITEM_RELATIONSHIPS,
    CHARACTER_LOCATION_RELATIONSHIPS,
    EVENT_LOCATION_RELATIONSHIPS,
    EVENT_ITEM_RELATIONSHIPS,
    EVENT_TEMPORAL_RELATIONSHIPS,
    LOCATION_SPATIAL_RELATIONSHIPS,
    LOCATION_ITEM_RELATIONSHIPS,
    SCENE_SEQUENTIAL_RELATIONSHIPS,
    SCENE_CHAPTER_RELATIONSHIPS,
    SCENE_EVENT_RELATIONSHIPS,
    SCENE_LOCATION_RELATIONSHIPS,
]


class TestValidNodeLabels:
    def test_exactly_six_labels(self):
        assert len(VALID_NODE_LABELS) == 6

    def test_expected_labels(self):
        assert VALID_NODE_LABELS == {
            "Character",
            "Location",
            "Event",
            "Item",
            "Chapter",
            "Scene",
        }


class TestLabelNormalizationMap:
    def test_all_targets_are_valid_node_labels(self):
        for source, target in LABEL_NORMALIZATION_MAP.items():
            assert target in VALID_NODE_LABELS, f"Normalization target '{target}' (from '{source}') " f"is not in VALID_NODE_LABELS"

    def test_no_canonical_label_maps_to_itself(self):
        for key in LABEL_NORMALIZATION_MAP:
            assert key not in VALID_NODE_LABELS, f"Canonical label '{key}' should not appear as a " f"normalization key"


class TestWorldItemLabels:
    def test_canonical_labels_are_valid_node_labels(self):
        for label in WORLD_ITEM_CANONICAL_LABELS:
            assert label in VALID_NODE_LABELS, f"Canonical world-item label '{label}' " f"is not in VALID_NODE_LABELS"

    def test_legacy_labels_are_in_normalization_map(self):
        for label in WORLD_ITEM_LEGACY_LABELS:
            assert label in LABEL_NORMALIZATION_MAP, f"Legacy label '{label}' is not in LABEL_NORMALIZATION_MAP"

    def test_legacy_labels_normalize_to_canonical(self):
        for label in WORLD_ITEM_LEGACY_LABELS:
            target = LABEL_NORMALIZATION_MAP[label]
            assert target in WORLD_ITEM_CANONICAL_LABELS, f"Legacy label '{label}' normalizes to '{target}', " f"which is not a canonical world-item label"


class TestSuggestedCategories:
    def test_keys_are_subset_of_valid_node_labels(self):
        assert set(SUGGESTED_CATEGORIES.keys()) <= VALID_NODE_LABELS

    def test_each_value_is_nonempty_list(self):
        for label, categories in SUGGESTED_CATEGORIES.items():
            assert isinstance(categories, list), f"SUGGESTED_CATEGORIES['{label}'] should be a list"
            assert len(categories) > 0, f"SUGGESTED_CATEGORIES['{label}'] should not be empty"


class TestContradictoryTraitPairs:
    def test_all_lowercase(self):
        for first, second in CONTRADICTORY_TRAIT_PAIRS:
            assert first == first.lower(), f"Trait '{first}' is not lowercase"
            assert second == second.lower(), f"Trait '{second}' is not lowercase"

    def test_no_self_contradictions(self):
        for first, second in CONTRADICTORY_TRAIT_PAIRS:
            assert first != second, f"Self-contradictory pair found: ('{first}', '{second}')"

    def test_all_pairs_are_two_element_tuples(self):
        for pair in CONTRADICTORY_TRAIT_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_pair_count(self):
        assert len(CONTRADICTORY_TRAIT_PAIRS) == 30


class TestRelationshipCategories:
    def test_all_category_values_exist_in_relationship_types(self):
        for category_name, category_set in RELATIONSHIP_CATEGORIES.items():
            for relationship_type in category_set:
                assert relationship_type in RELATIONSHIP_TYPES, f"'{relationship_type}' from category " f"'{category_name}' is not in RELATIONSHIP_TYPES"

    def test_relationship_types_is_exact_union(self):
        expected_union = set()
        for category_set in RELATIONSHIP_CATEGORIES.values():
            expected_union.update(category_set)
        assert RELATIONSHIP_TYPES == expected_union

    def test_fourteen_categories(self):
        assert len(RELATIONSHIP_CATEGORIES) == 14

    def test_expected_category_names(self):
        assert set(RELATIONSHIP_CATEGORIES.keys()) == {
            "character_social",
            "character_emotional",
            "character_event",
            "character_item",
            "character_location",
            "event_location",
            "event_item",
            "event_temporal",
            "location_spatial",
            "location_item",
            "scene_sequential",
            "scene_chapter",
            "scene_event",
            "scene_location",
        }


class TestRelationshipCategoryOverlaps:
    # Some relationship types intentionally appear in multiple categories.
    # This test documents the expected overlaps and ensures no unexpected ones.

    def test_document_expected_overlaps(self):
        seen: dict[str, list[str]] = {}
        for category_name, category_set in RELATIONSHIP_CATEGORIES.items():
            for relationship_type in category_set:
                seen.setdefault(relationship_type, []).append(category_name)

        overlapping = {relationship_type: categories for relationship_type, categories in seen.items() if len(categories) > 1}

        expected_overlapping_types = {
            "LOCATED_AT",
            "OCCURS_AT",
            "PART_OF",
        }

        assert set(overlapping.keys()) == expected_overlapping_types


class TestStaticRelationshipMap:
    def test_all_targets_exist_in_relationship_types(self):
        for synonym, canonical in STATIC_RELATIONSHIP_MAP.items():
            assert canonical in RELATIONSHIP_TYPES, f"STATIC_RELATIONSHIP_MAP target '{canonical}' " f"(from '{synonym}') is not in RELATIONSHIP_TYPES"

    def test_no_source_is_already_canonical(self):
        for synonym in STATIC_RELATIONSHIP_MAP:
            assert synonym not in RELATIONSHIP_TYPES, f"Synonym '{synonym}' is already in RELATIONSHIP_TYPES; " f"mapping is redundant"


class TestNovelInfoAllowedPropertyKeys:
    def test_is_frozenset(self):
        assert isinstance(NOVEL_INFO_ALLOWED_PROPERTY_KEYS, frozenset)

    def test_expected_keys(self):
        assert NOVEL_INFO_ALLOWED_PROPERTY_KEYS == frozenset(
            {
                "title",
                "genre",
                "setting",
                "theme",
                "central_conflict",
                "thematic_progression",
            }
        )

    def test_all_keys_are_lowercase_snake_case(self):
        for key in NOVEL_INFO_ALLOWED_PROPERTY_KEYS:
            assert key == key.lower(), f"Property key '{key}' is not lowercase"
            assert " " not in key, f"Property key '{key}' contains spaces"
