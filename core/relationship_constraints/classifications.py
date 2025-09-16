# core/relationship_constraints/classifications.py
"""Node classifications for relationship constraint system.

This module now imports from the unified enhanced_node_taxonomy system
and provides backward compatibility for existing constraint plugins.
"""

from ..enhanced_node_taxonomy import NodeClassification


class NodeClassifications:
    """Enhanced classification of node types into semantic categories.

    This class now serves as a compatibility layer that imports all
    classifications from the unified enhanced_node_taxonomy system.
    """

    # Import all classifications from the unified system
    SENTIENT = NodeClassification.SENTIENT
    CONSCIOUS = NodeClassification.CONSCIOUS
    PHYSICAL_PRESENCE = NodeClassification.PHYSICAL_PRESENCE
    LOCATABLE = NodeClassification.LOCATABLE
    OWNABLE = NodeClassification.OWNABLE
    SOCIAL = NodeClassification.SOCIAL
    CONTAINERS = NodeClassification.CONTAINERS
    TEMPORAL = NodeClassification.TEMPORAL
    ABSTRACT = NodeClassification.ABSTRACT
    ORGANIZATIONAL = NodeClassification.ORGANIZATIONAL
    SYSTEM_ENTITIES = NodeClassification.SYSTEM_ENTITIES
    INFORMATIONAL = NodeClassification.INFORMATIONAL

    # Legacy classifications for backward compatibility
    INANIMATE = NodeClassification.INANIMATE
    SPATIAL = NodeClassification.SPATIAL

    # Additional compatibility classifications that some plugins might still reference
    # These handle any legacy "WorldElement" references by filtering them out
    @classmethod
    def _filter_legacy_types(cls, classification_set):
        """Filter out legacy WorldElement references from classification sets."""
        return {item for item in classification_set if item != "WorldElement"}

    # Override sets to filter legacy types if needed
    @property
    def PHYSICAL_PRESENCE_FILTERED(self):
        """PHYSICAL_PRESENCE without legacy WorldElement references."""
        return self._filter_legacy_types(self.PHYSICAL_PRESENCE)

    @property
    def LOCATABLE_FILTERED(self):
        """LOCATABLE without legacy WorldElement references."""
        return self._filter_legacy_types(self.LOCATABLE)

    @property
    def OWNABLE_FILTERED(self):
        """OWNABLE without legacy WorldElement references."""
        return self._filter_legacy_types(self.OWNABLE)


# For direct module-level access (some plugins might import directly)
SENTIENT = NodeClassifications.SENTIENT
CONSCIOUS = NodeClassifications.CONSCIOUS
PHYSICAL_PRESENCE = NodeClassifications.PHYSICAL_PRESENCE
LOCATABLE = NodeClassifications.LOCATABLE
OWNABLE = NodeClassifications.OWNABLE
SOCIAL = NodeClassifications.SOCIAL
CONTAINERS = NodeClassifications.CONTAINERS
TEMPORAL = NodeClassifications.TEMPORAL
ABSTRACT = NodeClassifications.ABSTRACT
ORGANIZATIONAL = NodeClassifications.ORGANIZATIONAL
SYSTEM_ENTITIES = NodeClassifications.SYSTEM_ENTITIES
INFORMATIONAL = NodeClassifications.INFORMATIONAL
INANIMATE = NodeClassifications.INANIMATE
SPATIAL = NodeClassifications.SPATIAL
