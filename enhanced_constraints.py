"""
Enhanced Relationship Constraints leveraging the expanded node taxonomy.

This module demonstrates how the enhanced node type system dramatically improves
relationship constraint precision and enables new semantic relationship types.
"""

from typing import Dict, Set, List, Any, Tuple
from enhanced_node_taxonomy import NodeClassification, ENHANCED_NODE_LABELS

# Enhanced Relationship Constraint Matrix
ENHANCED_RELATIONSHIP_CONSTRAINTS = {
    # === ENHANCED EMOTIONAL RELATIONSHIPS ===
    "LOVES": {
        "valid_subject_types": NodeClassification.CONSCIOUS,
        "valid_object_types": NodeClassification.CONSCIOUS | {"Location", "Object", "Artifact", "Concept"},
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Deep emotional affection - now supports loving places and objects",
        "examples_valid": [
            "Character:Hero | LOVES | Character:Princess",
            "Character:Explorer | LOVES | Location:Hometown", 
            "Person:King | LOVES | Artifact:Crown",
            "Character:Philosopher | LOVES | Concept:Truth"
        ]
    },
    
    "FEARS": {
        "valid_subject_types": NodeClassification.CONSCIOUS,
        "valid_object_types": ENHANCED_NODE_LABELS - {"Entity"},  # Can fear almost anything
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Fear response - much more nuanced with specific types",
        "examples_valid": [
            "Character:Child | FEARS | Creature:Dragon",
            "Person:Sailor | FEARS | Event:Storm",
            "Character:Thief | FEARS | Law:Punishment",
            "Character:Mage | FEARS | Magic:DarkArts"
        ]
    },
    
    "WORSHIPS": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Culture", "Religion"},
        "valid_object_types": {"Deity", "Spirit", "Concept", "Symbol", "Artifact"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Religious devotion with specific divine/sacred objects",
        "examples_valid": [
            "Character:Priest | WORSHIPS | Deity:Sun_God",
            "Culture:Mountain_Folk | WORSHIPS | Spirit:Earth_Spirit",
            "Character:Knight | WORSHIPS | Concept:Honor",
            "Religion:Sun_Cult | WORSHIPS | Symbol:Solar_Disc"
        ]
    },

    # === PHYSICAL INTERACTION RELATIONSHIPS ===
    "WIELDS": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Creature"},
        "valid_object_types": {"Object", "Artifact", "Item"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Active use of physical tools/weapons",
        "examples_valid": [
            "Character:Warrior | WIELDS | Object:Sword",
            "Character:Mage | WIELDS | Artifact:Staff_of_Power",
            "Creature:Dragon | WIELDS | Object:Ancient_Claw"
        ]
    },
    
    "CRAFTED_BY": {
        "valid_subject_types": {"Object", "Artifact", "Item", "Structure", "Document"},
        "valid_object_types": NodeClassification.CONSCIOUS | {"Organization", "Culture"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Creation relationship with specific crafted items",
        "examples_valid": [
            "Artifact:Excalibur | CRAFTED_BY | Person:Merlin",
            "Structure:Cathedral | CRAFTED_BY | Organization:Masons_Guild",
            "Document:Treaty | CRAFTED_BY | Council:Peace_Council"
        ]
    },
    
    "FORGED_FROM": {
        "valid_subject_types": {"Object", "Artifact", "Item"},
        "valid_object_types": {"Material", "Resource"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Material composition relationship",
        "examples_valid": [
            "Object:Sword | FORGED_FROM | Material:Steel",
            "Artifact:Ring | FORGED_FROM | Material:Mithril",
            "Item:Crown | FORGED_FROM | Resource:Gold"
        ]
    },

    # === KNOWLEDGE & COMMUNICATION RELATIONSHIPS ===
    "SPEAKS": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Culture", "Region"},
        "valid_object_types": {"Language"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Language communication ability",
        "examples_valid": [
            "Character:Ambassador | SPEAKS | Language:Elvish",
            "Culture:Northern_Tribes | SPEAKS | Language:Ancient_Runic",
            "Region:Eastern_Kingdoms | SPEAKS | Language:Common_Tongue"
        ]
    },
    
    "TEACHES": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Document", "Organization", "System"},
        "valid_object_types": {"Skill", "Knowledge", "Magic", "Technology", "Concept"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Knowledge transfer relationship",
        "examples_valid": [
            "Character:Master | TEACHES | Skill:Swordsmanship",
            "Document:Spellbook | TEACHES | Magic:Fire_Magic",
            "Organization:Academy | TEACHES | Knowledge:History",
            "System:Apprenticeship | TEACHES | Skill:Crafting"
        ]
    },
    
    "RECORDS": {
        "valid_subject_types": {"Document", "Record", "Library", "Archive"},
        "valid_object_types": {"Event", "Person", "Lore", "Knowledge", "Law"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Information preservation relationship",
        "examples_valid": [
            "Document:Chronicle | RECORDS | Event:Great_War",
            "Archive:Royal_Records | RECORDS | Person:Kings_Lineage",
            "Library:Sage_Collection | RECORDS | Lore:Ancient_Prophecies"
        ]
    },

    # === CULTURAL & SOCIAL RELATIONSHIPS ===
    "PRACTICES": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Culture", "Religion", "Organization"},
        "valid_object_types": {"Tradition", "Religion", "Skill", "Magic"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Cultural or religious practice",
        "examples_valid": [
            "Character:Monk | PRACTICES | Tradition:Meditation",
            "Culture:Desert_Nomads | PRACTICES | Tradition:Star_Reading",
            "Religion:Sun_Worship | PRACTICES | Tradition:Dawn_Prayers"
        ]
    },
    
    "CELEBRATES": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Culture", "Settlement"},
        "valid_object_types": {"Event", "Tradition", "Person", "Deity"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Ceremonial celebration relationship",
        "examples_valid": [
            "Culture:Harvest_Folk | CELEBRATES | Event:Autumn_Festival",
            "Settlement:Capital_City | CELEBRATES | Person:Founder_King",
            "Character:Priest | CELEBRATES | Deity:Harvest_Goddess"
        ]
    },
    
    "FOUNDED": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Organization"},
        "valid_object_types": {"Settlement", "Organization", "Religion", "Tradition"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Establishment relationship",
        "examples_valid": [
            "Person:Saint_Marcus | FOUNDED | Religion:Order_of_Light",
            "Character:Explorer | FOUNDED | Settlement:New_Haven",
            "Organization:Merchants | FOUNDED | Tradition:Trade_Festival"
        ]
    },

    # === HIERARCHICAL & ORGANIZATIONAL RELATIONSHIPS ===
    "HOLDS_RANK": {
        "valid_subject_types": NodeClassification.CONSCIOUS,
        "valid_object_types": {"Rank", "Role"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Hierarchical position relationship",
        "examples_valid": [
            "Character:General | HOLDS_RANK | Rank:Field_Marshal",
            "Character:Diplomat | HOLDS_RANK | Role:Ambassador",
            "Person:Noble | HOLDS_RANK | Role:Duke"
        ]
    },
    
    "GOVERNS": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Organization", "Government"},
        "valid_object_types": {"Territory", "Region", "Settlement", "Organization"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Political control relationship",
        "examples_valid": [
            "Character:King | GOVERNS | Territory:Royal_Lands",
            "Government:Council | GOVERNS | Settlement:Capital",
            "Organization:Guild | GOVERNS | Territory:Trade_District"
        ]
    },
    
    "SERVES_IN": {
        "valid_subject_types": NodeClassification.CONSCIOUS,
        "valid_object_types": {"Organization", "Government", "Order", "Guild"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Service within structured organizations",
        "examples_valid": [
            "Character:Knight | SERVES_IN | Order:Knights_Templar",
            "Character:Merchant | SERVES_IN | Guild:Traders_Alliance",
            "Character:Judge | SERVES_IN | Government:High_Court"
        ]
    },

    # === TEMPORAL RELATIONSHIPS ===
    "OCCURRED_DURING": {
        "valid_subject_types": {"Event", "DevelopmentEvent", "WorldElaborationEvent"},
        "valid_object_types": {"Era", "Season", "Timeline", "Event"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Temporal occurrence relationship",
        "examples_valid": [
            "Event:Great_Battle | OCCURRED_DURING | Era:Age_of_War",
            "DevelopmentEvent:Coming_of_Age | OCCURRED_DURING | Season:Spring",
            "Event:Coronation | OCCURRED_DURING | Event:Peace_Festival"
        ]
    },
    
    "PRECEDED_BY": {
        "valid_subject_types": {"Event", "Era", "Season"},
        "valid_object_types": {"Event", "Era", "Season"},
        "invalid_combinations": [
            # Prevent temporal paradoxes
        ],
        "bidirectional": False,
        "description": "Temporal sequence relationship",
        "examples_valid": [
            "Era:Golden_Age | PRECEDED_BY | Era:Dark_Times",
            "Event:Victory | PRECEDED_BY | Event:Great_Battle",
            "Season:Harvest | PRECEDED_BY | Season:Growing_Season"
        ]
    },
    
    "COMMEMORATES": {
        "valid_subject_types": {"Tradition", "Structure", "Settlement", "Event"},
        "valid_object_types": {"Event", "Person", "Era"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Memorial/commemorative relationship",
        "examples_valid": [
            "Tradition:Heroes_Day | COMMEMORATES | Event:Final_Victory",
            "Structure:Monument | COMMEMORATES | Person:Fallen_Hero",
            "Settlement:Memorial_City | COMMEMORATES | Era:War_Years"
        ]
    },

    # === ECONOMIC RELATIONSHIPS ===
    "TRADES_WITH": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Organization", "Settlement"},
        "valid_object_types": NodeClassification.CONSCIOUS | {"Organization", "Settlement"},
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Commercial relationship",
        "examples_valid": [
            "Character:Merchant | TRADES_WITH | Character:Farmer",
            "Settlement:Port_City | TRADES_WITH | Settlement:Mountain_Town",
            "Organization:Guild | TRADES_WITH | Organization:Foreign_Company"
        ]
    },
    
    "PRODUCES": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Settlement", "Region", "Organization"},
        "valid_object_types": {"Resource", "Material", "Food", "Object", "Currency"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Production relationship",
        "examples_valid": [
            "Settlement:Mining_Town | PRODUCES | Resource:Iron_Ore",
            "Region:Farmlands | PRODUCES | Food:Wheat",
            "Organization:Mint | PRODUCES | Currency:Gold_Coins"
        ]
    },
    
    "CONSUMES": {
        "valid_subject_types": NodeClassification.CONSCIOUS | {"Settlement", "Organization", "System"},
        "valid_object_types": {"Resource", "Material", "Food", "Energy"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Consumption relationship",
        "examples_valid": [
            "Character:Warrior | CONSUMES | Food:Rations",
            "Settlement:City | CONSUMES | Resource:Water",
            "System:Magic_Barrier | CONSUMES | Energy:Mana"
        ]
    },

    # === MAGICAL & SUPERNATURAL RELATIONSHIPS ===
    "ENCHANTED_BY": {
        "valid_subject_types": {"Object", "Artifact", "Location", "Creature"},
        "valid_object_types": {"Magic", "Deity", "Spirit", "Person"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Magical enhancement relationship",
        "examples_valid": [
            "Artifact:Sword | ENCHANTED_BY | Magic:Fire_Magic",
            "Location:Forest | ENCHANTED_BY | Spirit:Forest_Guardian",
            "Creature:Wolf | ENCHANTED_BY | Person:Druid"
        ]
    },
    
    "POWERED_BY": {
        "valid_subject_types": {"Magic", "Technology", "System", "Object", "Artifact"},
        "valid_object_types": {"Resource", "Energy", "Material", "Concept"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Power source relationship",
        "examples_valid": [
            "Magic:Teleportation | POWERED_BY | Energy:Ether",
            "Technology:Airship | POWERED_BY | Resource:Steam",
            "Artifact:Crystal_Ball | POWERED_BY | Concept:Future_Sight"
        ]
    },
    
    "CURSED_BY": {
        "valid_subject_types": {"Character", "Object", "Artifact", "Location"},
        "valid_object_types": {"Person", "Deity", "Spirit", "Magic"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Magical curse relationship",
        "examples_valid": [
            "Character:Prince | CURSED_BY | Person:Witch",
            "Artifact:Ring | CURSED_BY | Deity:Dark_God",
            "Location:Castle | CURSED_BY | Spirit:Vengeful_Ghost"
        ]
    },

    # === ENHANCED SPATIAL RELATIONSHIPS ===
    "BORDERS": {
        "valid_subject_types": {"Territory", "Region", "Settlement"},
        "valid_object_types": {"Territory", "Region", "Settlement"},
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Geographical boundary relationship",
        "examples_valid": [
            "Territory:Northern_Kingdom | BORDERS | Territory:Southern_Empire",
            "Region:Coastal_Plains | BORDERS | Region:Mountain_Range",
            "Settlement:Border_Town | BORDERS | Settlement:Foreign_City"
        ]
    },
    
    "OVERLOOKS": {
        "valid_subject_types": {"Location", "Structure", "Settlement"},
        "valid_object_types": {"Location", "Region", "Territory", "Path"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Visual/positional dominance relationship",
        "examples_valid": [
            "Structure:Watchtower | OVERLOOKS | Path:Mountain_Pass",
            "Settlement:Hill_City | OVERLOOKS | Region:Valley",
            "Location:Cliff | OVERLOOKS | Location:Harbor"
        ]
    },

    # === INFORMATION RELATIONSHIPS ===
    "REVEALS": {
        "valid_subject_types": {"Document", "Character", "Event", "Message"},
        "valid_object_types": {"Secret", "Knowledge", "Lore", "Person"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Information disclosure relationship",
        "examples_valid": [
            "Document:Ancient_Text | REVEALS | Secret:Hidden_Treasure",
            "Character:Spy | REVEALS | Knowledge:Enemy_Plans",
            "Event:Discovery | REVEALS | Person:True_Identity"
        ]
    },
    
    "CONCEALS": {
        "valid_subject_types": {"Character", "Organization", "Magic", "Structure"},
        "valid_object_types": {"Secret", "Knowledge", "Location", "Object"},
        "invalid_combinations": [],
        "bidirectional": False,
        "description": "Information hiding relationship",
        "examples_valid": [
            "Character:Guardian | CONCEALS | Secret:Ancient_Prophecy",
            "Magic:Illusion | CONCEALS | Location:Hidden_City",
            "Structure:False_Wall | CONCEALS | Object:Treasure"
        ]
    },

    # === FALLBACK RELATIONSHIPS ===
    "RELATES_TO": {
        "valid_subject_types": ENHANCED_NODE_LABELS,
        "valid_object_types": ENHANCED_NODE_LABELS, 
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "Generic fallback - should be minimized with enhanced types",
        "confidence_modifier": 0.3,  # Low confidence for generic relationships
    },
    
    "ASSOCIATED_WITH": {
        "valid_subject_types": ENHANCED_NODE_LABELS,
        "valid_object_types": ENHANCED_NODE_LABELS,
        "invalid_combinations": [],
        "bidirectional": True,
        "description": "General association - better than RELATES_TO but still generic",
        "confidence_modifier": 0.5,
    }
}

# Enhanced semantic rules leveraging specific node types
ENHANCED_SEMANTIC_RULES = {
    "divine_interaction": {
        "relationships": ["WORSHIPS", "BLESSED_BY", "CURSED_BY"],
        "rule": "Divine interactions require conscious beings or sacred objects",
        "validator": lambda s, o: (s in NodeClassification.CONSCIOUS) or 
                                 (o in {"Deity", "Spirit"} and s in {"Artifact", "Symbol", "Structure"})
    },
    
    "crafting_logic": {
        "relationships": ["CRAFTED_BY", "FORGED_FROM", "MADE_OF"],
        "rule": "Only physical objects can be crafted or made from materials",
        "validator": lambda s, o: s in {"Object", "Artifact", "Item", "Structure"} and 
                                 (o in {"Material", "Resource"} or o in NodeClassification.CONSCIOUS)
    },
    
    "communication_requirements": {
        "relationships": ["SPEAKS", "TEACHES", "TELLS_OF"],
        "rule": "Communication requires conscious entities or information systems",
        "validator": lambda s, o: (s in NodeClassification.CONSCIOUS or 
                                  s in {"System", "Document", "Organization"})
    },
    
    "temporal_consistency": {
        "relationships": ["PRECEDED_BY", "FOLLOWED_BY", "OCCURRED_DURING"],
        "rule": "Temporal relationships require temporal entities",
        "validator": lambda s, o: (s in NodeClassification.TEMPORAL and 
                                  o in NodeClassification.TEMPORAL)
    },
    
    "spatial_logic": {
        "relationships": ["BORDERS", "OVERLOOKS", "CONNECTS_TO"],
        "rule": "Spatial relationships require physical presence",
        "validator": lambda s, o: (s in NodeClassification.PHYSICAL_PRESENCE and 
                                  o in NodeClassification.PHYSICAL_PRESENCE)
    },
    
    "magical_interaction": {
        "relationships": ["ENCHANTED_BY", "POWERED_BY", "CURSED_BY"],
        "rule": "Magical interactions require magic systems or supernatural entities",
        "validator": lambda s, o: o in {"Magic", "Deity", "Spirit", "Person"} or s in {"Magic", "System"}
    }
}


def get_enhanced_relationship_suggestions(subject_type: str, object_type: str) -> List[Tuple[str, str, float]]:
    """
    Get relationship suggestions with confidence scores using enhanced node types.
    
    Returns list of (relationship_type, description, confidence) tuples.
    """
    suggestions = []
    
    for rel_type, constraints in ENHANCED_RELATIONSHIP_CONSTRAINTS.items():
        if (subject_type in constraints["valid_subject_types"] and 
            object_type in constraints["valid_object_types"]):
            
            # Check for invalid combinations
            if (subject_type, object_type) not in constraints.get("invalid_combinations", []):
                confidence = constraints.get("confidence_modifier", 1.0)
                
                # Boost confidence for specific type matches
                if subject_type != "Entity" and object_type != "Entity":
                    confidence *= 1.2
                
                # Boost confidence for highly specific relationships
                if rel_type in ["WIELDS", "SPEAKS", "CRAFTED_BY", "GOVERNS"]:
                    confidence *= 1.1
                
                suggestions.append((rel_type, constraints["description"], min(confidence, 1.0)))
    
    # Sort by confidence (highest first)
    return sorted(suggestions, key=lambda x: x[2], reverse=True)


def analyze_constraint_improvement() -> Dict[str, Any]:
    """Analyze how enhanced node types improve constraint effectiveness."""
    
    # Count constraints by specificity
    generic_constraints = 0
    specific_constraints = 0
    
    for rel_type, constraints in ENHANCED_RELATIONSHIP_CONSTRAINTS.items():
        subject_types = constraints["valid_subject_types"]
        object_types = constraints["valid_object_types"]
        
        if "Entity" in subject_types or "Entity" in object_types:
            generic_constraints += 1
        else:
            specific_constraints += 1
    
    # Analyze relationship coverage
    total_relationships = len(ENHANCED_RELATIONSHIP_CONSTRAINTS)
    enhanced_relationships = sum(1 for rel in ENHANCED_RELATIONSHIP_CONSTRAINTS.keys() 
                                if rel not in ["RELATES_TO", "ASSOCIATED_WITH"])
    
    return {
        "total_relationships": total_relationships,
        "specific_constraints": specific_constraints,
        "generic_constraints": generic_constraints,
        "specificity_ratio": specific_constraints / total_relationships * 100,
        "enhanced_relationships": enhanced_relationships,
        "semantic_rules": len(ENHANCED_SEMANTIC_RULES),
        "supported_node_types": len(ENHANCED_NODE_LABELS),
        "improvement_metrics": {
            "relationship_precision": specific_constraints / total_relationships * 100,
            "semantic_coverage": enhanced_relationships / total_relationships * 100,
            "type_utilization": len(ENHANCED_NODE_LABELS) / 15 * 100,  # vs original 15 types
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Test constraint improvements
    analysis = analyze_constraint_improvement()
    print("=== Enhanced Constraint System Analysis ===")
    print(f"Total relationships: {analysis['total_relationships']}")
    print(f"Specific constraints: {analysis['specific_constraints']}")
    print(f"Generic constraints: {analysis['generic_constraints']}")
    print(f"Specificity ratio: {analysis['specificity_ratio']:.1f}%")
    print(f"Enhanced relationships: {analysis['enhanced_relationships']}")
    
    # Test relationship suggestions
    print("\n=== Relationship Suggestions Examples ===")
    test_cases = [
        ("Character", "Artifact"),
        ("Deity", "Character"),
        ("Settlement", "Resource"),
        ("Document", "Knowledge"),
    ]
    
    for subject, obj in test_cases:
        suggestions = get_enhanced_relationship_suggestions(subject, obj)
        print(f"\n{subject} -> {obj}:")
        for rel, desc, conf in suggestions[:3]:  # Top 3 suggestions
            print(f"  {rel}: {desc} (confidence: {conf:.2f})")