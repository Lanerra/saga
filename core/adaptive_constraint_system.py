# core/adaptive_constraint_system.py
"""
Adaptive relationship constraint system that learns from actual data patterns.

This system discovers relationship constraints by analyzing the actual relationships
in the knowledge graph, providing data-driven validation instead of static rules.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.db_manager import neo4j_manager

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints that can be learned or applied."""
    LEARNED_STATISTICAL = "learned_statistical"
    LEARNED_PATTERN = "learned_pattern"
    RULE_BASED = "rule_based"
    FALLBACK = "fallback"


@dataclass
class RelationshipConstraint:
    """Represents a constraint on a relationship type."""
    relationship_type: str
    subject_types: Set[str]
    object_types: Set[str]
    confidence: float
    constraint_type: ConstraintType
    sample_size: int = 0
    examples: List[Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


class AdaptiveConstraintSystem:
    """Self-learning constraint system based on actual relationship patterns."""
    
    def __init__(self, schema_introspector):
        self.introspector = schema_introspector
        self.constraints: Dict[str, RelationshipConstraint] = {}
        self.min_confidence = 0.5
        self.min_samples = 3
        self.max_type_combinations_ratio = 0.7  # Max ratio of actual vs possible combinations
        self.last_learning_update = None
        
    async def learn_constraints_from_data(self, min_frequency: int = 3):
        """Learn relationship constraints from existing data patterns."""
        try:
            logger.info("Learning relationship constraints from database patterns...")
            
            # Get relationship patterns from introspector
            patterns = await self.introspector.get_relationship_patterns()
            
            if not patterns:
                logger.warning("No relationship patterns found for constraint learning")
                return
            
            # Group patterns by relationship type
            rel_patterns = defaultdict(list)
            for pattern in patterns:
                if pattern['frequency'] >= min_frequency:
                    rel_type = pattern['rel_type']
                    rel_patterns[rel_type].append(pattern)
            
            # Generate constraints for each relationship type
            new_constraints = {}
            for rel_type, type_patterns in rel_patterns.items():
                if len(type_patterns) >= self.min_samples:
                    constraint = self._generate_statistical_constraint(rel_type, type_patterns)
                    if constraint and constraint.confidence >= self.min_confidence:
                        new_constraints[rel_type] = constraint
            
            # Update constraints
            self.constraints.update(new_constraints)
            self.last_learning_update = datetime.utcnow()
            
            logger.info(f"Learned {len(new_constraints)} relationship constraints from {len(patterns)} patterns")
            
            # Log some examples for debugging
            for rel_type, constraint in list(new_constraints.items())[:5]:
                logger.debug(f"Learned constraint for {rel_type}: "
                           f"{len(constraint.subject_types)} subject types, "
                           f"{len(constraint.object_types)} object types, "
                           f"confidence {constraint.confidence:.3f}")
                
        except Exception as e:
            logger.error(f"Failed to learn constraints from data: {e}", exc_info=True)
    
    def _generate_statistical_constraint(
        self, rel_type: str, patterns: List[Dict[str, Any]]
    ) -> Optional[RelationshipConstraint]:
        """Generate statistical constraint from relationship usage patterns."""
        try:
            subject_types = set()
            object_types = set()
            total_frequency = 0
            examples = []
            
            # Collect all observed type combinations
            for pattern in patterns:
                source_type = pattern['source_type']
                target_type = pattern['target_type']
                frequency = pattern['frequency']
                
                subject_types.add(source_type)
                object_types.add(target_type)
                total_frequency += frequency
                
                # Keep examples of the most common patterns
                if len(examples) < 5:
                    examples.append((source_type, target_type))
            
            if not subject_types or not object_types:
                return None
            
            # Calculate constraint confidence based on pattern consistency
            observed_combinations = len(patterns)
            max_possible_combinations = len(subject_types) * len(object_types)
            
            # High confidence if we see consistent patterns (few combinations relative to possibilities)
            combination_ratio = observed_combinations / max_possible_combinations
            
            # Confidence calculation:
            # - Lower combination ratio = higher confidence (more consistent usage)
            # - Higher total frequency = higher confidence (more data)
            # - Balanced to avoid over-confidence
            base_confidence = 1.0 - min(combination_ratio, self.max_type_combinations_ratio)
            frequency_boost = min(total_frequency / 100, 0.3)  # Max 30% boost from frequency
            confidence = min(base_confidence + frequency_boost, 0.95)  # Cap at 95%
            
            return RelationshipConstraint(
                relationship_type=rel_type,
                subject_types=subject_types,
                object_types=object_types,
                confidence=confidence,
                constraint_type=ConstraintType.LEARNED_STATISTICAL,
                sample_size=total_frequency,
                examples=examples
            )
            
        except Exception as e:
            logger.error(f"Failed to generate constraint for {rel_type}: {e}")
            return None
    
    def validate_relationship(
        self, subject_type: str, rel_type: str, object_type: str
    ) -> Tuple[bool, float, str]:
        """Validate relationship with confidence score and explanation."""
        
        # Check if we have learned constraints for this relationship
        if rel_type not in self.constraints:
            # No learned constraints - be permissive but with low confidence
            return True, 0.3, f"No learned constraints for {rel_type} - allowing with low confidence"
        
        constraint = self.constraints[rel_type]
        
        subject_valid = subject_type in constraint.subject_types
        object_valid = object_type in constraint.object_types
        
        if subject_valid and object_valid:
            return True, constraint.confidence, f"Matches learned pattern (samples: {constraint.sample_size})"
        
        # Build detailed error message
        error_parts = []
        if not subject_valid:
            valid_subjects = sorted(list(constraint.subject_types)[:3])  # Show first 3
            more_subjects = len(constraint.subject_types) - 3
            subject_list = ', '.join(valid_subjects)
            if more_subjects > 0:
                subject_list += f" (+{more_subjects} more)"
            error_parts.append(f"subject '{subject_type}' not in learned types [{subject_list}]")
        
        if not object_valid:
            valid_objects = sorted(list(constraint.object_types)[:3])  # Show first 3
            more_objects = len(constraint.object_types) - 3
            object_list = ', '.join(valid_objects)
            if more_objects > 0:
                object_list += f" (+{more_objects} more)"
            error_parts.append(f"object '{object_type}' not in learned types [{object_list}]")
        
        error_message = f"Violates learned pattern: {'; '.join(error_parts)}"
        
        return False, constraint.confidence, error_message
    
    def get_valid_relationships_for_types(
        self, subject_type: str, object_type: str
    ) -> List[Tuple[str, float]]:
        """Get all valid relationship types between two node types with confidence."""
        valid_relationships = []
        
        for rel_type, constraint in self.constraints.items():
            if (subject_type in constraint.subject_types and 
                object_type in constraint.object_types):
                valid_relationships.append((rel_type, constraint.confidence))
        
        # Sort by confidence, highest first
        return sorted(valid_relationships, key=lambda x: x[1], reverse=True)
    
    def suggest_relationship_types(
        self, subject_type: str, object_type: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Suggest valid relationship types with explanations."""
        suggestions = []
        
        valid_rels = self.get_valid_relationships_for_types(subject_type, object_type)
        
        for rel_type, confidence in valid_rels[:limit]:
            constraint = self.constraints[rel_type]
            
            # Find examples with these exact types if possible
            matching_examples = [
                ex for ex in constraint.examples 
                if ex[0] == subject_type and ex[1] == object_type
            ]
            
            suggestion = {
                "relationship_type": rel_type,
                "confidence": confidence,
                "sample_size": constraint.sample_size,
                "examples": matching_examples[:3] if matching_examples else constraint.examples[:3]
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of learned constraints for debugging."""
        if not self.constraints:
            return {
                "status": "No constraints learned",
                "last_update": None,
                "total_constraints": 0
            }
        
        constraint_types = defaultdict(int)
        confidence_distribution = defaultdict(int)
        total_samples = 0
        
        for constraint in self.constraints.values():
            constraint_types[constraint.constraint_type.value] += 1
            
            # Bin confidence for distribution
            confidence_bin = f"{int(constraint.confidence * 10) * 10}-{int(constraint.confidence * 10) * 10 + 9}%"
            confidence_distribution[confidence_bin] += 1
            
            total_samples += constraint.sample_size
        
        # Get top constraints by confidence
        top_constraints = sorted(
            [(rel_type, constraint.confidence, constraint.sample_size) 
             for rel_type, constraint in self.constraints.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_constraints": len(self.constraints),
            "constraint_types": dict(constraint_types),
            "confidence_distribution": dict(confidence_distribution),
            "total_samples": total_samples,
            "top_constraints": [
                {"relationship": rel, "confidence": conf, "samples": samples}
                for rel, conf, samples in top_constraints
            ],
            "last_update": self.last_learning_update.isoformat() if self.last_learning_update else None
        }
    
    def get_relationship_stats(self, rel_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a specific relationship type."""
        if rel_type not in self.constraints:
            return None
        
        constraint = self.constraints[rel_type]
        
        return {
            "relationship_type": rel_type,
            "subject_types": sorted(list(constraint.subject_types)),
            "object_types": sorted(list(constraint.object_types)),
            "confidence": constraint.confidence,
            "sample_size": constraint.sample_size,
            "constraint_type": constraint.constraint_type.value,
            "examples": constraint.examples,
            "type_combinations": len(constraint.subject_types) * len(constraint.object_types)
        }
    
    async def refresh_constraints_if_needed(self, max_age_hours: int = 12):
        """Refresh learned constraints if they're stale."""
        if not self.last_learning_update:
            await self.learn_constraints_from_data()
            return
        
        age = datetime.utcnow() - self.last_learning_update
        if age.total_seconds() > (max_age_hours * 3600):
            logger.info(f"Constraints are {age.total_seconds() / 3600:.1f} hours old, refreshing...")
            await self.learn_constraints_from_data()
    
    def clear_constraints(self):
        """Clear all learned constraints (useful for testing or reset)."""
        self.constraints.clear()
        self.last_learning_update = None
        logger.info("All learned constraints cleared")