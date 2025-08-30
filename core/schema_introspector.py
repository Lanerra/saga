# core/schema_introspector.py
"""
Dynamic Neo4j schema discovery using introspection procedures.

This module provides real-time schema discovery by querying the Neo4j database
directly, replacing static mappings with dynamic, data-driven schema understanding.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Set, Tuple, Optional
from async_lru import alru_cache

from core.db_manager import neo4j_manager

logger = logging.getLogger(__name__)


class SchemaIntrospector:
    """Dynamic schema discovery using Neo4j built-in introspection procedures."""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache
        self.last_schema_update = None
        
    @alru_cache(maxsize=1, ttl=300)
    async def get_active_labels(self) -> Set[str]:
        """Get all labels currently used in the database."""
        try:
            results = await neo4j_manager.execute_read_query(
                "CALL db.labels() YIELD label RETURN label"
            )
            labels = {r['label'] for r in results if r.get('label')}
            logger.debug(f"Found {len(labels)} active labels in database")
            return labels
        except Exception as e:
            logger.error(f"Failed to get database labels: {e}")
            # Fallback to constants if introspection fails
            from models.kg_constants import NODE_LABELS
            return NODE_LABELS.copy()
    
    @alru_cache(maxsize=1, ttl=300)
    async def get_active_relationship_types(self) -> Set[str]:
        """Get all relationship types currently used in the database."""
        try:
            results = await neo4j_manager.execute_read_query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            rel_types = {r['relationshipType'] for r in results if r.get('relationshipType')}
            logger.debug(f"Found {len(rel_types)} active relationship types in database")
            return rel_types
        except Exception as e:
            logger.error(f"Failed to get relationship types: {e}")
            # Fallback to constants if introspection fails
            from models.kg_constants import RELATIONSHIP_TYPES
            return RELATIONSHIP_TYPES.copy()
    
    @alru_cache(maxsize=1, ttl=600)  # Longer cache for expensive operation
    async def get_label_frequencies(self) -> Dict[str, int]:
        """Get frequency count for each label in the database."""
        try:
            labels = await self.get_active_labels()
            frequencies = {}
            
            # Use parallel queries for better performance
            for label in labels:
                try:
                    # Escape label name to handle special characters
                    escaped_label = label.replace('`', '``')
                    query = f"MATCH (n:`{escaped_label}`) RETURN count(n) as count"
                    result = await neo4j_manager.execute_read_query(query)
                    frequencies[label] = result[0]['count'] if result else 0
                except Exception as label_error:
                    logger.warning(f"Failed to count nodes for label '{label}': {label_error}")
                    frequencies[label] = 0
            
            logger.info(f"Retrieved frequencies for {len(frequencies)} labels")
            return frequencies
            
        except Exception as e:
            logger.error(f"Failed to get label frequencies: {e}")
            return {}
    
    @alru_cache(maxsize=1, ttl=600)
    async def get_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Discover relationship patterns between node types."""
        query = """
        MATCH (a)-[r]->(b)
        WHERE size(labels(a)) > 0 AND size(labels(b)) > 0
        WITH labels(a)[0] as source_type, type(r) as rel_type, labels(b)[0] as target_type
        RETURN 
            source_type,
            rel_type,
            target_type,
            count(*) as frequency
        ORDER BY frequency DESC
        LIMIT 200
        """
        try:
            results = await neo4j_manager.execute_read_query(query)
            patterns = [dict(r) for r in results]
            logger.info(f"Discovered {len(patterns)} relationship patterns")
            return patterns
        except Exception as e:
            logger.error(f"Failed to get relationship patterns: {e}")
            return []
    
    @alru_cache(maxsize=1, ttl=900)  # 15 minutes for expensive sampling
    async def sample_node_properties(self, sample_size: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Sample node properties to understand data patterns."""
        query = f"""
        MATCH (n)
        WHERE n.name IS NOT NULL AND size(labels(n)) > 0
        WITH n, labels(n)[0] as primary_label, rand() as r
        ORDER BY r
        LIMIT {sample_size}
        RETURN 
            primary_label,
            n.name as name,
            coalesce(n.category, '') as category,
            coalesce(n.description, '') as description,
            coalesce(n.type, '') as type_property
        """
        try:
            results = await neo4j_manager.execute_read_query(query)
            
            # Group by label
            samples_by_label = {}
            for record in results:
                label = record['primary_label']
                if label not in samples_by_label:
                    samples_by_label[label] = []
                
                samples_by_label[label].append({
                    'name': record.get('name', ''),
                    'category': record.get('category', ''),
                    'description': record.get('description', ''),
                    'type_property': record.get('type_property', '')
                })
            
            logger.info(f"Sampled {len(results)} nodes across {len(samples_by_label)} labels")
            return samples_by_label
            
        except Exception as e:
            logger.error(f"Failed to sample node properties: {e}")
            return {}
    
    async def get_schema_summary(self) -> Dict[str, Any]:
        """Get comprehensive schema summary."""
        try:
            labels = await self.get_active_labels()
            relationships = await self.get_active_relationship_types()
            frequencies = await self.get_label_frequencies()
            patterns = await self.get_relationship_patterns()
            
            # Calculate schema metrics
            total_nodes = sum(frequencies.values())
            most_common_labels = sorted(
                frequencies.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            relationship_counts = {}
            for pattern in patterns:
                rel_type = pattern['rel_type']
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + pattern['frequency']
            
            most_common_relationships = sorted(
                relationship_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                'total_labels': len(labels),
                'total_relationship_types': len(relationships),
                'total_nodes': total_nodes,
                'most_common_labels': most_common_labels,
                'most_common_relationships': most_common_relationships,
                'relationship_patterns': len(patterns),
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate schema summary: {e}")
            return {
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }
    
    async def invalidate_cache(self):
        """Force cache invalidation for fresh schema discovery."""
        # Clear all cached methods
        self.get_active_labels.cache_clear()
        self.get_active_relationship_types.cache_clear()
        self.get_label_frequencies.cache_clear()
        self.get_relationship_patterns.cache_clear()
        self.sample_node_properties.cache_clear()
        logger.info("Schema introspector cache invalidated")
    
    async def is_schema_fresh(self, max_age_minutes: int = 60) -> bool:
        """Check if cached schema data is still fresh."""
        if not self.last_schema_update:
            return False
        
        age = datetime.utcnow() - self.last_schema_update
        return age < timedelta(minutes=max_age_minutes)


# Global instance for easy access
schema_introspector = SchemaIntrospector()