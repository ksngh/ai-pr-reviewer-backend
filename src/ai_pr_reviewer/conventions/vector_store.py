"""
Vector Store

Manages vector storage and retrieval using Qdrant vector database.
Handles convention rule storage, updates, and similarity search.
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Qdrant client not installed: {e}")
    logger.error("Install with: pip install qdrant-client")
    raise

from ..models.convention import ConventionRule, EmbeddedConvention


logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Vector store related errors"""
    pass


class VectorStore:
    """
    Vector database interface for storing and retrieving convention rules.
    
    Uses Qdrant for efficient similarity search and vector operations.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "conventions",
        embedding_dim: int = 384  # Default for all-MiniLM-L6-v2
    ):
        """
        Initialize vector store.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            embedding_dim: Dimension of embeddings
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Initialize client
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise VectorStoreError(f"Failed to connect to Qdrant: {e}")
        
        # Initialize collection
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, create if not."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise VectorStoreError(f"Failed to ensure collection exists: {e}")
    
    def store_convention(self, embedded_convention: EmbeddedConvention) -> str:
        """
        Store a convention rule with its embedding.
        
        Args:
            embedded_convention: EmbeddedConvention to store
            
        Returns:
            Point ID of stored convention
        """
        rule = embedded_convention.rule
        
        # Generate point ID
        point_id = str(uuid.uuid4())
        
        # Prepare payload
        payload = {
            'rule_id': rule.id,
            'title': rule.title,
            'description': rule.description,
            'rule_type': rule.rule_type,
            'category': rule.category,
            'source_wiki_page': rule.source_wiki_page,
            'examples': rule.examples,
            'counter_examples': rule.counter_examples,
            'created_at': rule.created_at.isoformat(),
            'updated_at': rule.updated_at.isoformat(),
            'version': rule.version,
            'embedding_model': embedded_convention.embedding_model,
            'text_used_for_embedding': embedded_convention.text_used_for_embedding
        }
        
        # Create point
        point = PointStruct(
            id=point_id,
            vector=embedded_convention.embedding,
            payload=payload
        )
        
        try:
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored convention: {rule.id} (point_id: {point_id})")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store convention {rule.id}: {e}")
            raise VectorStoreError(f"Failed to store convention: {e}")
    
    def batch_store_conventions(self, embedded_conventions: List[EmbeddedConvention]) -> List[str]:
        """
        Store multiple conventions efficiently.
        
        Args:
            embedded_conventions: List of EmbeddedConvention objects
            
        Returns:
            List of point IDs
        """
        if not embedded_conventions:
            return []
        
        logger.info(f"Batch storing {len(embedded_conventions)} conventions")
        
        points = []
        point_ids = []
        
        for embedded_convention in embedded_conventions:
            rule = embedded_convention.rule
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            payload = {
                'rule_id': rule.id,
                'title': rule.title,
                'description': rule.description,
                'rule_type': rule.rule_type,
                'category': rule.category,
                'source_wiki_page': rule.source_wiki_page,
                'examples': rule.examples,
                'counter_examples': rule.counter_examples,
                'created_at': rule.created_at.isoformat(),
                'updated_at': rule.updated_at.isoformat(),
                'version': rule.version,
                'embedding_model': embedded_convention.embedding_model,
                'text_used_for_embedding': embedded_convention.text_used_for_embedding
            }
            
            point = PointStruct(
                id=point_id,
                vector=embedded_convention.embedding,
                payload=payload
            )
            
            points.append(point)
        
        try:
            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully stored {len(points)} conventions")
            return point_ids
            
        except Exception as e:
            logger.error(f"Failed to batch store conventions: {e}")
            raise VectorStoreError(f"Failed to batch store conventions: {e}")
    
    def search_similar_conventions(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[ConventionRule, float]]:
        """
        Search for similar conventions using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Optional filters (category, rule_type, etc.)
            
        Returns:
            List of (ConventionRule, similarity_score) tuples
        """
        try:
            # Build filter conditions
            filter_conditions = None
            if filters:
                conditions = []
                
                for field, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values (OR condition)
                        for v in value:
                            conditions.append(
                                FieldCondition(key=field, match=MatchValue(value=v))
                            )
                    else:
                        # Single value
                        conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )
                
                if conditions:
                    filter_conditions = Filter(should=conditions)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Convert results to ConventionRule objects
            results = []
            for scored_point in search_result:
                payload = scored_point.payload
                
                # Reconstruct ConventionRule
                rule = ConventionRule(
                    id=payload['rule_id'],
                    title=payload['title'],
                    description=payload['description'],
                    rule_type=payload['rule_type'],
                    category=payload['category'],
                    source_wiki_page=payload['source_wiki_page'],
                    examples=payload.get('examples', []),
                    counter_examples=payload.get('counter_examples', []),
                    created_at=datetime.fromisoformat(payload['created_at']),
                    updated_at=datetime.fromisoformat(payload['updated_at']),
                    version=payload['version']
                )
                
                results.append((rule, scored_point.score))
            
            logger.info(f"Found {len(results)} similar conventions")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search conventions: {e}")
            raise VectorStoreError(f"Failed to search conventions: {e}")
    
    def get_convention_by_id(self, rule_id: str) -> Optional[ConventionRule]:
        """
        Get convention by rule ID.
        
        Args:
            rule_id: Convention rule ID
            
        Returns:
            ConventionRule or None if not found
        """
        try:
            # Search by rule_id
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="rule_id", match=MatchValue(value=rule_id))]
                ),
                limit=1
            )
            
            if search_result[0]:  # Points found
                payload = search_result[0][0].payload
                
                rule = ConventionRule(
                    id=payload['rule_id'],
                    title=payload['title'],
                    description=payload['description'],
                    rule_type=payload['rule_type'],
                    category=payload['category'],
                    source_wiki_page=payload['source_wiki_page'],
                    examples=payload.get('examples', []),
                    counter_examples=payload.get('counter_examples', []),
                    created_at=datetime.fromisoformat(payload['created_at']),
                    updated_at=datetime.fromisoformat(payload['updated_at']),
                    version=payload['version']
                )
                
                return rule
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get convention {rule_id}: {e}")
            return None
    
    def update_convention(self, embedded_convention: EmbeddedConvention) -> bool:
        """
        Update existing convention.
        
        Args:
            embedded_convention: Updated EmbeddedConvention
            
        Returns:
            True if successful, False otherwise
        """
        rule = embedded_convention.rule
        
        try:
            # Find existing points with this rule_id
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="rule_id", match=MatchValue(value=rule.id))]
                ),
                limit=100  # Should be only one, but just in case
            )
            
            if not search_result[0]:  # No existing points
                logger.warning(f"No existing convention found for rule_id: {rule.id}")
                return False
            
            # Delete old points
            point_ids = [point.id for point in search_result[0]]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            
            # Store updated convention
            self.store_convention(embedded_convention)
            
            logger.info(f"Updated convention: {rule.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update convention {rule.id}: {e}")
            return False
    
    def delete_convention(self, rule_id: str) -> bool:
        """
        Delete convention by rule ID.
        
        Args:
            rule_id: Convention rule ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find points with this rule_id
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="rule_id", match=MatchValue(value=rule_id))]
                ),
                limit=100
            )
            
            if not search_result[0]:  # No points found
                logger.warning(f"No convention found for rule_id: {rule_id}")
                return False
            
            # Delete points
            point_ids = [point.id for point in search_result[0]]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            
            logger.info(f"Deleted convention: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete convention {rule_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'collection_name': self.collection_name,
                'points_count': collection_info.points_count,
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'status': collection_info.status.value,
                'embedding_dim': self.embedding_dim
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all conventions from collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete collection and recreate
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False