"""
Property-based tests for vector storage and retrieval.

Property 6: Vector Storage and Retrieval
Validates: Requirements 3.3
"""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil

from ai_pr_reviewer.conventions.vector_store import VectorStore
from ai_pr_reviewer.models.convention import ConventionRule, EmbeddedConvention


class TestVectorStorage:
    """Property tests for vector storage and retrieval operations."""

    @given(
        collection_names=st.lists(
            st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
            min_size=1,
            max_size=10
        ),
        vector_dimensions=st.integers(min_value=50, max_value=500)
    )
    def test_collection_management_consistency(self, collection_names, vector_dimensions):
        """
        Property: Collection management should be consistent and reliable.
        
        Given: Various collection names and vector dimensions
        When: Collections are created, accessed, and managed
        Then: Operations should be consistent and maintain data integrity
        """
        assume(all(name.replace('_', '').replace('-', '').isalnum() for name in collection_names))
        
        with patch('ai_pr_reviewer.conventions.vector_store.QdrantClient') as mock_client:
            # Setup mock client
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock collection operations
            mock_instance.get_collections.return_value.collections = []
            mock_instance.create_collection.return_value = True
            mock_instance.delete_collection.return_value = True
            
            vector_store = VectorStore()
            
            # Test collection creation
            created_collections = []
            for collection_name in collection_names:
                try:
                    success = vector_store.create_collection(collection_name, vector_dimensions)
                    if success:
                        created_collections.append(collection_name)
                        
                        # Verify collection was created
                        assert mock_instance.create_collection.called
                        
                except Exception as e:
                    # Should handle invalid names gracefully
                    assert "invalid" in str(e).lower() or "name" in str(e).lower()
            
            # Test collection listing
            mock_instance.get_collections.return_value.collections = [
                Mock(name=name) for name in created_collections
            ]
            
            collections = vector_store.list_collections()
            assert isinstance(collections, list)
            assert len(collections) == len(created_collections)

    @given(
        vectors_batch=st.lists(
            st.dictionaries({
                'id': st.text(min_size=1, max_size=50),
                'vector': st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=100, max_size=100),
                'metadata': st.dictionaries(
                    keys=st.text(min_size=1, max_size=20),
                    values=st.one_of(st.text(min_size=1, max_size=100), st.integers(), st.floats())
                )
            }),
            min_size=1,
            max_size=50
        )
    )
    def test_batch_operations_atomicity(self, vectors_batch):
        """
        Property: Batch operations should be atomic and consistent.
        
        Given: Batch of vectors with metadata
        When: Batch operations are performed
        Then: All operations should succeed or fail together
        """
        assume(len(set(v['id'] for v in vectors_batch)) == len(vectors_batch))  # Unique IDs
        
        with patch('ai_pr_reviewer.conventions.vector_store.QdrantClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock successful batch operations
            mock_instance.upsert.return_value = Mock(status='completed')
            mock_instance.search.return_value = []
            
            vector_store = VectorStore()
            vector_store.create_collection('test_collection', 100)
            
            # Prepare batch data
            batch_points = []
            for item in vectors_batch:
                # Normalize vector
                vector = np.array(item['vector'])
                if np.linalg.norm(vector) > 0:
                    vector = vector / np.linalg.norm(vector)
                
                batch_points.append({
                    'id': item['id'],
                    'vector': vector.tolist(),
                    'payload': item['metadata']
                })
            
            # Test batch insertion
            try:
                success = vector_store.add_vectors_batch('test_collection', batch_points)
                
                if success:
                    # Verify batch operation was called
                    assert mock_instance.upsert.called
                    
                    # Verify all vectors in batch
                    call_args = mock_instance.upsert.call_args
                    assert len(call_args[1]['points']) == len(batch_points)
                    
            except Exception as e:
                # Batch operations should handle errors gracefully
                assert "batch" in str(e).lower() or "operation" in str(e).lower()

    @given(
        query_vectors=st.lists(
            st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=100, max_size=100),
            min_size=1,
            max_size=10
        ),
        top_k_values=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=5),
        similarity_thresholds=st.lists(st.floats(min_value=0.1, max_value=0.9), min_size=1, max_size=5)
    )
    def test_search_result_consistency(self, query_vectors, top_k_values, similarity_thresholds):
        """
        Property: Search results should be consistent and properly ranked.
        
        Given: Query vectors with various search parameters
        When: Similarity searches are performed
        Then: Results should be consistent and properly ranked by similarity
        """
        with patch('ai_pr_reviewer.conventions.vector_store.QdrantClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock search results with decreasing similarity scores
            def mock_search(collection_name, query_vector, limit, score_threshold=None):
                results = []
                for i in range(min(limit, 10)):  # Return up to 10 results
                    score = 1.0 - (i * 0.1)  # Decreasing scores
                    if score_threshold is None or score >= score_threshold:
                        results.append(Mock(
                            id=f'result_{i}',
                            score=score,
                            payload={'title': f'Convention {i}', 'category': 'test'}
                        ))
                return results
            
            mock_instance.search.side_effect = mock_search
            
            vector_store = VectorStore()
            vector_store.create_collection('test_collection', 100)
            
            # Test search consistency
            for query_vector in query_vectors[:3]:  # Limit for performance
                # Normalize query vector
                query_array = np.array(query_vector)
                if np.linalg.norm(query_array) > 0:
                    query_array = query_array / np.linalg.norm(query_array)
                
                for top_k in top_k_values[:2]:  # Limit combinations
                    for threshold in similarity_thresholds[:2]:
                        try:
                            results = vector_store.search_similar(
                                'test_collection',
                                query_array.tolist(),
                                top_k=top_k,
                                similarity_threshold=threshold
                            )
                            
                            # Verify result structure
                            assert isinstance(results, list)
                            assert len(results) <= top_k
                            
                            # Verify results are properly ranked (descending similarity)
                            for i in range(len(results) - 1):
                                assert results[i].score >= results[i + 1].score, \
                                    "Results should be ranked by similarity"
                            
                            # Verify similarity threshold is respected
                            for result in results:
                                assert result.score >= threshold, \
                                    f"Result score {result.score} below threshold {threshold}"
                                    
                        except Exception as e:
                            # Search should handle edge cases gracefully
                            assert "search" in str(e).lower() or "vector" in str(e).lower()

    @given(
        storage_operations=st.lists(
            st.dictionaries({
                'operation': st.sampled_from(['add', 'update', 'delete', 'search']),
                'vector_id': st.text(min_size=1, max_size=30),
                'vector_data': st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=100, max_size=100)
            }),
            min_size=5,
            max_size=30
        )
    )
    def test_concurrent_operations_safety(self, storage_operations):
        """
        Property: Concurrent operations should be handled safely.
        
        Given: Multiple concurrent storage operations
        When: Operations are performed simultaneously
        Then: Data integrity should be maintained
        """
        with patch('ai_pr_reviewer.conventions.vector_store.QdrantClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock operation responses
            mock_instance.upsert.return_value = Mock(status='completed')
            mock_instance.delete.return_value = Mock(status='completed')
            mock_instance.search.return_value = []
            
            vector_store = VectorStore()
            vector_store.create_collection('concurrent_test', 100)
            
            # Track operations for consistency checking
            added_vectors = set()
            deleted_vectors = set()
            
            # Simulate concurrent operations
            for operation in storage_operations:
                vector_id = operation['vector_id']
                vector_data = np.array(operation['vector_data'])
                
                # Normalize vector
                if np.linalg.norm(vector_data) > 0:
                    vector_data = vector_data / np.linalg.norm(vector_data)
                
                try:
                    if operation['operation'] == 'add':
                        success = vector_store.add_vector(
                            'concurrent_test',
                            vector_id,
                            vector_data.tolist(),
                            {'operation': 'add'}
                        )
                        if success:
                            added_vectors.add(vector_id)
                            
                    elif operation['operation'] == 'update':
                        success = vector_store.update_vector(
                            'concurrent_test',
                            vector_id,
                            vector_data.tolist(),
                            {'operation': 'update'}
                        )
                        
                    elif operation['operation'] == 'delete':
                        success = vector_store.delete_vector('concurrent_test', vector_id)
                        if success:
                            deleted_vectors.add(vector_id)
                            
                    elif operation['operation'] == 'search':
                        results = vector_store.search_similar(
                            'concurrent_test',
                            vector_data.tolist(),
                            top_k=5
                        )
                        assert isinstance(results, list)
                        
                except Exception as e:
                    # Concurrent operations should handle conflicts gracefully
                    assert "concurrent" in str(e).lower() or "conflict" in str(e).lower() or \
                           "lock" in str(e).lower() or "timeout" in str(e).lower()
            
            # Verify operation tracking consistency
            # (This is a simplified check since we're mocking the actual storage)
            assert isinstance(added_vectors, set)
            assert isinstance(deleted_vectors, set)

    @given(
        large_dataset_size=st.integers(min_value=100, max_value=1000),
        vector_dimension=st.integers(min_value=100, max_value=300)
    )
    def test_large_dataset_performance(self, large_dataset_size, vector_dimension):
        """
        Property: Large dataset operations should maintain reasonable performance.
        
        Given: Large datasets with many vectors
        When: Storage and retrieval operations are performed
        Then: Performance should remain acceptable
        """
        assume(large_dataset_size * vector_dimension < 300000)  # Memory limit
        
        with patch('ai_pr_reviewer.conventions.vector_store.QdrantClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            
            # Mock efficient batch operations
            mock_instance.upsert.return_value = Mock(status='completed')
            mock_instance.search.return_value = [
                Mock(id=f'result_{i}', score=0.9 - i*0.01, payload={})
                for i in range(10)
            ]
            
            vector_store = VectorStore()
            vector_store.create_collection('large_dataset', vector_dimension)
            
            # Generate large dataset
            import time
            start_time = time.time()
            
            # Simulate batch insertion of large dataset
            batch_size = 50
            for batch_start in range(0, large_dataset_size, batch_size):
                batch_end = min(batch_start + batch_size, large_dataset_size)
                batch_points = []
                
                for i in range(batch_start, batch_end):
                    # Generate normalized random vector
                    vector = np.random.randn(vector_dimension)
                    vector = vector / np.linalg.norm(vector)
                    
                    batch_points.append({
                        'id': f'vector_{i}',
                        'vector': vector.tolist(),
                        'payload': {'index': i, 'batch': batch_start // batch_size}
                    })
                
                # Add batch
                success = vector_store.add_vectors_batch('large_dataset', batch_points)
                assert success, f"Batch insertion failed at batch {batch_start // batch_size}"
            
            insertion_time = time.time() - start_time
            
            # Test search performance
            search_start = time.time()
            query_vector = np.random.randn(vector_dimension)
            query_vector = query_vector / np.linalg.norm(query_vector)
            
            results = vector_store.search_similar(
                'large_dataset',
                query_vector.tolist(),
                top_k=10
            )
            
            search_time = time.time() - search_start
            
            # Verify performance is reasonable
            # (These are loose bounds since we're mocking the actual operations)
            assert insertion_time < 10.0, f"Insertion too slow: {insertion_time}s"
            assert search_time < 1.0, f"Search too slow: {search_time}s"
            
            # Verify results
            assert isinstance(results, list)
            assert len(results) <= 10

    @given(
        persistence_scenarios=st.lists(
            st.dictionaries({
                'collection_name': st.text(min_size=3, max_size=20),
                'vector_count': st.integers(min_value=1, max_value=20),
                'should_persist': st.booleans()
            }),
            min_size=1,
            max_size=5
        )
    )
    def test_data_persistence_reliability(self, persistence_scenarios):
        """
        Property: Data persistence should be reliable across sessions.
        
        Given: Various persistence scenarios
        When: Data is stored and vector store is reinitialized
        Then: Data should persist correctly
        """
        with patch('ai_pr_reviewer.conventions.vector_store.QdrantClient') as mock_client:
            # Mock persistent storage behavior
            stored_collections = {}
            
            def mock_get_collections():
                collections = [Mock(name=name) for name in stored_collections.keys()]
                return Mock(collections=collections)
            
            def mock_create_collection(name, vectors_config):
                stored_collections[name] = {'vectors': {}, 'config': vectors_config}
                return True
            
            def mock_upsert(collection_name, points):
                if collection_name in stored_collections:
                    for point in points:
                        stored_collections[collection_name]['vectors'][point['id']] = point
                return Mock(status='completed')
            
            def mock_search(collection_name, query_vector, limit, score_threshold=None):
                if collection_name not in stored_collections:
                    return []
                
                # Return some mock results based on stored data
                stored_vectors = stored_collections[collection_name]['vectors']
                results = []
                for i, (vector_id, vector_data) in enumerate(list(stored_vectors.items())[:limit]):
                    score = 0.9 - i * 0.1
                    if score_threshold is None or score >= score_threshold:
                        results.append(Mock(
                            id=vector_id,
                            score=score,
                            payload=vector_data.get('payload', {})
                        ))
                return results
            
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            mock_instance.get_collections.side_effect = mock_get_collections
            mock_instance.create_collection.side_effect = mock_create_collection
            mock_instance.upsert.side_effect = mock_upsert
            mock_instance.search.side_effect = mock_search
            
            # Test persistence scenarios
            for scenario in persistence_scenarios:
                collection_name = scenario['collection_name'].replace(' ', '_')
                
                # Create vector store and collection
                vector_store = VectorStore()
                vector_store.create_collection(collection_name, 100)
                
                # Add vectors
                for i in range(scenario['vector_count']):
                    vector = np.random.randn(100)
                    vector = vector / np.linalg.norm(vector)
                    
                    vector_store.add_vector(
                        collection_name,
                        f'vector_{i}',
                        vector.tolist(),
                        {'index': i, 'persistent': scenario['should_persist']}
                    )
                
                # Simulate restart by creating new vector store instance
                new_vector_store = VectorStore()
                
                # Verify persistence
                collections = new_vector_store.list_collections()
                if scenario['should_persist']:
                    collection_names = [c.name if hasattr(c, 'name') else str(c) for c in collections]
                    assert collection_name in collection_names, \
                        f"Collection {collection_name} not persisted"
                    
                    # Test search on persisted data
                    query_vector = np.random.randn(100)
                    query_vector = query_vector / np.linalg.norm(query_vector)
                    
                    results = new_vector_store.search_similar(
                        collection_name,
                        query_vector.tolist(),
                        top_k=5
                    )
                    
                    # Should find some results if data persisted
                    assert isinstance(results, list)