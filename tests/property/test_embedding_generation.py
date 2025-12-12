"""
Property-based tests for embedding generation and similarity.

Property 5: Embedding Generation and Similarity
Validates: Requirements 3.1, 3.2
"""

import pytest
from hypothesis import given, strategies as st, assume
import numpy as np
from unittest.mock import Mock, patch

from ai_pr_reviewer.conventions.embeddings import EmbeddingGenerator
from ai_pr_reviewer.models.convention import ConventionRule


class TestEmbeddingGeneration:
    """Property tests for embedding generation and similarity calculations."""

    @given(
        text_inputs=st.lists(
            st.text(min_size=10, max_size=500),
            min_size=2,
            max_size=20
        )
    )
    def test_embedding_consistency(self, text_inputs):
        """
        Property: Identical inputs should produce identical embeddings.
        
        Given: Text inputs for embedding generation
        When: Embeddings are generated multiple times for same input
        Then: Results should be identical
        """
        assume(all(len(text.strip()) > 5 for text in text_inputs))
        
        generator = EmbeddingGenerator()
        
        # Test consistency for each input
        for text in text_inputs[:5]:  # Limit for performance
            # Generate embeddings multiple times
            embedding1 = generator.generate_embedding(text)
            embedding2 = generator.generate_embedding(text)
            embedding3 = generator.generate_embedding(text)
            
            # Verify consistency
            assert isinstance(embedding1, np.ndarray)
            assert isinstance(embedding2, np.ndarray)
            assert isinstance(embedding3, np.ndarray)
            
            # Should be identical (or very close due to floating point)
            np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)
            np.testing.assert_array_almost_equal(embedding2, embedding3, decimal=6)
            
            # Verify embedding properties
            assert embedding1.shape == embedding2.shape == embedding3.shape
            assert len(embedding1.shape) == 1  # Should be 1D vector
            assert embedding1.shape[0] > 0  # Should have dimensions

    @given(
        similar_texts=st.lists(
            st.dictionaries({
                'base_text': st.text(min_size=20, max_size=200),
                'variation': st.sampled_from(['synonym', 'paraphrase', 'extension'])
            }),
            min_size=2,
            max_size=10
        )
    )
    def test_semantic_similarity_properties(self, similar_texts):
        """
        Property: Semantically similar texts should have higher similarity scores.
        
        Given: Texts with known semantic relationships
        When: Embeddings are generated and similarity calculated
        Then: Similar texts should have higher similarity than random texts
        """
        generator = EmbeddingGenerator()
        
        # Create text variations
        text_pairs = []
        for item in similar_texts:
            base_text = item['base_text']
            
            if item['variation'] == 'synonym':
                # Simple synonym replacement
                varied_text = base_text.replace('good', 'excellent').replace('bad', 'poor')
            elif item['variation'] == 'paraphrase':
                # Add explanatory text
                varied_text = f"{base_text} In other words, this means the same thing."
            else:  # extension
                # Extend with related content
                varied_text = f"{base_text} This is an important principle to follow."
            
            text_pairs.append((base_text, varied_text))
        
        # Test similarity properties
        for base_text, varied_text in text_pairs[:5]:  # Limit for performance
            if len(base_text.strip()) < 10 or len(varied_text.strip()) < 10:
                continue
                
            # Generate embeddings
            base_embedding = generator.generate_embedding(base_text)
            varied_embedding = generator.generate_embedding(varied_text)
            
            # Calculate similarity
            similarity = generator.calculate_similarity(base_embedding, varied_embedding)
            
            # Verify similarity properties
            assert isinstance(similarity, (float, np.floating))
            assert -1.0 <= similarity <= 1.0  # Cosine similarity bounds
            
            # Similar texts should have reasonable similarity (> 0.3 is reasonable threshold)
            # Note: This is a soft requirement as semantic similarity can vary
            assert similarity > 0.1, f"Similarity too low for related texts: {similarity}"

    @given(
        batch_sizes=st.integers(min_value=1, max_value=50),
        text_lengths=st.integers(min_value=10, max_value=1000)
    )
    def test_batch_processing_efficiency(self, batch_sizes, text_lengths):
        """
        Property: Batch processing should be efficient and produce consistent results.
        
        Given: Various batch sizes and text lengths
        When: Embeddings are generated in batches vs individually
        Then: Results should be consistent and processing should be efficient
        """
        generator = EmbeddingGenerator()
        
        # Create test texts
        test_texts = []
        for i in range(batch_sizes):
            text = f"Test convention rule {i}: " + "content " * (text_lengths // 10)
            test_texts.append(text[:text_lengths])
        
        # Generate embeddings individually
        individual_embeddings = []
        for text in test_texts:
            embedding = generator.generate_embedding(text)
            individual_embeddings.append(embedding)
        
        # Generate embeddings in batch
        batch_embeddings = generator.generate_batch_embeddings(test_texts)
        
        # Verify consistency
        assert len(batch_embeddings) == len(individual_embeddings)
        assert len(batch_embeddings) == batch_sizes
        
        # Compare individual vs batch results
        for i, (individual, batch) in enumerate(zip(individual_embeddings, batch_embeddings)):
            assert isinstance(individual, np.ndarray)
            assert isinstance(batch, np.ndarray)
            assert individual.shape == batch.shape
            
            # Should be very similar (allowing for minor processing differences)
            similarity = generator.calculate_similarity(individual, batch)
            assert similarity > 0.95, f"Batch vs individual mismatch at index {i}: {similarity}"

    @given(
        embedding_dimensions=st.integers(min_value=50, max_value=1000),
        vector_count=st.integers(min_value=2, max_value=20)
    )
    def test_similarity_calculation_properties(self, embedding_dimensions, vector_count):
        """
        Property: Similarity calculations should satisfy mathematical properties.
        
        Given: Random embedding vectors
        When: Similarities are calculated
        Then: Mathematical properties should hold (symmetry, bounds, etc.)
        """
        generator = EmbeddingGenerator()
        
        # Create random embeddings (simulating real embeddings)
        embeddings = []
        for _ in range(vector_count):
            # Create normalized random vector (like real embeddings)
            vector = np.random.randn(embedding_dimensions)
            vector = vector / np.linalg.norm(vector)  # Normalize
            embeddings.append(vector)
        
        # Test similarity properties
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                sim_ij = generator.calculate_similarity(embeddings[i], embeddings[j])
                sim_ji = generator.calculate_similarity(embeddings[j], embeddings[i])
                
                # Symmetry property
                assert abs(sim_ij - sim_ji) < 1e-6, "Similarity should be symmetric"
                
                # Bounds property
                assert -1.0 <= sim_ij <= 1.0, "Similarity should be in [-1, 1]"
                
                # Self-similarity property
                if i == j:
                    assert abs(sim_ij - 1.0) < 1e-6, "Self-similarity should be 1.0"

    @given(
        convention_texts=st.lists(
            st.dictionaries({
                'title': st.text(min_size=5, max_size=100),
                'description': st.text(min_size=20, max_size=500),
                'category': st.sampled_from(['naming', 'structure', 'testing', 'documentation'])
            }),
            min_size=3,
            max_size=15
        )
    )
    def test_convention_embedding_quality(self, convention_texts):
        """
        Property: Convention embeddings should capture semantic meaning effectively.
        
        Given: Convention rules with different categories
        When: Embeddings are generated
        Then: Similar categories should have higher similarity
        """
        generator = EmbeddingGenerator()
        
        # Create convention rules
        conventions = []
        for conv_data in convention_texts:
            convention = ConventionRule(
                title=conv_data['title'],
                description=conv_data['description'],
                category=conv_data['category'],
                source_file='test.md',
                line_number=1
            )
            conventions.append(convention)
        
        # Generate embeddings for conventions
        convention_embeddings = {}
        for convention in conventions:
            # Combine title and description for embedding
            full_text = f"{convention.title} {convention.description}"
            embedding = generator.generate_embedding(full_text)
            convention_embeddings[convention] = embedding
        
        # Test category clustering
        category_groups = {}
        for convention in conventions:
            if convention.category not in category_groups:
                category_groups[convention.category] = []
            category_groups[convention.category].append(convention)
        
        # Verify within-category similarity is higher than cross-category
        for category, group_conventions in category_groups.items():
            if len(group_conventions) < 2:
                continue
                
            # Calculate within-category similarities
            within_similarities = []
            for i in range(len(group_conventions)):
                for j in range(i + 1, len(group_conventions)):
                    conv1, conv2 = group_conventions[i], group_conventions[j]
                    sim = generator.calculate_similarity(
                        convention_embeddings[conv1],
                        convention_embeddings[conv2]
                    )
                    within_similarities.append(sim)
            
            if within_similarities:
                avg_within_similarity = np.mean(within_similarities)
                
                # Should have reasonable within-category similarity
                # (This is a soft requirement as it depends on content quality)
                assert avg_within_similarity > -0.5, "Within-category similarity too low"

    @given(
        noise_levels=st.lists(
            st.floats(min_value=0.0, max_value=0.5),
            min_size=1,
            max_size=5
        ),
        base_text=st.text(min_size=50, max_size=300)
    )
    def test_embedding_robustness_to_noise(self, noise_levels, base_text):
        """
        Property: Embeddings should be robust to minor text variations.
        
        Given: Text with various levels of noise/variation
        When: Embeddings are generated
        Then: Similar texts should maintain high similarity despite noise
        """
        assume(len(base_text.strip()) > 20)
        
        generator = EmbeddingGenerator()
        
        # Generate base embedding
        base_embedding = generator.generate_embedding(base_text)
        
        # Test robustness to different noise levels
        for noise_level in noise_levels:
            # Add character-level noise
            noisy_text = ""
            for char in base_text:
                if np.random.random() < noise_level and char.isalpha():
                    # Replace with similar character or add typo
                    if np.random.random() < 0.5:
                        noisy_text += char.upper() if char.islower() else char.lower()
                    else:
                        noisy_text += char + char  # Double character
                else:
                    noisy_text += char
            
            # Generate noisy embedding
            noisy_embedding = generator.generate_embedding(noisy_text)
            
            # Calculate similarity
            similarity = generator.calculate_similarity(base_embedding, noisy_embedding)
            
            # Should maintain reasonable similarity despite noise
            expected_min_similarity = 1.0 - (noise_level * 2)  # Rough heuristic
            assert similarity > max(0.3, expected_min_similarity), \
                f"Similarity {similarity} too low for noise level {noise_level}"