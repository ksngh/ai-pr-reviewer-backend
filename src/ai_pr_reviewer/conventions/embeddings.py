"""
Embedding Generator

Generates vector embeddings for convention rules using sentence transformers.
Handles batch processing and caching for efficient embedding generation.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Required ML dependencies not installed: {e}")
    logger.error("Install with: pip install sentence-transformers torch")
    raise

from ..models.convention import ConventionRule, EmbeddedConvention


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates vector embeddings for convention rules.
    
    Uses sentence-transformers to create semantic embeddings
    that can be used for similarity matching.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of sentence transformer model
            cache_dir: Directory for caching embeddings
            device: Device to run model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.ai_pr_reviewer' / 'embeddings'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        logger.info(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Cache for embeddings
        self._embedding_cache = {}
        self._load_cache()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.embedding_dim
        
        # Check cache first
        text_hash = self._hash_text(text)
        if text_hash in self._embedding_cache:
            logger.debug("Using cached embedding")
            return self._embedding_cache[text_hash]
        
        try:
            # Generate embedding
            logger.debug(f"Generating embedding for text: {text[:100]}...")
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list and cache
            embedding_list = embedding.tolist()
            self._embedding_cache[text_hash] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dim
    
    def batch_generate(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding lists
        """
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        cached_count = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            texts_to_process = []
            indices_to_process = []
            
            # Check cache for each text in batch
            for j, text in enumerate(batch_texts):
                if not text.strip():
                    batch_embeddings.append([0.0] * self.embedding_dim)
                    continue
                
                text_hash = self._hash_text(text)
                if text_hash in self._embedding_cache:
                    batch_embeddings.append(self._embedding_cache[text_hash])
                    cached_count += 1
                else:
                    batch_embeddings.append(None)  # Placeholder
                    texts_to_process.append(text)
                    indices_to_process.append(j)
            
            # Process uncached texts
            if texts_to_process:
                try:
                    logger.debug(f"Processing batch {i//batch_size + 1}: {len(texts_to_process)} new embeddings")
                    new_embeddings = self.model.encode(texts_to_process, convert_to_tensor=False)
                    
                    # Update cache and results
                    for k, (text, embedding) in enumerate(zip(texts_to_process, new_embeddings)):
                        embedding_list = embedding.tolist()
                        text_hash = self._hash_text(text)
                        self._embedding_cache[text_hash] = embedding_list
                        
                        # Update batch results
                        batch_index = indices_to_process[k]
                        batch_embeddings[batch_index] = embedding_list
                        
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    # Fill with zero embeddings
                    for k in indices_to_process:
                        if batch_embeddings[k] is None:
                            batch_embeddings[k] = [0.0] * self.embedding_dim
            
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated {len(embeddings)} embeddings ({cached_count} from cache)")
        return embeddings
    
    def embed_convention_rule(self, rule: ConventionRule) -> EmbeddedConvention:
        """
        Generate embedding for a convention rule.
        
        Args:
            rule: ConventionRule to embed
            
        Returns:
            EmbeddedConvention with generated embedding
        """
        # Combine rule components for embedding
        text_components = [
            rule.title,
            rule.description,
            f"Category: {rule.category}",
            f"Type: {rule.rule_type}"
        ]
        
        # Add examples if available
        if rule.examples:
            text_components.append("Examples: " + " ".join(rule.examples[:3]))  # Limit examples
        
        # Combine all components
        combined_text = " | ".join(filter(None, text_components))
        
        # Generate embedding
        embedding = self.generate_embedding(combined_text)
        
        return EmbeddedConvention(
            rule=rule,
            embedding=embedding,
            embedding_model=self.model_name,
            text_used_for_embedding=combined_text
        )
    
    def embed_code_snippet(self, code: str, language: Optional[str] = None) -> List[float]:
        """
        Generate embedding for code snippet.
        
        Args:
            code: Code snippet to embed
            language: Programming language (optional)
            
        Returns:
            List of embedding values
        """
        # Prepare text for embedding
        text_components = []
        
        if language:
            text_components.append(f"Language: {language}")
        
        text_components.append(f"Code: {code}")
        
        combined_text = " | ".join(text_components)
        return self.generate_embedding(combined_text)
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self._embedding_cache = {}
    
    def save_cache(self) -> None:
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            logger.info(f"Saved {len(self._embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache = {}
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_embeddings': len(self._embedding_cache),
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name
        }