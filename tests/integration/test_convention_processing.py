"""
Integration tests for Convention Processing Layer.

Tests the complete flow from wiki content extraction to vector storage.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from ai_pr_reviewer.conventions.extractor import ConventionExtractor
from ai_pr_reviewer.conventions.embeddings import EmbeddingGenerator
from ai_pr_reviewer.conventions.vector_store import VectorStore, VectorStoreError
from ai_pr_reviewer.models.convention import ConventionRule


class TestConventionExtractor:
    """Test convention rule extraction from wiki content."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ConventionExtractor()
    
    def test_extract_rules_from_simple_wiki(self):
        """Test extracting rules from simple wiki content."""
        wiki_content = """
# Naming Conventions

## Variables
- Variable names must use camelCase
- Avoid single letter variables except for loop counters
- Use descriptive names that explain the purpose

## Functions
- Function names should be verbs that describe what they do
- Use camelCase for function names
- Functions must not exceed 50 lines

## Classes
- Class names must use PascalCase
- Classes should have a single responsibility
"""
        
        rules = self.extractor.extract_rules(wiki_content, "naming-conventions")
        
        assert len(rules) >= 6  # Should extract multiple rules
        
        # Check rule types are determined correctly
        mandatory_rules = [r for r in rules if r.rule_type == 'mandatory']
        assert len(mandatory_rules) >= 2  # "must" rules
        
        # Check categories are assigned
        categories = {r.category for r in rules}
        assert 'naming' in categories
        
        # Check source page is set
        for rule in rules:
            assert rule.source_wiki_page == "naming-conventions"
    
    def test_extract_rules_with_code_examples(self):
        """Test extracting rules with code examples."""
        wiki_content = """
# Code Style

## Good Example
Use meaningful variable names:

```python
user_count = len(users)
total_price = sum(item.price for item in cart)
```

## Bad Example
Avoid unclear names:

```python
n = len(users)  # Bad: unclear what n represents
x = sum(item.price for item in cart)  # Bad: x is meaningless
```
"""
        
        rules = self.extractor.extract_rules(wiki_content, "code-style")
        
        # Should extract rules and categorize examples
        assert len(rules) >= 1
        
        # Check that examples are extracted
        rules_with_examples = [r for r in rules if r.examples or r.counter_examples]
        assert len(rules_with_examples) >= 1
    
    def test_extract_korean_rules(self):
        """Test extracting rules from Korean wiki content."""
        wiki_content = """
# 코딩 컨벤션

## 변수명 규칙
- 변수명은 반드시 camelCase를 사용해야 합니다
- 한 글자 변수는 피해야 합니다
- 의미있는 이름을 사용하는 것이 좋습니다

## 함수 규칙
- 함수명은 동사로 시작하는 것이 권장됩니다
- 함수는 50줄을 넘지 말아야 합니다
"""
        
        rules = self.extractor.extract_rules(wiki_content, "korean-conventions")
        
        assert len(rules) >= 4
        
        # Check Korean rule type detection
        mandatory_rules = [r for r in rules if r.rule_type == 'mandatory']
        recommended_rules = [r for r in rules if r.rule_type == 'recommended']
        prohibited_rules = [r for r in rules if r.rule_type == 'prohibited']
        
        assert len(mandatory_rules) >= 1  # "반드시", "말아야"
        assert len(recommended_rules) >= 1  # "좋습니다", "권장"
        assert len(prohibited_rules) >= 1  # "피해야"


class TestEmbeddingGenerator:
    """Test embedding generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a smaller model for testing
        self.generator = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=tempfile.mkdtemp()
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'generator') and self.generator.cache_dir.exists():
            shutil.rmtree(self.generator.cache_dir)
    
    def test_generate_single_embedding(self):
        """Test generating embedding for single text."""
        text = "Variable names must use camelCase"
        embedding = self.generator.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == self.generator.embedding_dim
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_generate_embeddings(self):
        """Test batch embedding generation."""
        texts = [
            "Use camelCase for variables",
            "Functions should be verbs",
            "Classes use PascalCase",
            "Avoid single letter variables"
        ]
        
        embeddings = self.generator.batch_generate(texts)
        
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert len(embedding) == self.generator.embedding_dim
    
    def test_embed_convention_rule(self):
        """Test embedding a convention rule."""
        rule = ConventionRule(
            id="test_rule_001",
            title="Use camelCase for variables",
            description="Variable names must use camelCase format",
            rule_type="mandatory",
            examples=["userName = 'john'", "itemCount = 5"],
            counter_examples=["user_name = 'john'", "item_count = 5"],
            category="naming",
            source_wiki_page="naming-conventions",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1
        )
        
        embedded_rule = self.generator.embed_convention_rule(rule)
        
        assert embedded_rule.rule == rule
        assert len(embedded_rule.embedding) == self.generator.embedding_dim
        assert embedded_rule.embedding_model == self.generator.model_name
        assert embedded_rule.text_used_for_embedding is not None
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        text1 = "Use camelCase for variables"
        text2 = "Variable names should use camelCase"
        text3 = "Functions should return values"
        
        emb1 = self.generator.generate_embedding(text1)
        emb2 = self.generator.generate_embedding(text2)
        emb3 = self.generator.generate_embedding(text3)
        
        # Similar texts should have higher similarity
        sim_12 = self.generator.compute_similarity(emb1, emb2)
        sim_13 = self.generator.compute_similarity(emb1, emb3)
        
        assert sim_12 > sim_13
        assert -1 <= sim_12 <= 1
        assert -1 <= sim_13 <= 1
    
    def test_caching_functionality(self):
        """Test embedding caching."""
        text = "Test caching functionality"
        
        # Generate embedding twice
        emb1 = self.generator.generate_embedding(text)
        emb2 = self.generator.generate_embedding(text)
        
        # Should be identical (from cache)
        assert emb1 == emb2
        
        # Check cache stats
        stats = self.generator.get_cache_stats()
        assert stats['cached_embeddings'] >= 1


@pytest.mark.integration
class TestVectorStore:
    """Test vector store functionality with Qdrant."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use test collection name
        self.collection_name = f"test_conventions_{int(datetime.now().timestamp())}"
        
        try:
            self.vector_store = VectorStore(
                host="localhost",
                port=6333,
                collection_name=self.collection_name,
                embedding_dim=384
            )
        except Exception as e:
            pytest.skip(f"Qdrant not available: {e}")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'vector_store'):
            try:
                self.vector_store.clear_collection()
            except Exception:
                pass  # Ignore cleanup errors
    
    def test_store_and_retrieve_convention(self):
        """Test storing and retrieving a convention."""
        # Create test rule and embedding
        rule = ConventionRule(
            id="test_rule_001",
            title="Use camelCase for variables",
            description="Variable names must use camelCase format",
            rule_type="mandatory",
            examples=["userName = 'john'"],
            counter_examples=["user_name = 'john'"],
            category="naming",
            source_wiki_page="naming-conventions",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1
        )
        
        # Create dummy embedding
        embedding = [0.1] * 384
        
        from ai_pr_reviewer.models.convention import EmbeddedConvention
        embedded_rule = EmbeddedConvention(
            rule=rule,
            embedding=embedding,
            embedding_model="test-model",
            text_used_for_embedding="test text"
        )
        
        # Store convention
        point_id = self.vector_store.store_convention(embedded_rule)
        assert point_id is not None
        
        # Retrieve by ID
        retrieved_rule = self.vector_store.get_convention_by_id(rule.id)
        assert retrieved_rule is not None
        assert retrieved_rule.id == rule.id
        assert retrieved_rule.title == rule.title
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Create and store multiple test rules
        rules_data = [
            ("Use camelCase", "naming", [0.1, 0.2] + [0.0] * 382),
            ("Use PascalCase", "naming", [0.2, 0.1] + [0.0] * 382),
            ("Write unit tests", "testing", [0.8, 0.9] + [0.0] * 382),
        ]
        
        from ai_pr_reviewer.models.convention import EmbeddedConvention
        
        for i, (title, category, embedding) in enumerate(rules_data):
            rule = ConventionRule(
                id=f"test_rule_{i:03d}",
                title=title,
                description=f"Description for {title}",
                rule_type="recommended",
                examples=[],
                counter_examples=[],
                category=category,
                source_wiki_page="test-page",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version=1
            )
            
            embedded_rule = EmbeddedConvention(
                rule=rule,
                embedding=embedding,
                embedding_model="test-model",
                text_used_for_embedding=title
            )
            
            self.vector_store.store_convention(embedded_rule)
        
        # Search for similar conventions
        query_embedding = [0.15, 0.15] + [0.0] * 382  # Should be similar to naming rules
        
        results = self.vector_store.search_similar_conventions(
            query_embedding=query_embedding,
            limit=5,
            score_threshold=0.0
        )
        
        assert len(results) >= 2  # Should find at least the naming rules
        
        # Results should be sorted by similarity
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_collection_stats(self):
        """Test getting collection statistics."""
        stats = self.vector_store.get_collection_stats()
        
        assert 'collection_name' in stats
        assert 'points_count' in stats
        assert stats['collection_name'] == self.collection_name


@pytest.mark.integration
class TestConventionProcessingFlow:
    """Test the complete convention processing flow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = ConventionExtractor()
        
        # Use smaller model for testing
        self.generator = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=tempfile.mkdtemp()
        )
        
        # Skip vector store tests if Qdrant not available
        try:
            self.vector_store = VectorStore(
                collection_name=f"test_flow_{int(datetime.now().timestamp())}",
                embedding_dim=self.generator.embedding_dim
            )
            self.vector_store_available = True
        except Exception:
            self.vector_store_available = False
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'generator') and self.generator.cache_dir.exists():
            shutil.rmtree(self.generator.cache_dir)
        
        if hasattr(self, 'vector_store') and self.vector_store_available:
            try:
                self.vector_store.clear_collection()
            except Exception:
                pass
    
    def test_complete_processing_flow(self):
        """Test the complete flow from wiki to vector storage."""
        wiki_content = """
# Coding Standards

## Variable Naming
- Variables must use camelCase format
- Avoid single letter variable names
- Use descriptive names that explain purpose

## Function Design
- Functions should do one thing well
- Function names must be verbs
- Keep functions under 50 lines

```python
# Good example
def calculateTotalPrice(items):
    return sum(item.price for item in items)

# Bad example
def calc(x):
    return sum(i.price for i in x)
```
"""
        
        # Step 1: Extract rules
        rules = self.extractor.extract_rules(wiki_content, "coding-standards")
        assert len(rules) >= 4
        
        # Step 2: Generate embeddings
        embedded_rules = []
        for rule in rules:
            embedded_rule = self.generator.embed_convention_rule(rule)
            embedded_rules.append(embedded_rule)
        
        assert len(embedded_rules) == len(rules)
        
        # Step 3: Store in vector database (if available)
        if self.vector_store_available:
            point_ids = self.vector_store.batch_store_conventions(embedded_rules)
            assert len(point_ids) == len(embedded_rules)
            
            # Step 4: Test similarity search
            query_text = "variable naming conventions"
            query_embedding = self.generator.generate_embedding(query_text)
            
            similar_rules = self.vector_store.search_similar_conventions(
                query_embedding=query_embedding,
                limit=3,
                score_threshold=0.1
            )
            
            assert len(similar_rules) >= 1
            
            # Should find naming-related rules
            naming_rules = [rule for rule, score in similar_rules if 'variable' in rule.title.lower() or 'naming' in rule.category.lower()]
            assert len(naming_rules) >= 1