#!/usr/bin/env python3
"""
Convention Processing Demo

Demonstrates the complete convention processing pipeline:
1. Extract rules from wiki content
2. Generate embeddings
3. Store in vector database
4. Perform similarity search

Usage:
    python examples/convention_processing_demo.py

Requirements:
    - Qdrant running on localhost:6333
    - sentence-transformers installed
"""

import sys
import os
import logging
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_pr_reviewer.conventions.extractor import ConventionExtractor
from ai_pr_reviewer.conventions.embeddings import EmbeddingGenerator
from ai_pr_reviewer.conventions.vector_store import VectorStore, VectorStoreError
from ai_pr_reviewer.models.convention import ConventionRule


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_sample_wiki_content() -> str:
    """Get sample wiki content for demonstration."""
    return """
# Coding Conventions

## Variable Naming Rules
- Variable names must use camelCase format
- Avoid single letter variables except for loop counters (i, j, k)
- Use descriptive names that clearly explain the variable's purpose
- Boolean variables should start with 'is', 'has', or 'can'

```python
# Good examples
userName = "john_doe"
isLoggedIn = True
itemCount = len(items)

# Bad examples
n = "john_doe"  # Too short, unclear
user_name = "john_doe"  # Wrong case format
loggedIn = True  # Missing 'is' prefix for boolean
```

## Function Design Principles
- Functions should have a single responsibility
- Function names must be verbs that describe what they do
- Functions should not exceed 50 lines of code
- Use meaningful parameter names

```python
# Good example
def calculateTotalPrice(items: List[Item]) -> float:
    return sum(item.price for item in items)

# Bad example
def calc(x):  # Unclear name and parameter
    total = 0
    for i in x:
        total += i.price
    return total
```

## Class Organization
- Class names must use PascalCase
- Classes should follow the Single Responsibility Principle
- Use composition over inheritance when possible
- Document all public methods

## Error Handling
- Always handle exceptions explicitly
- Use specific exception types, avoid bare except clauses
- Log errors with appropriate context
- Never ignore exceptions silently

```python
# Good example
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise

# Bad example
try:
    result = risky_operation()
except:  # Too broad
    pass  # Silent failure
```

## Testing Requirements
- All public functions must have unit tests
- Test coverage should be at least 80%
- Use descriptive test names that explain what is being tested
- Mock external dependencies in unit tests

## Documentation Standards
- All modules must have docstrings
- Public functions and classes must be documented
- Use type hints for all function parameters and return values
- Keep documentation up to date with code changes
"""


def demonstrate_rule_extraction():
    """Demonstrate rule extraction from wiki content."""
    print("🔍 Step 1: Extracting Convention Rules")
    print("=" * 50)
    
    extractor = ConventionExtractor()
    wiki_content = get_sample_wiki_content()
    
    rules = extractor.extract_rules(wiki_content, "coding-conventions")
    
    print(f"✓ Extracted {len(rules)} convention rules")
    print()
    
    # Show rule details
    for i, rule in enumerate(rules[:5], 1):  # Show first 5 rules
        print(f"Rule {i}: {rule.title}")
        print(f"   Type: {rule.rule_type}")
        print(f"   Category: {rule.category}")
        print(f"   Examples: {len(rule.examples)}")
        print(f"   Counter-examples: {len(rule.counter_examples)}")
        print()
    
    if len(rules) > 5:
        print(f"... and {len(rules) - 5} more rules")
        print()
    
    return rules


def demonstrate_embedding_generation(rules: List[ConventionRule]):
    """Demonstrate embedding generation for rules."""
    print("🧠 Step 2: Generating Embeddings")
    print("=" * 50)
    
    try:
        generator = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print(f"✓ Loaded embedding model: {generator.model_name}")
        print(f"✓ Embedding dimension: {generator.embedding_dim}")
        
        # Generate embeddings for all rules
        embedded_rules = []
        for rule in rules:
            embedded_rule = generator.embed_convention_rule(rule)
            embedded_rules.append(embedded_rule)
        
        print(f"✓ Generated embeddings for {len(embedded_rules)} rules")
        
        # Demonstrate similarity computation
        if len(embedded_rules) >= 2:
            emb1 = embedded_rules[0].embedding
            emb2 = embedded_rules[1].embedding
            similarity = generator.compute_similarity(emb1, emb2)
            
            print(f"✓ Sample similarity score: {similarity:.3f}")
            print(f"   Between: '{embedded_rules[0].rule.title}'")
            print(f"   And: '{embedded_rules[1].rule.title}'")
        
        print()
        return embedded_rules, generator
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install required packages: pip install sentence-transformers torch")
        return [], None


def demonstrate_vector_storage(embedded_rules, generator):
    """Demonstrate vector storage and search."""
    print("🗄️  Step 3: Vector Storage and Search")
    print("=" * 50)
    
    try:
        # Initialize vector store
        vector_store = VectorStore(
            host="localhost",
            port=6333,
            collection_name="demo_conventions",
            embedding_dim=generator.embedding_dim
        )
        
        print("✓ Connected to Qdrant vector database")
        
        # Clear existing data
        vector_store.clear_collection()
        print("✓ Cleared existing data")
        
        # Store conventions
        point_ids = vector_store.batch_store_conventions(embedded_rules)
        print(f"✓ Stored {len(point_ids)} conventions")
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        print(f"✓ Collection stats: {stats['points_count']} points")
        
        # Demonstrate similarity search
        print("\n🔎 Similarity Search Examples:")
        print("-" * 30)
        
        search_queries = [
            "variable naming conventions",
            "function design best practices",
            "error handling guidelines",
            "testing requirements"
        ]
        
        for query in search_queries:
            print(f"\nQuery: '{query}'")
            
            # Generate query embedding
            query_embedding = generator.generate_embedding(query)
            
            # Search for similar conventions
            results = vector_store.search_similar_conventions(
                query_embedding=query_embedding,
                limit=3,
                score_threshold=0.3
            )
            
            if results:
                for i, (rule, score) in enumerate(results, 1):
                    print(f"  {i}. {rule.title} (score: {score:.3f})")
                    print(f"     Category: {rule.category}, Type: {rule.rule_type}")
            else:
                print("  No similar conventions found")
        
        print()
        return vector_store
        
    except VectorStoreError as e:
        print(f"❌ Vector store error: {e}")
        print("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def demonstrate_category_filtering(vector_store, generator):
    """Demonstrate filtering by category."""
    if not vector_store or not generator:
        return
    
    print("🏷️  Step 4: Category-based Filtering")
    print("=" * 50)
    
    # Search with category filter
    query = "best practices for code quality"
    query_embedding = generator.generate_embedding(query)
    
    categories = ["naming", "testing", "documentation"]
    
    for category in categories:
        print(f"\nSearching in category: '{category}'")
        
        results = vector_store.search_similar_conventions(
            query_embedding=query_embedding,
            limit=2,
            score_threshold=0.1,
            filters={"category": category}
        )
        
        if results:
            for rule, score in results:
                print(f"  • {rule.title} (score: {score:.3f})")
        else:
            print(f"  No conventions found in '{category}' category")
    
    print()


def main():
    """Main demo function."""
    setup_logging()
    
    print("🚀 Convention Processing Demo")
    print("=" * 50)
    print("This demo shows the complete convention processing pipeline:")
    print("1. Extract rules from wiki content")
    print("2. Generate semantic embeddings")
    print("3. Store in vector database")
    print("4. Perform similarity search")
    print()
    
    try:
        # Step 1: Extract rules
        rules = demonstrate_rule_extraction()
        
        if not rules:
            print("❌ No rules extracted. Exiting.")
            return
        
        # Step 2: Generate embeddings
        embedded_rules, generator = demonstrate_embedding_generation(rules)
        
        if not embedded_rules:
            print("❌ Failed to generate embeddings. Exiting.")
            return
        
        # Step 3: Vector storage and search
        vector_store = demonstrate_vector_storage(embedded_rules, generator)
        
        # Step 4: Category filtering (optional)
        demonstrate_category_filtering(vector_store, generator)
        
        print("✅ Demo completed successfully!")
        print("\nNext steps:")
        print("- Try modifying the wiki content to see different rules")
        print("- Experiment with different search queries")
        print("- Add your own convention rules")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()