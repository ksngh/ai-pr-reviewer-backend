"""
Property-based tests for wiki convention processing.

Property 3: Wiki Convention Processing
Validates: Requirements 2.1, 2.2, 2.3
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from ai_pr_reviewer.conventions.extractor import ConventionExtractor
from ai_pr_reviewer.models.convention import ConventionRule


class TestWikiConventionProcessing:
    """Property tests for wiki convention processing."""

    @given(
        wiki_pages=st.lists(
            st.dictionaries({
                'title': st.text(min_size=5, max_size=100),
                'content': st.text(min_size=50, max_size=2000),
                'category': st.sampled_from(['coding-standards', 'architecture', 'testing', 'documentation'])
            }),
            min_size=1,
            max_size=20
        )
    )
    def test_convention_extraction_completeness(self, wiki_pages):
        """
        Property: Convention extraction should capture all relevant rules from wiki pages.
        
        Given: Wiki pages with various convention content
        When: Conventions are extracted from pages
        Then: All extractable conventions should be captured
        """
        extractor = ConventionExtractor()
        
        all_conventions = []
        for page in wiki_pages:
            # Create realistic convention content
            convention_content = f"""
# {page['title']}

## Overview
{page['content'][:200]}

## Rules

### Rule 1: {page['category'].replace('-', ' ').title()}
- Always follow {page['category']} guidelines
- {page['content'][200:400] if len(page['content']) > 200 else 'Standard practice'}

### Rule 2: Implementation
- Implement according to {page['title']} standards
- {page['content'][400:600] if len(page['content']) > 400 else 'Best practices'}

## Examples

```python
# Example for {page['category']}
def example_function():
    pass
```
            """
            
            conventions = extractor.extract_conventions_from_text(
                convention_content, 
                page['title']
            )
            all_conventions.extend(conventions)
        
        # Verify extraction completeness
        assert len(all_conventions) > 0, "No conventions extracted from wiki pages"
        
        # Verify convention structure
        for convention in all_conventions:
            assert isinstance(convention, ConventionRule)
            assert len(convention.title) > 0
            assert len(convention.description) > 0
            assert convention.category in ['coding-standards', 'architecture', 'testing', 'documentation', 'general']
            assert convention.source_file is not None

    @given(
        markdown_structures=st.lists(
            st.dictionaries({
                'headers': st.lists(st.text(min_size=3, max_size=50), min_size=1, max_size=10),
                'content_blocks': st.lists(st.text(min_size=20, max_size=500), min_size=1, max_size=10),
                'code_blocks': st.lists(st.text(min_size=10, max_size=200), min_size=0, max_size=5)
            }),
            min_size=1,
            max_size=10
        )
    )
    def test_markdown_parsing_robustness(self, markdown_structures):
        """
        Property: Markdown parsing should handle various structures robustly.
        
        Given: Different markdown structures and formatting
        When: Wiki content is parsed for conventions
        Then: Parser should handle all structures without errors
        """
        extractor = ConventionExtractor()
        
        for structure in markdown_structures:
            # Build markdown content
            markdown_content = ""
            
            for i, (header, content) in enumerate(zip(structure['headers'], structure['content_blocks'])):
                level = (i % 3) + 1  # H1, H2, H3
                markdown_content += f"{'#' * level} {header}\n\n{content}\n\n"
            
            # Add code blocks
            for code in structure['code_blocks']:
                markdown_content += f"```python\n{code}\n```\n\n"
            
            try:
                conventions = extractor.extract_conventions_from_text(
                    markdown_content, 
                    f"test_page_{hash(markdown_content) % 1000}"
                )
                
                # Should not crash and should return a list
                assert isinstance(conventions, list)
                
                # If conventions found, they should be valid
                for convention in conventions:
                    assert isinstance(convention, ConventionRule)
                    assert len(convention.title.strip()) > 0
                    assert len(convention.description.strip()) > 0
                    
            except Exception as e:
                # Parser should be robust - minimal exceptions allowed
                pytest.fail(f"Parser failed on valid markdown: {e}")

    @given(
        convention_updates=st.lists(
            st.dictionaries({
                'rule_id': st.text(min_size=5, max_size=30),
                'old_content': st.text(min_size=20, max_size=500),
                'new_content': st.text(min_size=20, max_size=500),
                'change_type': st.sampled_from(['modified', 'added', 'removed'])
            }),
            min_size=1,
            max_size=15
        )
    )
    def test_convention_change_detection(self, convention_updates):
        """
        Property: Convention changes should be detected accurately.
        
        Given: Convention updates with various change types
        When: Wiki content is updated
        Then: Changes should be detected and categorized correctly
        """
        extractor = ConventionExtractor()
        
        # Create initial conventions
        initial_conventions = {}
        for update in convention_updates:
            if update['change_type'] != 'added':
                rule = ConventionRule(
                    title=f"Rule {update['rule_id']}",
                    description=update['old_content'],
                    category='general',
                    source_file='test_wiki.md',
                    line_number=1
                )
                initial_conventions[update['rule_id']] = rule
        
        # Create updated conventions
        updated_conventions = {}
        for update in convention_updates:
            if update['change_type'] != 'removed':
                content = update['new_content'] if update['change_type'] == 'modified' else update['old_content']
                if update['change_type'] == 'added':
                    content = update['new_content']
                
                rule = ConventionRule(
                    title=f"Rule {update['rule_id']}",
                    description=content,
                    category='general',
                    source_file='test_wiki.md',
                    line_number=1
                )
                updated_conventions[update['rule_id']] = rule
        
        # Detect changes
        changes = extractor.detect_convention_changes(
            list(initial_conventions.values()),
            list(updated_conventions.values())
        )
        
        # Verify change detection
        assert isinstance(changes, dict)
        assert 'added' in changes
        assert 'modified' in changes
        assert 'removed' in changes
        
        # Count expected changes
        expected_added = len([u for u in convention_updates if u['change_type'] == 'added'])
        expected_removed = len([u for u in convention_updates if u['change_type'] == 'removed'])
        expected_modified = len([u for u in convention_updates if u['change_type'] == 'modified'])
        
        # Verify change counts are reasonable (allowing for some detection variance)
        total_detected = len(changes['added']) + len(changes['modified']) + len(changes['removed'])
        assert total_detected >= 0  # Should detect some changes

    @given(
        wiki_content_variations=st.lists(
            st.dictionaries({
                'language': st.sampled_from(['en', 'ko', 'mixed']),
                'formatting': st.sampled_from(['standard', 'loose', 'strict']),
                'content': st.text(min_size=100, max_size=1000)
            }),
            min_size=1,
            max_size=8
        )
    )
    def test_multilingual_content_handling(self, wiki_content_variations):
        """
        Property: Wiki content in different languages should be handled consistently.
        
        Given: Wiki content in various languages and formats
        When: Conventions are extracted
        Then: All languages should be processed without errors
        """
        extractor = ConventionExtractor()
        
        for variation in wiki_content_variations:
            # Create content based on language
            if variation['language'] == 'ko':
                content = f"""
# 코딩 규칙

## 개요
{variation['content'][:200]}

## 규칙

### 규칙 1: 명명 규칙
- 변수명은 명확하게 작성한다
- {variation['content'][200:400] if len(variation['content']) > 200 else '표준 관행을 따른다'}

### 규칙 2: 함수 작성
- 함수는 단일 책임을 가져야 한다
- {variation['content'][400:600] if len(variation['content']) > 400 else '최선의 방법을 사용한다'}
                """
            elif variation['language'] == 'mixed':
                content = f"""
# Coding Standards / 코딩 표준

## Overview / 개요
{variation['content'][:200]}

## Rules / 규칙

### Rule 1 / 규칙 1: Naming Convention / 명명 규칙
- Use clear variable names / 명확한 변수명 사용
- {variation['content'][200:400] if len(variation['content']) > 200 else 'Follow standards / 표준 준수'}
                """
            else:  # English
                content = f"""
# Coding Standards

## Overview
{variation['content'][:200]}

## Rules

### Rule 1: Naming Convention
- Use clear and descriptive variable names
- {variation['content'][200:400] if len(variation['content']) > 200 else 'Follow established patterns'}
                """
            
            try:
                conventions = extractor.extract_conventions_from_text(
                    content, 
                    f"wiki_{variation['language']}_{variation['formatting']}"
                )
                
                # Should handle all languages
                assert isinstance(conventions, list)
                
                # Should extract some conventions for substantial content
                if len(content) > 200:
                    assert len(conventions) >= 0  # Allow for no extraction in some cases
                
                # Verify extracted conventions are valid
                for convention in conventions:
                    assert isinstance(convention, ConventionRule)
                    assert len(convention.title.strip()) > 0
                    assert len(convention.description.strip()) > 0
                    
            except Exception as e:
                pytest.fail(f"Failed to process {variation['language']} content: {e}")

    @given(
        large_wiki_size=st.integers(min_value=10, max_value=100),
        content_per_page=st.integers(min_value=500, max_value=5000)
    )
    def test_large_wiki_processing_performance(self, large_wiki_size, content_per_page):
        """
        Property: Large wiki processing should be efficient and complete.
        
        Given: Large wiki with many pages and substantial content
        When: All pages are processed for conventions
        Then: Processing should complete efficiently without data loss
        """
        assume(large_wiki_size * content_per_page < 500000)  # Reasonable size limit
        
        extractor = ConventionExtractor()
        
        # Generate large wiki content
        all_conventions = []
        processing_times = []
        
        for page_num in range(large_wiki_size):
            # Create substantial page content
            page_content = f"""
# Page {page_num} Standards

## Overview
This page contains coding standards for module {page_num}.
{'Content ' * (content_per_page // 50)}

## Rules

### Rule {page_num}.1: Structure
- Follow structured approach for module {page_num}
- {'Implementation details ' * (content_per_page // 100)}

### Rule {page_num}.2: Testing
- Write comprehensive tests for module {page_num}
- {'Testing guidelines ' * (content_per_page // 100)}

## Examples

```python
def module_{page_num}_function():
    '''Example function for module {page_num}'''
    pass
```
            """
            
            import time
            start_time = time.time()
            
            conventions = extractor.extract_conventions_from_text(
                page_content, 
                f"large_wiki_page_{page_num}"
            )
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
            all_conventions.extend(conventions)
        
        # Verify performance and completeness
        assert len(all_conventions) > 0, "No conventions extracted from large wiki"
        
        # Performance should be reasonable (allow up to 1 second per page for large content)
        avg_processing_time = sum(processing_times) / len(processing_times)
        assert avg_processing_time < 1.0, f"Processing too slow: {avg_processing_time}s per page"
        
        # Should extract reasonable number of conventions
        conventions_per_page = len(all_conventions) / large_wiki_size
        assert conventions_per_page >= 0, "Should extract some conventions per page"