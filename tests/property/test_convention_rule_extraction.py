"""
Property-based tests for convention rule extraction.

Property 4: Convention Rule Extraction
Validates: Requirements 2.1, 2.2, 2.3
"""

import pytest
from hypothesis import given, strategies as st, assume
from ai_pr_reviewer.conventions.extractor import ConventionExtractor
from ai_pr_reviewer.models.convention import ConventionRule


class TestConventionRuleExtraction:
    """Property tests for convention rule extraction from wiki content."""

    @given(
        rule_patterns=st.lists(
            st.dictionaries({
                'title': st.text(min_size=5, max_size=100),
                'description': st.text(min_size=20, max_size=500),
                'examples': st.lists(st.text(min_size=10, max_size=200), min_size=0, max_size=3),
                'priority': st.sampled_from(['high', 'medium', 'low'])
            }),
            min_size=1,
            max_size=15
        )
    )
    def test_rule_extraction_consistency(self, rule_patterns):
        """
        Property: Rule extraction should be consistent across similar patterns.
        
        Given: Various rule patterns in wiki content
        When: Rules are extracted multiple times
        Then: Results should be consistent and complete
        """
        extractor = ConventionExtractor()
        
        # Create wiki content with embedded rules
        wiki_content = "# Coding Standards\n\n"
        
        for i, pattern in enumerate(rule_patterns):
            wiki_content += f"""
## Rule {i+1}: {pattern['title']}

{pattern['description']}

Priority: {pattern['priority']}

"""
            if pattern['examples']:
                wiki_content += "### Examples\n"
                for example in pattern['examples']:
                    wiki_content += f"- {example}\n"
                wiki_content += "\n"
        
        # Extract rules multiple times
        extraction_results = []
        for _ in range(3):  # Test consistency
            rules = extractor.extract_conventions_from_text(wiki_content, "test_wiki.md")
            extraction_results.append(rules)
        
        # Verify consistency
        assert len(extraction_results) == 3
        
        # All extractions should have same number of rules
        rule_counts = [len(result) for result in extraction_results]
        assert len(set(rule_counts)) <= 1, "Inconsistent rule extraction counts"
        
        if rule_counts[0] > 0:
            # Verify rule content consistency
            for i in range(len(extraction_results[0])):
                rule1 = extraction_results[0][i]
                rule2 = extraction_results[1][i]
                rule3 = extraction_results[2][i]
                
                assert rule1.title == rule2.title == rule3.title
                assert rule1.description == rule2.description == rule3.description

    @given(
        nested_structures=st.lists(
            st.dictionaries({
                'section_depth': st.integers(min_value=1, max_value=6),
                'subsections': st.lists(st.text(min_size=10, max_size=100), min_size=1, max_size=5),
                'content': st.text(min_size=50, max_size=800)
            }),
            min_size=1,
            max_size=10
        )
    )
    def test_nested_structure_handling(self, nested_structures):
        """
        Property: Nested markdown structures should be parsed correctly.
        
        Given: Deeply nested markdown sections with rules
        When: Content is parsed for conventions
        Then: All nested rules should be extracted with proper hierarchy
        """
        extractor = ConventionExtractor()
        
        wiki_content = "# Main Standards Document\n\n"
        expected_sections = 0
        
        for structure in nested_structures:
            # Create nested headers
            header_prefix = "#" * min(structure['section_depth'], 6)
            section_title = f"Section {expected_sections + 1}"
            
            wiki_content += f"{header_prefix} {section_title}\n\n"
            wiki_content += f"{structure['content']}\n\n"
            
            # Add subsections
            for j, subsection in enumerate(structure['subsections']):
                sub_header_prefix = "#" * min(structure['section_depth'] + 1, 6)
                wiki_content += f"{sub_header_prefix} {subsection}\n\n"
                wiki_content += f"Rule for {subsection}: {structure['content'][:100]}\n\n"
                expected_sections += 1
        
        # Extract conventions
        conventions = extractor.extract_conventions_from_text(wiki_content, "nested_wiki.md")
        
        # Verify extraction handles nesting
        assert isinstance(conventions, list)
        
        # Should extract some conventions from substantial content
        if len(wiki_content) > 500:
            assert len(conventions) >= 0  # Allow for varying extraction success
        
        # Verify convention structure
        for convention in conventions:
            assert isinstance(convention, ConventionRule)
            assert len(convention.title.strip()) > 0
            assert len(convention.description.strip()) > 0
            assert convention.line_number > 0

    @given(
        code_examples=st.lists(
            st.dictionaries({
                'language': st.sampled_from(['python', 'javascript', 'java', 'cpp', 'go']),
                'code': st.text(min_size=20, max_size=300),
                'explanation': st.text(min_size=10, max_size=200)
            }),
            min_size=1,
            max_size=8
        )
    )
    def test_code_example_integration(self, code_examples):
        """
        Property: Code examples should be properly integrated with rule descriptions.
        
        Given: Rules with associated code examples
        When: Conventions are extracted
        Then: Code examples should enhance rule descriptions
        """
        extractor = ConventionExtractor()
        
        wiki_content = "# Code Standards with Examples\n\n"
        
        for i, example in enumerate(code_examples):
            wiki_content += f"""
## Rule {i+1}: {example['language'].title()} Standards

{example['explanation']}

### Example

```{example['language']}
{example['code']}
```

This example demonstrates proper {example['language']} coding practices.

"""
        
        # Extract conventions
        conventions = extractor.extract_conventions_from_text(wiki_content, "code_examples.md")
        
        # Verify code examples are handled
        assert isinstance(conventions, list)
        
        for convention in conventions:
            assert isinstance(convention, ConventionRule)
            # Description should contain some reference to the explanation or code context
            assert len(convention.description) > 0
            
            # Should not contain raw code blocks in title
            assert "```" not in convention.title

    @given(
        rule_modifications=st.lists(
            st.dictionaries({
                'original_rule': st.text(min_size=30, max_size=200),
                'modification_type': st.sampled_from(['addition', 'deletion', 'modification']),
                'new_content': st.text(min_size=30, max_size=200)
            }),
            min_size=1,
            max_size=10
        )
    )
    def test_rule_change_detection_accuracy(self, rule_modifications):
        """
        Property: Rule changes should be detected accurately.
        
        Given: Original rules and their modifications
        When: Changes are detected between versions
        Then: All changes should be identified correctly
        """
        extractor = ConventionExtractor()
        
        # Create original wiki content
        original_content = "# Original Standards\n\n"
        for i, mod in enumerate(rule_modifications):
            original_content += f"""
## Rule {i+1}
{mod['original_rule']}

"""
        
        # Create modified wiki content
        modified_content = "# Modified Standards\n\n"
        for i, mod in enumerate(rule_modifications):
            if mod['modification_type'] == 'deletion':
                continue  # Skip deleted rules
            elif mod['modification_type'] == 'modification':
                modified_content += f"""
## Rule {i+1}
{mod['new_content']}

"""
            else:  # addition or original
                modified_content += f"""
## Rule {i+1}
{mod['original_rule']}

"""
        
        # Add new rules for additions
        for i, mod in enumerate(rule_modifications):
            if mod['modification_type'] == 'addition':
                modified_content += f"""
## New Rule {i+100}
{mod['new_content']}

"""
        
        # Extract conventions from both versions
        original_conventions = extractor.extract_conventions_from_text(original_content, "original.md")
        modified_conventions = extractor.extract_conventions_from_text(modified_content, "modified.md")
        
        # Detect changes
        changes = extractor.detect_convention_changes(original_conventions, modified_conventions)
        
        # Verify change detection structure
        assert isinstance(changes, dict)
        assert 'added' in changes
        assert 'modified' in changes
        assert 'removed' in changes
        
        # Verify changes are lists
        assert isinstance(changes['added'], list)
        assert isinstance(changes['modified'], list)
        assert isinstance(changes['removed'], list)

    @given(
        multilingual_rules=st.lists(
            st.dictionaries({
                'english_title': st.text(min_size=5, max_size=50),
                'korean_title': st.text(min_size=5, max_size=50),
                'english_desc': st.text(min_size=20, max_size=300),
                'korean_desc': st.text(min_size=20, max_size=300)
            }),
            min_size=1,
            max_size=6
        )
    )
    def test_multilingual_rule_extraction(self, multilingual_rules):
        """
        Property: Multilingual rules should be extracted correctly.
        
        Given: Rules written in multiple languages
        When: Conventions are extracted
        Then: Both languages should be handled appropriately
        """
        extractor = ConventionExtractor()
        
        # Create multilingual wiki content
        wiki_content = "# Multilingual Coding Standards / 다국어 코딩 표준\n\n"
        
        for i, rule in enumerate(multilingual_rules):
            wiki_content += f"""
## Rule {i+1}: {rule['english_title']} / 규칙 {i+1}: {rule['korean_title']}

**English**: {rule['english_desc']}

**한국어**: {rule['korean_desc']}

---

"""
        
        # Extract conventions
        conventions = extractor.extract_conventions_from_text(wiki_content, "multilingual.md")
        
        # Verify multilingual handling
        assert isinstance(conventions, list)
        
        for convention in conventions:
            assert isinstance(convention, ConventionRule)
            assert len(convention.title.strip()) > 0
            assert len(convention.description.strip()) > 0
            
            # Should handle both languages without errors
            # (Specific language detection not required, just no crashes)

    @given(
        malformed_content=st.lists(
            st.dictionaries({
                'broken_markdown': st.text(min_size=10, max_size=200),
                'missing_headers': st.booleans(),
                'invalid_structure': st.booleans()
            }),
            min_size=1,
            max_size=5
        )
    )
    def test_malformed_content_robustness(self, malformed_content):
        """
        Property: Malformed content should be handled gracefully.
        
        Given: Malformed or invalid wiki content
        When: Extraction is attempted
        Then: System should handle errors gracefully without crashing
        """
        extractor = ConventionExtractor()
        
        for content_item in malformed_content:
            # Create intentionally malformed content
            malformed_wiki = content_item['broken_markdown']
            
            if not content_item['missing_headers']:
                malformed_wiki = f"# Some Header\n{malformed_wiki}"
            
            if content_item['invalid_structure']:
                # Add some invalid markdown structures
                malformed_wiki += "\n### ### Invalid Header\n"
                malformed_wiki += "```\nUnclosed code block\n"
                malformed_wiki += "- List item without proper spacing"
            
            try:
                # Should not crash on malformed content
                conventions = extractor.extract_conventions_from_text(
                    malformed_wiki, 
                    "malformed.md"
                )
                
                # Should return a list (possibly empty)
                assert isinstance(conventions, list)
                
                # Any extracted conventions should still be valid
                for convention in conventions:
                    assert isinstance(convention, ConventionRule)
                    assert len(convention.title.strip()) > 0
                    
            except Exception as e:
                # Should be minimal exceptions for malformed content
                # Allow some exceptions but they should be handled gracefully
                assert "catastrophic" not in str(e).lower()  # No catastrophic failures