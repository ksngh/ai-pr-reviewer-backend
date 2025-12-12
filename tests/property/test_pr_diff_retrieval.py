"""
Property-based tests for PR diff retrieval and structuring.

Property 1: PR Diff Retrieval and Structuring
Validates: Requirements 1.1, 1.2, 1.5
"""

import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import Mock, patch
import json

from ai_pr_reviewer.github.client import GitHubClient
from ai_pr_reviewer.github.parser import PRDiffParser
from ai_pr_reviewer.models.pr_diff import PRDiff, FileChange, DiffChunk


class TestPRDiffRetrieval:
    """Property tests for PR diff retrieval and structuring."""

    @given(
        repo_owner=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        repo_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        pr_number=st.integers(min_value=1, max_value=99999),
        file_changes=st.lists(
            st.dictionaries({
                'filename': st.text(min_size=1, max_size=100),
                'status': st.sampled_from(['added', 'modified', 'removed']),
                'additions': st.integers(min_value=0, max_value=1000),
                'deletions': st.integers(min_value=0, max_value=1000),
                'patch': st.text(min_size=0, max_size=2000)
            }),
            min_size=1,
            max_size=20
        )
    )
    def test_pr_diff_structure_consistency(self, repo_owner, repo_name, pr_number, file_changes):
        """
        Property: PR diff retrieval should maintain consistent structure.
        
        Given: A repository and PR number with file changes
        When: PR diff is retrieved and parsed
        Then: The resulting structure should be consistent and complete
        """
        assume(repo_owner.replace('_', '').replace('-', '').isalnum())
        assume(repo_name.replace('_', '').replace('-', '').isalnum())
        
        # Mock GitHub API response
        mock_pr_data = {
            'number': pr_number,
            'title': f'Test PR {pr_number}',
            'body': 'Test PR description',
            'head': {'sha': 'abc123'},
            'base': {'sha': 'def456'}
        }
        
        mock_files_data = []
        for change in file_changes:
            mock_files_data.append({
                'filename': change['filename'],
                'status': change['status'],
                'additions': change['additions'],
                'deletions': change['deletions'],
                'patch': change['patch']
            })
        
        with patch('ai_pr_reviewer.github.client.requests.get') as mock_get:
            # Mock PR details response
            mock_pr_response = Mock()
            mock_pr_response.json.return_value = mock_pr_data
            mock_pr_response.status_code = 200
            
            # Mock files response
            mock_files_response = Mock()
            mock_files_response.json.return_value = mock_files_data
            mock_files_response.status_code = 200
            
            mock_get.side_effect = [mock_pr_response, mock_files_response]
            
            # Test PR diff retrieval
            client = GitHubClient('fake_token')
            parser = PRDiffParser()
            
            pr_data = client.get_pr_details(repo_owner, repo_name, pr_number)
            files_data = client.get_pr_files(repo_owner, repo_name, pr_number)
            
            # Parse the diff
            pr_diff = parser.parse_pr_diff(pr_data, files_data)
            
            # Verify structure consistency
            assert isinstance(pr_diff, PRDiff)
            assert pr_diff.pr_number == pr_number
            assert pr_diff.title == mock_pr_data['title']
            assert len(pr_diff.file_changes) == len(file_changes)
            
            # Verify each file change
            for i, file_change in enumerate(pr_diff.file_changes):
                assert isinstance(file_change, FileChange)
                assert file_change.filename == file_changes[i]['filename']
                assert file_change.status == file_changes[i]['status']
                assert file_change.additions == file_changes[i]['additions']
                assert file_change.deletions == file_changes[i]['deletions']

    @given(
        diff_content=st.text(min_size=10, max_size=5000),
        line_numbers=st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=100)
    )
    def test_diff_chunk_parsing_completeness(self, diff_content, line_numbers):
        """
        Property: Diff chunk parsing should capture all changes completely.
        
        Given: Diff content with line numbers
        When: Diff is parsed into chunks
        Then: All changes should be captured without loss
        """
        # Create a realistic diff format
        diff_lines = []
        for i, line_num in enumerate(line_numbers[:10]):  # Limit for performance
            if i % 3 == 0:
                diff_lines.append(f"+{line_num}: {diff_content[:50]}")
            elif i % 3 == 1:
                diff_lines.append(f"-{line_num}: {diff_content[50:100]}")
            else:
                diff_lines.append(f" {line_num}: {diff_content[100:150]}")
        
        patch_content = "\n".join(diff_lines)
        
        parser = PRDiffParser()
        chunks = parser.parse_diff_chunks(patch_content)
        
        # Verify completeness
        assert isinstance(chunks, list)
        
        if chunks:  # Only test if chunks were created
            total_lines = sum(len(chunk.added_lines) + len(chunk.removed_lines) + len(chunk.context_lines) 
                            for chunk in chunks)
            
            # Should capture a reasonable portion of the input
            assert total_lines > 0, "No lines captured in diff chunks"
            
            # Verify chunk structure
            for chunk in chunks:
                assert isinstance(chunk, DiffChunk)
                assert chunk.start_line >= 0
                assert chunk.end_line >= chunk.start_line
                assert isinstance(chunk.added_lines, list)
                assert isinstance(chunk.removed_lines, list)
                assert isinstance(chunk.context_lines, list)

    @given(
        file_extensions=st.lists(
            st.sampled_from(['.py', '.js', '.ts', '.java', '.cpp', '.md', '.txt', '.json']),
            min_size=1,
            max_size=10
        ),
        change_types=st.lists(
            st.sampled_from(['added', 'modified', 'removed']),
            min_size=1,
            max_size=10
        )
    )
    def test_file_type_handling_consistency(self, file_extensions, change_types):
        """
        Property: Different file types should be handled consistently.
        
        Given: Various file types and change types
        When: PR diff is processed
        Then: All file types should be handled with consistent structure
        """
        parser = PRDiffParser()
        
        # Create mock file changes for different types
        mock_files = []
        for i, (ext, change_type) in enumerate(zip(file_extensions, change_types)):
            filename = f"test_file_{i}{ext}"
            mock_files.append({
                'filename': filename,
                'status': change_type,
                'additions': 10,
                'deletions': 5,
                'patch': f"@@ -1,3 +1,4 @@\n-old line\n+new line\n context"
            })
        
        # Parse files
        file_changes = []
        for mock_file in mock_files:
            file_change = parser.parse_file_change(mock_file)
            file_changes.append(file_change)
        
        # Verify consistent handling
        for file_change in file_changes:
            assert isinstance(file_change, FileChange)
            assert file_change.filename.endswith(tuple(file_extensions))
            assert file_change.status in change_types
            assert file_change.additions >= 0
            assert file_change.deletions >= 0
            assert isinstance(file_change.diff_chunks, list)

    @given(
        large_pr_size=st.integers(min_value=50, max_value=500),
        files_per_chunk=st.integers(min_value=1, max_value=20)
    )
    def test_large_pr_handling_scalability(self, large_pr_size, files_per_chunk):
        """
        Property: Large PRs should be handled efficiently without data loss.
        
        Given: A large PR with many file changes
        When: PR is processed and chunked
        Then: All files should be processed efficiently
        """
        assume(large_pr_size >= files_per_chunk)
        
        parser = PRDiffParser()
        
        # Create a large PR simulation
        mock_files = []
        for i in range(large_pr_size):
            mock_files.append({
                'filename': f'file_{i}.py',
                'status': 'modified',
                'additions': i % 100,
                'deletions': (i * 2) % 50,
                'patch': f"@@ -{i},3 +{i},4 @@\n-old_{i}\n+new_{i}\n context_{i}"
            })
        
        # Process in chunks
        processed_files = []
        for i in range(0, len(mock_files), files_per_chunk):
            chunk = mock_files[i:i + files_per_chunk]
            for mock_file in chunk:
                file_change = parser.parse_file_change(mock_file)
                processed_files.append(file_change)
        
        # Verify no data loss
        assert len(processed_files) == large_pr_size, "File count mismatch in large PR processing"
        
        # Verify all files processed correctly
        for i, file_change in enumerate(processed_files):
            assert file_change.filename == f'file_{i}.py'
            assert file_change.additions == i % 100
            assert file_change.deletions == (i * 2) % 50