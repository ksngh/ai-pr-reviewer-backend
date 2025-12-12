"""
Integration tests for GitHub API client and parser.

These tests verify the GitHub integration layer works correctly
with real API responses (mocked for testing).
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from ai_pr_reviewer.github.client import GitHubClient, GitHubAPIError, RateLimitExceeded
from ai_pr_reviewer.github.parser import PRDiffParser
from ai_pr_reviewer.models.pr_diff import PRDiff, FileChange, DiffChunk


class TestGitHubClient:
    """Test GitHub API client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = GitHubClient("test_token")
        self.parser = PRDiffParser()
    
    @patch('requests.Session.request')
    def test_authentication_success(self, mock_request):
        """Test successful GitHub authentication."""
        # Mock successful response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'login': 'testuser',
            'id': 12345,
            'type': 'User'
        }
        mock_response.headers = {'X-RateLimit-Remaining': '4999'}
        mock_request.return_value = mock_response
        
        success, user_info = self.client.test_authentication()
        
        assert success is True
        assert user_info['login'] == 'testuser'
        assert user_info['id'] == 12345
    
    @patch('requests.Session.request')
    def test_authentication_failure(self, mock_request):
        """Test GitHub authentication failure."""
        # Mock failed response
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.json.return_value = {
            'message': 'Bad credentials'
        }
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        with pytest.raises(GitHubAPIError) as exc_info:
            self.client.test_authentication()
        
        assert exc_info.value.status_code == 401
        assert 'Bad credentials' in str(exc_info.value)
    
    @patch('requests.Session.request')
    def test_rate_limit_handling(self, mock_request):
        """Test rate limit handling."""
        # Mock rate limit exceeded response
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.headers = {
            'X-RateLimit-Reset': str(int(datetime.now().timestamp()) + 3600)
        }
        mock_request.return_value = mock_response
        
        with pytest.raises(RateLimitExceeded):
            self.client.get_repository_info('owner', 'repo')
    
    @patch('requests.Session.request')
    def test_get_pull_request_files(self, mock_request):
        """Test fetching PR files."""
        # Mock PR files response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'filename': 'src/main.py',
                'status': 'modified',
                'additions': 10,
                'deletions': 5,
                'patch': '@@ -1,3 +1,4 @@\n import os\n+import sys\n def main():\n     pass'
            },
            {
                'filename': 'tests/test_main.py',
                'status': 'added',
                'additions': 20,
                'deletions': 0,
                'patch': '@@ -0,0 +1,20 @@\n+import pytest\n+\n+def test_main():\n+    assert True'
            }
        ]
        mock_response.headers = {'X-RateLimit-Remaining': '4998'}
        mock_request.return_value = mock_response
        
        files = self.client.get_pull_request_files('owner', 'repo', 123)
        
        assert len(files) == 2
        assert files[0]['filename'] == 'src/main.py'
        assert files[0]['status'] == 'modified'
        assert files[1]['filename'] == 'tests/test_main.py'
        assert files[1]['status'] == 'added'


class TestPRDiffParser:
    """Test PR diff parser functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PRDiffParser()
    
    def test_parse_pr_diff(self):
        """Test parsing complete PR diff."""
        pr_data = {
            'number': 123,
            'base': {
                'repo': {
                    'full_name': 'owner/repo'
                }
            },
            'created_at': '2023-01-01T12:00:00Z'
        }
        
        files_data = [
            {
                'filename': 'src/main.py',
                'status': 'modified',
                'additions': 10,
                'deletions': 5,
                'patch': '@@ -1,3 +1,4 @@\n import os\n+import sys\n def main():\n     pass'
            }
        ]
        
        pr_diff = self.parser.parse_pr_diff(pr_data, files_data)
        
        assert isinstance(pr_diff, PRDiff)
        assert pr_diff.repository == 'owner/repo'
        assert pr_diff.pr_number == 123
        assert len(pr_diff.files_changed) == 1
        assert pr_diff.total_additions == 10
        assert pr_diff.total_deletions == 5
    
    def test_parse_file_change(self):
        """Test parsing individual file change."""
        file_data = {
            'filename': 'src/utils.py',
            'status': 'modified',
            'additions': 5,
            'deletions': 2,
            'patch': '@@ -10,4 +10,7 @@\n def helper():\n-    return None\n+    return True\n+\n+def new_helper():\n+    return False'
        }
        
        file_change = self.parser._parse_file_change(file_data)
        
        assert isinstance(file_change, FileChange)
        assert file_change.file_path == 'src/utils.py'
        assert file_change.change_type == 'modified'
        assert file_change.additions == 5
        assert file_change.deletions == 2
        assert len(file_change.chunks) == 1
    
    def test_parse_diff_chunks(self):
        """Test parsing diff chunks."""
        patch = '''@@ -1,3 +1,4 @@
 import os
+import sys
 def main():
     pass
@@ -10,2 +11,3 @@
 def helper():
-    return None
+    return True
+    print("done")'''
        
        chunks = self.parser._parse_diff_chunks(patch)
        
        assert len(chunks) == 2
        
        # First chunk
        assert chunks[0].old_start == 1
        assert chunks[0].old_lines == 3
        assert chunks[0].new_start == 1
        assert chunks[0].new_lines == 4
        
        # Second chunk
        assert chunks[1].old_start == 10
        assert chunks[1].old_lines == 2
        assert chunks[1].new_start == 11
        assert chunks[1].new_lines == 3
    
    def test_extract_changed_lines(self):
        """Test extracting added and removed lines."""
        chunk = DiffChunk(
            old_start=1,
            old_lines=3,
            new_start=1,
            new_lines=4,
            content=' import os\n+import sys\n def main():\n-    pass\n+    return True',
            context_lines=['import os', 'def main():']
        )
        
        added_lines, removed_lines = self.parser.extract_changed_lines(chunk)
        
        assert added_lines == ['import sys', '    return True']
        assert removed_lines == ['    pass']
    
    def test_is_code_file(self):
        """Test code file detection."""
        assert self.parser.is_code_file('src/main.py') is True
        assert self.parser.is_code_file('components/Button.tsx') is True
        assert self.parser.is_code_file('styles/main.css') is True
        assert self.parser.is_code_file('README.md') is False
        assert self.parser.is_code_file('image.png') is False
        assert self.parser.is_code_file('data.json') is True
    
    def test_filter_relevant_files(self):
        """Test filtering relevant files for review."""
        pr_diff = PRDiff(
            repository='owner/repo',
            pr_number=123,
            files_changed=[
                FileChange(
                    file_path='src/main.py',
                    change_type='modified',
                    additions=5,
                    deletions=2,
                    chunks=[]
                ),
                FileChange(
                    file_path='README.md',
                    change_type='modified',
                    additions=1,
                    deletions=0,
                    chunks=[]
                ),
                FileChange(
                    file_path='tests/test_main.py',
                    change_type='added',
                    additions=10,
                    deletions=0,
                    chunks=[]
                ),
                FileChange(
                    file_path='old_file.py',
                    change_type='deleted',
                    additions=0,
                    deletions=20,
                    chunks=[]
                )
            ],
            total_additions=16,
            total_deletions=22,
            created_at=datetime.now()
        )
        
        # Filter with tests included
        relevant_with_tests = self.parser.filter_relevant_files(pr_diff, include_tests=True)
        assert len(relevant_with_tests) == 2  # main.py and test_main.py
        
        # Filter without tests
        relevant_without_tests = self.parser.filter_relevant_files(pr_diff, include_tests=False)
        assert len(relevant_without_tests) == 1  # only main.py