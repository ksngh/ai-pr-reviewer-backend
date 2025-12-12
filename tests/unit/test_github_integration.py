"""
Unit tests for GitHub Integration Layer.

Tests: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException, HTTPError
import time

from ai_pr_reviewer.github.client import GitHubClient
from ai_pr_reviewer.github.parser import PRDiffParser
from ai_pr_reviewer.models.pr_diff import PRDiff, FileChange, DiffChunk


class TestGitHubClient:
    """Unit tests for GitHubClient class."""

    def test_client_initialization(self):
        """Test GitHubClient initialization."""
        token = "ghp_test_token_123456789"
        client = GitHubClient(token)
        
        assert client.token == token
        assert client.base_url == "https://api.github.com"
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == f"token {token}"
        assert client.headers["Accept"] == "application/vnd.github.v3+json"

    def test_client_initialization_validation(self):
        """Test GitHubClient initialization validation."""
        # Valid token
        valid_token = "ghp_" + "a" * 36  # GitHub token format
        client = GitHubClient(valid_token)
        assert client.token == valid_token
        
        # Invalid token (too short)
        with pytest.raises(ValueError):
            GitHubClient("short_token")
        
        # Empty token
        with pytest.raises(ValueError):
            GitHubClient("")
        
        # None token
        with pytest.raises(ValueError):
            GitHubClient(None)

    @patch('ai_pr_reviewer.github.client.requests.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": "success"}
        mock_response.headers = {
            "X-RateLimit-Remaining": "4999",
            "X-RateLimit-Reset": str(int(time.time()) + 3600)
        }
        mock_get.return_value = mock_response
        
        client = GitHubClient("test_token")
        result = client._make_request("/test/endpoint")
        
        assert result == {"message": "success"}
        mock_get.assert_called_once()

    @patch('ai_pr_reviewer.github.client.requests.get')
    def test_make_request_rate_limit_handling(self, mock_get):
        """Test rate limit handling."""
        # Setup rate limit response
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time()) + 3600)
        }
        mock_get.return_value = mock_response
        
        client = GitHubClient("test_token")
        
        # Should raise RateLimitExceeded
        with pytest.raises(Exception):  # Adjust based on actual exception type
            client._make_request("GET", "/test/endpoint")

    @patch('ai_pr_reviewer.github.client.requests.get')
    def test_get_pull_request(self, mock_get):
        """Test getting pull request information."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "number": 123,
            "title": "Test PR",
            "state": "open"
        }
        mock_response.headers = {
            "X-RateLimit-Remaining": "4999",
            "X-RateLimit-Reset": str(int(time.time()) + 3600)
        }
        mock_get.return_value = mock_response
        
        client = GitHubClient("test_token")
        result = client.get_pull_request("owner", "repo", 123)
        
        assert result["number"] == 123
        assert result["title"] == "Test PR"
        assert result["state"] == "open"


class TestPRDiffParser:
    """Unit tests for PRDiffParser class."""

    def test_parser_initialization(self):
        """Test PRDiffParser initialization."""
        parser = PRDiffParser()
        assert parser is not None

    def test_parse_diff_basic(self):
        """Test basic diff parsing."""
        parser = PRDiffParser()
        
        # Mock file data from GitHub API
        file_data = {
            "filename": "test.py",
            "status": "modified",
            "patch": "@@ -1,3 +1,3 @@\n line1\n-old line\n+new line\n line3"
        }
        
        result = parser.parse_file_diff(file_data)
        
        assert result.filename == "test.py"
        assert result.status == "modified"
        assert len(result.diff_chunks) > 0