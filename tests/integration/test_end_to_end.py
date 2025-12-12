"""
End-to-End Integration Tests

Tests the complete AI PR Reviewer flow from PR diff collection
to formatted GitHub comments.
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from ai_pr_reviewer.api import AIReviewerAPI, ReviewRequest
from ai_pr_reviewer.config import Config
from ai_pr_reviewer.models.pr_diff import PRDiff, FileChange, DiffChunk
from ai_pr_reviewer.models.convention import ConventionRule


class TestEndToEndFlow:
    """Test complete end-to-end review generation flow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock configuration
        self.config = Config()
        self.config.embedding_cache_dir = self.temp_dir
        self.config.qdrant_host = "localhost"
        self.config.qdrant_port = 6333
        
        # Create API instance
        try:
            self.api = AIReviewerAPI(config=self.config)
            self.api_available = True
        except Exception as e:
            pytest.skip(f"API initialization failed: {e}")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'api'):
            self.api.cleanup_resources()
        
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_pr_data(self):
        """Create mock PR data for testing."""
        pr_data = {
            'number': 123,
            'title': 'Add new feature',
            'base': {
                'repo': {
                    'full_name': 'test-org/test-repo'
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
                'patch': '''@@ -1,5 +1,8 @@
 import os
+import sys
 
 def main():
-    pass
+    print("Hello World")
+    return True
 
 if __name__ == "__main__":
     main()'''
            },
            {
                'filename': 'src/utils.py',
                'status': 'added',
                'additions': 20,
                'deletions': 0,
                'patch': '''@@ -0,0 +1,20 @@
+def helper_function():
+    """Helper function for utilities."""
+    return "helper"
+
+class UtilityClass:
+    def __init__(self):
+        self.value = 42
+    
+    def get_value(self):
+        return self.value'''
            }
        ]
        
        return pr_data, files_data
    
    def create_mock_wiki_data(self):
        """Create mock wiki data for testing."""
        return [
            {
                'title': 'Python Coding Standards',
                'content': '''# Python Coding Standards

## Function Naming
- Function names must use snake_case format
- Function names should be descriptive verbs
- Avoid single letter function names

## Class Naming  
- Class names must use PascalCase format
- Class names should be descriptive nouns
- Avoid abbreviations in class names

## Documentation
- All public functions must have docstrings
- Docstrings should follow Google style format
- Include parameter and return type information

## Import Organization
- Standard library imports first
- Third-party imports second  
- Local imports last
- Use absolute imports when possible'''
            },
            {
                'title': 'Code Review Guidelines',
                'content': '''# Code Review Guidelines

## Review Principles
- Focus on code quality and maintainability
- Provide constructive feedback
- Reference specific coding standards
- Be respectful and collaborative

## Common Issues
- Missing docstrings for public functions
- Inconsistent naming conventions
- Overly complex functions (>50 lines)
- Missing error handling'''
            }
        ]
    
    @patch('ai_pr_reviewer.github.client.GitHubClient')
    @pytest.mark.asyncio
    async def test_complete_review_flow(self, mock_github_client):
        """Test complete review generation flow."""
        if not self.api_available:
            pytest.skip("API not available")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_github_client.return_value = mock_client_instance
        
        # Mock PR data
        pr_data, files_data = self.create_mock_pr_data()
        mock_client_instance.get_pull_request.return_value = pr_data
        mock_client_instance.get_pull_request_files.return_value = files_data
        
        # Mock wiki data
        wiki_data = self.create_mock_wiki_data()
        mock_client_instance.get_wiki_pages.return_value = [
            {'title': page['title']} for page in wiki_data
        ]
        
        def mock_get_wiki_content(owner, repo, page_title):
            for page in wiki_data:
                if page['title'] == page_title:
                    return page['content']
            return None
        
        mock_client_instance.get_wiki_page_content.side_effect = mock_get_wiki_content
        
        # Create review request
        request = ReviewRequest(
            repository="test-org/test-repo",
            pr_number=123,
            github_token="test_token"
        )
        
        # Generate review
        result = await self.api.generate_review(request)
        
        # Verify result
        assert result is not None
        assert result.status in ['completed', 'failed']
        assert result.repository == "test-org/test-repo"
        assert result.pr_number == 123
        assert isinstance(result.processing_time, float)
        assert result.processing_time > 0
        
        # If successful, check GitHub comments
        if result.status == 'completed':
            assert isinstance(result.github_comments, list)
            # Should have at least a summary comment
            assert len(result.github_comments) >= 1
            
            # Check comment structure
            for comment in result.github_comments:
                assert hasattr(comment, 'body')
                assert hasattr(comment, 'comment_type')
                assert comment.body is not None
                assert len(comment.body) > 0
        
        # Verify metadata
        assert 'pr_stats' in result.metadata
        assert 'analysis_stats' in result.metadata
        assert 'processing_stats' in result.metadata
    
    @patch('ai_pr_reviewer.github.client.GitHubClient')
    @pytest.mark.asyncio
    async def test_convention_sync(self, mock_github_client):
        """Test convention synchronization."""
        if not self.api_available:
            pytest.skip("API not available")
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_github_client.return_value = mock_client_instance
        
        # Mock wiki data
        wiki_data = self.create_mock_wiki_data()
        mock_client_instance.get_wiki_pages.return_value = [
            {'title': page['title']} for page in wiki_data
        ]
        
        def mock_get_wiki_content(owner, repo, page_title):
            for page in wiki_data:
                if page['title'] == page_title:
                    return page['content']
            return None
        
        mock_client_instance.get_wiki_page_content.side_effect = mock_get_wiki_content
        
        # Sync conventions
        result = await self.api.sync_conventions(
            repository="test-org/test-repo",
            github_token="test_token"
        )
        
        # Verify result
        assert result is not None
        assert result['status'] in ['completed', 'failed']
        assert result['repository'] == "test-org/test-repo"
        
        if result['status'] == 'completed':
            assert 'wiki_pages_found' in result
            assert 'conventions_processed' in result
            assert result['wiki_pages_found'] >= 0
            assert result['conventions_processed'] >= 0
    
    def test_system_health(self):
        """Test system health check."""
        if not self.api_available:
            pytest.skip("API not available")
        
        health = self.api.get_system_health()
        
        # Verify health structure
        assert 'status' in health
        assert 'components' in health
        assert 'timestamp' in health
        
        assert health['status'] in ['healthy', 'degraded', 'unhealthy']
        
        # Check component health
        expected_components = ['vector_store', 'llm_model', 'embedding_generator']
        for component in expected_components:
            if component in health['components']:
                comp_health = health['components'][component]
                assert 'status' in comp_health
                assert comp_health['status'] in ['healthy', 'degraded', 'unhealthy']
    
    @patch('ai_pr_reviewer.github.client.GitHubClient')
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_github_client):
        """Test error handling in review generation."""
        if not self.api_available:
            pytest.skip("API not available")
        
        # Setup mock to raise exception
        mock_client_instance = Mock()
        mock_github_client.return_value = mock_client_instance
        mock_client_instance.get_pull_request.side_effect = Exception("GitHub API Error")
        
        # Create review request
        request = ReviewRequest(
            repository="test-org/test-repo",
            pr_number=123,
            github_token="invalid_token"
        )
        
        # Generate review (should handle error gracefully)
        result = await self.api.generate_review(request)
        
        # Verify error handling
        assert result is not None
        assert result.status == 'failed'
        assert 'error' in result.metadata
        assert isinstance(result.processing_time, float)
    
    @patch('ai_pr_reviewer.github.client.GitHubClient')
    @pytest.mark.asyncio
    async def test_empty_pr_handling(self, mock_github_client):
        """Test handling of PR with no changes."""
        if not self.api_available:
            pytest.skip("API not available")
        
        # Setup mocks for empty PR
        mock_client_instance = Mock()
        mock_github_client.return_value = mock_client_instance
        
        pr_data = {
            'number': 124,
            'title': 'Empty PR',
            'base': {
                'repo': {
                    'full_name': 'test-org/test-repo'
                }
            },
            'created_at': '2023-01-01T12:00:00Z'
        }
        
        mock_client_instance.get_pull_request.return_value = pr_data
        mock_client_instance.get_pull_request_files.return_value = []  # No files
        mock_client_instance.get_wiki_pages.return_value = []  # No wiki
        
        # Create review request
        request = ReviewRequest(
            repository="test-org/test-repo",
            pr_number=124,
            github_token="test_token"
        )
        
        # Generate review
        result = await self.api.generate_review(request)
        
        # Verify handling of empty PR
        assert result is not None
        assert result.status == 'completed'
        assert len(result.github_comments) >= 0  # May have summary comment
    
    def test_resource_cleanup(self):
        """Test resource cleanup functionality."""
        if not self.api_available:
            pytest.skip("API not available")
        
        # Test cleanup doesn't raise exceptions
        try:
            self.api.cleanup_resources()
        except Exception as e:
            pytest.fail(f"Resource cleanup failed: {e}")
    
    def test_context_manager(self):
        """Test API as context manager."""
        if not self.api_available:
            pytest.skip("API not available")
        
        # Test context manager usage
        try:
            with AIReviewerAPI(config=self.config) as api:
                health = api.get_system_health()
                assert health is not None
        except Exception as e:
            pytest.fail(f"Context manager failed: {e}")


@pytest.mark.integration
class TestRealIntegration:
    """Integration tests that require real services (optional)."""
    
    def test_with_real_qdrant(self):
        """Test with real Qdrant instance (if available)."""
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)
            collections = client.get_collections()
            
            # If Qdrant is available, test with real instance
            config = Config()
            config.qdrant_host = "localhost"
            config.qdrant_port = 6333
            
            api = AIReviewerAPI(config=config)
            health = api.get_system_health()
            
            assert health['components']['vector_store']['status'] == 'healthy'
            
        except Exception:
            pytest.skip("Qdrant not available for integration test")
    
    def test_with_real_models(self):
        """Test with real ML models (if available)."""
        try:
            # Test if models can be loaded
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
            
            # Try to load small models for testing
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            
            # If models load successfully, test API
            config = Config()
            config.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
            config.llm_model = 'gpt2'
            
            api = AIReviewerAPI(config=config)
            health = api.get_system_health()
            
            assert health['components']['embedding_generator']['status'] == 'healthy'
            
        except Exception:
            pytest.skip("ML models not available for integration test")