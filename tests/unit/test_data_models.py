"""
Unit tests for data models and configuration.

Tests: Requirements 5.1, 5.2, 5.3
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
import tempfile
import json
from pathlib import Path

from ai_pr_reviewer.models.pr_diff import PRDiff, FileChange, DiffChunk
from ai_pr_reviewer.models.convention import ConventionRule, EmbeddedConvention
from ai_pr_reviewer.models.review import ReviewComment, GitHubComment
from ai_pr_reviewer.config import Config


class TestPRDiffModels:
    """Unit tests for PR diff data models."""

    def test_diff_chunk_creation(self):
        """Test DiffChunk model creation and validation."""
        chunk = DiffChunk(
            start_line=10,
            end_line=15,
            added_lines=["+ new line 1", "+ new line 2"],
            removed_lines=["- old line 1"],
            context_lines=[" context line 1", " context line 2"]
        )
        
        assert chunk.start_line == 10
        assert chunk.end_line == 15
        assert len(chunk.added_lines) == 2
        assert len(chunk.removed_lines) == 1
        assert len(chunk.context_lines) == 2

    def test_diff_chunk_validation(self):
        """Test DiffChunk validation rules."""
        # Test invalid line numbers
        with pytest.raises(ValidationError):
            DiffChunk(
                start_line=-1,  # Invalid negative line number
                end_line=10,
                added_lines=[],
                removed_lines=[],
                context_lines=[]
            )
        
        # Test end_line before start_line
        with pytest.raises(ValidationError):
            DiffChunk(
                start_line=15,
                end_line=10,  # End before start
                added_lines=[],
                removed_lines=[],
                context_lines=[]
            )

    def test_file_change_creation(self):
        """Test FileChange model creation."""
        chunk = DiffChunk(
            start_line=1,
            end_line=5,
            added_lines=["+ added"],
            removed_lines=["- removed"],
            context_lines=[" context"]
        )
        
        file_change = FileChange(
            filename="test.py",
            status="modified",
            additions=10,
            deletions=5,
            diff_chunks=[chunk]
        )
        
        assert file_change.filename == "test.py"
        assert file_change.status == "modified"
        assert file_change.additions == 10
        assert file_change.deletions == 5
        assert len(file_change.diff_chunks) == 1

    def test_file_change_status_validation(self):
        """Test FileChange status validation."""
        # Valid statuses
        for status in ["added", "modified", "removed", "renamed"]:
            file_change = FileChange(
                filename="test.py",
                status=status,
                additions=0,
                deletions=0,
                diff_chunks=[]
            )
            assert file_change.status == status
        
        # Invalid status
        with pytest.raises(ValidationError):
            FileChange(
                filename="test.py",
                status="invalid_status",
                additions=0,
                deletions=0,
                diff_chunks=[]
            )

    def test_pr_diff_creation(self):
        """Test PRDiff model creation."""
        file_change = FileChange(
            filename="test.py",
            status="modified",
            additions=5,
            deletions=2,
            diff_chunks=[]
        )
        
        pr_diff = PRDiff(
            pr_number=123,
            title="Test PR",
            description="Test description",
            author="testuser",
            base_branch="main",
            head_branch="feature",
            file_changes=[file_change]
        )
        
        assert pr_diff.pr_number == 123
        assert pr_diff.title == "Test PR"
        assert pr_diff.author == "testuser"
        assert len(pr_diff.file_changes) == 1

    def test_pr_diff_serialization(self):
        """Test PRDiff serialization and deserialization."""
        chunk = DiffChunk(
            start_line=1,
            end_line=3,
            added_lines=["+ line"],
            removed_lines=[],
            context_lines=[]
        )
        
        file_change = FileChange(
            filename="test.py",
            status="added",
            additions=1,
            deletions=0,
            diff_chunks=[chunk]
        )
        
        pr_diff = PRDiff(
            pr_number=456,
            title="Serialization Test",
            description="Testing serialization",
            author="testuser",
            base_branch="main",
            head_branch="feature",
            file_changes=[file_change]
        )
        
        # Serialize to dict
        pr_dict = pr_diff.model_dump()
        assert isinstance(pr_dict, dict)
        assert pr_dict['pr_number'] == 456
        
        # Deserialize from dict
        restored_pr = PRDiff.model_validate(pr_dict)
        assert restored_pr.pr_number == pr_diff.pr_number
        assert restored_pr.title == pr_diff.title
        assert len(restored_pr.file_changes) == len(pr_diff.file_changes)


class TestConventionModels:
    """Unit tests for convention data models."""

    def test_convention_rule_creation(self):
        """Test ConventionRule model creation."""
        rule = ConventionRule(
            title="Naming Convention",
            description="Use descriptive variable names",
            category="naming",
            source_file="conventions.md",
            line_number=42
        )
        
        assert rule.title == "Naming Convention"
        assert rule.description == "Use descriptive variable names"
        assert rule.category == "naming"
        assert rule.source_file == "conventions.md"
        assert rule.line_number == 42

    def test_convention_rule_validation(self):
        """Test ConventionRule validation."""
        # Valid categories
        valid_categories = ["naming", "structure", "testing", "documentation", "performance", "security", "general"]
        
        for category in valid_categories:
            rule = ConventionRule(
                title="Test Rule",
                description="Test description",
                category=category,
                source_file="test.md",
                line_number=1
            )
            assert rule.category == category
        
        # Invalid category
        with pytest.raises(ValidationError):
            ConventionRule(
                title="Test Rule",
                description="Test description",
                category="invalid_category",
                source_file="test.md",
                line_number=1
            )

    def test_convention_rule_optional_fields(self):
        """Test ConventionRule optional fields."""
        # With all optional fields
        rule = ConventionRule(
            title="Complete Rule",
            description="Complete description",
            category="general",
            source_file="test.md",
            line_number=1,
            examples=["example 1", "example 2"],
            tags=["tag1", "tag2"],
            priority="high"
        )
        
        assert len(rule.examples) == 2
        assert len(rule.tags) == 2
        assert rule.priority == "high"
        
        # Without optional fields
        minimal_rule = ConventionRule(
            title="Minimal Rule",
            description="Minimal description",
            category="general",
            source_file="test.md",
            line_number=1
        )
        
        assert minimal_rule.examples == []
        assert minimal_rule.tags == []
        assert minimal_rule.priority == "medium"

    def test_embedded_convention_creation(self):
        """Test EmbeddedConvention model creation."""
        rule = ConventionRule(
            title="Test Rule",
            description="Test description",
            category="general",
            source_file="test.md",
            line_number=1
        )
        
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        embedded_convention = EmbeddedConvention(
            convention=rule,
            embedding=embedding,
            embedding_model="test-model",
            created_at=datetime.now()
        )
        
        assert embedded_convention.convention == rule
        assert embedded_convention.embedding == embedding
        assert embedded_convention.embedding_model == "test-model"
        assert isinstance(embedded_convention.created_at, datetime)

    def test_embedded_convention_validation(self):
        """Test EmbeddedConvention validation."""
        rule = ConventionRule(
            title="Test Rule",
            description="Test description",
            category="general",
            source_file="test.md",
            line_number=1
        )
        
        # Valid embedding
        valid_embedding = [0.1] * 384  # Typical embedding size
        embedded_convention = EmbeddedConvention(
            convention=rule,
            embedding=valid_embedding,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert len(embedded_convention.embedding) == 384
        
        # Empty embedding should fail
        with pytest.raises(ValidationError):
            EmbeddedConvention(
                convention=rule,
                embedding=[],  # Empty embedding
                embedding_model="test-model"
            )


class TestReviewModels:
    """Unit tests for review data models."""

    def test_review_comment_creation(self):
        """Test ReviewComment model creation."""
        comment = ReviewComment(
            file_path="src/test.py",
            line_number=25,
            comment_text="Consider using a more descriptive variable name",
            suggestion="Use 'user_count' instead of 'n'",
            convention_reference="naming-conventions.md#descriptive-names",
            severity="medium"
        )
        
        assert comment.file_path == "src/test.py"
        assert comment.line_number == 25
        assert comment.severity == "medium"
        assert comment.convention_reference is not None

    def test_review_comment_validation(self):
        """Test ReviewComment validation."""
        # Valid severities
        for severity in ["low", "medium", "high", "critical"]:
            comment = ReviewComment(
                file_path="test.py",
                line_number=1,
                comment_text="Test comment",
                severity=severity
            )
            assert comment.severity == severity
        
        # Invalid severity
        with pytest.raises(ValidationError):
            ReviewComment(
                file_path="test.py",
                line_number=1,
                comment_text="Test comment",
                severity="invalid_severity"
            )

    def test_github_comment_creation(self):
        """Test GitHubComment model creation."""
        review_comment = ReviewComment(
            file_path="test.py",
            line_number=10,
            comment_text="Test comment",
            severity="medium"
        )
        
        github_comment = GitHubComment(
            review_comment=review_comment,
            github_file_path="test.py",
            github_line_number=10,
            comment_body="**Medium Priority**: Test comment\n\nSuggestion: Consider improvement",
            position=5
        )
        
        assert github_comment.review_comment == review_comment
        assert github_comment.github_file_path == "test.py"
        assert github_comment.github_line_number == 10
        assert github_comment.position == 5

    def test_github_comment_formatting(self):
        """Test GitHubComment body formatting."""
        review_comment = ReviewComment(
            file_path="src/utils.py",
            line_number=42,
            comment_text="함수명이 명확하지 않습니다",
            suggestion="더 구체적인 함수명을 사용해보세요",
            convention_reference="naming.md#function-names",
            severity="high"
        )
        
        github_comment = GitHubComment(
            review_comment=review_comment,
            github_file_path="src/utils.py",
            github_line_number=42,
            comment_body="**높은 우선순위**: 함수명이 명확하지 않습니다\n\n💡 **제안**: 더 구체적인 함수명을 사용해보세요\n\n📖 **참고**: naming.md#function-names",
            position=15
        )
        
        assert "높은 우선순위" in github_comment.comment_body
        assert "제안" in github_comment.comment_body
        assert "참고" in github_comment.comment_body


class TestConfiguration:
    """Unit tests for configuration management."""

    def test_config_creation(self):
        """Test Config model creation with default values."""
        config = Config()
        
        # Test default values
        assert config.github_token is None
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.llm_model == "microsoft/DialoGPT-medium"

    def test_config_validation(self):
        """Test Config validation."""
        # Valid configuration
        config = Config(
            github_token="ghp_test_token_123",
            qdrant_host="localhost",
            qdrant_port=6333,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            llm_model="microsoft/DialoGPT-medium",
            max_tokens=2048,
            temperature=0.7
        )
        
        assert config.github_token == "ghp_test_token_123"
        assert config.max_tokens == 2048
        assert config.temperature == 0.7

    def test_config_from_file(self):
        """Test Config loading from file."""
        config_data = {
            "github_token": "test_token",
            "qdrant_host": "remote-qdrant.com",
            "qdrant_port": 6334,
            "embedding_model": "custom-model",
            "max_tokens": 4096,
            "temperature": 0.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = Config.from_file(config_file)
            
            assert config.github_token == "test_token"
            assert config.qdrant_host == "remote-qdrant.com"
            assert config.qdrant_port == 6334
            assert config.embedding_model == "custom-model"
            assert config.max_tokens == 4096
            assert config.temperature == 0.5
            
        finally:
            Path(config_file).unlink()

    def test_config_environment_variables(self):
        """Test Config loading from environment variables."""
        import os
        
        # Set environment variables
        env_vars = {
            'GITHUB_TOKEN': 'env_test_token',
            'QDRANT_HOST': 'env-qdrant.com',
            'QDRANT_PORT': '6335',
            'EMBEDDING_MODEL': 'env-embedding-model',
            'LLM_MODEL': 'env-llm-model'
        }
        
        # Backup original values
        original_values = {}
        for key in env_vars:
            original_values[key] = os.environ.get(key)
            os.environ[key] = env_vars[key]
        
        try:
            config = Config.from_env()
            
            assert config.github_token == 'env_test_token'
            assert config.qdrant_host == 'env-qdrant.com'
            assert config.qdrant_port == 6335
            assert config.embedding_model == 'env-embedding-model'
            assert config.llm_model == 'env-llm-model'
            
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_config_serialization(self):
        """Test Config serialization and deserialization."""
        config = Config(
            github_token="test_token_123",
            qdrant_host="test-host.com",
            qdrant_port=6333,
            embedding_model="test-embedding-model",
            llm_model="test-llm-model",
            max_tokens=1024,
            temperature=0.8
        )
        
        # Serialize to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict['github_token'] == "test_token_123"
        assert config_dict['qdrant_host'] == "test-host.com"
        
        # Deserialize from dict
        restored_config = Config.model_validate(config_dict)
        assert restored_config.github_token == config.github_token
        assert restored_config.qdrant_host == config.qdrant_host
        assert restored_config.max_tokens == config.max_tokens

    def test_config_validation_errors(self):
        """Test Config validation error handling."""
        # Invalid port number
        with pytest.raises(ValidationError):
            Config(qdrant_port=-1)
        
        with pytest.raises(ValidationError):
            Config(qdrant_port=70000)  # Port too high
        
        # Invalid temperature
        with pytest.raises(ValidationError):
            Config(temperature=-1.0)  # Negative temperature
        
        with pytest.raises(ValidationError):
            Config(temperature=2.0)  # Temperature too high
        
        # Invalid max_tokens
        with pytest.raises(ValidationError):
            Config(max_tokens=0)  # Zero tokens
        
        with pytest.raises(ValidationError):
            Config(max_tokens=-100)  # Negative tokens