"""
PR Diff Data Models

Pull Request diff 관련 데이터 모델들
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, validator


@dataclass
class DiffChunk:
    """PR diff의 개별 청크"""
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str
    context_lines: List[str]

    def __post_init__(self):
        """데이터 검증"""
        if self.old_start < 0 or self.new_start < 0:
            raise ValueError("Line numbers must be non-negative")
        if self.old_lines < 0 or self.new_lines < 0:
            raise ValueError("Line counts must be non-negative")


@dataclass
class FileChange:
    """파일 변경사항"""
    file_path: str
    change_type: str  # 'added', 'modified', 'deleted'
    additions: int
    deletions: int
    chunks: List[DiffChunk]

    def __post_init__(self):
        """데이터 검증"""
        valid_types = {'added', 'modified', 'deleted'}
        if self.change_type not in valid_types:
            raise ValueError(f"Invalid change_type: {self.change_type}")
        if self.additions < 0 or self.deletions < 0:
            raise ValueError("Addition and deletion counts must be non-negative")

    @property
    def is_binary(self) -> bool:
        """바이너리 파일 여부 확인"""
        binary_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.exe', '.dll'}
        return any(self.file_path.lower().endswith(ext) for ext in binary_extensions)


@dataclass
class PRDiff:
    """Pull Request diff 전체"""
    repository: str
    pr_number: int
    files_changed: List[FileChange]
    total_additions: int
    total_deletions: int
    created_at: datetime

    def __post_init__(self):
        """데이터 검증"""
        if self.pr_number <= 0:
            raise ValueError("PR number must be positive")
        if '/' not in self.repository:
            raise ValueError("Repository must be in format 'owner/repo'")
        if self.total_additions < 0 or self.total_deletions < 0:
            raise ValueError("Total counts must be non-negative")

    @property
    def non_binary_files(self) -> List[FileChange]:
        """바이너리가 아닌 파일들만 반환"""
        return [f for f in self.files_changed if not f.is_binary]

    def get_files_by_extension(self, extension: str) -> List[FileChange]:
        """특정 확장자의 파일들 반환"""
        return [f for f in self.files_changed if f.file_path.endswith(extension)]


# Pydantic models for API validation
class DiffChunkRequest(BaseModel):
    """API 요청용 DiffChunk 모델"""
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str
    context_lines: List[str]

    @validator('old_start', 'new_start')
    def validate_line_numbers(cls, v):
        if v < 0:
            raise ValueError('Line numbers must be non-negative')
        return v

    @validator('old_lines', 'new_lines')
    def validate_line_counts(cls, v):
        if v < 0:
            raise ValueError('Line counts must be non-negative')
        return v


class FileChangeRequest(BaseModel):
    """API 요청용 FileChange 모델"""
    file_path: str
    change_type: str
    additions: int
    deletions: int
    chunks: List[DiffChunkRequest]

    @validator('change_type')
    def validate_change_type(cls, v):
        if v not in {'added', 'modified', 'deleted'}:
            raise ValueError('Invalid change type')
        return v

    @validator('additions', 'deletions')
    def validate_counts(cls, v):
        if v < 0:
            raise ValueError('Counts must be non-negative')
        return v


class PRDiffRequest(BaseModel):
    """API 요청용 PRDiff 모델"""
    repository: str
    pr_number: int
    files_changed: List[FileChangeRequest]
    total_additions: int
    total_deletions: int

    @validator('repository')
    def validate_repository(cls, v):
        if '/' not in v:
            raise ValueError('Repository must be in format "owner/repo"')
        return v

    @validator('pr_number')
    def validate_pr_number(cls, v):
        if v <= 0:
            raise ValueError('PR number must be positive')
        return v

    @validator('total_additions', 'total_deletions')
    def validate_totals(cls, v):
        if v < 0:
            raise ValueError('Total counts must be non-negative')
        return v