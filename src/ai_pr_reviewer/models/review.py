"""
Review Data Models

코드 리뷰 관련 데이터 모델들
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, validator
import uuid

from .convention import ConventionRule


@dataclass
class ReviewComment:
    """개별 리뷰 코멘트"""
    file_path: str
    line_start: int
    line_end: int
    comment_type: str  # 'violation', 'suggestion', 'question'
    severity: str  # 'high', 'medium', 'low'
    title: str
    description: str
    code_snippet: str
    suggestion: Optional[str]
    convention_reference: ConventionRule
    created_at: datetime

    def __post_init__(self):
        """데이터 검증"""
        valid_comment_types = {'violation', 'suggestion', 'question'}
        if self.comment_type not in valid_comment_types:
            raise ValueError(f"Invalid comment_type: {self.comment_type}")
        
        valid_severities = {'high', 'medium', 'low'}
        if self.severity not in valid_severities:
            raise ValueError(f"Invalid severity: {self.severity}")
        
        if self.line_start <= 0 or self.line_end <= 0:
            raise ValueError("Line numbers must be positive")
        
        if self.line_start > self.line_end:
            raise ValueError("line_start cannot be greater than line_end")
        
        if not self.title.strip():
            raise ValueError("Title cannot be empty")
        
        if not self.description.strip():
            raise ValueError("Description cannot be empty")

    @property
    def is_critical(self) -> bool:
        """중요한 코멘트인지 확인"""
        return (
            self.severity == 'high' and 
            self.comment_type == 'violation' and
            self.convention_reference.rule_type == 'mandatory'
        )

    @property
    def line_range(self) -> str:
        """라인 범위 문자열 반환"""
        if self.line_start == self.line_end:
            return str(self.line_start)
        return f"{self.line_start}-{self.line_end}"


@dataclass
class GitHubComment:
    """GitHub PR 코멘트 형식"""
    path: str
    line: int
    body: str
    side: str  # 'RIGHT' for new code, 'LEFT' for old code

    def __post_init__(self):
        """데이터 검증"""
        valid_sides = {'RIGHT', 'LEFT'}
        if self.side not in valid_sides:
            raise ValueError(f"Invalid side: {self.side}")
        
        if self.line <= 0:
            raise ValueError("Line number must be positive")
        
        if not self.body.strip():
            raise ValueError("Comment body cannot be empty")


@dataclass
class ReviewResult:
    """전체 리뷰 결과"""
    review_id: str
    repository: str
    pr_number: int
    total_comments: int
    comments_by_file: Dict[str, List[ReviewComment]]
    processing_time: float
    conventions_used: List[str]
    created_at: datetime
    status: str

    def __post_init__(self):
        """데이터 검증"""
        if self.pr_number <= 0:
            raise ValueError("PR number must be positive")
        
        if '/' not in self.repository:
            raise ValueError("Repository must be in format 'owner/repo'")
        
        if self.total_comments < 0:
            raise ValueError("Total comments must be non-negative")
        
        if self.processing_time < 0:
            raise ValueError("Processing time must be non-negative")
        
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}")

    @classmethod
    def create_new(
        cls,
        repository: str,
        pr_number: int,
        comments_by_file: Dict[str, List[ReviewComment]],
        processing_time: float,
        conventions_used: List[str],
    ) -> "ReviewResult":
        """새로운 리뷰 결과 생성"""
        total_comments = sum(len(comments) for comments in comments_by_file.values())
        
        return cls(
            review_id=str(uuid.uuid4()),
            repository=repository,
            pr_number=pr_number,
            total_comments=total_comments,
            comments_by_file=comments_by_file,
            processing_time=processing_time,
            conventions_used=conventions_used,
            created_at=datetime.utcnow(),
            status='completed',
        )

    @property
    def critical_comments(self) -> List[ReviewComment]:
        """중요한 코멘트들만 반환"""
        critical = []
        for comments in self.comments_by_file.values():
            critical.extend([c for c in comments if c.is_critical])
        return critical

    @property
    def files_with_issues(self) -> List[str]:
        """이슈가 있는 파일 목록"""
        return [file_path for file_path, comments in self.comments_by_file.items() if comments]

    def get_comments_by_severity(self, severity: str) -> List[ReviewComment]:
        """특정 심각도의 코멘트들 반환"""
        result = []
        for comments in self.comments_by_file.values():
            result.extend([c for c in comments if c.severity == severity])
        return result


# Pydantic models for API validation
class ReviewCommentRequest(BaseModel):
    """API 요청용 ReviewComment 모델"""
    file_path: str
    line_start: int
    line_end: int
    comment_type: str
    severity: str
    title: str
    description: str
    code_snippet: str
    suggestion: Optional[str] = None

    @validator('comment_type')
    def validate_comment_type(cls, v):
        if v not in {'violation', 'suggestion', 'question'}:
            raise ValueError('Invalid comment type')
        return v

    @validator('severity')
    def validate_severity(cls, v):
        if v not in {'high', 'medium', 'low'}:
            raise ValueError('Invalid severity')
        return v

    @validator('line_start', 'line_end')
    def validate_line_numbers(cls, v):
        if v <= 0:
            raise ValueError('Line numbers must be positive')
        return v

    @validator('title', 'description')
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()


class ReviewCommentResponse(BaseModel):
    """API 응답용 ReviewComment 모델"""
    file_path: str
    line_start: int
    line_end: int
    comment_type: str
    severity: str
    title: str
    description: str
    code_snippet: str
    suggestion: Optional[str]
    convention_reference: dict  # ConventionRule을 dict로 직렬화
    created_at: datetime
    is_critical: bool
    line_range: str

    class Config:
        from_attributes = True


class ReviewResultResponse(BaseModel):
    """API 응답용 ReviewResult 모델"""
    review_id: str
    repository: str
    pr_number: int
    total_comments: int
    processing_time: float
    conventions_used: List[str]
    created_at: datetime
    status: str
    files_with_issues: List[str]
    critical_comment_count: int

    class Config:
        from_attributes = True


class GitHubCommentRequest(BaseModel):
    """API 요청용 GitHubComment 모델"""
    path: str
    line: int
    body: str
    side: str = 'RIGHT'

    @validator('side')
    def validate_side(cls, v):
        if v not in {'RIGHT', 'LEFT'}:
            raise ValueError('Invalid side')
        return v

    @validator('line')
    def validate_line(cls, v):
        if v <= 0:
            raise ValueError('Line number must be positive')
        return v

    @validator('body')
    def validate_body(cls, v):
        if not v.strip():
            raise ValueError('Comment body cannot be empty')
        return v.strip()