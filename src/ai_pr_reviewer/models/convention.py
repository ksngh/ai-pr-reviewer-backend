"""
Convention Data Models

컨벤션 규칙 관련 데이터 모델들
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, validator
import uuid


@dataclass
class ConventionRule:
    """개별 컨벤션 규칙"""
    id: str
    title: str
    description: str
    rule_type: str  # 'mandatory', 'recommended', 'prohibited'
    examples: List[str]
    counter_examples: List[str]
    category: str  # 'naming', 'structure', 'documentation', etc.
    source_wiki_page: str
    created_at: datetime
    updated_at: datetime
    version: int

    def __post_init__(self):
        """데이터 검증"""
        valid_types = {'mandatory', 'recommended', 'prohibited'}
        if self.rule_type not in valid_types:
            raise ValueError(f"Invalid rule_type: {self.rule_type}")
        if self.version <= 0:
            raise ValueError("Version must be positive")
        if not self.title.strip():
            raise ValueError("Title cannot be empty")
        if not self.description.strip():
            raise ValueError("Description cannot be empty")

    @classmethod
    def create_new(
        cls,
        title: str,
        description: str,
        rule_type: str,
        examples: List[str],
        counter_examples: List[str],
        category: str,
        source_wiki_page: str,
    ) -> "ConventionRule":
        """새로운 컨벤션 규칙 생성"""
        now = datetime.utcnow()
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            rule_type=rule_type,
            examples=examples,
            counter_examples=counter_examples,
            category=category,
            source_wiki_page=source_wiki_page,
            created_at=now,
            updated_at=now,
            version=1,
        )

    def update(
        self,
        title: Optional[str] = None,
        description: Optional[str] = None,
        rule_type: Optional[str] = None,
        examples: Optional[List[str]] = None,
        counter_examples: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> "ConventionRule":
        """컨벤션 규칙 업데이트"""
        return ConventionRule(
            id=self.id,
            title=title or self.title,
            description=description or self.description,
            rule_type=rule_type or self.rule_type,
            examples=examples or self.examples,
            counter_examples=counter_examples or self.counter_examples,
            category=category or self.category,
            source_wiki_page=self.source_wiki_page,
            created_at=self.created_at,
            updated_at=datetime.utcnow(),
            version=self.version + 1,
        )


@dataclass
class EmbeddedConvention:
    """임베딩된 컨벤션"""
    rule_id: str
    embedding: List[float]
    embedding_model: str
    created_at: datetime

    def __post_init__(self):
        """데이터 검증"""
        if not self.embedding:
            raise ValueError("Embedding cannot be empty")
        if len(self.embedding) == 0:
            raise ValueError("Embedding must have at least one dimension")


@dataclass
class ConventionMatch:
    """컨벤션 매칭 결과"""
    rule: ConventionRule
    similarity_score: float
    relevance_context: str

    def __post_init__(self):
        """데이터 검증"""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")

    @property
    def is_high_confidence(self) -> bool:
        """높은 신뢰도 매칭인지 확인"""
        return self.similarity_score >= 0.8

    @property
    def is_mandatory_violation(self) -> bool:
        """필수 규칙 위반인지 확인"""
        return self.rule.rule_type == 'mandatory' and self.is_high_confidence


# Pydantic models for API validation
class ConventionRuleRequest(BaseModel):
    """API 요청용 ConventionRule 모델"""
    title: str
    description: str
    rule_type: str
    examples: List[str] = []
    counter_examples: List[str] = []
    category: str
    source_wiki_page: str

    @validator('title', 'description')
    def validate_non_empty(cls, v):
        if not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()

    @validator('rule_type')
    def validate_rule_type(cls, v):
        if v not in {'mandatory', 'recommended', 'prohibited'}:
            raise ValueError('Invalid rule type')
        return v


class ConventionRuleResponse(BaseModel):
    """API 응답용 ConventionRule 모델"""
    id: str
    title: str
    description: str
    rule_type: str
    examples: List[str]
    counter_examples: List[str]
    category: str
    source_wiki_page: str
    created_at: datetime
    updated_at: datetime
    version: int

    class Config:
        from_attributes = True


class ConventionMatchResponse(BaseModel):
    """API 응답용 ConventionMatch 모델"""
    rule: ConventionRuleResponse
    similarity_score: float
    relevance_context: str
    is_high_confidence: bool
    is_mandatory_violation: bool

    class Config:
        from_attributes = True