"""
Configuration Management

시스템 설정 관리
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import logging


@dataclass
class ModelConfig:
    """ML 모델 설정"""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "microsoft/DialoGPT-medium"
    cache_dir: str = "./models"
    max_context_tokens: int = 2000
    batch_size: int = 32


@dataclass
class QdrantConfig:
    """Qdrant 벡터 데이터베이스 설정"""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "conventions"
    vector_size: int = 384  # all-MiniLM-L6-v2 embedding size
    distance_metric: str = "cosine"


@dataclass
class GitHubConfig:
    """GitHub API 설정"""
    token: Optional[str] = None
    api_base_url: str = "https://api.github.com"
    rate_limit_per_hour: int = 5000
    timeout_seconds: int = 30


@dataclass
class ReviewConfig:
    """리뷰 생성 설정"""
    max_comments_per_file: int = 10
    min_similarity_threshold: float = 0.7
    hwahae_style_enabled: bool = True
    include_suggestions: bool = True
    consolidate_violations: bool = True


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """전체 애플리케이션 설정"""
    model: ModelConfig
    qdrant: QdrantConfig
    github: GitHubConfig
    review: ReviewConfig
    logging: LoggingConfig
    debug: bool = False
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """환경 변수에서 설정 로드"""
        return cls(
            model=ModelConfig(
                embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                llm_model=os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium"),
                cache_dir=os.getenv("MODEL_CACHE_DIR", "./models"),
                max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "2000")),
                batch_size=int(os.getenv("BATCH_SIZE", "32")),
            ),
            qdrant=QdrantConfig(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                collection_name=os.getenv("QDRANT_COLLECTION", "conventions"),
                vector_size=int(os.getenv("VECTOR_SIZE", "384")),
                distance_metric=os.getenv("DISTANCE_METRIC", "cosine"),
            ),
            github=GitHubConfig(
                token=os.getenv("GITHUB_TOKEN"),
                api_base_url=os.getenv("GITHUB_API_URL", "https://api.github.com"),
                rate_limit_per_hour=int(os.getenv("GITHUB_RATE_LIMIT", "5000")),
                timeout_seconds=int(os.getenv("GITHUB_TIMEOUT", "30")),
            ),
            review=ReviewConfig(
                max_comments_per_file=int(os.getenv("MAX_COMMENTS_PER_FILE", "10")),
                min_similarity_threshold=float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.7")),
                hwahae_style_enabled=os.getenv("HWAHAE_STYLE", "true").lower() == "true",
                include_suggestions=os.getenv("INCLUDE_SUGGESTIONS", "true").lower() == "true",
                consolidate_violations=os.getenv("CONSOLIDATE_VIOLATIONS", "true").lower() == "true",
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
                file_path=os.getenv("LOG_FILE"),
                max_file_size=int(os.getenv("LOG_MAX_SIZE", str(10 * 1024 * 1024))),
                backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            ),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "AppConfig":
        """YAML 파일에서 설정 로드"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_data.get('model', {})),
            qdrant=QdrantConfig(**config_data.get('qdrant', {})),
            github=GitHubConfig(**config_data.get('github', {})),
            review=ReviewConfig(**config_data.get('review', {})),
            logging=LoggingConfig(**config_data.get('logging', {})),
            debug=config_data.get('debug', False),
        )
    
    def validate(self) -> None:
        """설정 유효성 검사"""
        errors = []
        
        # GitHub 토큰 필수 확인
        if not self.github.token:
            errors.append("GitHub token is required")
        
        # 모델 캐시 디렉토리 확인
        cache_dir = Path(self.model.cache_dir)
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create model cache directory: {e}")
        
        # 벡터 크기 검증
        if self.qdrant.vector_size <= 0:
            errors.append("Vector size must be positive")
        
        # 유사도 임계값 검증
        if not 0.0 <= self.review.min_similarity_threshold <= 1.0:
            errors.append("Similarity threshold must be between 0.0 and 1.0")
        
        # 로그 레벨 검증
        valid_log_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.logging.level.upper() not in valid_log_levels:
            errors.append(f"Invalid log level: {self.logging.level}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'model': {
                'embedding_model': self.model.embedding_model,
                'llm_model': self.model.llm_model,
                'cache_dir': self.model.cache_dir,
                'max_context_tokens': self.model.max_context_tokens,
                'batch_size': self.model.batch_size,
            },
            'qdrant': {
                'host': self.qdrant.host,
                'port': self.qdrant.port,
                'collection_name': self.qdrant.collection_name,
                'vector_size': self.qdrant.vector_size,
                'distance_metric': self.qdrant.distance_metric,
            },
            'github': {
                'api_base_url': self.github.api_base_url,
                'rate_limit_per_hour': self.github.rate_limit_per_hour,
                'timeout_seconds': self.github.timeout_seconds,
                # 보안상 토큰은 제외
            },
            'review': {
                'max_comments_per_file': self.review.max_comments_per_file,
                'min_similarity_threshold': self.review.min_similarity_threshold,
                'hwahae_style_enabled': self.review.hwahae_style_enabled,
                'include_suggestions': self.review.include_suggestions,
                'consolidate_violations': self.review.consolidate_violations,
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size': self.logging.max_file_size,
                'backup_count': self.logging.backup_count,
            },
            'debug': self.debug,
        }


class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self._config = config or AppConfig.from_env()
        self._config.validate()
        self._setup_logging()
    
    @property
    def config(self) -> AppConfig:
        """현재 설정 반환"""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """설정 업데이트"""
        # 새로운 설정으로 업데이트
        config_dict = self._config.to_dict()
        
        for key, value in kwargs.items():
            if '.' in key:
                # 중첩된 설정 (예: 'model.batch_size')
                section, field = key.split('.', 1)
                if section in config_dict:
                    config_dict[section][field] = value
            else:
                # 최상위 설정
                config_dict[key] = value
        
        # 새로운 설정 객체 생성
        self._config = AppConfig(
            model=ModelConfig(**config_dict['model']),
            qdrant=QdrantConfig(**config_dict['qdrant']),
            github=GitHubConfig(**config_dict['github']),
            review=ReviewConfig(**config_dict['review']),
            logging=LoggingConfig(**config_dict['logging']),
            debug=config_dict['debug'],
        )
        
        self._config.validate()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """로깅 설정"""
        logging.basicConfig(
            level=getattr(logging, self._config.logging.level.upper()),
            format=self._config.logging.format,
            filename=self._config.logging.file_path,
        )
        
        # 파일 로깅이 설정된 경우 로테이션 설정
        if self._config.logging.file_path:
            from logging.handlers import RotatingFileHandler
            
            handler = RotatingFileHandler(
                self._config.logging.file_path,
                maxBytes=self._config.logging.max_file_size,
                backupCount=self._config.logging.backup_count,
            )
            handler.setFormatter(logging.Formatter(self._config.logging.format))
            
            # 루트 로거에 핸들러 추가
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)


# 전역 설정 관리자 인스턴스
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """현재 설정 반환"""
    return config_manager.config

def update_config(**kwargs) -> None:
    """설정 업데이트"""
    config_manager.update_config(**kwargs)