# AI PR Reviewer Backend

Backend service for the AI PR Reviewer system that analyzes GitHub Pull Requests against team-defined coding conventions and generates structured, actionable feedback in Korean following Hwahae's collaborative review culture.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (for Qdrant vector database)
- GitHub Personal Access Token

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start Qdrant vector database:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your GitHub token and configuration
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

### Basic Usage

```python
import asyncio
from ai_pr_reviewer.api import AIReviewerAPI, ReviewRequest

async def main():
    # Initialize API
    api = AIReviewerAPI()
    
    # Create review request
    request = ReviewRequest(
        repository="owner/repo",
        pr_number=123,
        github_token="your_github_token"
    )
    
    # Generate review
    result = await api.generate_review(request)
    
    # Print results
    for comment in result.github_comments:
        print(f"📝 {comment.file_path}: {comment.body}")

asyncio.run(main())
```

## 🎯 Features

- **Convention-Based Reviews**: Only provides feedback based on documented team conventions
- **Vector-Powered Matching**: Uses semantic similarity to match code changes with relevant rules
- **Hwahae Style**: Generates reviews in Korean with collaborative, constructive tone
- **GitHub Integration**: Seamlessly integrates with GitHub PR workflow and Wiki pages
- **Open Source Stack**: Built entirely on open-source models and tools

## 구조

```
ai-pr-reviewer-backend/
├── src/
│   └── ai_pr_reviewer/
│       ├── __init__.py
│       ├── api.py                    # Main AIReviewerAPI interface
│       ├── github/                   # GitHub Integration Layer
│       │   ├── __init__.py
│       │   ├── client.py            # GitHubClient class
│       │   └── parser.py            # PRDiffParser class
│       ├── conventions/              # Convention Processing Layer
│       │   ├── __init__.py
│       │   ├── extractor.py         # ConventionExtractor class
│       │   ├── embeddings.py        # EmbeddingGenerator class
│       │   └── vector_store.py      # VectorStore class
│       ├── review/                   # Review Context Builder
│       │   ├── __init__.py
│       │   ├── analyzer.py          # DiffAnalyzer class
│       │   ├── matcher.py           # ConventionMatcher class
│       │   └── context.py           # ContextBuilder class
│       ├── llm/                      # LLM Review Engine
│       │   ├── __init__.py
│       │   ├── prompts.py           # PromptBuilder class
│       │   ├── generator.py         # ReviewGenerator class
│       │   └── quality.py           # QualityController class
│       ├── formatting/               # Review Formatter
│       │   ├── __init__.py
│       │   ├── hwahae.py            # HwahaeStyleFormatter class
│       │   └── github.py            # GitHubCommentFormatter class
│       └── models/                   # Data Models
│           ├── __init__.py
│           ├── pr_diff.py           # PRDiff, FileChange, DiffChunk
│           ├── convention.py        # ConventionRule, EmbeddedConvention
│           └── review.py            # ReviewComment, GitHubComment
├── tests/
│   ├── unit/                        # Unit tests for each component
│   ├── integration/                 # End-to-end integration tests
│   └── property/                    # Property-based tests (hypothesis)
├── config/
│   ├── models.yaml                  # Model configuration
│   └── prompts.yaml                 # Prompt templates
├── .kiro/
│   └── specs/                       # Shared specifications (git submodule)
├── requirements.txt
├── setup.py
├── Dockerfile
└── docker-compose.yml
```

## 설치 및 실행

### 개발 환경 설정

```bash
# 1. 레포지토리 클론 및 submodule 초기화
git clone <backend-repo-url>
cd ai-pr-reviewer-backend
git submodule update --init --recursive

# 2. Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. Qdrant 벡터 데이터베이스 시작
docker run -p 6333:6333 qdrant/qdrant
```

### 테스트 실행

```bash
# 모든 테스트 실행
pytest

# Property-based 테스트만 실행
pytest tests/property/ -v

# 커버리지와 함께 실행
pytest --cov=ai_pr_reviewer --cov-report=html
```

### API 서버 실행

```bash
# 개발 서버 시작
python -m ai_pr_reviewer.api

# 또는 Docker로 실행
docker-compose up
```

## API 엔드포인트

- `POST /api/v1/reviews/generate` - PR 리뷰 생성
- `GET /api/v1/reviews/{review_id}` - 리뷰 결과 조회
- `GET /api/v1/system/health` - 시스템 상태 확인

자세한 API 문서는 `.kiro/specs/api-contract.md`를 참조하세요.

## 환경 변수

```bash
GITHUB_TOKEN=your_github_token
QDRANT_HOST=localhost
QDRANT_PORT=6333
MODEL_CACHE_DIR=./models
LOG_LEVEL=INFO
```