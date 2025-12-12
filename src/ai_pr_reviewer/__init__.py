"""
AI PR Reviewer Backend

GitHub Pull Request 자동 코드 리뷰 시스템의 백엔드 구현체
"""

__version__ = "1.0.0"
__author__ = "Hwahae Team"
__email__ = "dev@hwahae.co.kr"

from .api import AIReviewerAPI

__all__ = ["AIReviewerAPI"]