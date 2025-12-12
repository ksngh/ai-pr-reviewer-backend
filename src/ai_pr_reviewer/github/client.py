"""
GitHub API Client

Handles GitHub API authentication, rate limiting, and communication.
Provides methods for PR diff retrieval and wiki page collection.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..models.pr_diff import PRDiff, FileChange, DiffChunk
from ..models.convention import ConventionRule


logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """GitHub API related errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class RateLimitExceeded(GitHubAPIError):
    """GitHub API rate limit exceeded"""
    def __init__(self, reset_time: datetime):
        super().__init__(f"Rate limit exceeded. Resets at {reset_time}")
        self.reset_time = reset_time


class GitHubClient:
    """
    GitHub API client with authentication, rate limiting, and error handling.
    
    Provides methods for:
    - PR diff retrieval and parsing
    - Wiki page collection and change detection
    - API rate limit management
    """
    
    def __init__(self, token: str, base_url: str = "https://api.github.com"):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token
            base_url: GitHub API base URL (default: https://api.github.com)
        """
        self.token = token
        self.base_url = base_url.rstrip('/')
        self.session = self._create_session()
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = datetime.now()
        
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy and authentication."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set authentication headers
        session.headers.update({
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'AI-PR-Reviewer/1.0'
        })
        
        return session
    
    def _check_rate_limit(self) -> None:
        """Check and handle GitHub API rate limits."""
        if self.rate_limit_remaining <= 10 and datetime.now() < self.rate_limit_reset:
            wait_time = (self.rate_limit_reset - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit low ({self.rate_limit_remaining}), waiting {wait_time:.1f}s")
                raise RateLimitExceeded(self.rate_limit_reset)
    
    def _update_rate_limit(self, response: requests.Response) -> None:
        """Update rate limit information from response headers."""
        if 'X-RateLimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
        
        if 'X-RateLimit-Reset' in response.headers:
            reset_timestamp = int(response.headers['X-RateLimit-Reset'])
            self.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make authenticated request to GitHub API with rate limiting.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            GitHubAPIError: For API errors
            RateLimitExceeded: When rate limit is exceeded
        """
        self._check_rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            self._update_rate_limit(response)
            
            if response.status_code == 429:
                reset_time = datetime.fromtimestamp(int(response.headers.get('X-RateLimit-Reset', time.time() + 3600)))
                raise RateLimitExceeded(reset_time)
            
            if not response.ok:
                error_data = response.json() if response.content else {}
                raise GitHubAPIError(
                    f"GitHub API error: {response.status_code} - {error_data.get('message', 'Unknown error')}",
                    status_code=response.status_code,
                    response_data=error_data
                )
            
            return response
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise GitHubAPIError(f"Request failed: {str(e)}")
    
    def get_pull_request(self, owner: str, repo: str, pr_number: int) -> Dict:
        """
        Get pull request information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            
        Returns:
            Pull request data
        """
        logger.info(f"Fetching PR {owner}/{repo}#{pr_number}")
        
        response = self._make_request('GET', f'/repos/{owner}/{repo}/pulls/{pr_number}')
        return response.json()
    
    def get_pull_request_files(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """
        Get files changed in a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            
        Returns:
            List of file change data
        """
        logger.info(f"Fetching PR files for {owner}/{repo}#{pr_number}")
        
        files = []
        page = 1
        per_page = 100
        
        while True:
            response = self._make_request(
                'GET', 
                f'/repos/{owner}/{repo}/pulls/{pr_number}/files',
                params={'page': page, 'per_page': per_page}
            )
            
            page_files = response.json()
            if not page_files:
                break
                
            files.extend(page_files)
            
            if len(page_files) < per_page:
                break
                
            page += 1
        
        logger.info(f"Found {len(files)} changed files")
        return files
    
    def get_wiki_pages(self, owner: str, repo: str) -> List[Dict]:
        """
        Get all wiki pages for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            List of wiki page metadata
        """
        logger.info(f"Fetching wiki pages for {owner}/{repo}")
        
        try:
            response = self._make_request('GET', f'/repos/{owner}/{repo}/wiki')
            return response.json()
        except GitHubAPIError as e:
            if e.status_code == 404:
                logger.warning(f"No wiki found for {owner}/{repo}")
                return []
            raise
    
    def get_wiki_page_content(self, owner: str, repo: str, page_name: str) -> Optional[str]:
        """
        Get content of a specific wiki page.
        
        Args:
            owner: Repository owner
            repo: Repository name
            page_name: Wiki page name
            
        Returns:
            Wiki page content or None if not found
        """
        logger.info(f"Fetching wiki page content: {owner}/{repo}/wiki/{page_name}")
        
        try:
            response = self._make_request('GET', f'/repos/{owner}/{repo}/wiki/{page_name}')
            return response.json().get('content', '')
        except GitHubAPIError as e:
            if e.status_code == 404:
                logger.warning(f"Wiki page not found: {page_name}")
                return None
            raise
    
    def get_repository_info(self, owner: str, repo: str) -> Dict:
        """
        Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository data
        """
        logger.info(f"Fetching repository info for {owner}/{repo}")
        
        response = self._make_request('GET', f'/repos/{owner}/{repo}')
        return response.json()
    
    def test_authentication(self) -> Tuple[bool, Dict]:
        """
        Test GitHub API authentication.
        
        Returns:
            Tuple of (success, user_info)
        """
        try:
            response = self._make_request('GET', '/user')
            user_data = response.json()
            logger.info(f"Authentication successful for user: {user_data.get('login')}")
            return True, user_data
        except GitHubAPIError as e:
            logger.error(f"Authentication failed: {e}")
            return False, {}
    
    def get_rate_limit_status(self) -> Dict:
        """
        Get current rate limit status.
        
        Returns:
            Rate limit information
        """
        try:
            response = self._make_request('GET', '/rate_limit')
            return response.json()
        except GitHubAPIError as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {
                'rate': {
                    'remaining': self.rate_limit_remaining,
                    'reset': int(self.rate_limit_reset.timestamp())
                }
            }