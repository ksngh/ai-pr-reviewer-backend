"""
GitHub Integration Layer

This module provides GitHub API integration for PR diff retrieval,
wiki page collection, and change detection.
"""

from .client import GitHubClient
from .parser import PRDiffParser

__all__ = ['GitHubClient', 'PRDiffParser']