#!/usr/bin/env python3
"""
GitHub Integration Demo

Demonstrates how to use the GitHub integration layer to fetch
PR diffs and parse them into structured format.

Usage:
    python examples/github_integration_demo.py <owner> <repo> <pr_number>

Example:
    python examples/github_integration_demo.py microsoft vscode 12345
"""

import sys
import os
import logging
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_pr_reviewer.github.client import GitHubClient, GitHubAPIError
from ai_pr_reviewer.github.parser import PRDiffParser
from ai_pr_reviewer.config import Config


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or config."""
    # Try environment variable first
    token = os.getenv('GITHUB_TOKEN')
    if token:
        return token
    
    # Try config file
    try:
        config = Config()
        return config.github_token
    except Exception:
        return None


def main():
    """Main demo function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python github_integration_demo.py <owner> <repo> <pr_number>")
        print("Example: python github_integration_demo.py microsoft vscode 12345")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    try:
        pr_number = int(sys.argv[3])
    except ValueError:
        print("Error: PR number must be an integer")
        sys.exit(1)
    
    # Get GitHub token
    token = get_github_token()
    if not token:
        print("Error: GitHub token not found. Set GITHUB_TOKEN environment variable or configure in config file.")
        sys.exit(1)
    
    try:
        # Initialize GitHub client and parser
        logger.info("Initializing GitHub client...")
        client = GitHubClient(token)
        parser = PRDiffParser()
        
        # Test authentication
        logger.info("Testing GitHub authentication...")
        success, user_info = client.test_authentication()
        if not success:
            print("Error: GitHub authentication failed")
            sys.exit(1)
        
        print(f"✓ Authenticated as: {user_info.get('login', 'Unknown')}")
        
        # Get rate limit status
        rate_limit = client.get_rate_limit_status()
        remaining = rate_limit.get('rate', {}).get('remaining', 'Unknown')
        print(f"✓ Rate limit remaining: {remaining}")
        
        # Fetch PR information
        logger.info(f"Fetching PR {owner}/{repo}#{pr_number}...")
        pr_data = client.get_pull_request(owner, repo, pr_number)
        
        print(f"\n📋 PR Information:")
        print(f"   Title: {pr_data.get('title', 'N/A')}")
        print(f"   State: {pr_data.get('state', 'N/A')}")
        print(f"   Author: {pr_data.get('user', {}).get('login', 'N/A')}")
        print(f"   Created: {pr_data.get('created_at', 'N/A')}")
        
        # Fetch PR files
        logger.info("Fetching PR files...")
        files_data = client.get_pull_request_files(owner, repo, pr_number)
        
        print(f"\n📁 Files Changed: {len(files_data)}")
        
        # Parse PR diff
        logger.info("Parsing PR diff...")
        pr_diff = parser.parse_pr_diff(pr_data, files_data)
        
        print(f"\n📊 Diff Summary:")
        print(f"   Repository: {pr_diff.repository}")
        print(f"   PR Number: {pr_diff.pr_number}")
        print(f"   Total Files: {len(pr_diff.files_changed)}")
        print(f"   Total Additions: +{pr_diff.total_additions}")
        print(f"   Total Deletions: -{pr_diff.total_deletions}")
        
        # Show file details
        print(f"\n📄 File Details:")
        for i, file_change in enumerate(pr_diff.files_changed[:5], 1):  # Show first 5 files
            print(f"   {i}. {file_change.file_path}")
            print(f"      Type: {file_change.change_type}")
            print(f"      Changes: +{file_change.additions}/-{file_change.deletions}")
            print(f"      Chunks: {len(file_change.chunks)}")
            
            # Show first chunk details if available
            if file_change.chunks:
                chunk = file_change.chunks[0]
                print(f"      First chunk: @@ -{chunk.old_start},{chunk.old_lines} +{chunk.new_start},{chunk.new_lines} @@")
        
        if len(pr_diff.files_changed) > 5:
            print(f"   ... and {len(pr_diff.files_changed) - 5} more files")
        
        # Filter relevant files
        relevant_files = parser.filter_relevant_files(pr_diff, include_tests=True)
        code_files = [f for f in relevant_files if parser.is_code_file(f.file_path)]
        
        print(f"\n🔍 Analysis:")
        print(f"   Relevant files for review: {len(relevant_files)}")
        print(f"   Code files: {len(code_files)}")
        
        # Show code file extensions
        extensions = {}
        for file_change in code_files:
            ext = parser.get_file_extension(file_change.file_path)
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
        
        if extensions:
            print(f"   File types: {', '.join(f'{ext}({count})' for ext, count in extensions.items())}")
        
        # Try to fetch wiki pages (optional)
        try:
            logger.info("Fetching wiki pages...")
            wiki_pages = client.get_wiki_pages(owner, repo)
            print(f"\n📚 Wiki Pages: {len(wiki_pages)}")
            
            for page in wiki_pages[:3]:  # Show first 3 pages
                print(f"   - {page.get('title', 'Untitled')}")
                
        except GitHubAPIError as e:
            if e.status_code == 404:
                print(f"\n📚 Wiki Pages: Not available (repository may not have wiki enabled)")
            else:
                print(f"\n📚 Wiki Pages: Error fetching ({e})")
        
        print(f"\n✅ Demo completed successfully!")
        
    except GitHubAPIError as e:
        logger.error(f"GitHub API error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()