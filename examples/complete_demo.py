#!/usr/bin/env python3
"""
Complete AI PR Reviewer Demo

Demonstrates the complete AI PR Reviewer system functionality
including PR analysis, convention matching, and review generation.

Usage:
    python examples/complete_demo.py <owner> <repo> <pr_number>

Example:
    python examples/complete_demo.py microsoft vscode 12345

Requirements:
    - GitHub token (set GITHUB_TOKEN environment variable)
    - Qdrant running on localhost:6333 (optional, will use mock if not available)
    - Internet connection for model downloads
"""

import sys
import os
import asyncio
import logging
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_pr_reviewer.api import AIReviewerAPI, ReviewRequest
from ai_pr_reviewer.config import Config


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ Error: GITHUB_TOKEN environment variable not set")
        print("Please set your GitHub token:")
        print("export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx")
        return None
    return token


def print_header():
    """Print demo header."""
    print("🚀 AI PR Reviewer - Complete Demo")
    print("=" * 50)
    print("This demo shows the complete AI PR Reviewer functionality:")
    print("1. 📥 Collect PR diff and wiki conventions")
    print("2. 🔍 Analyze code changes and match conventions")
    print("3. 🤖 Generate LLM reviews with quality control")
    print("4. 📝 Format reviews for GitHub PR comments")
    print()


def print_system_info(api: AIReviewerAPI):
    """Print system information and health."""
    print("🔧 System Information")
    print("-" * 30)
    
    # Get system health
    health = api.get_system_health()
    
    print(f"Overall Status: {health['status'].upper()}")
    print()
    
    # Component status
    for component, info in health['components'].items():
        status_icon = "✅" if info['status'] == 'healthy' else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}: {info['status']}")
        
        if 'stats' in info:
            stats = info['stats']
            if isinstance(stats, dict):
                for key, value in list(stats.items())[:3]:  # Show first 3 stats
                    print(f"   - {key}: {value}")
        
        if 'error' in info:
            print(f"   - Error: {info['error']}")
    
    print()


async def demonstrate_convention_sync(api: AIReviewerAPI, repository: str, github_token: str):
    """Demonstrate convention synchronization."""
    print("📚 Step 1: Synchronizing Conventions")
    print("-" * 40)
    
    try:
        sync_result = await api.sync_conventions(
            repository=repository,
            github_token=github_token,
            force_update=True
        )
        
        if sync_result['status'] == 'completed':
            print(f"✅ Convention sync successful!")
            print(f"   📄 Wiki pages found: {sync_result['wiki_pages_found']}")
            print(f"   📋 Conventions processed: {sync_result['conventions_processed']}")
            print(f"   🔄 Conventions updated: {sync_result['conventions_updated']}")
            print(f"   ⏱️  Processing time: {sync_result['processing_time']:.2f}s")
        else:
            print(f"❌ Convention sync failed: {sync_result.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"❌ Convention sync error: {e}")
        return False
    
    print()
    return True


async def demonstrate_review_generation(
    api: AIReviewerAPI, 
    repository: str, 
    pr_number: int, 
    github_token: str
):
    """Demonstrate PR review generation."""
    print("🤖 Step 2: Generating PR Review")
    print("-" * 40)
    
    try:
        # Create review request
        request = ReviewRequest(
            repository=repository,
            pr_number=pr_number,
            github_token=github_token,
            options={
                'include_suggestions': True,
                'max_comments': 10
            }
        )
        
        print(f"📥 Analyzing PR {repository}#{pr_number}...")
        
        # Generate review
        result = await api.generate_review(request)
        
        if result.status == 'completed':
            print(f"✅ Review generation successful!")
            print(f"   ⏱️  Processing time: {result.processing_time:.2f}s")
            print(f"   💬 GitHub comments generated: {len(result.github_comments)}")
            
            # Show metadata
            metadata = result.metadata
            if 'pr_stats' in metadata:
                pr_stats = metadata['pr_stats']
                print(f"   📊 PR Statistics:")
                print(f"      - Files changed: {pr_stats['files_changed']}")
                print(f"      - Lines added: +{pr_stats['total_additions']}")
                print(f"      - Lines removed: -{pr_stats['total_deletions']}")
            
            if 'analysis_stats' in metadata:
                analysis_stats = metadata['analysis_stats']
                print(f"   🔍 Analysis Statistics:")
                print(f"      - Files analyzed: {analysis_stats['files_analyzed']}")
                print(f"      - Contexts created: {analysis_stats['contexts_created']}")
                print(f"      - Reviews generated: {analysis_stats['reviews_generated']}")
            
            return result
        
        else:
            print(f"❌ Review generation failed: {result.metadata.get('error', 'Unknown error')}")
            return None
    
    except Exception as e:
        print(f"❌ Review generation error: {e}")
        return None


def display_github_comments(result):
    """Display generated GitHub comments."""
    print("\n📝 Step 3: Generated GitHub Comments")
    print("-" * 40)
    
    if not result or not result.github_comments:
        print("No comments generated (PR follows conventions well)")
        return
    
    for i, comment in enumerate(result.github_comments, 1):
        print(f"\n💬 Comment {i}:")
        print(f"   Type: {comment.comment_type}")
        print(f"   Severity: {comment.severity}")
        
        if comment.file_path:
            line_info = f":{comment.line_start}"
            if comment.line_end and comment.line_end != comment.line_start:
                line_info += f"-{comment.line_end}"
            print(f"   Location: {comment.file_path}{line_info}")
        
        # Show preview of comment body
        body_preview = comment.body[:200].replace('\n', ' ')
        if len(comment.body) > 200:
            body_preview += "..."
        print(f"   Preview: {body_preview}")
        
        # Show full comment for first few
        if i <= 2:
            print(f"\n   Full Comment:")
            print("   " + "\n   ".join(comment.body.split('\n')[:10]))
            if len(comment.body.split('\n')) > 10:
                print("   ... (truncated)")


def show_usage_examples():
    """Show usage examples and next steps."""
    print("\n🎯 Next Steps and Usage Examples")
    print("-" * 40)
    
    print("1. 🔧 Integration with GitHub Actions:")
    print("   Create .github/workflows/ai-review.yml:")
    print("""   ```yaml
   name: AI Code Review
   on: [pull_request]
   jobs:
     ai-review:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Run AI Review
           run: |
             python -m ai_pr_reviewer.cli \\
               --repo ${{ github.repository }} \\
               --pr ${{ github.event.number }} \\
               --token ${{ secrets.GITHUB_TOKEN }}
   ```""")
    
    print("\n2. 🐳 Docker Deployment:")
    print("   docker run -d -p 8000:8000 \\")
    print("     -e GITHUB_TOKEN=$GITHUB_TOKEN \\")
    print("     -e QDRANT_HOST=qdrant \\")
    print("     ai-pr-reviewer:latest")
    
    print("\n3. 📚 Convention Management:")
    print("   - Add conventions to your repository wiki")
    print("   - Use structured markdown with clear rules")
    print("   - Include examples and counter-examples")
    print("   - Sync regularly with: api.sync_conventions()")
    
    print("\n4. 🎛️  Configuration Options:")
    print("   - Adjust similarity thresholds")
    print("   - Configure review language (Korean/English)")
    print("   - Set custom model endpoints")
    print("   - Tune quality control parameters")


async def main():
    """Main demo function."""
    setup_logging()
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python complete_demo.py <owner> <repo> <pr_number>")
        print("Example: python complete_demo.py microsoft vscode 12345")
        sys.exit(1)
    
    owner = sys.argv[1]
    repo = sys.argv[2]
    try:
        pr_number = int(sys.argv[3])
    except ValueError:
        print("Error: PR number must be an integer")
        sys.exit(1)
    
    repository = f"{owner}/{repo}"
    
    # Get GitHub token
    github_token = get_github_token()
    if not github_token:
        sys.exit(1)
    
    print_header()
    
    try:
        # Initialize API
        print("🔧 Initializing AI PR Reviewer...")
        config = Config()
        
        # Configure for demo (use smaller models if available)
        config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.llm_model = "microsoft/DialoGPT-medium"
        config.review_language = "korean"
        
        with AIReviewerAPI(config=config) as api:
            print("✅ AI PR Reviewer initialized successfully!")
            print()
            
            # Show system information
            print_system_info(api)
            
            # Step 1: Sync conventions
            sync_success = await demonstrate_convention_sync(
                api, repository, github_token
            )
            
            if not sync_success:
                print("⚠️  Continuing with existing conventions...")
            
            # Step 2: Generate review
            result = await demonstrate_review_generation(
                api, repository, pr_number, github_token
            )
            
            # Step 3: Display results
            display_github_comments(result)
            
            # Show usage examples
            show_usage_examples()
            
            print("\n🎉 Demo completed successfully!")
            print("Check the generated comments above for the AI review results.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())