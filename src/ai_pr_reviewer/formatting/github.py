"""
GitHub Comment Formatter

Formats reviews for GitHub PR comments with proper structure,
file organization, and markdown formatting.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..models.review import ReviewComment, GitHubComment
from ..review.context import ReviewContext
from .hwahae import HwahaeStyleFormatter


logger = logging.getLogger(__name__)


@dataclass
class CommentGroup:
    """Group of comments for a file or section."""
    file_path: str
    comments: List[ReviewComment]
    line_ranges: List[Tuple[int, int]]
    total_severity_score: float


class GitHubCommentFormatter:
    """
    Formats reviews for GitHub PR comments.
    
    Organizes comments by file, handles line positioning,
    and generates proper GitHub markdown format.
    """
    
    def __init__(self, language: str = "korean"):
        """
        Initialize GitHub comment formatter.
        
        Args:
            language: Language for formatting ("korean" or "english")
        """
        self.language = language
        self.hwahae_formatter = HwahaeStyleFormatter(language=language)
        
        # GitHub-specific formatting settings
        self.max_comment_length = 65536  # GitHub's comment limit
        self.max_comments_per_file = 10  # Reasonable limit
        self.line_context_range = 5  # Lines of context around issues
    
    def format_for_github(
        self, 
        reviews: List[ReviewComment], 
        contexts: List[ReviewContext]
    ) -> List[GitHubComment]:
        """
        Format reviews for GitHub PR comments.
        
        Args:
            reviews: List of ReviewComment objects
            contexts: List of ReviewContext objects for additional info
            
        Returns:
            List of GitHubComment objects ready for GitHub API
        """
        logger.info(f"Formatting {len(reviews)} reviews for GitHub")
        
        if not reviews:
            # Create summary comment for clean PR
            summary_comment = self._create_summary_comment([], contexts)
            return [summary_comment] if summary_comment else []
        
        # Group comments by file
        comment_groups = self._group_comments_by_file(reviews)
        
        # Create GitHub comments
        github_comments = []
        
        # Individual file comments
        for group in comment_groups:
            file_comments = self._create_file_comments(group, contexts)
            github_comments.extend(file_comments)
        
        # Summary comment
        summary_comment = self._create_summary_comment(reviews, contexts)
        if summary_comment:
            github_comments.append(summary_comment)
        
        logger.info(f"Created {len(github_comments)} GitHub comments")
        return github_comments
    
    def _group_comments_by_file(self, reviews: List[ReviewComment]) -> List[CommentGroup]:
        """Group comments by file path."""
        file_groups = {}
        
        for review in reviews:
            file_path = review.file_path
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(review)
        
        # Create CommentGroup objects
        groups = []
        for file_path, comments in file_groups.items():
            # Sort comments by line number
            comments.sort(key=lambda x: x.line_start)
            
            # Calculate line ranges
            line_ranges = [(c.line_start, c.line_end) for c in comments]
            
            # Calculate severity score
            severity_scores = {"high": 3, "medium": 2, "low": 1}
            total_score = sum(severity_scores.get(c.severity, 1) for c in comments)
            
            group = CommentGroup(
                file_path=file_path,
                comments=comments,
                line_ranges=line_ranges,
                total_severity_score=total_score
            )
            groups.append(group)
        
        # Sort groups by severity (most important first)
        groups.sort(key=lambda x: x.total_severity_score, reverse=True)
        
        return groups
    
    def _create_file_comments(
        self, 
        group: CommentGroup, 
        contexts: List[ReviewContext]
    ) -> List[GitHubComment]:
        """Create GitHub comments for a file group."""
        github_comments = []
        
        # Find relevant context for this file
        file_context = None
        for context in contexts:
            if context.file_path == group.file_path:
                file_context = context
                break
        
        # Limit comments per file
        comments_to_process = group.comments[:self.max_comments_per_file]
        
        if len(group.comments) > self.max_comments_per_file:
            logger.warning(f"Limiting comments for {group.file_path} to {self.max_comments_per_file}")
        
        # Create individual comments or grouped comment
        if len(comments_to_process) == 1:
            # Single comment
            comment = self._create_single_github_comment(
                comments_to_process[0], file_context
            )
            if comment:
                github_comments.append(comment)
        
        elif len(comments_to_process) <= 3:
            # Create individual comments for small groups
            for review in comments_to_process:
                comment = self._create_single_github_comment(review, file_context)
                if comment:
                    github_comments.append(comment)
        
        else:
            # Create grouped comment for many issues
            grouped_comment = self._create_grouped_github_comment(
                comments_to_process, file_context
            )
            if grouped_comment:
                github_comments.append(grouped_comment)
        
        return github_comments
    
    def _create_single_github_comment(
        self, 
        review: ReviewComment, 
        context: Optional[ReviewContext]
    ) -> Optional[GitHubComment]:
        """Create a single GitHub comment."""
        # Format with Hwahae style
        formatted_body = self.hwahae_formatter.format_review_comment(review, context)
        
        # Check length limit
        if len(formatted_body) > self.max_comment_length:
            formatted_body = self._truncate_comment(formatted_body)
        
        return GitHubComment(
            file_path=review.file_path,
            line_start=review.line_start,
            line_end=review.line_end,
            body=formatted_body,
            comment_type="review",
            severity=review.severity,
            created_at=datetime.now()
        )
    
    def _create_grouped_github_comment(
        self, 
        reviews: List[ReviewComment], 
        context: Optional[ReviewContext]
    ) -> Optional[GitHubComment]:
        """Create a grouped GitHub comment for multiple issues."""
        if not reviews:
            return None
        
        # Use first comment's position for the group
        primary_review = reviews[0]
        
        # Build grouped comment body
        body_parts = []
        
        if self.language == "korean":
            header = f"## 📋 {len(reviews)}개 검토 사항 - {primary_review.file_path}"
        else:
            header = f"## 📋 {len(reviews)} Review Items - {primary_review.file_path}"
        
        body_parts.append(header)
        body_parts.append("")
        
        # Add each review as a section
        for i, review in enumerate(reviews, 1):
            section_header = f"### {i}. {review.title}"
            body_parts.append(section_header)
            body_parts.append("")
            
            # Format individual review (simplified)
            review_body = self.hwahae_formatter.format_review_comment(review, context)
            # Remove the title since we already have it
            review_lines = review_body.split('\n')
            if review_lines and review_lines[0].startswith('#'):
                review_lines = review_lines[2:]  # Skip title and empty line
            
            body_parts.append('\n'.join(review_lines))
            body_parts.append("")
            body_parts.append("---")
            body_parts.append("")
        
        # Add summary
        if self.language == "korean":
            summary = "위 사항들을 검토해보시고, 궁금한 점이 있으시면 언제든 말씀해주세요! 😊"
        else:
            summary = "Please review the above items and feel free to ask if you have any questions! 😊"
        
        body_parts.append(summary)
        
        formatted_body = '\n'.join(body_parts)
        
        # Check length limit
        if len(formatted_body) > self.max_comment_length:
            formatted_body = self._truncate_grouped_comment(formatted_body, reviews)
        
        return GitHubComment(
            file_path=primary_review.file_path,
            line_start=primary_review.line_start,
            line_end=max(r.line_end for r in reviews),
            body=formatted_body,
            comment_type="review",
            severity=self._get_highest_severity(reviews),
            created_at=datetime.now()
        )
    
    def _create_summary_comment(
        self, 
        reviews: List[ReviewComment], 
        contexts: List[ReviewContext]
    ) -> Optional[GitHubComment]:
        """Create summary comment for the PR."""
        # Format summary with Hwahae style
        summary_body = self.hwahae_formatter.format_summary_comment(reviews, contexts)
        
        # Add metadata
        metadata_parts = []
        
        if reviews:
            # Statistics
            files_affected = len(set(r.file_path for r in reviews))
            if self.language == "korean":
                stats = f"**📊 통계**\n"
                stats += f"- 검토한 파일: {files_affected}개\n"
                stats += f"- 총 검토 사항: {len(reviews)}개\n"
            else:
                stats = f"**📊 Statistics**\n"
                stats += f"- Files reviewed: {files_affected}\n"
                stats += f"- Total review items: {len(reviews)}\n"
            
            metadata_parts.append(stats)
        
        # Add review metadata
        if self.language == "korean":
            metadata = f"**🤖 AI 리뷰 정보**\n"
            metadata += f"- 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            metadata += f"- 언어: 한국어\n"
            metadata += f"- 스타일: 화해 협력적 톤"
        else:
            metadata = f"**🤖 AI Review Info**\n"
            metadata += f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            metadata += f"- Language: English\n"
            metadata += f"- Style: Hwahae Collaborative Tone"
        
        metadata_parts.append(metadata)
        
        # Combine summary and metadata
        full_body = summary_body + "\n\n" + "\n\n".join(metadata_parts)
        
        return GitHubComment(
            file_path=None,  # Summary comment not tied to specific file
            line_start=None,
            line_end=None,
            body=full_body,
            comment_type="summary",
            severity="info",
            created_at=datetime.now()
        )
    
    def _truncate_comment(self, comment: str) -> str:
        """Truncate comment to fit GitHub limits."""
        if len(comment) <= self.max_comment_length:
            return comment
        
        # Calculate truncation point
        truncate_at = self.max_comment_length - 200  # Leave room for truncation message
        
        # Try to truncate at a natural break point
        truncated = comment[:truncate_at]
        
        # Find last complete line
        last_newline = truncated.rfind('\n')
        if last_newline > truncate_at - 500:  # If reasonably close
            truncated = truncated[:last_newline]
        
        # Add truncation message
        if self.language == "korean":
            truncation_msg = "\n\n---\n*⚠️ 코멘트가 너무 길어서 일부가 생략되었습니다.*"
        else:
            truncation_msg = "\n\n---\n*⚠️ Comment truncated due to length limit.*"
        
        return truncated + truncation_msg
    
    def _truncate_grouped_comment(
        self, 
        comment: str, 
        reviews: List[ReviewComment]
    ) -> str:
        """Truncate grouped comment by reducing number of items."""
        # Keep only high and medium severity items
        important_reviews = [
            r for r in reviews 
            if r.severity in ['high', 'medium']
        ]
        
        if len(important_reviews) < len(reviews):
            # Recreate comment with only important items
            logger.info(f"Truncating grouped comment: keeping {len(important_reviews)} of {len(reviews)} items")
            return self._create_grouped_github_comment(important_reviews, None).body
        
        # If still too long, use basic truncation
        return self._truncate_comment(comment)
    
    def _get_highest_severity(self, reviews: List[ReviewComment]) -> str:
        """Get highest severity from a list of reviews."""
        severity_order = {"high": 3, "medium": 2, "low": 1}
        
        highest = "low"
        highest_score = 0
        
        for review in reviews:
            score = severity_order.get(review.severity, 0)
            if score > highest_score:
                highest_score = score
                highest = review.severity
        
        return highest
    
    def format_diff_comment(
        self, 
        review: ReviewComment, 
        diff_content: str
    ) -> GitHubComment:
        """
        Format comment with diff context.
        
        Args:
            review: ReviewComment to format
            diff_content: Relevant diff content
            
        Returns:
            GitHubComment with diff context
        """
        # Format base comment
        base_comment = self._create_single_github_comment(review, None)
        
        if not base_comment:
            return None
        
        # Add diff context
        if diff_content:
            diff_section = f"\n\n**📝 관련 변경사항**\n```diff\n{diff_content}\n```"
            base_comment.body += diff_section
        
        return base_comment
    
    def get_comment_statistics(self, comments: List[GitHubComment]) -> Dict[str, any]:
        """Get statistics about generated comments."""
        if not comments:
            return {}
        
        # Count by type
        by_type = {}
        by_severity = {}
        files_affected = set()
        
        for comment in comments:
            # Count by type
            comment_type = comment.comment_type
            by_type[comment_type] = by_type.get(comment_type, 0) + 1
            
            # Count by severity
            severity = comment.severity
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Track files
            if comment.file_path:
                files_affected.add(comment.file_path)
        
        # Calculate total body length
        total_length = sum(len(c.body) for c in comments)
        avg_length = total_length / len(comments)
        
        return {
            'total_comments': len(comments),
            'by_type': by_type,
            'by_severity': by_severity,
            'files_affected': len(files_affected),
            'total_body_length': total_length,
            'average_body_length': avg_length,
            'language': self.language,
        }