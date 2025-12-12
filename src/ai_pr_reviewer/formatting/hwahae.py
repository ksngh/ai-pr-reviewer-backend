"""
Hwahae Style Formatter

Formats code reviews with Hwahae's collaborative and constructive tone.
Applies Korean language patterns and team culture guidelines.
"""

import logging
import re
from typing import Dict, List, Optional
from datetime import datetime

from ..models.review import ReviewComment
from ..review.context import ReviewContext


logger = logging.getLogger(__name__)


class HwahaeStyleFormatter:
    """
    Formats reviews with Hwahae collaborative style.
    
    Applies team-specific tone guidelines, Korean language patterns,
    and constructive feedback principles.
    """
    
    def __init__(self, language: str = "korean"):
        """
        Initialize Hwahae style formatter.
        
        Args:
            language: Language for formatting ("korean" or "english")
        """
        self.language = language
        
        # Hwahae tone guidelines
        self.tone_replacements = {
            "korean": {
                # Make suggestions more collaborative
                "해야 합니다": "하면 좋을 것 같습니다",
                "해야만 합니다": "하시는 것을 추천드립니다",
                "잘못되었습니다": "개선할 수 있을 것 같습니다",
                "틀렸습니다": "다른 방법을 고려해보면 어떨까요",
                "문제가 있습니다": "더 나은 방법이 있을 것 같습니다",
                "오류입니다": "수정이 필요할 것 같습니다",
                
                # Add collaborative phrases
                "수정하세요": "수정해보시면 어떨까요",
                "변경하세요": "변경을 고려해보시면 좋을 것 같습니다",
                "사용하세요": "사용해보시는 것을 추천드립니다",
                "적용하세요": "적용해보시면 어떨까요",
                
                # Soften imperatives
                "하지 마세요": "피하시는 것이 좋을 것 같습니다",
                "사용하지 마세요": "다른 방법을 고려해보시면 어떨까요",
                "피하세요": "피하시는 것을 추천드립니다",
                
                # Add positive framing
                "안 좋습니다": "개선할 여지가 있을 것 같습니다",
                "부족합니다": "더 보완하면 좋을 것 같습니다",
            },
            
            "english": {
                # Make suggestions more collaborative
                "you must": "you might want to",
                "you should": "consider",
                "this is wrong": "this could be improved",
                "this is incorrect": "there might be a better approach",
                "you need to": "it would be good to",
                
                # Add collaborative phrases
                "fix this": "consider fixing this",
                "change this": "you might want to change this",
                "use this": "consider using this",
                
                # Soften imperatives
                "don't do": "consider avoiding",
                "avoid": "it might be better to avoid",
                "never": "it's generally better not to",
            }
        }
        
        # Positive reinforcement phrases
        self.positive_phrases = {
            "korean": [
                "좋은 접근 방식이네요!",
                "잘 구현하셨습니다.",
                "깔끔하게 작성해주셨네요.",
                "이 부분은 잘 되어 있습니다.",
                "컨벤션을 잘 따라주셨네요.",
            ],
            "english": [
                "Nice approach!",
                "Well implemented.",
                "Clean implementation.",
                "This part looks good.",
                "Good adherence to conventions.",
            ]
        }
        
        # Severity-specific formatting
        self.severity_formats = {
            "korean": {
                "high": {
                    "icon": "🚨",
                    "prefix": "중요",
                    "tone": "urgent_collaborative"
                },
                "medium": {
                    "icon": "💡",
                    "prefix": "제안",
                    "tone": "collaborative"
                },
                "low": {
                    "icon": "💭",
                    "prefix": "참고",
                    "tone": "gentle"
                }
            },
            "english": {
                "high": {
                    "icon": "🚨",
                    "prefix": "Important",
                    "tone": "urgent_collaborative"
                },
                "medium": {
                    "icon": "💡",
                    "prefix": "Suggestion",
                    "tone": "collaborative"
                },
                "low": {
                    "icon": "💭",
                    "prefix": "Note",
                    "tone": "gentle"
                }
            }
        }
    
    def format_review_comment(
        self, 
        review: ReviewComment, 
        context: Optional[ReviewContext] = None
    ) -> str:
        """
        Format a review comment with Hwahae style.
        
        Args:
            review: ReviewComment to format
            context: Optional ReviewContext for additional information
            
        Returns:
            Formatted review comment string
        """
        logger.debug(f"Formatting review comment for {review.file_path}")
        
        # Apply tone transformations
        formatted_description = self._apply_hwahae_tone(review.description)
        formatted_title = self._apply_hwahae_tone(review.title)
        
        # Get severity formatting
        severity_format = self.severity_formats[self.language][review.severity]
        
        # Build formatted comment
        comment_parts = []
        
        # Header with icon and severity
        header = f"{severity_format['icon']} **{severity_format['prefix']}: {formatted_title}**"
        comment_parts.append(header)
        comment_parts.append("")  # Empty line
        
        # Main description
        comment_parts.append(formatted_description)
        comment_parts.append("")
        
        # Convention reference
        if review.convention_reference:
            conv_ref = self._format_convention_reference(review.convention_reference)
            comment_parts.append(conv_ref)
            comment_parts.append("")
        
        # Suggestion section
        if review.suggestion:
            suggestion_section = self._format_suggestion_section(review.suggestion)
            comment_parts.append(suggestion_section)
            comment_parts.append("")
        
        # Code snippet if available
        if review.code_snippet and len(review.code_snippet.strip()) > 0:
            code_section = self._format_code_section(review.code_snippet, context)
            comment_parts.append(code_section)
            comment_parts.append("")
        
        # Metadata footer
        footer = self._format_metadata_footer(review, context)
        comment_parts.append(footer)
        
        return "\n".join(comment_parts)
    
    def _apply_hwahae_tone(self, text: str) -> str:
        """Apply Hwahae collaborative tone to text."""
        if not text:
            return text
        
        modified_text = text
        replacements = self.tone_replacements.get(self.language, {})
        
        # Apply tone replacements
        for original, replacement in replacements.items():
            modified_text = modified_text.replace(original, replacement)
        
        # Add collaborative endings for Korean
        if self.language == "korean":
            modified_text = self._add_collaborative_endings(modified_text)
        
        return modified_text
    
    def _add_collaborative_endings(self, text: str) -> str:
        """Add collaborative endings to Korean sentences."""
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        modified_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Add collaborative endings
            if sentence.endswith(('습니다', '입니다', '됩니다')):
                # Already polite, check if we can make it more collaborative
                if not any(phrase in sentence for phrase in ['어떨까요', '좋을 것 같습니다', '추천드립니다']):
                    # Add collaborative suggestion
                    if '습니다' in sentence:
                        sentence = sentence.replace('습니다', '시면 어떨까요')
                    elif '입니다' in sentence:
                        sentence = sentence.replace('입니다', '일 것 같습니다')
            
            modified_sentences.append(sentence)
        
        return '. '.join(modified_sentences) + '.' if modified_sentences else text
    
    def _format_convention_reference(self, convention_rule) -> str:
        """Format convention reference section."""
        if self.language == "korean":
            ref_text = f"**📋 관련 컨벤션**\n"
            ref_text += f"- **규칙**: {convention_rule.title}\n"
            ref_text += f"- **카테고리**: {convention_rule.category}\n"
            ref_text += f"- **유형**: {self._get_rule_type_korean(convention_rule.rule_type)}\n"
            ref_text += f"- **출처**: {convention_rule.source_wiki_page}"
        else:
            ref_text = f"**📋 Related Convention**\n"
            ref_text += f"- **Rule**: {convention_rule.title}\n"
            ref_text += f"- **Category**: {convention_rule.category}\n"
            ref_text += f"- **Type**: {convention_rule.rule_type.title()}\n"
            ref_text += f"- **Source**: {convention_rule.source_wiki_page}"
        
        return ref_text
    
    def _get_rule_type_korean(self, rule_type: str) -> str:
        """Get Korean translation for rule type."""
        type_map = {
            "mandatory": "필수",
            "recommended": "권장",
            "prohibited": "금지"
        }
        return type_map.get(rule_type, rule_type)
    
    def _format_suggestion_section(self, suggestion: str) -> str:
        """Format suggestion section."""
        formatted_suggestion = self._apply_hwahae_tone(suggestion)
        
        if self.language == "korean":
            return f"**💡 제안사항**\n{formatted_suggestion}"
        else:
            return f"**💡 Suggestion**\n{formatted_suggestion}"
    
    def _format_code_section(
        self, 
        code_snippet: str, 
        context: Optional[ReviewContext] = None
    ) -> str:
        """Format code section with syntax highlighting."""
        # Determine language for syntax highlighting
        language = "text"
        if context and context.language:
            language = context.language
        elif context:
            # Try to detect from file extension
            file_ext = context.file_path.split('.')[-1].lower()
            lang_map = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
                'go': 'go',
                'rs': 'rust',
            }
            language = lang_map.get(file_ext, 'text')
        
        if self.language == "korean":
            header = "**📝 관련 코드**"
        else:
            header = "**📝 Related Code**"
        
        return f"{header}\n```{language}\n{code_snippet.strip()}\n```"
    
    def _format_metadata_footer(
        self, 
        review: ReviewComment, 
        context: Optional[ReviewContext] = None
    ) -> str:
        """Format metadata footer."""
        footer_parts = []
        
        # File location
        line_info = f"{review.line_start}"
        if review.line_end != review.line_start:
            line_info += f"-{review.line_end}"
        
        if self.language == "korean":
            location = f"📍 **위치**: `{review.file_path}:{line_info}`"
        else:
            location = f"📍 **Location**: `{review.file_path}:{line_info}`"
        
        footer_parts.append(location)
        
        # Comment type and severity
        if self.language == "korean":
            type_map = {
                "violation": "컨벤션 위반",
                "suggestion": "개선 제안", 
                "question": "확인 요청"
            }
            severity_map = {
                "high": "높음",
                "medium": "보통",
                "low": "낮음"
            }
            
            comment_info = f"🏷️ **유형**: {type_map.get(review.comment_type, review.comment_type)}"
            comment_info += f" | **중요도**: {severity_map.get(review.severity, review.severity)}"
        else:
            comment_info = f"🏷️ **Type**: {review.comment_type.title()}"
            comment_info += f" | **Severity**: {review.severity.title()}"
        
        footer_parts.append(comment_info)
        
        return "\n".join(footer_parts)
    
    def format_positive_feedback(self, context: ReviewContext) -> str:
        """Format positive feedback for good code."""
        phrases = self.positive_phrases[self.language]
        
        # Select appropriate phrase based on context
        if context.relevant_conventions:
            # Code follows conventions well
            if self.language == "korean":
                feedback = "✅ **잘 작성되었습니다!**\n\n"
                feedback += "컨벤션을 잘 따라주셨네요. 깔끔하고 읽기 좋은 코드입니다."
            else:
                feedback = "✅ **Well done!**\n\n"
                feedback += "Good adherence to conventions. Clean and readable code."
        else:
            # General positive feedback
            if self.language == "korean":
                feedback = "✅ **좋습니다!**\n\n"
                feedback += "코드가 깔끔하게 작성되었습니다."
            else:
                feedback = "✅ **Looks good!**\n\n"
                feedback += "Code is well written."
        
        return feedback
    
    def format_summary_comment(
        self, 
        reviews: List[ReviewComment], 
        contexts: List[ReviewContext]
    ) -> str:
        """Format summary comment for multiple reviews."""
        if not reviews:
            if self.language == "korean":
                return self._format_no_issues_summary(contexts)
            else:
                return "## Review Summary\n\nNo issues found. All code follows conventions well. ✅"
        
        # Group reviews by severity
        by_severity = {"high": [], "medium": [], "low": []}
        for review in reviews:
            by_severity[review.severity].append(review)
        
        summary_parts = []
        
        if self.language == "korean":
            summary_parts.append("## 📋 리뷰 요약")
            summary_parts.append("")
            
            total_issues = len(reviews)
            summary_parts.append(f"총 **{total_issues}개**의 검토 사항이 있습니다.")
            summary_parts.append("")
            
            # Severity breakdown
            if by_severity["high"]:
                summary_parts.append(f"🚨 **중요 사항**: {len(by_severity['high'])}개")
            if by_severity["medium"]:
                summary_parts.append(f"💡 **개선 제안**: {len(by_severity['medium'])}개")
            if by_severity["low"]:
                summary_parts.append(f"💭 **참고 사항**: {len(by_severity['low'])}개")
            
            summary_parts.append("")
            summary_parts.append("각 파일별 상세 리뷰를 확인해주세요. 궁금한 점이 있으시면 언제든 말씀해주세요! 😊")
        
        else:
            summary_parts.append("## 📋 Review Summary")
            summary_parts.append("")
            
            total_issues = len(reviews)
            summary_parts.append(f"Found **{total_issues}** review items.")
            summary_parts.append("")
            
            # Severity breakdown
            if by_severity["high"]:
                summary_parts.append(f"🚨 **Important**: {len(by_severity['high'])} items")
            if by_severity["medium"]:
                summary_parts.append(f"💡 **Suggestions**: {len(by_severity['medium'])} items")
            if by_severity["low"]:
                summary_parts.append(f"💭 **Notes**: {len(by_severity['low'])} items")
            
            summary_parts.append("")
            summary_parts.append("Please check the detailed reviews for each file. Feel free to ask if you have any questions! 😊")
        
        return "\n".join(summary_parts)
    
    def _format_no_issues_summary(self, contexts: List[ReviewContext]) -> str:
        """Format summary when no issues are found."""
        if self.language == "korean":
            summary = "## 🎉 리뷰 완료\n\n"
            summary += "모든 코드가 팀 컨벤션을 잘 따르고 있습니다! "
            summary += "깔끔하고 일관성 있게 작성해주셨네요.\n\n"
            
            if contexts:
                files_count = len(set(c.file_path for c in contexts))
                summary += f"**검토한 파일**: {files_count}개\n"
                summary += "**결과**: 컨벤션 준수 ✅\n\n"
            
            summary += "계속해서 좋은 코드 작성해주세요! 👍"
        else:
            summary = "## 🎉 Review Complete\n\n"
            summary += "All code follows team conventions well! "
            summary += "Clean and consistent implementation.\n\n"
            
            if contexts:
                files_count = len(set(c.file_path for c in contexts))
                summary += f"**Files Reviewed**: {files_count}\n"
                summary += "**Result**: Convention Compliant ✅\n\n"
            
            summary += "Keep up the great work! 👍"
        
        return summary