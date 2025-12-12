"""
Quality Controller

Controls the quality of generated reviews to prevent hallucinations
and ensure convention-grounded feedback.
"""

import logging
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from ..models.review import ReviewComment
from ..review.context import ReviewContext
from ..review.matcher import ConventionMatch


logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a review."""
    convention_grounding_score: float
    relevance_score: float
    specificity_score: float
    tone_score: float
    overall_score: float
    issues: List[str]


class QualityController:
    """
    Controls quality of generated reviews.
    
    Validates that reviews are grounded in conventions,
    relevant to code changes, and maintain appropriate tone.
    """
    
    def __init__(self, min_quality_score: float = 0.6):
        """
        Initialize quality controller.
        
        Args:
            min_quality_score: Minimum quality score to pass validation
        """
        self.min_quality_score = min_quality_score
        
        # Hallucination detection patterns
        self.hallucination_patterns = [
            # Generic programming advice not based on conventions
            r'일반적으로.*좋습니다',
            r'보통.*합니다',
            r'대부분.*권장',
            r'generally.*good',
            r'usually.*recommended',
            
            # Vague statements without specifics
            r'더 나은 방법이 있을 것입니다',
            r'개선할 수 있습니다',
            r'could be better',
            r'might be improved',
            
            # Non-convention based suggestions
            r'성능상.*좋습니다',
            r'가독성.*향상',
            r'for performance',
            r'for readability',
        ]
        
        # Required convention grounding patterns
        self.grounding_patterns = [
            r'컨벤션.*규칙',
            r'팀.*정의',
            r'convention.*rule',
            r'team.*defined',
            r'규칙.*따라',
            r'according.*rule',
        ]
        
        # Inappropriate tone patterns (too harsh)
        self.harsh_tone_patterns = [
            r'잘못.*했습니다',
            r'틀렸습니다',
            r'문제가.*있습니다',
            r'you.*wrong',
            r'this.*incorrect',
            r'you.*should.*not',
        ]
        
        # Positive tone indicators
        self.positive_tone_patterns = [
            r'.*어떨까요',
            r'.*좋을 것 같습니다',
            r'.*추천드립니다',
            r'.*consider',
            r'.*might.*want',
            r'.*suggest',
        ]
    
    def validate_review(
        self, 
        review: ReviewComment, 
        context: ReviewContext
    ) -> bool:
        """
        Validate review quality.
        
        Args:
            review: ReviewComment to validate
            context: ReviewContext used for generation
            
        Returns:
            True if review passes quality checks
        """
        logger.debug(f"Validating review for {review.file_path}")
        
        # Calculate quality metrics
        metrics = self.calculate_quality_metrics(review, context)
        
        # Log quality issues
        if metrics.issues:
            logger.warning(f"Quality issues for {review.file_path}: {metrics.issues}")
        
        # Check overall quality score
        passed = metrics.overall_score >= self.min_quality_score
        
        if passed:
            logger.debug(f"Review passed quality check (score: {metrics.overall_score:.2f})")
        else:
            logger.warning(f"Review failed quality check (score: {metrics.overall_score:.2f})")
        
        return passed
    
    def calculate_quality_metrics(
        self, 
        review: ReviewComment, 
        context: ReviewContext
    ) -> QualityMetrics:
        """Calculate detailed quality metrics for a review."""
        issues = []
        
        # 1. Convention grounding score
        grounding_score = self._check_convention_grounding(review, context, issues)
        
        # 2. Relevance score
        relevance_score = self._check_relevance(review, context, issues)
        
        # 3. Specificity score
        specificity_score = self._check_specificity(review, issues)
        
        # 4. Tone score
        tone_score = self._check_tone(review, issues)
        
        # Calculate overall score (weighted average)
        overall_score = (
            grounding_score * 0.4 +  # Most important
            relevance_score * 0.3 +
            specificity_score * 0.2 +
            tone_score * 0.1
        )
        
        return QualityMetrics(
            convention_grounding_score=grounding_score,
            relevance_score=relevance_score,
            specificity_score=specificity_score,
            tone_score=tone_score,
            overall_score=overall_score,
            issues=issues
        )
    
    def _check_convention_grounding(
        self, 
        review: ReviewComment, 
        context: ReviewContext, 
        issues: List[str]
    ) -> float:
        """Check if review is grounded in conventions."""
        score = 0.0
        
        # Check if review references a convention
        if review.convention_reference:
            score += 0.4
        else:
            issues.append("No convention reference")
        
        # Check if review mentions convention-related terms
        description = review.description.lower()
        
        grounding_found = False
        for pattern in self.grounding_patterns:
            if re.search(pattern, description):
                grounding_found = True
                break
        
        if grounding_found:
            score += 0.3
        else:
            issues.append("No convention grounding language")
        
        # Check if review content matches convention rules
        if context.relevant_conventions:
            for match in context.relevant_conventions:
                rule = match.rule
                
                # Check if rule title or key terms appear in review
                rule_terms = [
                    rule.title.lower(),
                    rule.category.lower(),
                    *[ex.lower() for ex in rule.examples[:2]]
                ]
                
                for term in rule_terms:
                    if len(term) > 3 and term in description:
                        score += 0.1
                        break
        
        # Check for hallucination patterns
        for pattern in self.hallucination_patterns:
            if re.search(pattern, description):
                score -= 0.2
                issues.append(f"Potential hallucination: {pattern}")
        
        return max(0.0, min(1.0, score))
    
    def _check_relevance(
        self, 
        review: ReviewComment, 
        context: ReviewContext, 
        issues: List[str]
    ) -> float:
        """Check if review is relevant to the code changes."""
        score = 0.0
        
        description = review.description.lower()
        
        # Check if review mentions file-specific elements
        metadata = context.context_metadata
        
        # Check for function names
        functions_mentioned = 0
        for func_name in metadata.get('functions_affected', []):
            if func_name.lower() in description:
                functions_mentioned += 1
        
        if functions_mentioned > 0:
            score += 0.3
        
        # Check for class names
        classes_mentioned = 0
        for class_name in metadata.get('classes_affected', []):
            if class_name.lower() in description:
                classes_mentioned += 1
        
        if classes_mentioned > 0:
            score += 0.2
        
        # Check for change type relevance
        change_types = metadata.get('change_types', [])
        change_type_terms = {
            'function_added': ['함수', 'function', '추가', 'add'],
            'function_modified': ['함수', 'function', '수정', 'modify'],
            'class_added': ['클래스', 'class', '추가', 'add'],
            'variable_modified': ['변수', 'variable', '수정', 'modify'],
        }
        
        for change_type in change_types:
            if change_type in change_type_terms:
                terms = change_type_terms[change_type]
                if any(term in description for term in terms):
                    score += 0.2
                    break
        
        # Check if review is too generic
        generic_phrases = [
            '일반적으로', '보통', '대부분', 'generally', 'usually', 'typically'
        ]
        
        for phrase in generic_phrases:
            if phrase in description:
                score -= 0.1
                issues.append(f"Generic phrase: {phrase}")
        
        return max(0.0, min(1.0, score))
    
    def _check_specificity(self, review: ReviewComment, issues: List[str]) -> float:
        """Check if review provides specific, actionable feedback."""
        score = 0.0
        
        description = review.description
        
        # Check length (too short or too long is bad)
        if len(description) < 20:
            issues.append("Review too short")
            score -= 0.3
        elif len(description) > 500:
            issues.append("Review too long")
            score -= 0.1
        else:
            score += 0.2
        
        # Check for specific suggestions
        if review.suggestion and len(review.suggestion) > 10:
            score += 0.3
        else:
            issues.append("No specific suggestion")
        
        # Check for code examples or specific references
        code_indicators = ['```', '`', 'code:', '코드:', 'example:', '예시:']
        if any(indicator in description for indicator in code_indicators):
            score += 0.2
        
        # Check for vague language
        vague_phrases = [
            '더 좋게', '개선', '향상', 'better', 'improve', 'enhance'
        ]
        
        vague_count = sum(1 for phrase in vague_phrases if phrase in description.lower())
        if vague_count > 2:
            score -= 0.2
            issues.append("Too much vague language")
        
        return max(0.0, min(1.0, score))
    
    def _check_tone(self, review: ReviewComment, issues: List[str]) -> float:
        """Check if review maintains appropriate collaborative tone."""
        score = 1.0  # Start with perfect score
        
        description = review.description.lower()
        
        # Check for harsh tone
        for pattern in self.harsh_tone_patterns:
            if re.search(pattern, description):
                score -= 0.3
                issues.append(f"Harsh tone: {pattern}")
        
        # Check for positive/collaborative tone
        positive_found = False
        for pattern in self.positive_tone_patterns:
            if re.search(pattern, description):
                positive_found = True
                break
        
        if positive_found:
            score += 0.1
        else:
            # Not necessarily bad, but could be better
            pass
        
        # Check for imperative vs. suggestive language
        imperatives = ['하세요', '하십시오', 'must', 'should', 'need to']
        suggestives = ['어떨까요', '좋을 것 같습니다', 'might', 'could', 'consider']
        
        imperative_count = sum(1 for imp in imperatives if imp in description)
        suggestive_count = sum(1 for sug in suggestives if sug in description)
        
        if imperative_count > suggestive_count:
            score -= 0.1
            issues.append("Too many imperatives")
        
        return max(0.0, min(1.0, score))
    
    def filter_hallucinations(self, review_text: str) -> str:
        """
        Filter out potential hallucinations from review text.
        
        Args:
            review_text: Original review text
            
        Returns:
            Filtered review text
        """
        filtered_text = review_text
        
        # Remove sentences with hallucination patterns
        sentences = re.split(r'[.!?]', filtered_text)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains hallucination patterns
            is_hallucination = False
            for pattern in self.hallucination_patterns:
                if re.search(pattern, sentence):
                    is_hallucination = True
                    logger.debug(f"Filtered hallucination: {sentence}")
                    break
            
            if not is_hallucination:
                clean_sentences.append(sentence)
        
        return '. '.join(clean_sentences) + '.' if clean_sentences else ""
    
    def enhance_convention_grounding(
        self, 
        review: ReviewComment, 
        context: ReviewContext
    ) -> ReviewComment:
        """
        Enhance review with better convention grounding.
        
        Args:
            review: Original ReviewComment
            context: ReviewContext with conventions
            
        Returns:
            Enhanced ReviewComment
        """
        enhanced_review = ReviewComment(
            file_path=review.file_path,
            line_start=review.line_start,
            line_end=review.line_end,
            comment_type=review.comment_type,
            severity=review.severity,
            title=review.title,
            description=review.description,
            code_snippet=review.code_snippet,
            suggestion=review.suggestion,
            convention_reference=review.convention_reference
        )
        
        # Add convention reference if missing
        if not enhanced_review.convention_reference and context.relevant_conventions:
            enhanced_review.convention_reference = context.relevant_conventions[0].rule
        
        # Enhance description with convention grounding
        if enhanced_review.convention_reference:
            rule = enhanced_review.convention_reference
            
            grounding_text = f"\n\n**관련 컨벤션:** {rule.title} ({rule.category})"
            if not grounding_text in enhanced_review.description:
                enhanced_review.description += grounding_text
        
        return enhanced_review
    
    def get_quality_summary(self, metrics: QualityMetrics) -> str:
        """Get human-readable quality summary."""
        if metrics.overall_score >= 0.8:
            quality_level = "Excellent"
        elif metrics.overall_score >= 0.6:
            quality_level = "Good"
        elif metrics.overall_score >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
        
        summary = f"Quality: {quality_level} ({metrics.overall_score:.2f})\n"
        summary += f"- Convention Grounding: {metrics.convention_grounding_score:.2f}\n"
        summary += f"- Relevance: {metrics.relevance_score:.2f}\n"
        summary += f"- Specificity: {metrics.specificity_score:.2f}\n"
        summary += f"- Tone: {metrics.tone_score:.2f}\n"
        
        if metrics.issues:
            summary += f"Issues: {', '.join(metrics.issues)}"
        
        return summary