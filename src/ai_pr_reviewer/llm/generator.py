"""
Review Generator

Generates code reviews using LLM with quality control and
convention-grounded feedback validation.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Required ML dependencies not installed: {e}")
    logger.error("Install with: pip install transformers torch")
    raise

from ..models.review import ReviewComment
from ..review.context import ReviewContext
from ..review.matcher import ConventionMatch


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for review generation."""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class ReviewResult:
    """Result of review generation."""
    comments: List[ReviewComment]
    generation_metadata: Dict[str, any]
    quality_score: float
    processing_time: float


class ReviewGenerator:
    """
    Generates code reviews using LLM with quality control.
    
    Uses open-source language models to generate convention-grounded
    reviews following Hwahae's collaborative style.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize review generator.
        
        Args:
            model_name: Name of the language model to use
            device: Device to run model on ('cpu', 'cuda', etc.)
            generation_config: Configuration for text generation
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.generation_config = generation_config or GenerationConfig()
        
        # Initialize model and tokenizer
        logger.info(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        # Comment parsing patterns
        self.comment_patterns = {
            'korean': {
                'file': r'파일:\s*(.+)',
                'line': r'라인:\s*(\d+)',
                'type': r'유형:\s*(violation|suggestion|praise)',
                'severity': r'심각도:\s*(high|medium|low)',
                'title': r'\*\*(.+?)\*\*',
                'content': r'\*\*.*?\*\*\n(.+?)(?=참조 컨벤션:|$)',
                'convention': r'참조 컨벤션:\s*(.+)'
            },
            'english': {
                'file': r'File:\s*(.+)',
                'line': r'Line:\s*(\d+)',
                'type': r'Type:\s*(violation|suggestion|praise)',
                'severity': r'Severity:\s*(high|medium|low)',
                'title': r'\*\*(.+?)\*\*',
                'content': r'\*\*.*?\*\*\n(.+?)(?=Referenced Convention:|$)',
                'convention': r'Referenced Convention:\s*(.+)'
            }
        }
    
    def generate_review(
        self,
        prompt: str,
        context: ReviewContext,
        language: str = "korean"
    ) -> ReviewResult:
        """
        Generate review for a single context.
        
        Args:
            prompt: Complete prompt for review generation
            context: ReviewContext being reviewed
            language: Language for review generation
            
        Returns:
            ReviewResult with generated comments
        """
        logger.debug(f"Generating review for {context.file_path}")
        
        import time
        start_time = time.time()
        
        try:
            # Generate text using the model
            generated_text = self._generate_text(prompt)
            
            # Parse generated text into structured comments
            comments = self._parse_generated_review(generated_text, context, language)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(generated_text, comments, context)
            
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                'model_name': self.model_name,
                'prompt_length': len(prompt),
                'generated_length': len(generated_text),
                'language': language,
                'generation_config': self.generation_config.__dict__,
                'device': self.device
            }
            
            logger.debug(f"Generated {len(comments)} comments in {processing_time:.2f}s")
            
            return ReviewResult(
                comments=comments,
                generation_metadata=metadata,
                quality_score=quality_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to generate review: {e}")
            return ReviewResult(
                comments=[],
                generation_metadata={'error': str(e)},
                quality_score=0.0,
                processing_time=time.time() - start_time
            )
    
    def generate_batch_review(
        self,
        prompt: str,
        contexts: List[ReviewContext],
        language: str = "korean"
    ) -> List[ReviewResult]:
        """
        Generate reviews for multiple contexts in batch.
        
        Args:
            prompt: Complete batch prompt
            contexts: List of ReviewContext objects
            language: Language for review generation
            
        Returns:
            List of ReviewResult objects
        """
        logger.info(f"Generating batch review for {len(contexts)} contexts")
        
        import time
        start_time = time.time()
        
        try:
            # Generate text for batch
            generated_text = self._generate_text(prompt)
            
            # Parse batch results
            results = self._parse_batch_review(generated_text, contexts, language)
            
            processing_time = time.time() - start_time
            logger.info(f"Generated batch review in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate batch review: {e}")
            # Return empty results for each context
            return [
                ReviewResult(
                    comments=[],
                    generation_metadata={'error': str(e)},
                    quality_score=0.0,
                    processing_time=0.0
                )
                for _ in contexts
            ]
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the language model."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.generation_config.max_length,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    do_sample=self.generation_config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove input prompt)
            generated_part = generated_text[len(prompt):].strip()
            
            return generated_part
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
    
    def _parse_generated_review(
        self,
        generated_text: str,
        context: ReviewContext,
        language: str
    ) -> List[ReviewComment]:
        """Parse generated text into structured ReviewComment objects."""
        comments = []
        
        if not generated_text.strip():
            return comments
        
        # Check for "no issues" response
        no_issues_patterns = {
            'korean': ['검토 완료', '컨벤션을 잘 준수', '문제없음', '개선점이 없'],
            'english': ['review complete', 'follows conventions well', 'no issues', 'no improvements']
        }
        
        text_lower = generated_text.lower()
        for pattern in no_issues_patterns.get(language, []):
            if pattern in text_lower:
                logger.debug("No issues found in generated review")
                return comments
        
        # Parse structured comments
        patterns = self.comment_patterns.get(language, self.comment_patterns['english'])
        
        # Split by comment blocks (look for file patterns)
        comment_blocks = re.split(r'(?=파일:|File:)', generated_text)
        
        for block in comment_blocks:
            if not block.strip():
                continue
            
            comment = self._parse_single_comment(block, context, patterns)
            if comment:
                comments.append(comment)
        
        return comments
    
    def _parse_single_comment(
        self,
        comment_block: str,
        context: ReviewContext,
        patterns: Dict[str, str]
    ) -> Optional[ReviewComment]:
        """Parse a single comment block into ReviewComment."""
        try:
            # Extract fields using regex patterns
            file_match = re.search(patterns['file'], comment_block)
            line_match = re.search(patterns['line'], comment_block)
            type_match = re.search(patterns['type'], comment_block)
            severity_match = re.search(patterns['severity'], comment_block)
            title_match = re.search(patterns['title'], comment_block)
            content_match = re.search(patterns['content'], comment_block, re.DOTALL)
            convention_match = re.search(patterns['convention'], comment_block)
            
            # Extract values with defaults
            file_path = file_match.group(1).strip() if file_match else context.file_path
            line_number = int(line_match.group(1)) if line_match else 1
            comment_type = type_match.group(1).strip() if type_match else 'suggestion'
            severity = severity_match.group(1).strip() if severity_match else 'medium'
            title = title_match.group(1).strip() if title_match else 'Code Review Comment'
            description = content_match.group(1).strip() if content_match else comment_block.strip()
            convention_ref = convention_match.group(1).strip() if convention_match else None
            
            # Find referenced convention
            referenced_convention = None
            if convention_ref:
                for match in context.relevant_conventions:
                    if (match.rule.id == convention_ref or 
                        convention_ref in match.rule.title or
                        match.rule.title in convention_ref):
                        referenced_convention = match.rule
                        break
            
            # Use first available convention if no specific reference found
            if not referenced_convention and context.relevant_conventions:
                referenced_convention = context.relevant_conventions[0].rule
            
            # Extract code snippet from context
            code_snippet = ""
            if context.code_after:
                lines = context.code_after.split('\n')
                if line_number <= len(lines):
                    start_line = max(0, line_number - 3)
                    end_line = min(len(lines), line_number + 2)
                    code_snippet = '\n'.join(lines[start_line:end_line])
            
            return ReviewComment(
                file_path=file_path,
                line_start=line_number,
                line_end=line_number,
                comment_type=comment_type,
                severity=severity,
                title=title,
                description=description,
                code_snippet=code_snippet,
                suggestion=None,  # Could be extracted from description
                convention_reference=referenced_convention,
                created_at=None  # Will be set by caller
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse comment block: {e}")
            return None
    
    def _parse_batch_review(
        self,
        generated_text: str,
        contexts: List[ReviewContext],
        language: str
    ) -> List[ReviewResult]:
        """Parse batch review results."""
        results = []
        
        # Split by context sections
        if language == "korean":
            context_pattern = r'## 컨텍스트 (\d+):'
        else:
            context_pattern = r'## Context (\d+):'
        
        sections = re.split(context_pattern, generated_text)
        
        # First section is usually the header, skip it
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                context_num = int(sections[i]) - 1  # Convert to 0-based index
                section_content = sections[i + 1]
                
                if context_num < len(contexts):
                    context = contexts[context_num]
                    comments = self._parse_generated_review(section_content, context, language)
                    
                    quality_score = self._calculate_quality_score(section_content, comments, context)
                    
                    result = ReviewResult(
                        comments=comments,
                        generation_metadata={
                            'context_number': context_num + 1,
                            'section_content_length': len(section_content)
                        },
                        quality_score=quality_score,
                        processing_time=0.0  # Will be set by caller
                    )
                    
                    results.append(result)
        
        # Ensure we have results for all contexts
        while len(results) < len(contexts):
            results.append(ReviewResult(
                comments=[],
                generation_metadata={'context_number': len(results) + 1},
                quality_score=0.0,
                processing_time=0.0
            ))
        
        return results
    
    def _calculate_quality_score(
        self,
        generated_text: str,
        comments: List[ReviewComment],
        context: ReviewContext
    ) -> float:
        """Calculate quality score for generated review."""
        score = 0.0
        
        # Base score for generating any content
        if generated_text.strip():
            score += 0.2
        
        # Score for structured comments
        if comments:
            score += 0.3
            
            # Bonus for multiple comments
            if len(comments) > 1:
                score += 0.1
        
        # Score for convention references
        convention_refs = sum(1 for c in comments if c.convention_reference)
        if convention_refs > 0:
            score += 0.2 * min(convention_refs / len(comments), 1.0)
        
        # Score for appropriate comment types
        appropriate_types = ['violation', 'suggestion', 'praise']
        type_score = sum(1 for c in comments if c.comment_type in appropriate_types)
        if comments:
            score += 0.1 * (type_score / len(comments))
        
        # Score for severity assignment
        severity_assigned = sum(1 for c in comments if c.severity in ['high', 'medium', 'low'])
        if comments:
            score += 0.1 * (severity_assigned / len(comments))
        
        # Penalty for very short or generic comments
        if comments:
            avg_description_length = sum(len(c.description) for c in comments) / len(comments)
            if avg_description_length < 20:
                score -= 0.1
        
        # Bonus for Korean language appropriateness (if applicable)
        if '존댓말' in generated_text or '습니다' in generated_text or '해보면' in generated_text:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def validate_review_quality(
        self,
        review_result: ReviewResult,
        context: ReviewContext
    ) -> bool:
        """
        Validate if the generated review meets quality standards.
        
        Args:
            review_result: Generated review result
            context: Original context
            
        Returns:
            True if review meets quality standards
        """
        # Minimum quality threshold
        if review_result.quality_score < 0.3:
            return False
        
        # Check for convention grounding
        if context.relevant_conventions and review_result.comments:
            grounded_comments = sum(
                1 for c in review_result.comments 
                if c.convention_reference is not None
            )
            
            # At least 50% of comments should reference conventions
            if grounded_comments / len(review_result.comments) < 0.5:
                return False
        
        # Check for appropriate comment structure
        for comment in review_result.comments:
            if not comment.title or not comment.description:
                return False
            
            if len(comment.description) < 10:  # Too short
                return False
        
        return True
    
    def get_generation_statistics(self) -> Dict[str, any]:
        """Get statistics about the generator."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'generation_config': self.generation_config.__dict__,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        }