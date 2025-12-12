"""
Review Generator

Generates code reviews using LLM with convention-grounded prompts.
Handles LLM inference and response processing.
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

from ..review.context import ReviewContext
from ..models.review import ReviewComment
from .prompts import PromptBuilder
from .quality import QualityController


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None


class ReviewGenerator:
    """
    Generates code reviews using LLM.
    
    Uses open-source language models to generate convention-based
    code reviews with Hwahae collaborative tone.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: Optional[str] = None,
        language: str = "korean"
    ):
        """
        Initialize review generator.
        
        Args:
            model_name: Name of the LLM model to use
            device: Device to run model on ('cpu', 'cuda', etc.)
            language: Language for reviews ("korean" or "english")
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        
        # Initialize components
        self.prompt_builder = PromptBuilder(language=language)
        self.quality_controller = QualityController()
        
        # Initialize model and tokenizer
        logger.info(f"Loading LLM model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id
        )
    
    def generate_review(self, context: ReviewContext) -> Optional[ReviewComment]:
        """
        Generate review for a single context.
        
        Args:
            context: ReviewContext to review
            
        Returns:
            ReviewComment or None if no review needed
        """
        logger.debug(f"Generating review for {context.file_path}")
        
        # Build prompt
        prompt = self.prompt_builder.build_review_prompt(context)
        
        # Generate response
        response = self._generate_text(prompt)
        
        if not response:
            logger.warning(f"No response generated for {context.file_path}")
            return None
        
        # Check if review is needed (LGTM check)
        if self._is_lgtm_response(response):
            logger.debug(f"LGTM response for {context.file_path}")
            return None
        
        # Parse response into ReviewComment
        review_comment = self._parse_review_response(response, context)
        
        # Quality control
        if review_comment and self.quality_controller.validate_review(review_comment, context):
            return review_comment
        
        logger.warning(f"Review failed quality control for {context.file_path}")
        return None
    
    def generate_batch_reviews(
        self, 
        contexts: List[ReviewContext]
    ) -> List[Optional[ReviewComment]]:
        """
        Generate reviews for multiple contexts.
        
        Args:
            contexts: List of ReviewContext objects
            
        Returns:
            List of ReviewComment objects (None for LGTM cases)
        """
        logger.info(f"Generating batch reviews for {len(contexts)} contexts")
        
        reviews = []
        
        for context in contexts:
            try:
                review = self.generate_review(context)
                reviews.append(review)
            except Exception as e:
                logger.error(f"Failed to generate review for {context.file_path}: {e}")
                reviews.append(None)
        
        logger.info(f"Generated {sum(1 for r in reviews if r is not None)} reviews")
        return reviews
    
    def _generate_text(self, prompt: str) -> Optional[str]:
        """Generate text using the LLM."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.generation_config.max_length,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    do_sample=self.generation_config.do_sample,
                    pad_token_id=self.generation_config.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return None
    
    def _is_lgtm_response(self, response: str) -> bool:
        """Check if response indicates LGTM (no review needed)."""
        lgtm_indicators = [
            "LGTM",
            "lgtm", 
            "looks good to me",
            "컨벤션을 잘 따르고 있습니다",
            "문제없습니다",
            "좋습니다",
            "잘 작성되었습니다"
        ]
        
        response_lower = response.lower().strip()
        
        # Check for exact LGTM
        if response_lower == "lgtm":
            return True
        
        # Check for LGTM indicators in short responses
        if len(response_lower) < 50:
            for indicator in lgtm_indicators:
                if indicator.lower() in response_lower:
                    return True
        
        return False
    
    def _parse_review_response(
        self, 
        response: str, 
        context: ReviewContext
    ) -> Optional[ReviewComment]:
        """Parse LLM response into ReviewComment."""
        if not response or len(response.strip()) < 10:
            return None
        
        # Apply Hwahae tone
        formatted_response = self.prompt_builder.format_hwahae_tone(response)
        
        # Extract title (first line or sentence)
        lines = formatted_response.split('\n')
        title = lines[0].strip()
        if len(title) > 100:
            title = title[:97] + "..."
        
        # Determine severity based on conventions
        severity = self._determine_severity(context, formatted_response)
        
        # Determine comment type
        comment_type = self._determine_comment_type(formatted_response)
        
        # Extract suggestion if available
        suggestion = self._extract_suggestion(formatted_response)
        
        # Get primary convention reference
        convention_ref = None
        if context.relevant_conventions:
            convention_ref = context.relevant_conventions[0].rule
        
        return ReviewComment(
            file_path=context.file_path,
            line_start=1,  # Default to first line
            line_end=1,
            comment_type=comment_type,
            severity=severity,
            title=title,
            description=formatted_response,
            code_snippet=context.code_snippet[:200],  # Limit snippet
            suggestion=suggestion,
            convention_reference=convention_ref
        )
    
    def _determine_severity(self, context: ReviewContext, response: str) -> str:
        """Determine severity based on context and response."""
        # Check for high-priority conventions
        for match in context.relevant_conventions:
            if match.rule.rule_type == 'prohibited':
                return 'high'
            elif match.rule.rule_type == 'mandatory':
                return 'medium'
        
        # Check response content for severity indicators
        high_severity_words = ['문제', '오류', '잘못', '위반', 'error', 'wrong', 'violation']
        medium_severity_words = ['개선', '수정', 'improve', 'fix', 'change']
        
        response_lower = response.lower()
        
        if any(word in response_lower for word in high_severity_words):
            return 'high'
        elif any(word in response_lower for word in medium_severity_words):
            return 'medium'
        
        return 'low'
    
    def _determine_comment_type(self, response: str) -> str:
        """Determine comment type from response content."""
        violation_words = ['위반', '문제', '오류', 'violation', 'error', 'wrong']
        suggestion_words = ['제안', '개선', '추천', 'suggest', 'recommend', 'improve']
        question_words = ['질문', '확인', '어떻게', 'question', 'how', 'why']
        
        response_lower = response.lower()
        
        if any(word in response_lower for word in violation_words):
            return 'violation'
        elif any(word in response_lower for word in question_words):
            return 'question'
        else:
            return 'suggestion'
    
    def _extract_suggestion(self, response: str) -> Optional[str]:
        """Extract suggestion from response."""
        # Look for suggestion patterns
        suggestion_patterns = [
            r'제안[:\s]*(.+)',
            r'추천[:\s]*(.+)',
            r'개선[:\s]*(.+)',
            r'suggestion[:\s]*(.+)',
            r'recommend[:\s]*(.+)',
        ]
        
        for pattern in suggestion_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                suggestion = match.group(1).strip()
                # Clean up suggestion
                suggestion = re.sub(r'\n+', ' ', suggestion)
                if len(suggestion) > 200:
                    suggestion = suggestion[:197] + "..."
                return suggestion
        
        # If no explicit suggestion pattern, use the whole response as suggestion
        if len(response) < 300:
            return response.strip()
        
        return None
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'language': self.language,
            'vocab_size': self.tokenizer.vocab_size,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'generation_config': {
                'max_length': self.generation_config.max_length,
                'temperature': self.generation_config.temperature,
                'top_p': self.generation_config.top_p,
            }
        }
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration."""
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)
                logger.info(f"Updated generation config: {key} = {value}")
            else:
                logger.warning(f"Unknown generation config parameter: {key}")
    
    def clear_cache(self):
        """Clear model cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared model cache")