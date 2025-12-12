"""
Prompt Builder

Builds Hwahae-style prompts for LLM review generation.
Creates convention-grounded prompts with Korean collaborative tone.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..review.context import ReviewContext
from ..review.matcher import ConventionMatch
from ..models.convention import ConventionRule


logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for LLM review generation.
    
    Creates structured prompts that incorporate convention rules
    and maintain Hwahae's collaborative review culture.
    """
    
    def __init__(self, language: str = "korean"):
        """
        Initialize prompt builder.
        
        Args:
            language: Language for prompts ("korean" or "english")
        """
        self.language = language
        
        # Hwahae style templates
        self.hwahae_templates = {
            "korean": {
                "system_prompt": """당신은 화해 팀의 코드 리뷰어입니다. 팀에서 정의한 컨벤션 규칙을 기반으로 건설적이고 협력적인 코드 리뷰를 제공합니다.

리뷰 원칙:
1. 팀 컨벤션에 명시된 규칙만을 기준으로 피드백 제공
2. 협력적이고 존중하는 톤 유지
3. 구체적인 개선 방안 제시
4. 긍정적인 측면도 함께 언급
5. 학습과 성장을 돕는 방향으로 소통

응답 형식:
- 컨벤션 위반이나 개선점이 있을 때만 코멘트 작성
- 각 코멘트는 관련 컨벤션 규칙을 명시
- 구체적인 수정 제안 포함
- 한국어로 작성""",
                
                "review_template": """## 코드 리뷰

### 변경 사항 분석
{change_summary}

### 코드 스니펫
```{language}
{code_snippet}
```

### 적용 가능한 컨벤션 규칙
{conventions}

### 리뷰 요청
위 컨벤션 규칙을 기반으로 코드 변경사항을 검토하고, 필요한 경우에만 개선 제안을 해주세요. 
컨벤션을 잘 따르고 있다면 별도 코멘트 없이 "LGTM"으로 응답해주세요."""
            }
        }
    
    def build_review_prompt(self, context: ReviewContext) -> str:
        """
        Build a complete review prompt for a context.
        
        Args:
            context: ReviewContext with code and conventions
            
        Returns:
            Formatted prompt string
        """
        logger.debug(f"Building review prompt for {context.file_path}")
        
        templates = self.hwahae_templates[self.language]
        
        # Format conventions
        conventions_text = self._format_conventions(context.relevant_conventions)
        
        # Build main prompt
        prompt = templates["review_template"].format(
            change_summary=context.change_summary,
            language=context.language or "text",
            code_snippet=context.code_snippet,
            conventions=conventions_text
        )
        
        return prompt
    
    def build_system_prompt(self) -> str:
        """Build system prompt for LLM."""
        return self.hwahae_templates[self.language]["system_prompt"]
    
    def _format_conventions(self, matches: List[ConventionMatch]) -> str:
        """Format convention matches for prompt."""
        if not matches:
            return "적용 가능한 컨벤션 규칙이 없습니다."
        
        formatted_conventions = []
        
        for i, match in enumerate(matches, 1):
            rule = match.rule
            
            # Format rule type in Korean
            rule_type_map = {
                "mandatory": "필수",
                "recommended": "권장",
                "prohibited": "금지"
            }
            
            rule_type = rule_type_map.get(rule.rule_type, rule.rule_type)
            
            convention_text = f"""**{rule_type} 규칙: {rule.title}**
- 설명: {rule.description}
- 카테고리: {rule.category}"""
            
            # Add examples if available
            if rule.examples:
                convention_text += "\n- 예시:"
                for example in rule.examples[:2]:  # Limit to 2 examples
                    convention_text += f"\n  ```\n  {example}\n  ```"
            
            formatted_conventions.append(f"{i}. {convention_text}")
        
        return "\n\n".join(formatted_conventions)