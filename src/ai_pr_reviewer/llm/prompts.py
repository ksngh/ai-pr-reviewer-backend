"""
Prompt Builder

Builds structured prompts for LLM review generation following
Hwahae's collaborative review style and convention-grounded approach.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..models.convention import ConventionRule
from ..review.context import ReviewContext
from ..review.matcher import ConventionMatch


logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds structured prompts for LLM review generation.
    
    Creates prompts that follow Hwahae's collaborative review culture
    and ensure convention-grounded feedback.
    """
    
    def __init__(self, language: str = "korean"):
        """
        Initialize prompt builder.
        
        Args:
            language: Language for review generation ("korean" or "english")
        """
        self.language = language
        
        # Hwahae style guidelines
        self.hwahae_guidelines = {
            "korean": {
                "tone": "협력적이고 건설적인",
                "approach": "함께 개선해나가는",
                "feedback_style": "구체적이고 실행 가능한",
                "politeness": "존댓말 사용",
                "focus": "팀 컨벤션 준수"
            },
            "english": {
                "tone": "collaborative and constructive",
                "approach": "working together to improve",
                "feedback_style": "specific and actionable",
                "politeness": "professional and respectful",
                "focus": "team convention adherence"
            }
        }
        
        # Template components
        self.templates = self._load_templates()
    
    def build_review_prompt(
        self,
        context: ReviewContext,
        system_context: Optional[str] = None
    ) -> str:
        """
        Build complete review prompt for a single context.
        
        Args:
            context: ReviewContext to generate prompt for
            system_context: Optional system context information
            
        Returns:
            Complete prompt string
        """
        logger.debug(f"Building review prompt for {context.file_path}")
        
        # Select template based on language
        template = self.templates[self.language]
        
        # Build prompt sections
        sections = []
        
        # System context
        sections.append(template["system_prompt"])
        
        if system_context:
            sections.append(f"\n{template['system_context_header']}\n{system_context}")
        
        # Hwahae style guidelines
        sections.append(template["hwahae_guidelines"])
        
        # Convention rules
        if context.relevant_conventions:
            conventions_text = self._format_conventions(context.relevant_conventions)
            sections.append(f"\n{template['conventions_header']}\n{conventions_text}")
        
        # Code context
        code_context_text = self._format_code_context(context)
        sections.append(f"\n{template['code_context_header']}\n{code_context_text}")
        
        # Review instructions
        sections.append(template["review_instructions"])
        
        # Output format
        sections.append(template["output_format"])
        
        return "\n".join(sections)
    
    def build_batch_review_prompt(
        self,
        contexts: List[ReviewContext],
        system_context: Optional[str] = None
    ) -> str:
        """
        Build prompt for reviewing multiple contexts together.
        
        Args:
            contexts: List of ReviewContext objects
            system_context: Optional system context information
            
        Returns:
            Complete batch review prompt
        """
        logger.debug(f"Building batch review prompt for {len(contexts)} contexts")
        
        template = self.templates[self.language]
        
        sections = []
        
        # System context
        sections.append(template["system_prompt"])
        
        if system_context:
            sections.append(f"\n{template['system_context_header']}\n{system_context}")
        
        # Hwahae style guidelines
        sections.append(template["hwahae_guidelines"])
        
        # Collect all unique conventions
        all_conventions = {}
        for context in contexts:
            for match in context.relevant_conventions:
                all_conventions[match.rule.id] = match
        
        if all_conventions:
            conventions_text = self._format_conventions(list(all_conventions.values()))
            sections.append(f"\n{template['conventions_header']}\n{conventions_text}")
        
        # Multiple code contexts
        sections.append(f"\n{template['batch_context_header']}")
        
        for i, context in enumerate(contexts, 1):
            code_context_text = self._format_code_context(context)
            sections.append(f"\n### {template['context_number'].format(number=i)}\n{code_context_text}")
        
        # Batch review instructions
        sections.append(template["batch_review_instructions"])
        
        # Output format
        sections.append(template["batch_output_format"])
        
        return "\n".join(sections)
    
    def _format_conventions(self, convention_matches: List[ConventionMatch]) -> str:
        """Format convention rules for prompt inclusion."""
        if self.language == "korean":
            formatted = ["다음은 팀에서 정의한 코딩 컨벤션입니다:"]
        else:
            formatted = ["The following are team-defined coding conventions:"]
        
        for i, match in enumerate(convention_matches, 1):
            rule = match.rule
            
            if self.language == "korean":
                rule_text = f"{i}. **{rule.title}** ({rule.rule_type})\n"
                rule_text += f"   설명: {rule.description}\n"
                rule_text += f"   카테고리: {rule.category}\n"
                
                if rule.examples:
                    rule_text += f"   좋은 예시:\n"
                    for example in rule.examples[:2]:  # Limit examples
                        rule_text += f"   ```\n   {example}\n   ```\n"
                
                if rule.counter_examples:
                    rule_text += f"   피해야 할 예시:\n"
                    for counter_example in rule.counter_examples[:2]:
                        rule_text += f"   ```\n   {counter_example}\n   ```\n"
            else:
                rule_text = f"{i}. **{rule.title}** ({rule.rule_type})\n"
                rule_text += f"   Description: {rule.description}\n"
                rule_text += f"   Category: {rule.category}\n"
                
                if rule.examples:
                    rule_text += f"   Good examples:\n"
                    for example in rule.examples[:2]:
                        rule_text += f"   ```\n   {example}\n   ```\n"
                
                if rule.counter_examples:
                    rule_text += f"   Examples to avoid:\n"
                    for counter_example in rule.counter_examples[:2]:
                        rule_text += f"   ```\n   {counter_example}\n   ```\n"
            
            rule_text += f"   관련성: {match.relevance_score:.2f} ({match.match_reason})\n"
            formatted.append(rule_text)
        
        return "\n".join(formatted)
    
    def _format_code_context(self, context: ReviewContext) -> str:
        """Format code context for prompt inclusion."""
        if self.language == "korean":
            formatted = [f"**파일**: {context.file_path}"]
            
            if context.language:
                formatted.append(f"**언어**: {context.language}")
            
            formatted.append(f"**변경 요약**: {context.change_summary}")
            
            if context.code_before:
                formatted.append("**변경 전 코드**:")
                formatted.append(f"```{context.language or ''}")
                formatted.append(context.code_before)
                formatted.append("```")
            
            if context.code_after:
                formatted.append("**변경 후 코드**:")
                formatted.append(f"```{context.language or ''}")
                formatted.append(context.code_after)
                formatted.append("```")
        else:
            formatted = [f"**File**: {context.file_path}"]
            
            if context.language:
                formatted.append(f"**Language**: {context.language}")
            
            formatted.append(f"**Change Summary**: {context.change_summary}")
            
            if context.code_before:
                formatted.append("**Code Before**:")
                formatted.append(f"```{context.language or ''}")
                formatted.append(context.code_before)
                formatted.append("```")
            
            if context.code_after:
                formatted.append("**Code After**:")
                formatted.append(f"```{context.language or ''}")
                formatted.append(context.code_after)
                formatted.append("```")
        
        return "\n".join(formatted)
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load prompt templates for different languages."""
        return {
            "korean": {
                "system_prompt": """당신은 화해 팀의 코드 리뷰어입니다. 팀에서 정의한 코딩 컨벤션을 기반으로 건설적이고 협력적인 코드 리뷰를 제공합니다.

**핵심 원칙**:
- 팀 컨벤션에 기반한 객관적인 피드백만 제공
- 협력적이고 존중하는 톤 유지
- 구체적이고 실행 가능한 개선 제안
- 좋은 점도 함께 언급하여 균형잡힌 리뷰""",
                
                "system_context_header": "**시스템 컨텍스트**:",
                
                "hwahae_guidelines": """**화해 리뷰 스타일 가이드라인**:
- 존댓말을 사용하여 정중하게 소통합니다
- "이 부분을 개선해보면 어떨까요?" 같은 제안형 표현을 사용합니다
- 문제점과 함께 구체적인 해결 방안을 제시합니다
- 코드의 좋은 점도 함께 언급하여 격려합니다
- 팀 컨벤션을 명시적으로 참조합니다""",
                
                "conventions_header": "**적용 가능한 팀 컨벤션**:",
                
                "code_context_header": "**검토할 코드 변경사항**:",
                
                "batch_context_header": "**검토할 코드 변경사항들**:",
                
                "context_number": "컨텍스트 {number}",
                
                "review_instructions": """**리뷰 지침**:
1. 제공된 팀 컨벤션만을 기준으로 리뷰합니다
2. 컨벤션 위반이나 개선점이 있는 경우에만 코멘트를 작성합니다
3. 각 코멘트는 구체적인 컨벤션 규칙을 참조해야 합니다
4. 코드의 좋은 점이 있다면 함께 언급합니다
5. 주관적인 의견이나 개인적 선호는 배제합니다""",
                
                "batch_review_instructions": """**배치 리뷰 지침**:
1. 각 컨텍스트를 개별적으로 검토합니다
2. 컨텍스트 간의 일관성도 확인합니다
3. 중복되는 이슈는 한 번만 언급합니다
4. 전체적인 변경사항의 품질을 평가합니다""",
                
                "output_format": """**출력 형식**:
각 코멘트는 다음 형식으로 작성해주세요:

```
파일: [파일경로]
라인: [라인번호] (해당하는 경우)
유형: [violation/suggestion/praise]
심각도: [high/medium/low]

**[컨벤션 규칙 제목]**
[구체적인 피드백 내용]

참조 컨벤션: [컨벤션 ID]
```

컨벤션 위반이나 개선점이 없다면 "검토 완료: 팀 컨벤션을 잘 준수하고 있습니다."라고 응답해주세요.""",
                
                "batch_output_format": """**배치 출력 형식**:
각 컨텍스트별로 코멘트를 그룹화하여 작성해주세요:

```
## 컨텍스트 1: [파일명]
[개별 코멘트들]

## 컨텍스트 2: [파일명]
[개별 코멘트들]

## 전체 요약
[전체적인 리뷰 요약]
```"""
            },
            
            "english": {
                "system_prompt": """You are a code reviewer for the Hwahae team. You provide constructive and collaborative code reviews based on team-defined coding conventions.

**Core Principles**:
- Provide objective feedback based only on team conventions
- Maintain a collaborative and respectful tone
- Offer specific and actionable improvement suggestions
- Mention positive aspects for balanced reviews""",
                
                "system_context_header": "**System Context**:",
                
                "hwahae_guidelines": """**Hwahae Review Style Guidelines**:
- Use polite and respectful language
- Use suggestive expressions like "How about improving this part?"
- Provide specific solutions along with problems
- Mention good aspects of the code for encouragement
- Explicitly reference team conventions""",
                
                "conventions_header": "**Applicable Team Conventions**:",
                
                "code_context_header": "**Code Changes to Review**:",
                
                "batch_context_header": "**Code Changes to Review**:",
                
                "context_number": "Context {number}",
                
                "review_instructions": """**Review Guidelines**:
1. Review based only on the provided team conventions
2. Write comments only when there are convention violations or improvements
3. Each comment must reference specific convention rules
4. Mention good aspects of the code if present
5. Exclude subjective opinions or personal preferences""",
                
                "batch_review_instructions": """**Batch Review Guidelines**:
1. Review each context individually
2. Check consistency between contexts
3. Mention duplicate issues only once
4. Evaluate overall quality of changes""",
                
                "output_format": """**Output Format**:
Please write each comment in the following format:

```
File: [file_path]
Line: [line_number] (if applicable)
Type: [violation/suggestion/praise]
Severity: [high/medium/low]

**[Convention Rule Title]**
[Specific feedback content]

Referenced Convention: [Convention ID]
```

If there are no convention violations or improvements, respond with "Review Complete: Code follows team conventions well.""",
                
                "batch_output_format": """**Batch Output Format**:
Please group comments by context:

```
## Context 1: [filename]
[Individual comments]

## Context 2: [filename]
[Individual comments]

## Overall Summary
[Overall review summary]
```"""
            }
        }
    
    def create_hwahae_style_template(self) -> str:
        """Create a template specifically for Hwahae's collaborative style."""
        if self.language == "korean":
            return """화해 팀 코드 리뷰 템플릿:

**좋은 점**:
- [코드의 긍정적인 측면들]

**개선 제안**:
- [구체적인 개선 방안들]

**컨벤션 관련**:
- [팀 컨벤션 준수 상태]

**추가 고려사항**:
- [성능, 보안, 유지보수성 등]"""
        else:
            return """Hwahae Team Code Review Template:

**Positive Aspects**:
- [Positive aspects of the code]

**Improvement Suggestions**:
- [Specific improvement recommendations]

**Convention Related**:
- [Team convention compliance status]

**Additional Considerations**:
- [Performance, security, maintainability, etc.]"""
    
    def format_conventions_summary(self, conventions: List[ConventionRule]) -> str:
        """Format a summary of conventions for context."""
        if not conventions:
            return ""
        
        if self.language == "korean":
            summary = ["**적용된 컨벤션 요약**:"]
            
            by_category = {}
            for conv in conventions:
                category = conv.category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(conv)
            
            for category, rules in by_category.items():
                summary.append(f"- {category}: {len(rules)}개 규칙")
                for rule in rules[:3]:  # Show first 3
                    summary.append(f"  • {rule.title}")
                if len(rules) > 3:
                    summary.append(f"  • ... 외 {len(rules) - 3}개")
        else:
            summary = ["**Applied Conventions Summary**:"]
            
            by_category = {}
            for conv in conventions:
                category = conv.category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(conv)
            
            for category, rules in by_category.items():
                summary.append(f"- {category}: {len(rules)} rules")
                for rule in rules[:3]:
                    summary.append(f"  • {rule.title}")
                if len(rules) > 3:
                    summary.append(f"  • ... and {len(rules) - 3} more")
        
        return "\n".join(summary)