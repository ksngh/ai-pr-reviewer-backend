"""
Convention Extractor

Extracts structured convention rules from wiki documents.
Parses markdown content and identifies rule patterns.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..models.convention import ConventionRule


logger = logging.getLogger(__name__)


@dataclass
class WikiSection:
    """Represents a section in a wiki document."""
    title: str
    level: int
    content: str
    line_start: int
    line_end: int


class ConventionExtractor:
    """
    Extracts convention rules from wiki markdown content.
    
    Identifies structured patterns in wiki documents and converts
    them into ConventionRule objects with proper categorization.
    """
    
    def __init__(self):
        """Initialize convention extractor."""
        # Patterns for identifying convention rules
        self.rule_patterns = {
            'mandatory': [
                r'(?i)(must|required|mandatory|shall|always)',
                r'(?i)(금지|필수|반드시|해야|하지 말)',
            ],
            'recommended': [
                r'(?i)(should|recommended|prefer|better|좋은)',
                r'(?i)(권장|추천|바람직|좋다)',
            ],
            'prohibited': [
                r'(?i)(must not|forbidden|prohibited|never|avoid)',
                r'(?i)(금지|하지 말|피해|안 됨)',
            ]
        }
        
        # Patterns for code examples
        self.code_block_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        
        # Header patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        # List patterns
        self.list_pattern = re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE)
        self.numbered_list_pattern = re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE)
    
    def extract_rules(self, wiki_content: str, source_page: str) -> List[ConventionRule]:
        """
        Extract convention rules from wiki content.
        
        Args:
            wiki_content: Raw wiki markdown content
            source_page: Name of the source wiki page
            
        Returns:
            List of extracted ConventionRule objects
        """
        logger.info(f"Extracting rules from wiki page: {source_page}")
        
        if not wiki_content.strip():
            logger.warning(f"Empty wiki content for page: {source_page}")
            return []
        
        # Parse document structure
        sections = self._parse_sections(wiki_content)
        logger.debug(f"Found {len(sections)} sections in {source_page}")
        
        # Extract rules from each section
        rules = []
        for section in sections:
            section_rules = self._extract_rules_from_section(section, source_page)
            rules.extend(section_rules)
        
        logger.info(f"Extracted {len(rules)} rules from {source_page}")
        return rules
    
    def _parse_sections(self, content: str) -> List[WikiSection]:
        """
        Parse wiki content into sections based on headers.
        
        Args:
            content: Wiki markdown content
            
        Returns:
            List of WikiSection objects
        """
        lines = content.split('\n')
        sections = []
        current_section = None
        
        for i, line in enumerate(lines):
            header_match = self.header_pattern.match(line)
            
            if header_match:
                # Save previous section
                if current_section:
                    current_section.line_end = i - 1
                    current_section.content = '\n'.join(lines[current_section.line_start:i])
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                current_section = WikiSection(
                    title=title,
                    level=level,
                    content='',
                    line_start=i + 1,
                    line_end=len(lines)
                )
        
        # Save last section
        if current_section:
            current_section.content = '\n'.join(lines[current_section.line_start:])
            sections.append(current_section)
        
        # If no headers found, treat entire content as one section
        if not sections and content.strip():
            sections.append(WikiSection(
                title='General Rules',
                level=1,
                content=content,
                line_start=0,
                line_end=len(lines)
            ))
        
        return sections
    
    def _extract_rules_from_section(self, section: WikiSection, source_page: str) -> List[ConventionRule]:
        """
        Extract rules from a specific section.
        
        Args:
            section: WikiSection to process
            source_page: Source wiki page name
            
        Returns:
            List of ConventionRule objects
        """
        rules = []
        
        # Extract list items as potential rules
        list_items = self._extract_list_items(section.content)
        
        for item in list_items:
            rule = self._parse_rule_from_text(item, section, source_page)
            if rule:
                rules.append(rule)
        
        # Also check for paragraph-based rules
        paragraphs = self._extract_paragraphs(section.content)
        for paragraph in paragraphs:
            if self._is_rule_paragraph(paragraph):
                rule = self._parse_rule_from_text(paragraph, section, source_page)
                if rule:
                    rules.append(rule)
        
        return rules
    
    def _extract_list_items(self, content: str) -> List[str]:
        """Extract list items from content."""
        items = []
        
        # Extract bullet list items
        bullet_matches = self.list_pattern.findall(content)
        items.extend(bullet_matches)
        
        # Extract numbered list items
        numbered_matches = self.numbered_list_pattern.findall(content)
        items.extend(numbered_matches)
        
        return [item.strip() for item in items if item.strip()]
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """Extract paragraphs from content."""
        # Remove code blocks first
        content_no_code = self.code_block_pattern.sub('', content)
        
        # Split by double newlines
        paragraphs = content_no_code.split('\n\n')
        
        return [p.strip() for p in paragraphs if p.strip() and not p.startswith('#')]
    
    def _is_rule_paragraph(self, paragraph: str) -> bool:
        """Check if paragraph contains a rule."""
        # Check for rule indicators
        for rule_type, patterns in self.rule_patterns.items():
            for pattern in patterns:
                if re.search(pattern, paragraph):
                    return True
        
        return False
    
    def _parse_rule_from_text(self, text: str, section: WikiSection, source_page: str) -> Optional[ConventionRule]:
        """
        Parse a convention rule from text.
        
        Args:
            text: Text containing the rule
            section: Section containing the rule
            source_page: Source wiki page
            
        Returns:
            ConventionRule object or None
        """
        if not text.strip():
            return None
        
        # Determine rule type
        rule_type = self._determine_rule_type(text)
        
        # Extract title (first sentence or up to first period)
        title = self._extract_title(text)
        
        # Extract description
        description = text.strip()
        
        # Extract code examples
        examples, counter_examples = self._extract_examples(text)
        
        # Generate unique ID
        rule_id = self._generate_rule_id(title, source_page)
        
        # Determine category from section title
        category = self._determine_category(section.title)
        
        return ConventionRule(
            id=rule_id,
            title=title,
            description=description,
            rule_type=rule_type,
            examples=examples,
            counter_examples=counter_examples,
            category=category,
            source_wiki_page=source_page,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            version=1
        )
    
    def _determine_rule_type(self, text: str) -> str:
        """Determine rule type from text content."""
        text_lower = text.lower()
        
        # Check for prohibited patterns first (most specific)
        for pattern in self.rule_patterns['prohibited']:
            if re.search(pattern, text):
                return 'prohibited'
        
        # Check for mandatory patterns
        for pattern in self.rule_patterns['mandatory']:
            if re.search(pattern, text):
                return 'mandatory'
        
        # Check for recommended patterns
        for pattern in self.rule_patterns['recommended']:
            if re.search(pattern, text):
                return 'recommended'
        
        # Default to recommended
        return 'recommended'
    
    def _extract_title(self, text: str) -> str:
        """Extract title from rule text."""
        # Remove markdown formatting
        clean_text = re.sub(r'[*_`]', '', text)
        
        # Take first sentence or first 100 characters
        sentences = re.split(r'[.!?]', clean_text)
        if sentences:
            title = sentences[0].strip()
            if len(title) > 100:
                title = title[:97] + '...'
            return title
        
        # Fallback to first 100 characters
        if len(clean_text) > 100:
            return clean_text[:97] + '...'
        
        return clean_text.strip()
    
    def _extract_examples(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract code examples and counter-examples."""
        examples = []
        counter_examples = []
        
        # Extract code blocks
        code_blocks = self.code_block_pattern.findall(text)
        for lang, code in code_blocks:
            code = code.strip()
            if code:
                # Determine if it's a good or bad example based on context
                if self._is_counter_example(text, code):
                    counter_examples.append(code)
                else:
                    examples.append(code)
        
        # Extract inline code
        inline_codes = self.inline_code_pattern.findall(text)
        for code in inline_codes:
            code = code.strip()
            if code and len(code) > 5:  # Only meaningful code snippets
                if self._is_counter_example(text, code):
                    counter_examples.append(code)
                else:
                    examples.append(code)
        
        return examples, counter_examples
    
    def _is_counter_example(self, context: str, code: str) -> bool:
        """Determine if code is a counter-example based on context."""
        # Look for negative indicators around the code
        negative_indicators = [
            'bad', 'wrong', 'incorrect', 'avoid', 'don\'t', 'not',
            '나쁜', '잘못', '피해', '하지 말', '안 됨'
        ]
        
        # Check context around code
        code_index = context.find(code)
        if code_index != -1:
            # Check 100 characters before and after
            start = max(0, code_index - 100)
            end = min(len(context), code_index + len(code) + 100)
            surrounding_text = context[start:end].lower()
            
            for indicator in negative_indicators:
                if indicator in surrounding_text:
                    return True
        
        return False
    
    def _determine_category(self, section_title: str) -> str:
        """Determine category from section title."""
        title_lower = section_title.lower()
        
        # Common category mappings
        category_keywords = {
            'naming': ['naming', 'name', '네이밍', '이름'],
            'formatting': ['format', 'style', 'indent', '포맷', '스타일', '들여쓰기'],
            'structure': ['structure', 'organization', '구조', '구성'],
            'documentation': ['doc', 'comment', '문서', '주석'],
            'testing': ['test', 'spec', '테스트', '검증'],
            'performance': ['performance', 'optimization', '성능', '최적화'],
            'security': ['security', 'auth', '보안', '인증'],
            'error_handling': ['error', 'exception', '에러', '예외'],
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return category
        
        return 'general'
    
    def _generate_rule_id(self, title: str, source_page: str) -> str:
        """Generate unique rule ID."""
        # Create ID from title and source page
        clean_title = re.sub(r'[^\w\s-]', '', title.lower())
        clean_title = re.sub(r'\s+', '_', clean_title)
        clean_page = re.sub(r'[^\w\s-]', '', source_page.lower())
        clean_page = re.sub(r'\s+', '_', clean_page)
        
        # Limit length
        if len(clean_title) > 50:
            clean_title = clean_title[:50]
        
        return f"{clean_page}_{clean_title}_{hash(title + source_page) % 10000:04d}"
    
    def update_rule_version(self, existing_rule: ConventionRule, new_content: str) -> ConventionRule:
        """
        Update existing rule with new content.
        
        Args:
            existing_rule: Current rule version
            new_content: New rule content
            
        Returns:
            Updated ConventionRule with incremented version
        """
        # Parse new content
        temp_section = WikiSection(
            title=existing_rule.category,
            level=1,
            content=new_content,
            line_start=0,
            line_end=0
        )
        
        new_rule = self._parse_rule_from_text(
            new_content, 
            temp_section, 
            existing_rule.source_wiki_page
        )
        
        if new_rule:
            # Preserve ID and increment version
            new_rule.id = existing_rule.id
            new_rule.version = existing_rule.version + 1
            new_rule.created_at = existing_rule.created_at
            new_rule.updated_at = datetime.now()
            
            return new_rule
        
        return existing_rule