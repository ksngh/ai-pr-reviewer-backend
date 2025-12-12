"""
Convention Matcher

Matches code changes against convention rules using vector similarity.
Provides ranked matches with relevance scores.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..models.convention import ConventionRule
from ..conventions.embeddings import EmbeddingGenerator
from ..conventions.vector_store import VectorStore
from .analyzer import AnalyzedChunk, FileAnalysis, ChangeType


logger = logging.getLogger(__name__)


@dataclass
class ConventionMatch:
    """A matched convention rule with relevance information."""
    rule: ConventionRule
    similarity_score: float
    relevance_factors: List[str]
    matched_code: str
    confidence: float


@dataclass
class MatchingContext:
    """Context information for convention matching."""
    file_path: str
    language: Optional[str]
    change_types: List[ChangeType]
    code_snippet: str
    functions_affected: List[str]
    classes_affected: List[str]
    variables_affected: List[str]


class ConventionMatcher:
    """
    Matches code changes against convention rules using semantic similarity.
    
    Uses vector embeddings to find relevant conventions and ranks them
    by similarity and contextual relevance.
    """
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        similarity_threshold: float = 0.3,
        max_matches: int = 10
    ):
        """
        Initialize convention matcher.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            similarity_threshold: Minimum similarity score for matches
            max_matches: Maximum number of matches to return
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.max_matches = max_matches
        
        # Category weights for different change types
        self.category_weights = {
            ChangeType.FUNCTION_ADDED: {
                'naming': 1.5,
                'structure': 1.3,
                'documentation': 1.2,
                'testing': 1.1,
            },
            ChangeType.FUNCTION_MODIFIED: {
                'naming': 1.2,
                'structure': 1.4,
                'performance': 1.3,
                'error_handling': 1.2,
            },
            ChangeType.CLASS_ADDED: {
                'naming': 1.6,
                'structure': 1.5,
                'documentation': 1.3,
            },
            ChangeType.CLASS_MODIFIED: {
                'structure': 1.4,
                'naming': 1.2,
                'documentation': 1.1,
            },
            ChangeType.VARIABLE_MODIFIED: {
                'naming': 1.8,
                'formatting': 1.2,
            },
            ChangeType.IMPORT_ADDED: {
                'structure': 1.3,
                'performance': 1.1,
            },
        }
    
    def find_relevant_conventions(
        self, 
        analyzed_chunk: AnalyzedChunk
    ) -> List[ConventionMatch]:
        """
        Find conventions relevant to an analyzed chunk.
        
        Args:
            analyzed_chunk: AnalyzedChunk to match against conventions
            
        Returns:
            List of ConventionMatch objects sorted by relevance
        """
        logger.debug(f"Finding conventions for chunk in {analyzed_chunk.file_path}")
        
        # Create matching context
        context = self._create_matching_context(analyzed_chunk)
        
        # Generate embedding for the code change
        query_embedding = self._generate_query_embedding(context)
        
        # Search for similar conventions
        similar_conventions = self.vector_store.search_similar_conventions(
            query_embedding=query_embedding,
            limit=self.max_matches * 2,  # Get more to filter later
            score_threshold=self.similarity_threshold
        )
        
        # Create matches with additional context
        matches = []
        for rule, similarity_score in similar_conventions:
            match = self._create_convention_match(
                rule, similarity_score, context, analyzed_chunk
            )
            if match:
                matches.append(match)
        
        # Rank matches by relevance
        ranked_matches = self._rank_matches(matches, context)
        
        logger.debug(f"Found {len(ranked_matches)} relevant conventions")
        return ranked_matches[:self.max_matches]
    
    def find_conventions_for_file(
        self, 
        file_analysis: FileAnalysis
    ) -> Dict[str, List[ConventionMatch]]:
        """
        Find conventions for all chunks in a file.
        
        Args:
            file_analysis: FileAnalysis object
            
        Returns:
            Dictionary mapping chunk indices to convention matches
        """
        logger.info(f"Finding conventions for file: {file_analysis.file_change.file_path}")
        
        file_matches = {}
        
        for i, chunk in enumerate(file_analysis.analyzed_chunks):
            matches = self.find_relevant_conventions(chunk)
            if matches:
                file_matches[f"chunk_{i}"] = matches
        
        return file_matches
    
    def _create_matching_context(self, analyzed_chunk: AnalyzedChunk) -> MatchingContext:
        """Create matching context from analyzed chunk."""
        # Combine added and removed lines for context
        code_lines = analyzed_chunk.added_lines + analyzed_chunk.removed_lines
        code_snippet = '\n'.join(code_lines[:10])  # Limit to first 10 lines
        
        return MatchingContext(
            file_path=analyzed_chunk.file_path,
            language=analyzed_chunk.language,
            change_types=analyzed_chunk.change_types,
            code_snippet=code_snippet,
            functions_affected=analyzed_chunk.functions_affected,
            classes_affected=analyzed_chunk.classes_affected,
            variables_affected=analyzed_chunk.variables_affected
        )
    
    def _generate_query_embedding(self, context: MatchingContext) -> List[float]:
        """Generate embedding for matching query."""
        # Build query text from context
        query_parts = []
        
        # Add language context
        if context.language:
            query_parts.append(f"Language: {context.language}")
        
        # Add change type context
        change_descriptions = {
            ChangeType.FUNCTION_ADDED: "adding new function",
            ChangeType.FUNCTION_MODIFIED: "modifying function",
            ChangeType.CLASS_ADDED: "adding new class",
            ChangeType.CLASS_MODIFIED: "modifying class",
            ChangeType.VARIABLE_MODIFIED: "variable assignment",
            ChangeType.IMPORT_ADDED: "adding import",
        }
        
        for change_type in context.change_types:
            if change_type in change_descriptions:
                query_parts.append(change_descriptions[change_type])
        
        # Add affected elements
        if context.functions_affected:
            query_parts.append(f"functions: {', '.join(context.functions_affected[:3])}")
        
        if context.classes_affected:
            query_parts.append(f"classes: {', '.join(context.classes_affected[:3])}")
        
        # Add code snippet (limited)
        if context.code_snippet:
            # Clean and limit code snippet
            clean_code = context.code_snippet.replace('\n', ' ').strip()
            if len(clean_code) > 200:
                clean_code = clean_code[:200] + "..."
            query_parts.append(f"code: {clean_code}")
        
        # Combine all parts
        query_text = " | ".join(query_parts)
        
        # Generate embedding
        return self.embedding_generator.generate_embedding(query_text)
    
    def _create_convention_match(
        self,
        rule: ConventionRule,
        similarity_score: float,
        context: MatchingContext,
        analyzed_chunk: AnalyzedChunk
    ) -> Optional[ConventionMatch]:
        """Create a ConventionMatch with relevance analysis."""
        # Calculate relevance factors
        relevance_factors = self._calculate_relevance_factors(rule, context)
        
        # Skip if no relevance factors found
        if not relevance_factors:
            return None
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            rule, similarity_score, context, relevance_factors
        )
        
        # Skip low-confidence matches
        if confidence < 0.3:
            return None
        
        return ConventionMatch(
            rule=rule,
            similarity_score=similarity_score,
            relevance_factors=relevance_factors,
            matched_code=context.code_snippet,
            confidence=confidence
        )
    
    def _calculate_relevance_factors(
        self, 
        rule: ConventionRule, 
        context: MatchingContext
    ) -> List[str]:
        """Calculate why a rule is relevant to the context."""
        factors = []
        
        # Language match
        if context.language and context.language.lower() in rule.description.lower():
            factors.append(f"language_match_{context.language}")
        
        # Category relevance for change types
        for change_type in context.change_types:
            if change_type in self.category_weights:
                category_weights = self.category_weights[change_type]
                if rule.category in category_weights:
                    weight = category_weights[rule.category]
                    if weight > 1.0:
                        factors.append(f"category_relevant_{rule.category}")
        
        # Function name matches
        for func_name in context.functions_affected:
            if func_name.lower() in rule.description.lower() or \
               any(func_name.lower() in example.lower() for example in rule.examples):
                factors.append(f"function_name_match_{func_name}")
        
        # Class name matches
        for class_name in context.classes_affected:
            if class_name.lower() in rule.description.lower() or \
               any(class_name.lower() in example.lower() for example in rule.examples):
                factors.append(f"class_name_match_{class_name}")
        
        # Rule type relevance
        high_priority_types = {'mandatory', 'prohibited'}
        if rule.rule_type in high_priority_types:
            factors.append(f"high_priority_{rule.rule_type}")
        
        # Code pattern matches
        if rule.examples:
            for example in rule.examples[:3]:  # Check first 3 examples
                if self._has_code_pattern_similarity(context.code_snippet, example):
                    factors.append("code_pattern_match")
                    break
        
        return factors
    
    def _has_code_pattern_similarity(self, code1: str, code2: str) -> bool:
        """Check if two code snippets have similar patterns."""
        if not code1 or not code2:
            return False
        
        # Simple pattern matching - could be enhanced
        code1_clean = code1.lower().replace(' ', '').replace('\n', '')
        code2_clean = code2.lower().replace(' ', '').replace('\n', '')
        
        # Check for common keywords
        keywords1 = set(code1_clean.split())
        keywords2 = set(code2_clean.split())
        
        if keywords1 and keywords2:
            overlap = len(keywords1.intersection(keywords2))
            total = len(keywords1.union(keywords2))
            similarity = overlap / total if total > 0 else 0
            return similarity > 0.3
        
        return False
    
    def _calculate_confidence(
        self,
        rule: ConventionRule,
        similarity_score: float,
        context: MatchingContext,
        relevance_factors: List[str]
    ) -> float:
        """Calculate confidence score for a match."""
        base_confidence = similarity_score
        
        # Boost confidence based on relevance factors
        factor_boost = len(relevance_factors) * 0.1
        base_confidence += factor_boost
        
        # Category-specific boosts
        for change_type in context.change_types:
            if change_type in self.category_weights:
                category_weights = self.category_weights[change_type]
                if rule.category in category_weights:
                    weight = category_weights[rule.category]
                    base_confidence *= weight
        
        # Rule type boost
        type_boosts = {
            'mandatory': 1.2,
            'prohibited': 1.3,
            'recommended': 1.0,
        }
        base_confidence *= type_boosts.get(rule.rule_type, 1.0)
        
        # Language match boost
        if context.language and context.language.lower() in rule.description.lower():
            base_confidence *= 1.1
        
        # Normalize to 0-1 range
        return min(base_confidence, 1.0)
    
    def _rank_matches(
        self, 
        matches: List[ConventionMatch], 
        context: MatchingContext
    ) -> List[ConventionMatch]:
        """Rank matches by relevance and confidence."""
        def ranking_score(match: ConventionMatch) -> float:
            score = match.confidence
            
            # Boost for more relevance factors
            score += len(match.relevance_factors) * 0.05
            
            # Boost for rule type priority
            type_priorities = {
                'prohibited': 3.0,
                'mandatory': 2.0,
                'recommended': 1.0,
            }
            score += type_priorities.get(match.rule.rule_type, 0.0)
            
            # Boost for recent rules (higher version)
            score += match.rule.version * 0.01
            
            return score
        
        # Sort by ranking score (descending)
        matches.sort(key=ranking_score, reverse=True)
        
        return matches
    
    def filter_matches_by_category(
        self, 
        matches: List[ConventionMatch], 
        categories: List[str]
    ) -> List[ConventionMatch]:
        """Filter matches by specific categories."""
        return [
            match for match in matches 
            if match.rule.category in categories
        ]
    
    def filter_matches_by_type(
        self, 
        matches: List[ConventionMatch], 
        rule_types: List[str]
    ) -> List[ConventionMatch]:
        """Filter matches by rule types."""
        return [
            match for match in matches 
            if match.rule.rule_type in rule_types
        ]
    
    def get_match_summary(self, matches: List[ConventionMatch]) -> Dict[str, int]:
        """Get summary statistics for matches."""
        summary = {
            'total_matches': len(matches),
            'mandatory_rules': 0,
            'recommended_rules': 0,
            'prohibited_rules': 0,
            'categories': set(),
            'avg_confidence': 0.0,
        }
        
        if not matches:
            return summary
        
        for match in matches:
            if match.rule.rule_type == 'mandatory':
                summary['mandatory_rules'] += 1
            elif match.rule.rule_type == 'recommended':
                summary['recommended_rules'] += 1
            elif match.rule.rule_type == 'prohibited':
                summary['prohibited_rules'] += 1
            
            summary['categories'].add(match.rule.category)
        
        summary['avg_confidence'] = sum(m.confidence for m in matches) / len(matches)
        summary['categories'] = list(summary['categories'])
        
        return summary