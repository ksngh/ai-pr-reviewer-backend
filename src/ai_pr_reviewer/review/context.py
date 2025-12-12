"""
Context Builder

Builds optimized context for LLM review generation by combining
PR diff analysis with matched conventions.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..models.pr_diff import PRDiff
from ..models.convention import ConventionRule
from .analyzer import FileAnalysis, AnalyzedChunk
from .matcher import ConventionMatch


logger = logging.getLogger(__name__)


@dataclass
class ReviewContext:
    """Context information for LLM review generation."""
    file_path: str
    language: Optional[str]
    change_summary: str
    code_snippet: str
    relevant_conventions: List[ConventionMatch]
    context_metadata: Dict[str, any]
    token_count: int
    priority_score: float


@dataclass
class OptimizedContext:
    """Optimized context that fits within token limits."""
    contexts: List[ReviewContext]
    total_tokens: int
    optimization_notes: List[str]


class ContextBuilder:
    """
    Builds optimized context for LLM review generation.
    
    Combines PR diff analysis with matched conventions to create
    focused context that fits within LLM token limits.
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        max_contexts_per_request: int = 5,
        min_context_tokens: int = 200
    ):
        """
        Initialize context builder.
        
        Args:
            max_tokens: Maximum tokens for complete context
            max_contexts_per_request: Maximum number of contexts per LLM request
            min_context_tokens: Minimum tokens required per context
        """
        self.max_tokens = max_tokens
        self.max_contexts_per_request = max_contexts_per_request
        self.min_context_tokens = min_context_tokens
        
        # Token estimation (rough approximation)
        self.chars_per_token = 4  # Average characters per token
    
    def build_review_contexts(
        self,
        file_analyses: List[FileAnalysis],
        convention_matches: Dict[str, Dict[str, List[ConventionMatch]]]
    ) -> List[ReviewContext]:
        """
        Build review contexts from file analyses and convention matches.
        
        Args:
            file_analyses: List of FileAnalysis objects
            convention_matches: Nested dict of convention matches by file and chunk
            
        Returns:
            List of ReviewContext objects
        """
        logger.info(f"Building review contexts for {len(file_analyses)} files")
        
        contexts = []
        
        for file_analysis in file_analyses:
            file_path = file_analysis.file_change.file_path
            file_matches = convention_matches.get(file_path, {})
            
            # Create contexts for each chunk with matches
            for i, chunk in enumerate(file_analysis.analyzed_chunks):
                chunk_key = f"chunk_{i}"
                chunk_matches = file_matches.get(chunk_key, [])
                
                if chunk_matches or self._is_significant_chunk(chunk):
                    context = self._build_chunk_context(
                        chunk, file_analysis, chunk_matches
                    )
                    if context:
                        contexts.append(context)
        
        # Sort by priority
        contexts.sort(key=lambda x: x.priority_score, reverse=True)
        
        logger.info(f"Built {len(contexts)} review contexts")
        return contexts
    
    def optimize_contexts(
        self, 
        contexts: List[ReviewContext]
    ) -> OptimizedContext:
        """
        Optimize contexts to fit within token limits.
        
        Args:
            contexts: List of ReviewContext objects
            
        Returns:
            OptimizedContext with selected and optimized contexts
        """
        logger.info(f"Optimizing {len(contexts)} contexts for token limit {self.max_tokens}")
        
        optimized_contexts = []
        total_tokens = 0
        optimization_notes = []
        
        # Select highest priority contexts that fit
        for context in contexts:
            if len(optimized_contexts) >= self.max_contexts_per_request:
                optimization_notes.append(
                    f"Reached maximum contexts limit ({self.max_contexts_per_request})"
                )
                break
            
            # Check if context fits
            if total_tokens + context.token_count <= self.max_tokens:
                optimized_contexts.append(context)
                total_tokens += context.token_count
            else:
                # Try to compress context
                compressed_context = self._compress_context(
                    context, 
                    self.max_tokens - total_tokens
                )
                
                if compressed_context and compressed_context.token_count >= self.min_context_tokens:
                    optimized_contexts.append(compressed_context)
                    total_tokens += compressed_context.token_count
                    optimization_notes.append(
                        f"Compressed context for {context.file_path}"
                    )
                else:
                    optimization_notes.append(
                        f"Skipped context for {context.file_path} (too large)"
                    )
        
        logger.info(f"Optimized to {len(optimized_contexts)} contexts, {total_tokens} tokens")
        
        return OptimizedContext(
            contexts=optimized_contexts,
            total_tokens=total_tokens,
            optimization_notes=optimization_notes
        )
    
    def _build_chunk_context(
        self,
        chunk: AnalyzedChunk,
        file_analysis: FileAnalysis,
        matches: List[ConventionMatch]
    ) -> Optional[ReviewContext]:
        """Build context for a single chunk."""
        # Create change summary
        change_summary = self._create_change_summary(chunk, file_analysis)
        
        # Create code snippet
        code_snippet = self._create_code_snippet(chunk)
        
        # Filter and rank matches
        relevant_matches = self._filter_relevant_matches(matches)
        
        # Create metadata
        metadata = {
            'change_types': [ct.value for ct in chunk.change_types],
            'functions_affected': chunk.functions_affected,
            'classes_affected': chunk.classes_affected,
            'variables_affected': chunk.variables_affected,
            'complexity_score': chunk.complexity_score,
            'lines_added': len(chunk.added_lines),
            'lines_removed': len(chunk.removed_lines),
        }
        
        # Estimate token count
        content_text = f"{change_summary}\n{code_snippet}"
        for match in relevant_matches:
            content_text += f"\n{match.rule.title}: {match.rule.description}"
        
        token_count = self._estimate_tokens(content_text)
        
        return ReviewContext(
            file_path=chunk.file_path,
            language=chunk.language,
            change_summary=change_summary,
            code_snippet=code_snippet,
            relevant_conventions=relevant_matches,
            context_metadata=metadata,
            token_count=token_count,
            priority_score=chunk.priority_score
        )
    
    def _create_change_summary(
        self, 
        chunk: AnalyzedChunk, 
        file_analysis: FileAnalysis
    ) -> str:
        """Create a summary of changes in the chunk."""
        summary_parts = []
        
        # File and change type
        summary_parts.append(f"File: {chunk.file_path}")
        if chunk.language:
            summary_parts.append(f"Language: {chunk.language}")
        
        # Change types
        change_descriptions = {
            'function_added': 'Added new function',
            'function_modified': 'Modified function',
            'class_added': 'Added new class',
            'class_modified': 'Modified class',
            'variable_modified': 'Modified variables',
            'import_added': 'Added imports',
        }
        
        for change_type in chunk.change_types:
            desc = change_descriptions.get(change_type.value, f"Changed: {change_type.value}")
            summary_parts.append(desc)
        
        # Affected elements
        if chunk.functions_affected:
            summary_parts.append(f"Functions: {', '.join(chunk.functions_affected[:3])}")
        
        if chunk.classes_affected:
            summary_parts.append(f"Classes: {', '.join(chunk.classes_affected[:3])}")
        
        # Change statistics
        summary_parts.append(f"Lines: +{len(chunk.added_lines)}/-{len(chunk.removed_lines)}")
        
        return " | ".join(summary_parts)
    
    def _create_code_snippet(self, chunk: AnalyzedChunk) -> str:
        """Create a focused code snippet from the chunk."""
        snippet_parts = []
        
        # Add context lines if available
        if chunk.context_lines:
            snippet_parts.append("Context:")
            for line in chunk.context_lines[:2]:  # Limit context
                snippet_parts.append(f"  {line}")
        
        # Add removed lines
        if chunk.removed_lines:
            snippet_parts.append("Removed:")
            for line in chunk.removed_lines[:5]:  # Limit lines
                snippet_parts.append(f"- {line}")
        
        # Add added lines
        if chunk.added_lines:
            snippet_parts.append("Added:")
            for line in chunk.added_lines[:5]:  # Limit lines
                snippet_parts.append(f"+ {line}")
        
        return "\n".join(snippet_parts)
    
    def _filter_relevant_matches(
        self, 
        matches: List[ConventionMatch]
    ) -> List[ConventionMatch]:
        """Filter and prioritize convention matches."""
        if not matches:
            return []
        
        # Sort by confidence and take top matches
        sorted_matches = sorted(matches, key=lambda x: x.confidence, reverse=True)
        
        # Prioritize mandatory and prohibited rules
        priority_matches = []
        other_matches = []
        
        for match in sorted_matches:
            if match.rule.rule_type in ['mandatory', 'prohibited']:
                priority_matches.append(match)
            else:
                other_matches.append(match)
        
        # Combine with priority rules first
        relevant_matches = priority_matches[:3] + other_matches[:2]  # Max 5 matches
        
        return relevant_matches
    
    def _is_significant_chunk(self, chunk: AnalyzedChunk) -> bool:
        """Check if chunk is significant enough to review without matches."""
        # Significant if it has meaningful changes
        if len(chunk.added_lines) + len(chunk.removed_lines) < 3:
            return False
        
        # Significant if it affects functions or classes
        if chunk.functions_affected or chunk.classes_affected:
            return True
        
        # Significant if complexity is high
        if chunk.complexity_score > 10:
            return True
        
        return False
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.chars_per_token
    
    def _compress_context(
        self, 
        context: ReviewContext, 
        max_tokens: int
    ) -> Optional[ReviewContext]:
        """Compress context to fit within token limit."""
        if max_tokens < self.min_context_tokens:
            return None
        
        # Start with original context
        compressed = ReviewContext(
            file_path=context.file_path,
            language=context.language,
            change_summary=context.change_summary,
            code_snippet=context.code_snippet,
            relevant_conventions=context.relevant_conventions.copy(),
            context_metadata=context.context_metadata.copy(),
            token_count=context.token_count,
            priority_score=context.priority_score
        )
        
        # Reduce code snippet if needed
        if compressed.token_count > max_tokens:
            lines = compressed.code_snippet.split('\n')
            if len(lines) > 10:
                compressed.code_snippet = '\n'.join(lines[:10]) + "\n... (truncated)"
                compressed.token_count = self._estimate_tokens(
                    f"{compressed.change_summary}\n{compressed.code_snippet}"
                )
        
        # Reduce conventions if needed
        while compressed.token_count > max_tokens and compressed.relevant_conventions:
            compressed.relevant_conventions.pop()  # Remove least relevant
            
            # Recalculate token count
            content_text = f"{compressed.change_summary}\n{compressed.code_snippet}"
            for match in compressed.relevant_conventions:
                content_text += f"\n{match.rule.title}: {match.rule.description}"
            
            compressed.token_count = self._estimate_tokens(content_text)
        
        # Add compression note to metadata
        if compressed.token_count != context.token_count:
            compressed.context_metadata['compressed'] = True
            compressed.context_metadata['original_tokens'] = context.token_count
        
        return compressed if compressed.token_count <= max_tokens else None
    
    def create_batch_contexts(
        self,
        contexts: List[ReviewContext],
        batch_size: Optional[int] = None
    ) -> List[List[ReviewContext]]:
        """
        Create batches of contexts for processing.
        
        Args:
            contexts: List of ReviewContext objects
            batch_size: Optional batch size (uses max_contexts_per_request if None)
            
        Returns:
            List of context batches
        """
        if batch_size is None:
            batch_size = self.max_contexts_per_request
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for context in contexts:
            # Check if adding this context would exceed limits
            if (len(current_batch) >= batch_size or 
                current_tokens + context.token_count > self.max_tokens):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(context)
            current_tokens += context.token_count
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} context batches")
        return batches
    
    def get_context_statistics(self, contexts: List[ReviewContext]) -> Dict[str, any]:
        """Get statistics about contexts."""
        if not contexts:
            return {}
        
        total_tokens = sum(c.token_count for c in contexts)
        languages = set(c.language for c in contexts if c.language)
        files = set(c.file_path for c in contexts)
        
        convention_counts = {}
        for context in contexts:
            for match in context.relevant_conventions:
                rule_type = match.rule.rule_type
                convention_counts[rule_type] = convention_counts.get(rule_type, 0) + 1
        
        return {
            'total_contexts': len(contexts),
            'total_tokens': total_tokens,
            'avg_tokens_per_context': total_tokens / len(contexts),
            'languages': list(languages),
            'files_affected': len(files),
            'convention_types': convention_counts,
            'avg_priority': sum(c.priority_score for c in contexts) / len(contexts),
        }