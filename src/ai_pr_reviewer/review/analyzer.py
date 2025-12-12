"""
Diff Analyzer

Analyzes PR diffs to identify code changes and extract meaningful chunks
for convention matching and review generation.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from ..models.pr_diff import PRDiff, FileChange, DiffChunk


logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of code changes detected in diffs."""
    FUNCTION_ADDED = "function_added"
    FUNCTION_MODIFIED = "function_modified"
    FUNCTION_REMOVED = "function_removed"
    CLASS_ADDED = "class_added"
    CLASS_MODIFIED = "class_modified"
    CLASS_REMOVED = "class_removed"
    VARIABLE_ADDED = "variable_added"
    VARIABLE_MODIFIED = "variable_modified"
    IMPORT_ADDED = "import_added"
    IMPORT_REMOVED = "import_removed"
    COMMENT_ADDED = "comment_added"
    COMMENT_MODIFIED = "comment_modified"
    WHITESPACE_ONLY = "whitespace_only"
    UNKNOWN = "unknown"


@dataclass
class AnalyzedChunk:
    """Analyzed diff chunk with extracted metadata."""
    original_chunk: DiffChunk
    change_types: List[ChangeType]
    added_lines: List[str]
    removed_lines: List[str]
    context_lines: List[str]
    file_path: str
    language: Optional[str]
    complexity_score: float
    priority_score: float
    functions_affected: List[str]
    classes_affected: List[str]
    variables_affected: List[str]


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    file_change: FileChange
    analyzed_chunks: List[AnalyzedChunk]
    language: Optional[str]
    total_complexity: float
    change_summary: Dict[ChangeType, int]
    review_priority: float


class DiffAnalyzer:
    """
    Analyzes PR diffs to extract meaningful information for review.
    
    Identifies code patterns, change types, and provides context
    for convention matching and LLM review generation.
    """
    
    def __init__(self, max_chunk_size: int = 1000):
        """
        Initialize diff analyzer.
        
        Args:
            max_chunk_size: Maximum size for diff chunks (in characters)
        """
        self.max_chunk_size = max_chunk_size
        
        # Language detection patterns
        self.language_patterns = {
            'python': [r'\.py$', r'def\s+\w+', r'class\s+\w+', r'import\s+\w+'],
            'javascript': [r'\.js$', r'function\s+\w+', r'const\s+\w+', r'let\s+\w+'],
            'typescript': [r'\.ts$', r'\.tsx$', r'interface\s+\w+', r'type\s+\w+'],
            'java': [r'\.java$', r'public\s+class', r'private\s+\w+', r'@\w+'],
            'cpp': [r'\.cpp$', r'\.h$', r'#include', r'namespace\s+\w+'],
            'go': [r'\.go$', r'func\s+\w+', r'package\s+\w+', r'type\s+\w+'],
            'rust': [r'\.rs$', r'fn\s+\w+', r'struct\s+\w+', r'impl\s+\w+'],
        }
        
        # Code pattern matchers
        self.function_patterns = {
            'python': r'def\s+(\w+)\s*\(',
            'javascript': r'function\s+(\w+)\s*\(|(\w+)\s*=\s*\(',
            'typescript': r'function\s+(\w+)\s*\(|(\w+)\s*=\s*\(',
            'java': r'(public|private|protected)?\s*\w+\s+(\w+)\s*\(',
            'cpp': r'\w+\s+(\w+)\s*\(',
            'go': r'func\s+(\w+)\s*\(',
            'rust': r'fn\s+(\w+)\s*\(',
        }
        
        self.class_patterns = {
            'python': r'class\s+(\w+)',
            'javascript': r'class\s+(\w+)',
            'typescript': r'class\s+(\w+)|interface\s+(\w+)',
            'java': r'(public|private)?\s*class\s+(\w+)',
            'cpp': r'class\s+(\w+)',
            'go': r'type\s+(\w+)\s+struct',
            'rust': r'struct\s+(\w+)',
        }
        
        self.variable_patterns = {
            'python': r'(\w+)\s*=\s*',
            'javascript': r'(const|let|var)\s+(\w+)',
            'typescript': r'(const|let|var)\s+(\w+)',
            'java': r'(private|public|protected)?\s*\w+\s+(\w+)\s*=',
            'cpp': r'\w+\s+(\w+)\s*=',
            'go': r'(\w+)\s*:=|var\s+(\w+)',
            'rust': r'let\s+(\w+)',
        }
    
    def analyze_pr_diff(self, pr_diff: PRDiff) -> List[FileAnalysis]:
        """
        Analyze complete PR diff.
        
        Args:
            pr_diff: PRDiff object to analyze
            
        Returns:
            List of FileAnalysis objects
        """
        logger.info(f"Analyzing PR diff for {pr_diff.repository}#{pr_diff.pr_number}")
        
        file_analyses = []
        
        for file_change in pr_diff.files_changed:
            analysis = self.analyze_file_change(file_change)
            file_analyses.append(analysis)
        
        # Sort by review priority
        file_analyses.sort(key=lambda x: x.review_priority, reverse=True)
        
        logger.info(f"Analyzed {len(file_analyses)} files")
        return file_analyses
    
    def analyze_file_change(self, file_change: FileChange) -> FileAnalysis:
        """
        Analyze a single file change.
        
        Args:
            file_change: FileChange object to analyze
            
        Returns:
            FileAnalysis object
        """
        logger.debug(f"Analyzing file: {file_change.file_path}")
        
        # Detect language
        language = self._detect_language(file_change.file_path)
        
        # Analyze each chunk
        analyzed_chunks = []
        total_complexity = 0.0
        change_summary = {}
        
        for chunk in file_change.chunks:
            analyzed_chunk = self._analyze_chunk(chunk, file_change.file_path, language)
            analyzed_chunks.append(analyzed_chunk)
            total_complexity += analyzed_chunk.complexity_score
            
            # Update change summary
            for change_type in analyzed_chunk.change_types:
                change_summary[change_type] = change_summary.get(change_type, 0) + 1
        
        # Calculate review priority
        review_priority = self._calculate_review_priority(
            file_change, analyzed_chunks, total_complexity
        )
        
        return FileAnalysis(
            file_change=file_change,
            analyzed_chunks=analyzed_chunks,
            language=language,
            total_complexity=total_complexity,
            change_summary=change_summary,
            review_priority=review_priority
        )
    
    def _analyze_chunk(self, chunk: DiffChunk, file_path: str, language: Optional[str]) -> AnalyzedChunk:
        """Analyze a single diff chunk."""
        # Extract added and removed lines
        added_lines = []
        removed_lines = []
        
        for line in chunk.content.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])  # Remove leading +
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:])  # Remove leading -
        
        # Identify change types
        change_types = self._identify_change_types(added_lines, removed_lines, language)
        
        # Extract affected code elements
        functions_affected = self._extract_functions(added_lines + removed_lines, language)
        classes_affected = self._extract_classes(added_lines + removed_lines, language)
        variables_affected = self._extract_variables(added_lines + removed_lines, language)
        
        # Calculate complexity and priority scores
        complexity_score = self._calculate_complexity_score(
            added_lines, removed_lines, change_types
        )
        priority_score = self._calculate_priority_score(
            change_types, functions_affected, classes_affected
        )
        
        return AnalyzedChunk(
            original_chunk=chunk,
            change_types=change_types,
            added_lines=added_lines,
            removed_lines=removed_lines,
            context_lines=chunk.context_lines,
            file_path=file_path,
            language=language,
            complexity_score=complexity_score,
            priority_score=priority_score,
            functions_affected=functions_affected,
            classes_affected=classes_affected,
            variables_affected=variables_affected
        )
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file path."""
        for language, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    return language
        return None
    
    def _identify_change_types(
        self, 
        added_lines: List[str], 
        removed_lines: List[str], 
        language: Optional[str]
    ) -> List[ChangeType]:
        """Identify types of changes in the diff."""
        change_types = []
        
        if not added_lines and not removed_lines:
            return [ChangeType.UNKNOWN]
        
        # Check for whitespace-only changes
        if self._is_whitespace_only_change(added_lines, removed_lines):
            return [ChangeType.WHITESPACE_ONLY]
        
        # Check for function changes
        if language and language in self.function_patterns:
            pattern = self.function_patterns[language]
            
            added_functions = set()
            removed_functions = set()
            
            for line in added_lines:
                matches = re.findall(pattern, line)
                if matches:
                    added_functions.update([m for m in matches if m])
            
            for line in removed_lines:
                matches = re.findall(pattern, line)
                if matches:
                    removed_functions.update([m for m in matches if m])
            
            if added_functions and not removed_functions:
                change_types.append(ChangeType.FUNCTION_ADDED)
            elif removed_functions and not added_functions:
                change_types.append(ChangeType.FUNCTION_REMOVED)
            elif added_functions or removed_functions:
                change_types.append(ChangeType.FUNCTION_MODIFIED)
        
        # Check for class changes
        if language and language in self.class_patterns:
            pattern = self.class_patterns[language]
            
            added_classes = any(re.search(pattern, line) for line in added_lines)
            removed_classes = any(re.search(pattern, line) for line in removed_lines)
            
            if added_classes and not removed_classes:
                change_types.append(ChangeType.CLASS_ADDED)
            elif removed_classes and not added_classes:
                change_types.append(ChangeType.CLASS_REMOVED)
            elif added_classes or removed_classes:
                change_types.append(ChangeType.CLASS_MODIFIED)
        
        # Check for variable changes
        if language and language in self.variable_patterns:
            pattern = self.variable_patterns[language]
            
            added_vars = any(re.search(pattern, line) for line in added_lines)
            removed_vars = any(re.search(pattern, line) for line in removed_lines)
            
            if added_vars or removed_vars:
                change_types.append(ChangeType.VARIABLE_MODIFIED)
        
        # Check for import changes
        import_patterns = [r'import\s+', r'from\s+\w+\s+import', r'#include', r'require\(']
        
        added_imports = any(
            any(re.search(pattern, line) for pattern in import_patterns)
            for line in added_lines
        )
        removed_imports = any(
            any(re.search(pattern, line) for pattern in import_patterns)
            for line in removed_lines
        )
        
        if added_imports:
            change_types.append(ChangeType.IMPORT_ADDED)
        if removed_imports:
            change_types.append(ChangeType.IMPORT_REMOVED)
        
        # Check for comment changes
        comment_patterns = [r'^\s*#', r'^\s*//', r'^\s*/\*', r'^\s*\*']
        
        added_comments = any(
            any(re.search(pattern, line) for pattern in comment_patterns)
            for line in added_lines
        )
        
        if added_comments:
            change_types.append(ChangeType.COMMENT_ADDED)
        
        # Default to unknown if no specific type identified
        if not change_types:
            change_types.append(ChangeType.UNKNOWN)
        
        return change_types
    
    def _is_whitespace_only_change(self, added_lines: List[str], removed_lines: List[str]) -> bool:
        """Check if changes are whitespace-only."""
        if len(added_lines) != len(removed_lines):
            return False
        
        for added, removed in zip(added_lines, removed_lines):
            if added.strip() != removed.strip():
                return False
        
        return True
    
    def _extract_functions(self, lines: List[str], language: Optional[str]) -> List[str]:
        """Extract function names from lines."""
        if not language or language not in self.function_patterns:
            return []
        
        pattern = self.function_patterns[language]
        functions = []
        
        for line in lines:
            matches = re.findall(pattern, line)
            if matches:
                # Handle tuple matches from regex groups
                for match in matches:
                    if isinstance(match, tuple):
                        functions.extend([m for m in match if m])
                    else:
                        functions.append(match)
        
        return list(set(functions))  # Remove duplicates
    
    def _extract_classes(self, lines: List[str], language: Optional[str]) -> List[str]:
        """Extract class names from lines."""
        if not language or language not in self.class_patterns:
            return []
        
        pattern = self.class_patterns[language]
        classes = []
        
        for line in lines:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        classes.extend([m for m in match if m])
                    else:
                        classes.append(match)
        
        return list(set(classes))
    
    def _extract_variables(self, lines: List[str], language: Optional[str]) -> List[str]:
        """Extract variable names from lines."""
        if not language or language not in self.variable_patterns:
            return []
        
        pattern = self.variable_patterns[language]
        variables = []
        
        for line in lines:
            matches = re.findall(pattern, line)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        variables.extend([m for m in match if m])
                    else:
                        variables.append(match)
        
        return list(set(variables))
    
    def _calculate_complexity_score(
        self, 
        added_lines: List[str], 
        removed_lines: List[str], 
        change_types: List[ChangeType]
    ) -> float:
        """Calculate complexity score for a chunk."""
        base_score = len(added_lines) + len(removed_lines)
        
        # Weight by change type complexity
        type_weights = {
            ChangeType.FUNCTION_ADDED: 3.0,
            ChangeType.FUNCTION_MODIFIED: 2.5,
            ChangeType.CLASS_ADDED: 4.0,
            ChangeType.CLASS_MODIFIED: 3.5,
            ChangeType.VARIABLE_MODIFIED: 1.5,
            ChangeType.IMPORT_ADDED: 1.0,
            ChangeType.COMMENT_ADDED: 0.5,
            ChangeType.WHITESPACE_ONLY: 0.1,
        }
        
        type_multiplier = max(
            type_weights.get(change_type, 1.0) 
            for change_type in change_types
        )
        
        return base_score * type_multiplier
    
    def _calculate_priority_score(
        self, 
        change_types: List[ChangeType], 
        functions_affected: List[str], 
        classes_affected: List[str]
    ) -> float:
        """Calculate priority score for review."""
        base_score = 1.0
        
        # High priority change types
        high_priority_types = {
            ChangeType.FUNCTION_ADDED,
            ChangeType.CLASS_ADDED,
            ChangeType.FUNCTION_MODIFIED,
            ChangeType.CLASS_MODIFIED,
        }
        
        if any(ct in high_priority_types for ct in change_types):
            base_score += 2.0
        
        # More affected elements = higher priority
        base_score += len(functions_affected) * 0.5
        base_score += len(classes_affected) * 1.0
        
        return base_score
    
    def _calculate_review_priority(
        self, 
        file_change: FileChange, 
        analyzed_chunks: List[AnalyzedChunk], 
        total_complexity: float
    ) -> float:
        """Calculate overall review priority for a file."""
        base_priority = total_complexity / max(len(analyzed_chunks), 1)
        
        # File type modifiers
        if file_change.file_path.endswith(('.py', '.js', '.ts', '.java')):
            base_priority *= 1.2  # Higher priority for main code files
        elif file_change.file_path.endswith(('.md', '.txt', '.yml', '.yaml')):
            base_priority *= 0.5  # Lower priority for documentation
        
        # Change type modifiers
        if file_change.change_type == 'added':
            base_priority *= 1.5  # New files need more attention
        elif file_change.change_type == 'deleted':
            base_priority *= 0.3  # Deleted files less important
        
        return base_priority
    
    def chunk_large_diff(self, file_analysis: FileAnalysis) -> List[AnalyzedChunk]:
        """
        Split large diffs into manageable chunks.
        
        Args:
            file_analysis: FileAnalysis with potentially large chunks
            
        Returns:
            List of smaller AnalyzedChunk objects
        """
        chunked_results = []
        
        for chunk in file_analysis.analyzed_chunks:
            if len(chunk.original_chunk.content) <= self.max_chunk_size:
                chunked_results.append(chunk)
            else:
                # Split large chunk
                sub_chunks = self._split_chunk(chunk)
                chunked_results.extend(sub_chunks)
        
        return chunked_results
    
    def _split_chunk(self, chunk: AnalyzedChunk) -> List[AnalyzedChunk]:
        """Split a large chunk into smaller pieces."""
        content_lines = chunk.original_chunk.content.split('\n')
        chunk_size = self.max_chunk_size // 50  # Approximate lines per chunk
        
        sub_chunks = []
        
        for i in range(0, len(content_lines), chunk_size):
            sub_content = '\n'.join(content_lines[i:i + chunk_size])
            
            # Create new DiffChunk for sub-content
            sub_diff_chunk = DiffChunk(
                old_start=chunk.original_chunk.old_start + i,
                old_lines=min(chunk_size, len(content_lines) - i),
                new_start=chunk.original_chunk.new_start + i,
                new_lines=min(chunk_size, len(content_lines) - i),
                content=sub_content,
                context_lines=chunk.context_lines[:3]  # Keep some context
            )
            
            # Analyze sub-chunk
            sub_analyzed = self._analyze_chunk(
                sub_diff_chunk, 
                chunk.file_path, 
                chunk.language
            )
            
            sub_chunks.append(sub_analyzed)
        
        return sub_chunks