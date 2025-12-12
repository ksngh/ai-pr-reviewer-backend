"""
PR Diff Parser

Parses GitHub PR diff data into structured format for analysis.
Handles file changes, diff chunks, and context extraction.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ..models.pr_diff import PRDiff, FileChange, DiffChunk


logger = logging.getLogger(__name__)


class PRDiffParser:
    """
    Parser for GitHub PR diff data.
    
    Converts GitHub API responses into structured PRDiff objects
    with proper file changes and diff chunks.
    """
    
    def __init__(self):
        """Initialize PR diff parser."""
        self.diff_header_pattern = re.compile(r'^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@(.*)$')
        self.binary_file_pattern = re.compile(r'Binary files? .* differ')
    
    def parse_pr_diff(self, pr_data: Dict, files_data: List[Dict]) -> PRDiff:
        """
        Parse PR data and files into structured PRDiff object.
        
        Args:
            pr_data: PR information from GitHub API
            files_data: List of file changes from GitHub API
            
        Returns:
            Structured PRDiff object
        """
        logger.info(f"Parsing PR diff for #{pr_data.get('number')}")
        
        # Parse file changes
        files_changed = []
        total_additions = 0
        total_deletions = 0
        
        for file_data in files_data:
            file_change = self._parse_file_change(file_data)
            files_changed.append(file_change)
            total_additions += file_change.additions
            total_deletions += file_change.deletions
        
        # Create PRDiff object
        pr_diff = PRDiff(
            repository=pr_data['base']['repo']['full_name'],
            pr_number=pr_data['number'],
            files_changed=files_changed,
            total_additions=total_additions,
            total_deletions=total_deletions,
            created_at=datetime.fromisoformat(pr_data['created_at'].replace('Z', '+00:00'))
        )
        
        logger.info(f"Parsed PR diff: {len(files_changed)} files, +{total_additions}/-{total_deletions}")
        return pr_diff
    
    def _parse_file_change(self, file_data: Dict) -> FileChange:
        """
        Parse individual file change data.
        
        Args:
            file_data: File change data from GitHub API
            
        Returns:
            Structured FileChange object
        """
        file_path = file_data['filename']
        logger.debug(f"Parsing file change: {file_path}")
        
        # Determine change type
        change_type = self._determine_change_type(file_data['status'])
        
        # Parse diff chunks
        chunks = []
        if 'patch' in file_data and file_data['patch']:
            chunks = self._parse_diff_chunks(file_data['patch'])
        
        return FileChange(
            file_path=file_path,
            change_type=change_type,
            additions=file_data.get('additions', 0),
            deletions=file_data.get('deletions', 0),
            chunks=chunks
        )
    
    def _determine_change_type(self, status: str) -> str:
        """
        Determine file change type from GitHub status.
        
        Args:
            status: GitHub file status
            
        Returns:
            Normalized change type
        """
        status_mapping = {
            'added': 'added',
            'removed': 'deleted',
            'modified': 'modified',
            'renamed': 'modified',
            'copied': 'added'
        }
        
        return status_mapping.get(status, 'modified')
    
    def _parse_diff_chunks(self, patch: str) -> List[DiffChunk]:
        """
        Parse diff patch into structured chunks.
        
        Args:
            patch: Raw diff patch string
            
        Returns:
            List of DiffChunk objects
        """
        if not patch:
            return []
        
        # Check for binary files
        if self.binary_file_pattern.search(patch):
            logger.debug("Skipping binary file diff")
            return []
        
        chunks = []
        lines = patch.split('\n')
        current_chunk = None
        chunk_content = []
        context_lines = []
        
        for line in lines:
            # Check for chunk header
            header_match = self.diff_header_pattern.match(line)
            if header_match:
                # Save previous chunk if exists
                if current_chunk:
                    current_chunk.content = '\n'.join(chunk_content)
                    current_chunk.context_lines = context_lines.copy()
                    chunks.append(current_chunk)
                
                # Start new chunk
                old_start = int(header_match.group(1))
                old_lines = int(header_match.group(2) or 1)
                new_start = int(header_match.group(3))
                new_lines = int(header_match.group(4) or 1)
                
                current_chunk = DiffChunk(
                    old_start=old_start,
                    old_lines=old_lines,
                    new_start=new_start,
                    new_lines=new_lines,
                    content='',
                    context_lines=[]
                )
                
                chunk_content = []
                context_lines = []
                
                # Include context from header if present
                context = header_match.group(5).strip()
                if context:
                    context_lines.append(context)
                
            elif current_chunk:
                # Add line to current chunk
                chunk_content.append(line)
                
                # Collect context lines (unchanged lines)
                if line.startswith(' ') and len(context_lines) < 3:
                    context_lines.append(line[1:])  # Remove leading space
        
        # Save last chunk
        if current_chunk:
            current_chunk.content = '\n'.join(chunk_content)
            current_chunk.context_lines = context_lines.copy()
            chunks.append(current_chunk)
        
        logger.debug(f"Parsed {len(chunks)} diff chunks")
        return chunks
    
    def extract_changed_lines(self, chunk: DiffChunk) -> Tuple[List[str], List[str]]:
        """
        Extract added and removed lines from a diff chunk.
        
        Args:
            chunk: DiffChunk to analyze
            
        Returns:
            Tuple of (added_lines, removed_lines)
        """
        added_lines = []
        removed_lines = []
        
        for line in chunk.content.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])  # Remove leading +
            elif line.startswith('-') and not line.startswith('---'):
                removed_lines.append(line[1:])  # Remove leading -
        
        return added_lines, removed_lines
    
    def get_file_extension(self, file_path: str) -> Optional[str]:
        """
        Get file extension from file path.
        
        Args:
            file_path: Path to file
            
        Returns:
            File extension or None
        """
        if '.' not in file_path:
            return None
        
        return file_path.split('.')[-1].lower()
    
    def is_code_file(self, file_path: str) -> bool:
        """
        Check if file is a code file based on extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is likely a code file
        """
        code_extensions = {
            'py', 'js', 'ts', 'jsx', 'tsx', 'java', 'cpp', 'c', 'h',
            'cs', 'php', 'rb', 'go', 'rs', 'kt', 'swift', 'scala',
            'html', 'css', 'scss', 'sass', 'less', 'vue', 'svelte',
            'sql', 'sh', 'bash', 'ps1', 'yaml', 'yml', 'json', 'xml'
        }
        
        extension = self.get_file_extension(file_path)
        return extension in code_extensions if extension else False
    
    def filter_relevant_files(self, pr_diff: PRDiff, include_tests: bool = True) -> List[FileChange]:
        """
        Filter files relevant for convention checking.
        
        Args:
            pr_diff: PRDiff object to filter
            include_tests: Whether to include test files
            
        Returns:
            List of relevant FileChange objects
        """
        relevant_files = []
        
        for file_change in pr_diff.files_changed:
            # Skip deleted files
            if file_change.change_type == 'deleted':
                continue
            
            # Only include code files
            if not self.is_code_file(file_change.file_path):
                continue
            
            # Skip test files if not included
            if not include_tests and self._is_test_file(file_change.file_path):
                continue
            
            relevant_files.append(file_change)
        
        logger.info(f"Filtered to {len(relevant_files)} relevant files")
        return relevant_files
    
    def _is_test_file(self, file_path: str) -> bool:
        """
        Check if file is a test file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is likely a test file
        """
        test_indicators = [
            'test_', '_test.', 'tests/', '/test/', 'spec_', '_spec.',
            'specs/', '/spec/', '__tests__/', '.test.', '.spec.'
        ]
        
        file_path_lower = file_path.lower()
        return any(indicator in file_path_lower for indicator in test_indicators)