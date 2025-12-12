"""
Main AI Reviewer API

Main interface that orchestrates the complete review process
from PR diff collection to formatted GitHub comments.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .github.client import GitHubClient
from .github.parser import PRDiffParser
from .conventions.extractor import ConventionExtractor
from .conventions.embeddings import EmbeddingGenerator
from .conventions.vector_store import VectorStore
from .review.analyzer import DiffAnalyzer
from .review.matcher import ConventionMatcher
from .review.context import ContextBuilder
from .llm.generator import ReviewGenerator
from .formatting.github import GitHubCommentFormatter
from .models.pr_diff import PRDiff
from .models.review import ReviewComment, GitHubComment
from .config import Config


logger = logging.getLogger(__name__)


@dataclass
class ReviewRequest:
    """Request for PR review generation."""
    repository: str
    pr_number: int
    github_token: str
    options: Optional[Dict] = None


@dataclass
class ReviewResult:
    """Result of PR review generation."""
    review_id: str
    repository: str
    pr_number: int
    status: str
    github_comments: List[GitHubComment]
    processing_time: float
    metadata: Dict
    created_at: datetime


class AIReviewerAPI:
    """
    Main AI Reviewer API interface.
    
    Orchestrates the complete review process:
    1. Collect PR diff and wiki conventions
    2. Analyze code changes and match conventions
    3. Generate LLM reviews with quality control
    4. Format for GitHub PR comments
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize AI Reviewer API.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        
        # Initialize components
        logger.info("Initializing AI Reviewer API components...")
        
        # GitHub integration
        self.pr_parser = PRDiffParser()
        
        # Convention processing
        self.convention_extractor = ConventionExtractor()
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.embedding_model,
            cache_dir=self.config.embedding_cache_dir
        )
        self.vector_store = VectorStore(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            collection_name=self.config.qdrant_collection,
            embedding_dim=self.embedding_generator.embedding_dim
        )
        
        # Review processing
        self.diff_analyzer = DiffAnalyzer(
            max_chunk_size=self.config.max_chunk_size
        )
        self.convention_matcher = ConventionMatcher(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            similarity_threshold=self.config.similarity_threshold
        )
        self.context_builder = ContextBuilder(
            max_tokens=self.config.max_context_tokens,
            max_contexts_per_request=self.config.max_contexts_per_request
        )
        
        # LLM and formatting
        self.review_generator = ReviewGenerator(
            model_name=self.config.llm_model,
            language=self.config.review_language
        )
        self.github_formatter = GitHubCommentFormatter(
            language=self.config.review_language
        )
        
        logger.info("AI Reviewer API initialized successfully")
    
    async def generate_review(self, request: ReviewRequest) -> ReviewResult:
        """
        Generate complete PR review.
        
        Args:
            request: ReviewRequest with PR information
            
        Returns:
            ReviewResult with formatted GitHub comments
        """
        start_time = datetime.now()
        review_id = f"{request.repository}_{request.pr_number}_{int(start_time.timestamp())}"
        
        logger.info(f"Starting review generation: {review_id}")
        
        try:
            # Step 1: Collect PR diff and conventions
            pr_diff, conventions = await self._collect_pr_data(request)
            
            # Step 2: Analyze and match
            file_analyses, convention_matches = await self._analyze_and_match(
                pr_diff, conventions
            )
            
            # Step 3: Build contexts and generate reviews
            contexts, reviews = await self._generate_reviews(
                file_analyses, convention_matches
            )
            
            # Step 4: Format for GitHub
            github_comments = await self._format_for_github(reviews, contexts)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ReviewResult(
                review_id=review_id,
                repository=request.repository,
                pr_number=request.pr_number,
                status="completed",
                github_comments=github_comments,
                processing_time=processing_time,
                metadata=self._create_metadata(
                    pr_diff, file_analyses, contexts, reviews
                ),
                created_at=start_time
            )
            
            logger.info(f"Review generation completed: {review_id} ({processing_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Review generation failed: {review_id} - {e}")
            
            # Return error result
            processing_time = (datetime.now() - start_time).total_seconds()
            return ReviewResult(
                review_id=review_id,
                repository=request.repository,
                pr_number=request.pr_number,
                status="failed",
                github_comments=[],
                processing_time=processing_time,
                metadata={"error": str(e)},
                created_at=start_time
            )
    
    async def _collect_pr_data(
        self, 
        request: ReviewRequest
    ) -> Tuple[PRDiff, List]:
        """Collect PR diff and convention data."""
        logger.info(f"Collecting PR data for {request.repository}#{request.pr_number}")
        
        # Initialize GitHub client
        github_client = GitHubClient(request.github_token)
        
        # Get PR information and files
        pr_data = github_client.get_pull_request(
            *request.repository.split('/'), request.pr_number
        )
        files_data = github_client.get_pull_request_files(
            *request.repository.split('/'), request.pr_number
        )
        
        # Parse PR diff
        pr_diff = self.pr_parser.parse_pr_diff(pr_data, files_data)
        
        # Get wiki pages and extract conventions
        wiki_pages = github_client.get_wiki_pages(*request.repository.split('/'))
        conventions = []
        
        for page in wiki_pages:
            page_content = github_client.get_wiki_page_content(
                *request.repository.split('/'), page['title']
            )
            if page_content:
                page_conventions = self.convention_extractor.extract_rules(
                    page_content, page['title']
                )
                conventions.extend(page_conventions)
        
        # Store conventions in vector database
        if conventions:
            embedded_conventions = []
            for convention in conventions:
                embedded_conv = self.embedding_generator.embed_convention_rule(convention)
                embedded_conventions.append(embedded_conv)
            
            self.vector_store.batch_store_conventions(embedded_conventions)
        
        logger.info(f"Collected {len(conventions)} conventions from {len(wiki_pages)} wiki pages")
        return pr_diff, conventions
    
    async def _analyze_and_match(
        self, 
        pr_diff: PRDiff, 
        conventions: List
    ) -> Tuple[List, Dict]:
        """Analyze PR diff and match with conventions."""
        logger.info("Analyzing PR diff and matching conventions")
        
        # Analyze PR diff
        file_analyses = self.diff_analyzer.analyze_pr_diff(pr_diff)
        
        # Match conventions for each file
        convention_matches = {}
        
        for file_analysis in file_analyses:
            file_path = file_analysis.file_change.file_path
            file_matches = self.convention_matcher.find_conventions_for_file(file_analysis)
            
            if file_matches:
                convention_matches[file_path] = file_matches
        
        logger.info(f"Analyzed {len(file_analyses)} files, found matches for {len(convention_matches)} files")
        return file_analyses, convention_matches
    
    async def _generate_reviews(
        self, 
        file_analyses: List, 
        convention_matches: Dict
    ) -> Tuple[List, List]:
        """Build contexts and generate LLM reviews."""
        logger.info("Building contexts and generating reviews")
        
        # Build review contexts
        contexts = self.context_builder.build_review_contexts(
            file_analyses, convention_matches
        )
        
        # Optimize contexts for token limits
        optimized = self.context_builder.optimize_contexts(contexts)
        
        # Generate reviews for each context
        reviews = []
        for context in optimized.contexts:
            review = self.review_generator.generate_review(context)
            if review:
                reviews.append(review)
        
        logger.info(f"Generated {len(reviews)} reviews from {len(optimized.contexts)} contexts")
        return optimized.contexts, reviews
    
    async def _format_for_github(
        self, 
        reviews: List[ReviewComment], 
        contexts: List
    ) -> List[GitHubComment]:
        """Format reviews for GitHub PR comments."""
        logger.info("Formatting reviews for GitHub")
        
        github_comments = self.github_formatter.format_for_github(reviews, contexts)
        
        logger.info(f"Created {len(github_comments)} GitHub comments")
        return github_comments
    
    def _create_metadata(
        self, 
        pr_diff: PRDiff, 
        file_analyses: List, 
        contexts: List, 
        reviews: List[ReviewComment]
    ) -> Dict:
        """Create metadata for the review result."""
        return {
            'pr_stats': {
                'files_changed': len(pr_diff.files_changed),
                'total_additions': pr_diff.total_additions,
                'total_deletions': pr_diff.total_deletions,
            },
            'analysis_stats': {
                'files_analyzed': len(file_analyses),
                'contexts_created': len(contexts),
                'reviews_generated': len(reviews),
            },
            'processing_stats': {
                'embedding_model': self.embedding_generator.model_name,
                'llm_model': self.review_generator.model_name,
                'language': self.config.review_language,
            }
        }
    
    async def sync_conventions(
        self, 
        repository: str, 
        github_token: str,
        force_update: bool = False
    ) -> Dict:
        """
        Sync conventions from repository wiki.
        
        Args:
            repository: Repository name (owner/repo)
            github_token: GitHub access token
            force_update: Whether to force update existing conventions
            
        Returns:
            Sync result information
        """
        logger.info(f"Syncing conventions for {repository}")
        
        start_time = datetime.now()
        
        try:
            # Initialize GitHub client
            github_client = GitHubClient(github_token)
            
            # Get wiki pages
            wiki_pages = github_client.get_wiki_pages(*repository.split('/'))
            
            conventions_processed = 0
            conventions_updated = 0
            
            for page in wiki_pages:
                page_content = github_client.get_wiki_page_content(
                    *repository.split('/'), page['title']
                )
                
                if page_content:
                    # Extract conventions
                    page_conventions = self.convention_extractor.extract_rules(
                        page_content, page['title']
                    )
                    
                    # Process each convention
                    for convention in page_conventions:
                        # Check if convention exists
                        existing = self.vector_store.get_convention_by_id(convention.id)
                        
                        if existing and not force_update:
                            # Skip existing conventions unless force update
                            continue
                        
                        # Generate embedding and store
                        embedded_conv = self.embedding_generator.embed_convention_rule(convention)
                        
                        if existing:
                            self.vector_store.update_convention(embedded_conv)
                            conventions_updated += 1
                        else:
                            self.vector_store.store_convention(embedded_conv)
                        
                        conventions_processed += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'completed',
                'repository': repository,
                'wiki_pages_found': len(wiki_pages),
                'conventions_processed': conventions_processed,
                'conventions_updated': conventions_updated,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Convention sync completed: {conventions_processed} processed, {conventions_updated} updated")
            return result
            
        except Exception as e:
            logger.error(f"Convention sync failed for {repository}: {e}")
            return {
                'status': 'failed',
                'repository': repository,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_system_health(self) -> Dict:
        """Get system health status."""
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check vector store
            vector_stats = self.vector_store.get_collection_stats()
            health['components']['vector_store'] = {
                'status': 'healthy' if vector_stats else 'unhealthy',
                'stats': vector_stats
            }
        except Exception as e:
            health['components']['vector_store'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        try:
            # Check LLM model
            model_info = self.review_generator.get_model_info()
            health['components']['llm_model'] = {
                'status': 'healthy',
                'info': model_info
            }
        except Exception as e:
            health['components']['llm_model'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        try:
            # Check embedding generator
            embedding_stats = self.embedding_generator.get_cache_stats()
            health['components']['embedding_generator'] = {
                'status': 'healthy',
                'stats': embedding_stats
            }
        except Exception as e:
            health['components']['embedding_generator'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health['components'].values()]
        if 'unhealthy' in component_statuses:
            health['status'] = 'unhealthy'
        elif 'degraded' in component_statuses:
            health['status'] = 'degraded'
        
        return health
    
    def cleanup_resources(self):
        """Clean up resources and caches."""
        logger.info("Cleaning up AI Reviewer API resources")
        
        try:
            # Clear model caches
            self.review_generator.clear_cache()
            
            # Save embedding cache
            self.embedding_generator.save_cache()
            
            logger.info("Resource cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_resources()