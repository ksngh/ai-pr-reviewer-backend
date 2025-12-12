#!/usr/bin/env python3
"""
AI PR Reviewer Backend Server

Simple Flask/FastAPI server to run the AI PR Reviewer system.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Installing...")
    os.system("pip install flask flask-cors")
    from flask import Flask, request, jsonify
    from flask_cors import CORS

from ai_pr_reviewer.api import AIReviewerAPI, ReviewRequest

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Initialize AI Reviewer API
reviewer_api = AIReviewerAPI()

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ai-pr-reviewer-backend',
        'version': '1.0.0'
    })

@app.route('/api/v1/reviews/generate', methods=['POST'])
def generate_review():
    """Generate PR review."""
    try:
        data = request.get_json()
        
        # Create review request
        review_request = ReviewRequest(
            repository=data['repository'],
            pr_number=data['pr_number'],
            github_token=data['github_token'],
            options=data.get('options', {})
        )
        
        # Generate review
        result = reviewer_api.review_pr(
            repository=review_request.repository,
            pr_number=review_request.pr_number,
            github_token=review_request.github_token
        )
        
        return jsonify({
            'review_id': result.review_id,
            'status': result.status,
            'repository': result.repository,
            'pr_number': result.pr_number,
            'total_comments': len(result.github_comments),
            'processing_time': result.processing_time
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/api/v1/conventions/sync', methods=['POST'])
def sync_conventions():
    """Sync conventions from repository wiki."""
    try:
        data = request.get_json()
        
        result = reviewer_api.update_conventions(
            repository=data['repository'],
            github_token=data['github_token']
        )
        
        return jsonify({
            'status': 'completed',
            'conventions_processed': result.get('conventions_processed', 0),
            'last_sync': result.get('last_sync')
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting AI PR Reviewer Backend Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📋 API Documentation:")
    print("   - Health Check: GET /api/v1/health")
    print("   - Generate Review: POST /api/v1/reviews/generate")
    print("   - Sync Conventions: POST /api/v1/conventions/sync")
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )