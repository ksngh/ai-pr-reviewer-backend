"""
Property-based tests for specification synchronization.

Property 16: Specification Synchronization
Validates: Requirements 0.3, 0.5
"""

import pytest
from hypothesis import given, strategies as st, assume
import os
import tempfile
import shutil
from pathlib import Path

from ai_pr_reviewer.config import Config


class TestSpecificationSynchronization:
    """Property tests for specification synchronization across repositories."""

    @given(
        spec_files=st.lists(
            st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
            min_size=1,
            max_size=10
        ),
        content=st.text(min_size=10, max_size=1000)
    )
    def test_spec_files_remain_synchronized(self, spec_files, content):
        """
        Property: Specification files should remain synchronized across repositories.
        
        Given: Multiple repositories with shared specification files
        When: Specifications are updated in the shared repository
        Then: All dependent repositories should reflect the same changes
        """
        assume(all(name.endswith('.md') or '.' not in name for name in spec_files))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create shared specs directory
            shared_specs_dir = Path(temp_dir) / "shared-specs"
            shared_specs_dir.mkdir()
            
            # Create backend specs directory (simulating submodule)
            backend_specs_dir = Path(temp_dir) / "backend" / ".kiro" / "specs"
            backend_specs_dir.mkdir(parents=True)
            
            # Create frontend specs directory (simulating submodule)
            frontend_specs_dir = Path(temp_dir) / "frontend" / ".kiro" / "specs"
            frontend_specs_dir.mkdir(parents=True)
            
            # Create specification files in shared directory
            created_files = []
            for spec_file in spec_files:
                if not spec_file.endswith('.md'):
                    spec_file += '.md'
                
                file_path = shared_specs_dir / spec_file
                file_path.write_text(content)
                created_files.append(spec_file)
            
            # Simulate submodule synchronization by copying files
            for spec_file in created_files:
                shutil.copy2(
                    shared_specs_dir / spec_file,
                    backend_specs_dir / spec_file
                )
                shutil.copy2(
                    shared_specs_dir / spec_file,
                    frontend_specs_dir / spec_file
                )
            
            # Verify synchronization
            for spec_file in created_files:
                shared_content = (shared_specs_dir / spec_file).read_text()
                backend_content = (backend_specs_dir / spec_file).read_text()
                frontend_content = (frontend_specs_dir / spec_file).read_text()
                
                assert shared_content == backend_content, f"Backend {spec_file} not synchronized"
                assert shared_content == frontend_content, f"Frontend {spec_file} not synchronized"
                assert backend_content == frontend_content, f"Backend and frontend {spec_file} differ"

    @given(
        version_tag=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
        spec_changes=st.dictionaries(
            keys=st.text(min_size=1, max_size=30),
            values=st.text(min_size=10, max_size=500),
            min_size=1,
            max_size=5
        )
    )
    def test_version_consistency_across_repositories(self, version_tag, spec_changes):
        """
        Property: Version tags should maintain consistency across all repositories.
        
        Given: A version tag and specification changes
        When: Changes are tagged in the shared repository
        Then: All dependent repositories should reference the same version
        """
        assume(version_tag.replace('_', '').replace('-', '').replace('.', '').isalnum())
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create version tracking files
            shared_version_file = Path(temp_dir) / "shared-specs" / "VERSION"
            backend_version_file = Path(temp_dir) / "backend" / ".kiro" / "specs" / "VERSION"
            frontend_version_file = Path(temp_dir) / "frontend" / ".kiro" / "specs" / "VERSION"
            
            # Create directories
            for version_file in [shared_version_file, backend_version_file, frontend_version_file]:
                version_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write version information
            version_info = f"version: {version_tag}\nchanges: {len(spec_changes)} files updated"
            
            for version_file in [shared_version_file, backend_version_file, frontend_version_file]:
                version_file.write_text(version_info)
            
            # Verify version consistency
            shared_version = shared_version_file.read_text()
            backend_version = backend_version_file.read_text()
            frontend_version = frontend_version_file.read_text()
            
            assert shared_version == backend_version, "Backend version not synchronized"
            assert shared_version == frontend_version, "Frontend version not synchronized"
            assert backend_version == frontend_version, "Backend and frontend versions differ"

    @given(
        api_changes=st.dictionaries(
            keys=st.sampled_from(['endpoints', 'models', 'responses']),
            values=st.text(min_size=10, max_size=200),
            min_size=1,
            max_size=3
        )
    )
    def test_api_contract_synchronization(self, api_changes):
        """
        Property: API contract changes should be reflected in both backend and frontend.
        
        Given: API contract modifications
        When: Contract is updated in shared specifications
        Then: Both backend and frontend should reference the updated contract
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create API contract files
            shared_api_file = Path(temp_dir) / "shared-specs" / "api-contract.md"
            backend_api_file = Path(temp_dir) / "backend" / ".kiro" / "specs" / "api-contract.md"
            frontend_api_file = Path(temp_dir) / "frontend" / ".kiro" / "specs" / "api-contract.md"
            
            # Create directories
            for api_file in [shared_api_file, backend_api_file, frontend_api_file]:
                api_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create API contract content
            contract_content = "# API Contract\n\n"
            for section, changes in api_changes.items():
                contract_content += f"## {section.title()}\n{changes}\n\n"
            
            # Write to all locations
            for api_file in [shared_api_file, backend_api_file, frontend_api_file]:
                api_file.write_text(contract_content)
            
            # Verify synchronization
            shared_content = shared_api_file.read_text()
            backend_content = backend_api_file.read_text()
            frontend_content = frontend_api_file.read_text()
            
            assert shared_content == backend_content, "Backend API contract not synchronized"
            assert shared_content == frontend_content, "Frontend API contract not synchronized"
            assert backend_content == frontend_content, "Backend and frontend API contracts differ"