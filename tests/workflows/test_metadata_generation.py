"""
Unit tests for build metadata generation.

Tests edge cases and specific scenarios for metadata generation including
missing environment variables, field validation, and JSON output format.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from generate_build_metadata import (
    generate_build_metadata,
    save_metadata,
    extract_git_sha,
    extract_git_ref,
    extract_workflow_run_id,
    extract_platform,
    REQUIRED_FIELDS,
)


class TestMissingEnvironmentVariables:
    """Test handling of missing environment variables."""

    def test_missing_github_sha(self):
        """Test that missing GITHUB_SHA is handled gracefully."""
        env = {
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                metadata = generate_build_metadata()
                
                assert not metadata.is_valid()
                assert any("git_sha" in error.lower() for error in metadata.errors)

    def test_missing_github_ref(self):
        """Test that missing GITHUB_REF is handled gracefully."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                metadata = generate_build_metadata()
                
                assert not metadata.is_valid()
                assert any("git_ref" in error.lower() for error in metadata.errors)

    def test_missing_github_run_id(self):
        """Test that missing GITHUB_RUN_ID is handled gracefully."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
        }
        
        with patch.dict(os.environ, env, clear=True):
            metadata = generate_build_metadata()
            
            assert not metadata.is_valid()
            assert any("workflow_run_id" in error.lower() for error in metadata.errors)

    def test_all_environment_variables_missing(self):
        """Test that all missing environment variables are reported."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                metadata = generate_build_metadata()
                
                assert not metadata.is_valid()
                assert len(metadata.errors) >= 3  # At least git_sha, git_ref, workflow_run_id


class TestFieldValidation:
    """Test field validation logic."""

    def test_valid_metadata_has_all_required_fields(self):
        """Test that valid metadata contains all required fields."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            assert metadata.is_valid()
            metadata_dict = metadata.to_dict()
            
            for field in REQUIRED_FIELDS:
                assert field in metadata_dict
                assert metadata_dict[field] is not None

    def test_optional_fields_not_required_for_validity(self):
        """Test that optional fields are not required for validity."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            # Generate without optional fields
            metadata = generate_build_metadata()
            
            # Should still be valid
            assert metadata.is_valid()
            
            # Should have warnings about missing optional fields
            assert len(metadata.warnings) > 0

    def test_image_field_included_when_provided(self):
        """Test that image field is included when provided."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata(image="ghcr.io/org/app:v1.0.0")
            
            assert metadata.is_valid()
            assert metadata.to_dict()["image"] == "ghcr.io/org/app:v1.0.0"

    def test_digest_field_included_when_provided(self):
        """Test that digest field is included when provided."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata(digest="sha256:abc123")
            
            assert metadata.is_valid()
            assert metadata.to_dict()["digest"] == "sha256:abc123"

    def test_platform_field_included_when_provided(self):
        """Test that platform field is included when provided."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata(platform="linux/amd64")
            
            assert metadata.is_valid()
            assert metadata.to_dict()["platform"] == "linux/amd64"


class TestJSONOutputFormat:
    """Test JSON output format and serialization."""

    def test_metadata_is_json_serializable(self):
        """Test that metadata can be serialized to JSON."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            # Should not raise exception
            json_str = json.dumps(metadata.to_dict())
            assert json_str is not None

    def test_json_output_can_be_deserialized(self):
        """Test that JSON output can be deserialized back."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            json_str = json.dumps(metadata.to_dict())
            deserialized = json.loads(json_str)
            
            # Verify all required fields are preserved
            for field in REQUIRED_FIELDS:
                assert field in deserialized

    def test_save_metadata_creates_valid_json_file(self):
        """Test that save_metadata creates a valid JSON file."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "metadata.json"
                save_metadata(output_path, metadata)
                
                # Verify file exists
                assert output_path.exists()
                
                # Verify file contains valid JSON
                with open(output_path, "r") as f:
                    loaded = json.load(f)
                
                # Verify all required fields are present
                for field in REQUIRED_FIELDS:
                    assert field in loaded

    def test_json_output_is_properly_formatted(self):
        """Test that JSON output is properly formatted with indentation."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "metadata.json"
                save_metadata(output_path, metadata)
                
                # Read file content
                with open(output_path, "r") as f:
                    content = f.read()
                
                # Verify it's formatted (has newlines and indentation)
                assert "\n" in content
                assert "  " in content  # Indentation


class TestEnvironmentExtraction:
    """Test environment variable extraction functions."""

    def test_extract_git_sha_from_github_actions(self):
        """Test extracting git SHA from GITHUB_SHA."""
        with patch.dict(os.environ, {"GITHUB_SHA": "abc123def456"}, clear=True):
            sha = extract_git_sha()
            assert sha == "abc123def456"

    def test_extract_git_ref_from_github_actions(self):
        """Test extracting git ref from GITHUB_REF."""
        with patch.dict(os.environ, {"GITHUB_REF": "refs/heads/main"}, clear=True):
            ref = extract_git_ref()
            assert ref == "refs/heads/main"

    def test_extract_workflow_run_id_from_github_actions(self):
        """Test extracting workflow run ID from GITHUB_RUN_ID."""
        with patch.dict(os.environ, {"GITHUB_RUN_ID": "123456"}, clear=True):
            run_id = extract_workflow_run_id()
            assert run_id == "123456"

    def test_extract_platform_from_platform_env(self):
        """Test extracting platform from PLATFORM environment variable."""
        with patch.dict(os.environ, {"PLATFORM": "linux/amd64"}, clear=True):
            platform = extract_platform()
            assert platform == "linux/amd64"

    def test_extract_platform_from_targetplatform_env(self):
        """Test extracting platform from TARGETPLATFORM environment variable."""
        with patch.dict(os.environ, {"TARGETPLATFORM": "linux/arm64"}, clear=True):
            platform = extract_platform()
            assert platform == "linux/arm64"

    def test_extract_platform_prefers_platform_over_targetplatform(self):
        """Test that PLATFORM is preferred over TARGETPLATFORM."""
        env = {
            "PLATFORM": "linux/amd64",
            "TARGETPLATFORM": "linux/arm64",
        }
        with patch.dict(os.environ, env, clear=True):
            platform = extract_platform()
            assert platform == "linux/amd64"


class TestBuilderField:
    """Test builder field handling."""

    def test_default_builder_is_github_actions(self):
        """Test that default builder is 'github-actions'."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            assert metadata.to_dict()["builder"] == "github-actions"

    def test_custom_builder_from_environment(self):
        """Test that custom builder can be set via BUILDER environment variable."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
            "BUILDER": "custom-builder",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            assert metadata.to_dict()["builder"] == "custom-builder"


class TestBuildTimestamp:
    """Test build timestamp generation."""

    def test_build_timestamp_is_iso_format(self):
        """Test that build timestamp is in ISO format."""
        env = {
            "GITHUB_SHA": "abc123def456",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_RUN_ID": "123456",
        }
        
        with patch.dict(os.environ, env, clear=False):
            metadata = generate_build_metadata()
            
            timestamp = metadata.to_dict()["build_timestamp"]
            
            # Should contain 'T' separator
            assert "T" in timestamp
            
            # Should be UTC (ends with Z or +00:00)
            assert timestamp.endswith("Z") or timestamp.endswith("+00:00")

    def test_build_timestamp_is_always_present(self):
        """Test that build timestamp is always present even with missing fields."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                metadata = generate_build_metadata()
                
                # Even though metadata is invalid, timestamp should be present
                assert "build_timestamp" in metadata.to_dict()
