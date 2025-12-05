"""
Property-based tests for build metadata generation.

**Feature: github-actions-cicd, Property 2: Build metadata includes all required fields**
**Validates: Requirements 9.2, 9.3**
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

from hypothesis import given
from hypothesis import strategies as st

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from generate_build_metadata import (
    REQUIRED_FIELDS,
    generate_build_metadata,
)

# Strategy for generating valid git SHAs (40 character hex strings)
git_sha = st.text(
    alphabet="0123456789abcdef",
    min_size=40,
    max_size=40,
)

# Strategy for generating git refs
git_ref = st.one_of(
    st.builds(lambda branch: f"refs/heads/{branch}", st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_/"))),
    st.builds(lambda tag: f"refs/tags/{tag}", st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=".-_"))),
)

# Strategy for generating workflow run IDs (numeric strings)
workflow_run_id = st.integers(min_value=1, max_value=999999999).map(str)

# Strategy for generating image names
image_name = st.builds(
    lambda org, repo, tag: f"ghcr.io/{org}/{repo}:{tag}",
    org=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="-")),
    repo=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="-_")),
    tag=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=".-_")),
)

# Strategy for generating platforms
platform = st.sampled_from(["linux/amd64", "linux/arm64", "linux/arm/v7"])


@given(
    sha=git_sha,
    ref=git_ref,
    run_id=workflow_run_id,
)
def test_metadata_includes_all_required_fields(sha, ref, run_id):
    """
    Property: For any valid build context (git SHA, git ref, workflow run ID),
    the metadata generation function should produce a metadata object that contains
    all required fields with valid values.

    This tests that the metadata generation always produces complete metadata
    when given valid environment variables.
    """
    # Set up environment variables
    env = {
        "GITHUB_SHA": sha,
        "GITHUB_REF": ref,
        "GITHUB_RUN_ID": run_id,
    }

    with patch.dict(os.environ, env, clear=False):
        # Generate metadata
        metadata = generate_build_metadata()

        # Verify metadata is valid
        assert metadata.is_valid(), (
            f"Metadata is not valid. Errors: {metadata.errors}"
        )

        # Verify all required fields are present
        metadata_dict = metadata.to_dict()
        for field in REQUIRED_FIELDS:
            assert field in metadata_dict, (
                f"Required field '{field}' is missing from metadata"
            )
            assert metadata_dict[field] is not None, (
                f"Required field '{field}' is None"
            )
            assert metadata_dict[field] != "", (
                f"Required field '{field}' is empty string"
            )

        # Verify the values match what we set
        assert metadata_dict["git_sha"] == sha
        assert metadata_dict["git_ref"] == ref
        assert metadata_dict["workflow_run_id"] == run_id
        assert metadata_dict["builder"] == "github-actions"

        # Verify build_timestamp is present and valid ISO format
        assert "build_timestamp" in metadata_dict
        assert "T" in metadata_dict["build_timestamp"]
        assert metadata_dict["build_timestamp"].endswith(("Z", "+00:00"))


@given(
    sha=git_sha,
    ref=git_ref,
    run_id=workflow_run_id,
    image=image_name,
    plat=platform,
)
def test_metadata_includes_optional_fields_when_provided(sha, ref, run_id, image, plat):
    """
    Property: For any valid build context with optional fields provided,
    the metadata generation function should include those optional fields
    in the output.

    This tests that optional fields are properly included when provided.
    """
    # Set up environment variables
    env = {
        "GITHUB_SHA": sha,
        "GITHUB_REF": ref,
        "GITHUB_RUN_ID": run_id,
    }

    with patch.dict(os.environ, env, clear=False):
        # Generate metadata with optional fields
        metadata = generate_build_metadata(
            image=image,
            platform=plat,
        )

        # Verify metadata is valid
        assert metadata.is_valid()

        # Verify optional fields are present
        metadata_dict = metadata.to_dict()
        assert metadata_dict["image"] == image
        assert metadata_dict["platform"] == plat


@given(
    sha=git_sha,
    ref=git_ref,
)
def test_metadata_invalid_when_missing_required_field(sha, ref):
    """
    Property: For any build context missing a required field (workflow_run_id),
    the metadata generation function should produce invalid metadata with
    appropriate error messages.

    This tests that the validation correctly identifies missing required fields.
    """
    # Set up environment variables WITHOUT workflow_run_id
    env = {
        "GITHUB_SHA": sha,
        "GITHUB_REF": ref,
        # GITHUB_RUN_ID is intentionally missing
    }

    # Clear GITHUB_RUN_ID if it exists
    with patch.dict(os.environ, env, clear=True):
        # Generate metadata
        metadata = generate_build_metadata()

        # Verify metadata is NOT valid
        assert not metadata.is_valid(), (
            "Metadata should be invalid when required field is missing"
        )

        # Verify there are errors
        assert len(metadata.errors) > 0, (
            "Should have errors when required field is missing"
        )

        # Verify the error mentions the missing field
        error_text = " ".join(metadata.errors)
        assert "workflow_run_id" in error_text.lower()


@given(
    sha=git_sha,
    ref=git_ref,
    run_id=workflow_run_id,
)
def test_metadata_structure_is_json_serializable(sha, ref, run_id):
    """
    Property: For any valid build context, the metadata should be JSON serializable
    without errors.

    This tests that the metadata structure can be safely serialized to JSON.
    """
    import json

    # Set up environment variables
    env = {
        "GITHUB_SHA": sha,
        "GITHUB_REF": ref,
        "GITHUB_RUN_ID": run_id,
    }

    with patch.dict(os.environ, env, clear=False):
        # Generate metadata
        metadata = generate_build_metadata()

        # Verify we can serialize to JSON without errors
        metadata_dict = metadata.to_dict()
        json_str = json.dumps(metadata_dict)

        # Verify we can deserialize back
        deserialized = json.loads(json_str)

        # Verify all required fields are preserved
        for field in REQUIRED_FIELDS:
            assert field in deserialized
            assert deserialized[field] == metadata_dict[field]


@given(
    sha=git_sha,
    ref=git_ref,
    run_id=workflow_run_id,
    builder=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")),
)
def test_metadata_respects_custom_builder(sha, ref, run_id, builder):
    """
    Property: For any valid build context with a custom BUILDER environment variable,
    the metadata should use that builder value instead of the default.

    This tests that custom builder values are properly respected.
    """
    # Set up environment variables with custom builder
    env = {
        "GITHUB_SHA": sha,
        "GITHUB_REF": ref,
        "GITHUB_RUN_ID": run_id,
        "BUILDER": builder,
    }

    with patch.dict(os.environ, env, clear=False):
        # Generate metadata
        metadata = generate_build_metadata()

        # Verify metadata is valid
        assert metadata.is_valid()

        # Verify builder field matches custom value
        metadata_dict = metadata.to_dict()
        assert metadata_dict["builder"] == builder
