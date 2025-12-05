#!/usr/bin/env python3
"""
Build metadata generation script for CI/CD pipeline.

Extracts build context from environment variables and generates structured
metadata for build provenance and traceability.

Usage:
    python generate_build_metadata.py --output metadata.json
    python generate_build_metadata.py --output metadata.json --image ghcr.io/org/app:v1.0.0
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Required fields for build metadata
REQUIRED_FIELDS = [
    "git_sha",
    "git_ref",
    "workflow_run_id",
    "build_timestamp",
    "builder",
]

# Optional fields that enhance metadata
OPTIONAL_FIELDS = [
    "image",
    "digest",
    "platform",
]


class BuildMetadata:
    """Build metadata container with validation."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_field(self, field: str, value: Any) -> None:
        """Add a field to the metadata."""
        if value is not None:
            self.data[field] = value

    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def is_valid(self) -> bool:
        """Check if metadata is valid (all required fields present)."""
        return len(self.errors) == 0 and all(
            field in self.data for field in REQUIRED_FIELDS
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return self.data


def extract_git_sha() -> str | None:
    """Extract git SHA from environment."""
    # GitHub Actions provides GITHUB_SHA
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha

    # Try git command as fallback
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def extract_git_ref() -> str | None:
    """Extract git ref from environment."""
    # GitHub Actions provides GITHUB_REF
    ref = os.getenv("GITHUB_REF")
    if ref:
        return ref

    # Try git command as fallback
    try:
        import subprocess

        result = subprocess.run(
            ["git", "symbolic-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def extract_workflow_run_id() -> str | None:
    """Extract workflow run ID from environment."""
    # GitHub Actions provides GITHUB_RUN_ID
    return os.getenv("GITHUB_RUN_ID")


def extract_platform() -> str | None:
    """Extract build platform from environment."""
    # Common environment variables for platform
    platform = os.getenv("PLATFORM") or os.getenv("TARGETPLATFORM")
    return platform


def generate_build_metadata(
    image: str | None = None,
    digest: str | None = None,
    platform: str | None = None,
) -> BuildMetadata:
    """Generate build metadata from environment and arguments."""
    metadata = BuildMetadata()

    # Extract required fields
    git_sha = extract_git_sha()
    if git_sha:
        metadata.add_field("git_sha", git_sha)
    else:
        metadata.add_error("Missing required field: git_sha")

    git_ref = extract_git_ref()
    if git_ref:
        metadata.add_field("git_ref", git_ref)
    else:
        metadata.add_error("Missing required field: git_ref")

    workflow_run_id = extract_workflow_run_id()
    if workflow_run_id:
        metadata.add_field("workflow_run_id", workflow_run_id)
    else:
        metadata.add_error("Missing required field: workflow_run_id")

    # Add build timestamp (always available)
    build_timestamp = datetime.now(timezone.utc).isoformat()
    metadata.add_field("build_timestamp", build_timestamp)

    # Add builder (default to github-actions)
    builder = os.getenv("BUILDER", "github-actions")
    metadata.add_field("builder", builder)

    # Add optional fields
    if image:
        metadata.add_field("image", image)
    else:
        metadata.add_warning("Optional field not provided: image")

    if digest:
        metadata.add_field("digest", digest)

    # Extract platform from environment if not provided
    if platform:
        metadata.add_field("platform", platform)
    else:
        env_platform = extract_platform()
        if env_platform:
            metadata.add_field("platform", env_platform)

    return metadata


def save_metadata(file_path: Path, metadata: BuildMetadata) -> None:
    """Save metadata to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)


def print_report(metadata: BuildMetadata) -> None:
    """Print human-readable metadata report."""
    print("\n" + "=" * 80)
    print("BUILD METADATA GENERATION REPORT")
    print("=" * 80)

    if metadata.is_valid():
        print("\n✅ METADATA VALID - All required fields present")
        print("\nGenerated Metadata:")
        for key, value in metadata.to_dict().items():
            print(f"  {key}: {value}")
    else:
        print("\n❌ METADATA INVALID - Missing required fields")
        for error in metadata.errors:
            print(f"  - {error}")

    if metadata.warnings:
        print("\n⚠️  Warnings:")
        for warning in metadata.warnings:
            print(f"  - {warning}")

    print("=" * 80 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate build metadata from environment context"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output metadata JSON file",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Container image name (e.g., ghcr.io/org/app:v1.0.0)",
    )
    parser.add_argument(
        "--digest",
        type=str,
        help="Container image digest (e.g., sha256:abc123...)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Build platform (e.g., linux/amd64, linux/arm64)",
    )

    args = parser.parse_args()

    # Generate metadata
    metadata = generate_build_metadata(
        image=args.image,
        digest=args.digest,
        platform=args.platform,
    )

    # Print report
    print_report(metadata)

    # Save metadata if valid
    if metadata.is_valid():
        save_metadata(args.output, metadata)
        print(f"✅ Metadata saved to: {args.output}")
        sys.exit(0)
    else:
        print(f"❌ Failed to generate valid metadata", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
