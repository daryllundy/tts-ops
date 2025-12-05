"""
Tests for validating GitHub Actions workflow YAML files.

These tests ensure that workflow files are syntactically valid and contain
required configuration elements.
"""

import pytest
import yaml
from pathlib import Path


WORKFLOWS_DIR = Path(__file__).parent.parent.parent / ".github" / "workflows"


def get_workflow_files():
    """Get all workflow YAML files."""
    if not WORKFLOWS_DIR.exists():
        return []
    return list(WORKFLOWS_DIR.glob("*.yml")) + list(WORKFLOWS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_yaml_syntax(workflow_file):
    """Test that workflow files are valid YAML."""
    with open(workflow_file, "r") as f:
        content = yaml.safe_load(f)
    
    assert content is not None, f"{workflow_file.name} is empty"
    assert isinstance(content, dict), f"{workflow_file.name} is not a valid YAML dictionary"


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_has_name(workflow_file):
    """Test that workflow files have a name field."""
    with open(workflow_file, "r") as f:
        content = yaml.safe_load(f)
    
    assert "name" in content, f"{workflow_file.name} missing 'name' field"
    assert isinstance(content["name"], str), f"{workflow_file.name} 'name' must be a string"
    assert len(content["name"]) > 0, f"{workflow_file.name} 'name' cannot be empty"


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_has_triggers(workflow_file):
    """Test that workflow files have trigger configuration."""
    with open(workflow_file, "r") as f:
        content = yaml.safe_load(f)
    
    # Workflow must have at least one trigger (on, true, or workflow_dispatch)
    has_trigger = "on" in content or "true" in content or "workflow_dispatch" in content
    assert has_trigger, f"{workflow_file.name} missing trigger configuration"


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_has_jobs(workflow_file):
    """Test that workflow files have jobs defined."""
    with open(workflow_file, "r") as f:
        content = yaml.safe_load(f)
    
    assert "jobs" in content, f"{workflow_file.name} missing 'jobs' section"
    assert isinstance(content["jobs"], dict), f"{workflow_file.name} 'jobs' must be a dictionary"
    assert len(content["jobs"]) > 0, f"{workflow_file.name} must have at least one job"


def test_workflows_directory_exists():
    """Test that the workflows directory exists."""
    assert WORKFLOWS_DIR.exists(), ".github/workflows directory does not exist"


def test_workflows_directory_has_readme():
    """Test that the workflows directory has a README."""
    readme_path = WORKFLOWS_DIR / "README.md"
    assert readme_path.exists(), ".github/workflows/README.md does not exist"
