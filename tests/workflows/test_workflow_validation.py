"""
Tests for validating GitHub Actions workflow YAML files.

These tests ensure that workflow files are syntactically valid and contain
required configuration elements.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).parent.parent.parent / ".github" / "workflows"


def get_workflow_files():
    """Get all workflow YAML files."""
    if not WORKFLOWS_DIR.exists():
        return []
    return list(WORKFLOWS_DIR.glob("*.yml")) + list(WORKFLOWS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_yaml_syntax(workflow_file):
    """Test that workflow files are valid YAML."""
    with open(workflow_file) as f:
        content = yaml.safe_load(f)

    assert content is not None, f"{workflow_file.name} is empty"
    assert isinstance(content, dict), f"{workflow_file.name} is not a valid YAML dictionary"


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_has_name(workflow_file):
    """Test that workflow files have a name field."""
    with open(workflow_file) as f:
        content = yaml.safe_load(f)

    assert "name" in content, f"{workflow_file.name} missing 'name' field"
    assert isinstance(content["name"], str), f"{workflow_file.name} 'name' must be a string"
    assert len(content["name"]) > 0, f"{workflow_file.name} 'name' cannot be empty"


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_has_triggers(workflow_file):
    """Test that workflow files have trigger configuration."""
    with open(workflow_file) as f:
        content = yaml.safe_load(f)

    # Workflow must have at least one trigger (on, True, or workflow_dispatch)
    # Note: YAML parses 'on' as boolean True
    has_trigger = "on" in content or True in content or "workflow_dispatch" in content
    assert has_trigger, f"{workflow_file.name} missing trigger configuration"


@pytest.mark.parametrize("workflow_file", get_workflow_files(), ids=lambda p: p.name)
def test_workflow_has_jobs(workflow_file):
    """Test that workflow files have jobs defined."""
    with open(workflow_file) as f:
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


def load_workflow(filename):
    """Load a workflow file by name."""
    path = WORKFLOWS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


class TestCiTestWorkflow:
    """Validation tests for ci-test.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-test.yml")

    def test_exists(self, workflow):
        assert workflow is not None, "ci-test.yml does not exist"

    def test_triggers(self, workflow):
        """Verify trigger configuration."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))
        assert "push" in triggers
        assert "pull_request" in triggers

    def test_python_setup(self, workflow):
        """Verify Python setup with version 3.11."""
        jobs = workflow.get("jobs", {})
        # Check in any job (e.g., test or lint)
        found_python = False
        for _job_name, job in jobs.items():
            steps = job.get("steps", [])
            for step in steps:
                uses = step.get("uses", "")
                if "actions/setup-python" in uses:
                    found_python = True
                    with_args = step.get("with", {})
                    assert with_args.get("python-version") == "3.11"

        assert found_python, "Python setup step not found"

    def test_caching(self, workflow):
        """Verify dependency caching."""
        jobs = workflow.get("jobs", {})
        found_cache = False
        for _job_name, job in jobs.items():
            steps = job.get("steps", [])
            for step in steps:
                uses = step.get("uses", "")
                if "actions/cache" in uses:
                    found_cache = True
                    break

        assert found_cache, "Dependency caching step not found"


class TestCiBuildPushWorkflow:
    """Validation tests for ci-build-push.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-build-push.yml")

    def test_exists(self, workflow):
        assert workflow is not None, "ci-build-push.yml does not exist"

    def test_triggers(self, workflow):
        """Verify triggers for tags and main branch."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))

        # Check push to main
        if "push" in triggers:
            push = triggers["push"]
            branches = push.get("branches", [])
            tags = push.get("tags", [])
            assert "main" in branches or not branches  # Empty branches means all
            assert tags  # Should trigger on tags
        else:
            pytest.fail("Missing push trigger")

    def test_build_matrix(self, workflow):
        """Verify build matrix includes amd64 and arm64."""
        # This might be in the strategy or in the docker buildx args
        # For docker buildx, it's usually platforms input
        jobs = workflow.get("jobs", {})
        build_job = jobs.get("build-and-push")
        assert build_job, "build-and-push job not found"

        steps = build_job.get("steps", [])
        found_build_push = False
        for step in steps:
            uses = step.get("uses", "")
            if "docker/build-push-action" in uses:
                found_build_push = True
                with_args = step.get("with", {})
                platforms = with_args.get("platforms", "")
                assert "linux/amd64" in platforms
                assert "linux/arm64" in platforms

        assert found_build_push, "docker/build-push-action step not found"

    def test_registry_auth(self, workflow):
        """Verify registry authentication."""
        jobs = workflow.get("jobs", {})
        build_job = jobs.get("build-and-push")

        steps = build_job.get("steps", [])
        found_login = False
        for step in steps:
            uses = step.get("uses", "")
            if "docker/login-action" in uses:
                found_login = True
                with_args = step.get("with", {})
                registry = with_args.get("registry", "")
                # Check for literal or env var
                assert "ghcr.io" in registry or "${{ env.REGISTRY }}" in registry

        assert found_login, "docker/login-action step not found"


class TestCiSmokeTestWorkflow:
    """Validation tests for ci-smoke-test.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-smoke-test.yml")

    def test_exists(self, workflow):
        assert workflow is not None, "ci-smoke-test.yml does not exist"

    def test_triggers(self, workflow):
        """Verify workflow_run trigger."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))
        assert "workflow_run" in triggers

        wf_run = triggers["workflow_run"]
        workflows = wf_run.get("workflows", [])
        assert "Build and Push" in workflows or "ci-build-push" in workflows
        assert "completed" in wf_run.get("types", [])

    def test_smoke_test_execution(self, workflow):
        """Verify smoke test script execution."""
        jobs = workflow.get("jobs", {})
        smoke_job = jobs.get("smoke-test")
        assert smoke_job, "smoke-test job not found"

        steps = smoke_job.get("steps", [])
        found_script = False
        for step in steps:
            run = step.get("run", "")
            if "scripts/smoke_test.py" in run:
                found_script = True
                break

        assert found_script, "Smoke test script execution step not found"


class TestCiPerformanceWorkflow:
    """Validation tests for ci-performance.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-performance.yml")

    def test_exists(self, workflow):
        """Validate YAML syntax - workflow file exists and is valid."""
        assert workflow is not None, "ci-performance.yml does not exist"

    def test_scheduled_trigger(self, workflow):
        """Verify scheduled trigger configuration (nightly at 2 AM UTC)."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))

        assert "schedule" in triggers, "Missing schedule trigger"
        schedule = triggers["schedule"]
        assert isinstance(schedule, list), "Schedule should be a list"
        assert len(schedule) > 0, "Schedule list is empty"

        # Check cron expression for 2 AM UTC
        cron_expr = schedule[0].get("cron", "")
        assert "0 2 * * *" in cron_expr, "Schedule should be nightly at 2 AM UTC"

        # Check workflow_dispatch for manual trigger
        assert "workflow_dispatch" in triggers, "Missing workflow_dispatch trigger"

    def test_job_dependencies(self, workflow):
        """Verify job dependencies between benchmark and regression check."""
        jobs = workflow.get("jobs", {})

        # Check benchmark job exists
        assert "benchmark" in jobs, "benchmark job not found"

        # Check regression-check job exists
        assert "regression-check" in jobs, "regression-check job not found"

        # Verify regression-check depends on benchmark
        regression_job = jobs["regression-check"]
        needs = regression_job.get("needs")
        assert needs == "benchmark" or "benchmark" in needs, \
            "regression-check should depend on benchmark job"

    def test_artifact_handling(self, workflow):
        """Verify artifact handling for metrics and baseline."""
        jobs = workflow.get("jobs", {})

        # Check benchmark job uploads metrics
        benchmark_job = jobs.get("benchmark", {})
        benchmark_steps = benchmark_job.get("steps", [])
        found_upload = False
        for step in benchmark_steps:
            uses = step.get("uses", "")
            if "actions/upload-artifact" in uses:
                found_upload = True
                with_args = step.get("with", {})
                assert with_args.get("name") == "performance-metrics"

        assert found_upload, "Benchmark job should upload performance-metrics artifact"

        # Check regression-check job downloads and uploads artifacts
        regression_job = jobs.get("regression-check", {})
        regression_steps = regression_job.get("steps", [])
        found_download_current = False
        found_download_baseline = False
        found_upload_baseline = False

        for step in regression_steps:
            uses = step.get("uses", "")
            with_args = step.get("with", {})

            if "actions/download-artifact" in uses:
                if with_args.get("name") == "performance-metrics":
                    found_download_current = True

            if "dawidd6/action-download-artifact" in uses:
                if with_args.get("name") == "performance-baseline":
                    found_download_baseline = True

            if "actions/upload-artifact" in uses:
                if with_args.get("name") == "performance-baseline":
                    found_upload_baseline = True

        assert found_download_current, "Should download current performance-metrics"
        assert found_download_baseline, "Should download baseline metrics from previous runs"
        assert found_upload_baseline, "Should upload new baseline on success"


class TestCiHelmValidateWorkflow:
    """Validation tests for ci-helm-validate.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-helm-validate.yml")

    def test_exists(self, workflow):
        """Validate YAML syntax - workflow file exists and is valid."""
        assert workflow is not None, "ci-helm-validate.yml does not exist"

    def test_path_based_triggers(self, workflow):
        """Verify path-based trigger configuration for infra/k8s/**."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))

        # Check push trigger with paths
        assert "push" in triggers, "Missing push trigger"
        push = triggers["push"]
        paths = push.get("paths", [])
        assert any("infra/k8s" in path for path in paths), "Missing infra/k8s/** path trigger"

        # Check pull_request trigger with paths
        assert "pull_request" in triggers, "Missing pull_request trigger"
        pr = triggers["pull_request"]
        pr_paths = pr.get("paths", [])
        assert any("infra/k8s" in path for path in pr_paths), "Missing infra/k8s/** path trigger in PR"

    def test_helm_setup_steps(self, workflow):
        """Verify Helm setup steps are present."""
        jobs = workflow.get("jobs", {})

        # Check helm-lint job
        helm_lint = jobs.get("helm-lint")
        assert helm_lint, "helm-lint job not found"

        lint_steps = helm_lint.get("steps", [])
        found_helm_setup = False
        found_helm_lint = False

        for step in lint_steps:
            uses = step.get("uses", "")
            run = step.get("run", "")

            if "azure/setup-helm" in uses or "setup-helm" in uses:
                found_helm_setup = True
                with_args = step.get("with", {})
                assert "version" in with_args, "Helm version not specified"

            if "helm lint" in run:
                found_helm_lint = True

        assert found_helm_setup, "Helm setup step not found in helm-lint job"
        assert found_helm_lint, "helm lint command not found in helm-lint job"

        # Check helm-template job
        helm_template = jobs.get("helm-template")
        assert helm_template, "helm-template job not found"

        template_steps = helm_template.get("steps", [])
        found_helm_template = False

        for step in template_steps:
            run = step.get("run", "")
            if "helm template" in run:
                found_helm_template = True

        assert found_helm_template, "helm template command not found in helm-template job"

    def test_optional_kind_job(self, workflow):
        """Verify optional Kind job configuration."""
        jobs = workflow.get("jobs", {})

        # Check kind-test job exists
        kind_test = jobs.get("kind-test")
        assert kind_test, "kind-test job not found"

        # Verify it's conditional (optional)
        if_condition = kind_test.get("if")
        assert if_condition, "kind-test job should have conditional execution (if)"
        assert "workflow_dispatch" in if_condition or "test-k8s" in if_condition, \
            "kind-test should be triggered by workflow_dispatch or commit message"

        # Verify it depends on lint and template jobs
        needs = kind_test.get("needs", [])
        assert "helm-lint" in needs, "kind-test should depend on helm-lint"
        assert "helm-template" in needs, "kind-test should depend on helm-template"

        # Verify Kind cluster creation
        steps = kind_test.get("steps", [])
        found_kind_action = False
        found_manifest_apply = False
        found_connectivity_test = False

        for step in steps:
            uses = step.get("uses", "")
            run = step.get("run", "")

            if "kind-action" in uses:
                found_kind_action = True

            if "helm install" in run or "kubectl apply" in run:
                found_manifest_apply = True

            if "kubectl get" in run or "kubectl describe" in run:
                found_connectivity_test = True

        assert found_kind_action, "Kind cluster creation step not found"
        assert found_manifest_apply, "Manifest application step not found"
        assert found_connectivity_test, "Connectivity test step not found"



class TestCiSecurityDepsWorkflow:
    """Validation tests for ci-security-deps.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-security-deps.yml")

    def test_exists(self, workflow):
        """Validate YAML syntax - workflow file exists and is valid."""
        assert workflow is not None, "ci-security-deps.yml does not exist"

    def test_trigger_configuration(self, workflow):
        """Verify trigger configuration including schedule."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))

        # Check push trigger
        assert "push" in triggers, "Missing push trigger"
        push = triggers["push"]
        branches = push.get("branches", [])
        assert "main" in branches, "Should trigger on push to main"

        # Check pull_request trigger
        assert "pull_request" in triggers, "Missing pull_request trigger"
        pr = triggers["pull_request"]
        pr_branches = pr.get("branches", [])
        assert "main" in pr_branches, "Should trigger on PRs to main"

        # Check schedule trigger (Monday 9 AM UTC)
        assert "schedule" in triggers, "Missing schedule trigger"
        schedule = triggers["schedule"]
        assert isinstance(schedule, list), "Schedule should be a list"
        assert len(schedule) > 0, "Schedule list is empty"

        cron_expr = schedule[0].get("cron", "")
        assert "0 9 * * 1" in cron_expr, "Schedule should be Monday at 9 AM UTC"

        # Check workflow_dispatch for manual trigger
        assert "workflow_dispatch" in triggers, "Missing workflow_dispatch trigger"

    def test_pip_audit_execution(self, workflow):
        """Verify pip-audit execution steps."""
        jobs = workflow.get("jobs", {})

        # Check scan-dependencies job exists
        scan_job = jobs.get("scan-dependencies")
        assert scan_job, "scan-dependencies job not found"

        steps = scan_job.get("steps", [])

        # Verify Python 3.11 setup
        found_python = False
        for step in steps:
            uses = step.get("uses", "")
            if "actions/setup-python" in uses:
                found_python = True
                with_args = step.get("with", {})
                assert with_args.get("python-version") == "3.11", "Should use Python 3.11"

        assert found_python, "Python setup step not found"

        # Verify pip-audit installation
        found_install = False
        for step in steps:
            run = step.get("run", "")
            if "pip install pip-audit" in run:
                found_install = True

        assert found_install, "pip-audit installation step not found"

        # Verify pip-audit execution
        found_audit = False
        for step in steps:
            run = step.get("run", "")
            if "pip-audit" in run and ("pyproject.toml" in run or "--desc" in run):
                found_audit = True

        assert found_audit, "pip-audit execution step not found"

        # Verify vulnerability check for critical/high CVEs
        found_check = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "critical" in name.lower() or "high" in name.lower() or "vulnerabilities" in run:
                found_check = True

        assert found_check, "Critical/high vulnerability check step not found"

    def test_sarif_upload_configuration(self, workflow):
        """Verify SARIF upload configuration."""
        jobs = workflow.get("jobs", {})
        scan_job = jobs.get("scan-dependencies", {})
        steps = scan_job.get("steps", [])

        # Verify artifact upload for vulnerability report
        found_upload = False
        for step in steps:
            uses = step.get("uses", "")
            if "actions/upload-artifact" in uses:
                found_upload = True
                with_args = step.get("with", {})
                name = with_args.get("name", "")
                assert "vulnerability" in name or "report" in name, \
                    "Artifact should be named appropriately for vulnerability reports"

        assert found_upload, "Vulnerability report upload step not found"

        # Note: pip-audit doesn't natively support SARIF format
        # The workflow generates JSON reports which can be converted to SARIF
        # or uploaded to GitHub Security tab using other tools

    def test_permissions(self, workflow):
        """Verify job has appropriate permissions for security scanning."""
        jobs = workflow.get("jobs", {})
        scan_job = jobs.get("scan-dependencies", {})

        # Check permissions are set
        permissions = scan_job.get("permissions", {})
        assert permissions, "scan-dependencies job should have permissions defined"

        # Should have read access to contents
        assert permissions.get("contents") == "read", \
            "Should have read access to contents"

        # Should have write access to security-events for SARIF upload
        assert permissions.get("security-events") == "write", \
            "Should have write access to security-events"

    def test_dependabot_check(self, workflow):
        """Verify Dependabot configuration check job."""
        jobs = workflow.get("jobs", {})

        # Check dependabot check job exists
        dependabot_job = jobs.get("check-dependabot")
        assert dependabot_job, "check-dependabot job not found"

        steps = dependabot_job.get("steps", [])
        found_check = False

        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "dependabot" in name.lower() or ".github/dependabot.yml" in run:
                found_check = True

        assert found_check, "Dependabot configuration check step not found"


class TestCiSecurityImagesWorkflow:
    """Validation tests for ci-security-images.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-security-images.yml")

    def test_exists(self, workflow):
        """Validate YAML syntax - workflow file exists and is valid."""
        assert workflow is not None, "ci-security-images.yml does not exist"

    def test_workflow_run_trigger(self, workflow):
        """Verify workflow_run trigger configuration."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))

        # Check workflow_run trigger
        assert "workflow_run" in triggers, "Missing workflow_run trigger"

        wf_run = triggers["workflow_run"]
        workflows = wf_run.get("workflows", [])
        assert "Build and Push" in workflows, \
            "Should trigger on 'Build and Push' workflow completion"

        types = wf_run.get("types", [])
        assert "completed" in types, "Should trigger on workflow completion"

        # Check workflow_dispatch for manual trigger
        assert "workflow_dispatch" in triggers, "Missing workflow_dispatch trigger"

    def test_trivy_setup_and_execution(self, workflow):
        """Verify Trivy setup and execution."""
        jobs = workflow.get("jobs", {})

        # Check scan-images job exists
        scan_job = jobs.get("scan-images")
        assert scan_job, "scan-images job not found"

        # Verify conditional execution on successful build
        if_condition = scan_job.get("if")
        assert if_condition, "scan-images job should have conditional execution"
        assert "workflow_run.conclusion == 'success'" in if_condition or \
               "workflow_dispatch" in if_condition, \
            "Should only run on successful build or manual dispatch"

        # Verify matrix strategy for multiple services
        strategy = scan_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        services = matrix.get("service", [])
        assert len(services) >= 2, "Should scan at least 2 services (TTS and agent)"
        assert "tts-service" in services or "tts" in str(services), \
            "Should include TTS service"
        assert "agent-service" in services or "agent" in str(services), \
            "Should include agent service"

        steps = scan_job.get("steps", [])

        # Verify image pull step
        found_pull = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "docker pull" in run or "pull" in name.lower():
                found_pull = True

        assert found_pull, "Image pull step not found"

        # Verify Trivy scanner execution
        found_trivy = False
        found_sarif = False
        found_json = False

        for step in steps:
            uses = step.get("uses", "")
            if "aquasecurity/trivy-action" in uses or "trivy" in uses.lower():
                found_trivy = True
                with_args = step.get("with", {})

                # Check for SARIF format output
                if with_args.get("format") == "sarif":
                    found_sarif = True

                # Check for JSON format output
                if with_args.get("format") == "json":
                    found_json = True

        assert found_trivy, "Trivy scanner execution step not found"
        assert found_sarif, "Trivy SARIF output not configured"
        assert found_json, "Trivy JSON output not configured for parsing"

        # Verify critical CVE check
        found_cve_check = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "critical" in name.lower() or ("CRITICAL" in run and "exit 1" in run):
                found_cve_check = True

        assert found_cve_check, "Critical CVE check step not found"

    def test_sarif_upload_steps(self, workflow):
        """Verify SARIF upload steps."""
        jobs = workflow.get("jobs", {})
        scan_job = jobs.get("scan-images", {})
        steps = scan_job.get("steps", [])

        # Verify SARIF upload to GitHub Security tab
        found_sarif_upload = False
        for step in steps:
            uses = step.get("uses", "")
            if "github/codeql-action/upload-sarif" in uses or "upload-sarif" in uses:
                found_sarif_upload = True
                with_args = step.get("with", {})

                # Verify SARIF file is specified
                sarif_file = with_args.get("sarif_file", "")
                assert "trivy-results" in sarif_file or ".sarif" in sarif_file, \
                    "SARIF file should be specified"

                # Verify category is set for proper organization
                category = with_args.get("category", "")
                assert "container" in category or "service" in category, \
                    "Category should identify container scans"

        assert found_sarif_upload, "SARIF upload to GitHub Security tab not found"

        # Verify artifact upload for scan results
        found_artifact_upload = False
        for step in steps:
            uses = step.get("uses", "")
            if "actions/upload-artifact" in uses:
                found_artifact_upload = True
                with_args = step.get("with", {})

                # Verify artifact name
                name = with_args.get("name", "")
                assert "trivy" in name or "scan" in name, \
                    "Artifact name should indicate Trivy scan results"

                # Verify retention period
                retention = with_args.get("retention-days")
                assert retention == 90 or retention == "90", \
                    "Scan results should be retained for 90 days"

        assert found_artifact_upload, "Artifact upload for scan results not found"

    def test_permissions(self, workflow):
        """Verify job has appropriate permissions for security scanning."""
        jobs = workflow.get("jobs", {})
        scan_job = jobs.get("scan-images", {})

        # Check permissions are set
        permissions = scan_job.get("permissions", {})
        assert permissions, "scan-images job should have permissions defined"

        # Should have read access to contents
        assert permissions.get("contents") == "read", \
            "Should have read access to contents"

        # Should have write access to security-events for SARIF upload
        assert permissions.get("security-events") == "write", \
            "Should have write access to security-events"

        # Should have read access to packages to pull images
        assert permissions.get("packages") == "read", \
            "Should have read access to packages"

    def test_registry_authentication(self, workflow):
        """Verify registry authentication for pulling images."""
        jobs = workflow.get("jobs", {})
        scan_job = jobs.get("scan-images", {})
        steps = scan_job.get("steps", [])

        # Verify registry login step
        found_login = False
        for step in steps:
            uses = step.get("uses", "")
            if "docker/login-action" in uses:
                found_login = True
                with_args = step.get("with", {})

                # Verify registry is specified
                registry = with_args.get("registry", "")
                assert "ghcr.io" in registry or "${{ env.REGISTRY }}" in registry, \
                    "Should authenticate to GitHub Container Registry"

                # Verify credentials
                username = with_args.get("username", "")
                password = with_args.get("password", "")
                assert username or password, "Should provide authentication credentials"

        assert found_login, "Registry authentication step not found"

    def test_failure_on_critical_cves(self, workflow):
        """Verify workflow fails on critical CVEs."""
        jobs = workflow.get("jobs", {})
        scan_job = jobs.get("scan-images", {})
        steps = scan_job.get("steps", [])

        # Verify critical CVE check with exit 1
        found_failure_logic = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")

            if "critical" in name.lower() and "cve" in name.lower():
                # Check that the step contains logic to fail
                assert "exit 1" in run, \
                    "Critical CVE check should exit with failure code"

                # Check that it parses the JSON report
                assert "jq" in run or "json" in run.lower(), \
                    "Should parse JSON report to check for critical CVEs"

                found_failure_logic = True

        assert found_failure_logic, \
            "Workflow should fail on critical CVEs with proper exit code"

    def test_summary_job(self, workflow):
        """Verify summary job aggregates scan results."""
        jobs = workflow.get("jobs", {})

        # Check summary job exists
        summary_job = jobs.get("summary")
        assert summary_job, "summary job not found"

        # Verify it depends on scan-images
        needs = summary_job.get("needs")
        assert needs == "scan-images" or "scan-images" in needs, \
            "summary job should depend on scan-images"

        # Verify it runs always (even on failure)
        if_condition = summary_job.get("if")
        assert if_condition and "always()" in if_condition, \
            "summary job should run always to report results"

        # Verify it checks scan results
        steps = summary_job.get("steps", [])
        found_result_check = False
        for step in steps:
            run = step.get("run", "")
            if "scan-images.result" in run or "failure" in run:
                found_result_check = True

        assert found_result_check, "summary job should check scan results"


class TestCiProvenanceWorkflow:
    """Validation tests for ci-provenance.yml."""

    @pytest.fixture
    def workflow(self):
        return load_workflow("ci-provenance.yml")

    def test_exists(self, workflow):
        """Validate YAML syntax - workflow file exists and is valid."""
        assert workflow is not None, "ci-provenance.yml does not exist"

    def test_conditional_trigger_for_releases(self, workflow):
        """Verify conditional trigger for releases only (tags matching v*.*.*)."""
        # Handle YAML quirk: 'on' is parsed as boolean True
        triggers = workflow.get("on", workflow.get(True, {}))

        # Check workflow_run trigger
        assert "workflow_run" in triggers, "Missing workflow_run trigger"

        wf_run = triggers["workflow_run"]
        workflows = wf_run.get("workflows", [])
        assert "Build and Push" in workflows, \
            "Should trigger on 'Build and Push' workflow completion"

        types = wf_run.get("types", [])
        assert "completed" in types, "Should trigger on workflow completion"

        # Verify branches filter for main (releases are tagged from main)
        branches = wf_run.get("branches", [])
        assert "main" in branches, "Should trigger on main branch"

        # Verify there's a check-release job that filters for release tags
        jobs = workflow.get("jobs", {})
        check_release = jobs.get("check-release")
        assert check_release, "check-release job not found"

        # Verify conditional execution on successful build
        if_condition = check_release.get("if")
        assert if_condition, "check-release job should have conditional execution"
        assert "workflow_run.conclusion == 'success'" in if_condition, \
            "Should only run on successful build"

        # Verify the job checks for release tags (v*.*.*)
        steps = check_release.get("steps", [])
        found_tag_check = False
        for step in steps:
            run = step.get("run", "")
            if "v[0-9]" in run or "^v" in run or "release" in run.lower():
                found_tag_check = True
                # Verify it checks for semantic version pattern
                assert "v[0-9]" in run or "^v" in run, \
                    "Should check for semantic version tags (v*.*.*)"

        assert found_tag_check, "Release tag check not found in check-release job"

        # Verify outputs for downstream jobs
        outputs = check_release.get("outputs", {})
        assert "is_release" in outputs, "Should output is_release flag"
        assert "tag" in outputs, "Should output tag value"

    def test_cosign_and_slsa_generator_setup(self, workflow):
        """Verify Cosign and SLSA generator setup."""
        jobs = workflow.get("jobs", {})

        # Check generate-provenance job exists
        provenance_job = jobs.get("generate-provenance")
        assert provenance_job, "generate-provenance job not found"

        # Verify it depends on check-release
        needs = provenance_job.get("needs")
        assert needs == "check-release" or "check-release" in needs, \
            "generate-provenance should depend on check-release"

        # Verify conditional execution only for releases
        if_condition = provenance_job.get("if")
        assert if_condition, "generate-provenance job should have conditional execution"
        assert "is_release == 'true'" in if_condition, \
            "Should only run when is_release is true"

        # Verify permissions for signing
        permissions = provenance_job.get("permissions", {})
        assert permissions, "generate-provenance job should have permissions defined"
        assert permissions.get("contents") == "read", "Should have read access to contents"
        assert permissions.get("packages") == "write", "Should have write access to packages"
        assert permissions.get("id-token") == "write", \
            "Should have write access to id-token for keyless signing"

        steps = provenance_job.get("steps", [])

        # Verify Cosign installation
        found_cosign = False
        for step in steps:
            uses = step.get("uses", "")
            if "sigstore/cosign-installer" in uses or "cosign-installer" in uses:
                found_cosign = True
                with_args = step.get("with", {})
                # Verify version is specified
                assert "cosign-release" in with_args or "version" in with_args, \
                    "Cosign version should be specified"

        assert found_cosign, "Cosign installation step not found"

        # Verify SLSA provenance generation
        found_slsa = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if ("slsa" in name.lower() or "provenance" in name.lower()) and "generate" in name.lower():
                found_slsa = True
                # Verify it generates provenance document (check run command or name)
                if run:
                    assert "provenance" in run or "slsa" in run.lower(), \
                        "Should generate SLSA provenance document"

        assert found_slsa, "SLSA provenance generation step not found"

    def test_attestation_attachment_steps(self, workflow):
        """Verify attestation attachment steps."""
        jobs = workflow.get("jobs", {})
        provenance_job = jobs.get("generate-provenance", {})
        steps = provenance_job.get("steps", [])

        # Verify build metadata generation
        found_metadata = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "metadata" in name.lower() and "generate" in name.lower():
                found_metadata = True
                # Verify it includes required fields
                assert "git_sha" in run or "workflow_run_id" in run, \
                    "Should include git SHA and workflow run ID in metadata"

        assert found_metadata, "Build metadata generation step not found"

        # Verify image signing with Cosign
        found_signing = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "sign" in name.lower() and "image" in name.lower():
                found_signing = True
                # Verify keyless signing
                assert "cosign sign" in run, "Should use cosign sign command"
                # Check for keyless signing flag
                env = step.get("env", {})
                assert env.get("COSIGN_EXPERIMENTAL") == 1 or \
                       env.get("COSIGN_EXPERIMENTAL") == "1", \
                    "Should enable COSIGN_EXPERIMENTAL for keyless signing"

        assert found_signing, "Image signing step not found"

        # Verify provenance attestation attachment
        found_provenance_attach = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "provenance" in name.lower() and "attest" in name.lower():
                found_provenance_attach = True
                # Verify cosign attest command
                assert "cosign attest" in run, "Should use cosign attest command"
                assert "--predicate" in run, "Should specify predicate file"
                assert "slsaprovenance" in run or "slsa" in run, \
                    "Should specify SLSA provenance type"

        assert found_provenance_attach, "Provenance attestation attachment step not found"

        # Verify build metadata attestation attachment
        found_metadata_attach = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "metadata" in name.lower() and "attest" in name.lower():
                found_metadata_attach = True
                # Verify cosign attest command
                assert "cosign attest" in run, "Should use cosign attest command"
                assert "--predicate" in run, "Should specify predicate file"
                assert "build-metadata" in run, "Should reference build metadata file"

        assert found_metadata_attach, "Build metadata attestation attachment step not found"

        # Verify signature verification
        found_verify = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "verify" in name.lower() and "signature" in name.lower():
                found_verify = True
                # Verify cosign verify command
                assert "cosign verify" in run, "Should use cosign verify command"
                assert "--certificate-identity" in run or "--certificate-oidc-issuer" in run, \
                    "Should verify certificate identity"

        assert found_verify, "Signature verification step not found"

        # Verify attestation verification
        found_verify_attest = False
        for step in steps:
            run = step.get("run", "")
            name = step.get("name", "")
            if "verify" in name.lower() and "attest" in name.lower():
                found_verify_attest = True
                # Verify cosign verify-attestation command
                assert "cosign verify-attestation" in run, \
                    "Should use cosign verify-attestation command"

        assert found_verify_attest, "Attestation verification step not found"

        # Verify artifact upload for provenance
        found_upload = False
        for step in steps:
            uses = step.get("uses", "")
            if "actions/upload-artifact" in uses:
                found_upload = True
                with_args = step.get("with", {})
                name = with_args.get("name", "")
                assert "provenance" in name, "Artifact should be named for provenance"

                # Verify retention period
                retention = with_args.get("retention-days")
                assert retention == 90 or retention == "90", \
                    "Provenance should be retained for 90 days"

        assert found_upload, "Provenance artifact upload step not found"

    def test_matrix_strategy(self, workflow):
        """Verify matrix strategy for multiple services."""
        jobs = workflow.get("jobs", {})
        provenance_job = jobs.get("generate-provenance", {})

        # Verify matrix strategy
        strategy = provenance_job.get("strategy", {})
        matrix = strategy.get("matrix", {})
        services = matrix.get("service", [])

        assert len(services) >= 2, "Should process at least 2 services (TTS and agent)"
        assert "tts-service" in services or "tts" in str(services), \
            "Should include TTS service"
        assert "agent-service" in services or "agent" in str(services), \
            "Should include agent service"

    def test_summary_job(self, workflow):
        """Verify summary job reports provenance results."""
        jobs = workflow.get("jobs", {})

        # Check summary job exists
        summary_job = jobs.get("summary")
        assert summary_job, "summary job not found"

        # Verify it depends on check-release and generate-provenance
        needs = summary_job.get("needs")
        assert "check-release" in needs, "summary should depend on check-release"
        assert "generate-provenance" in needs, "summary should depend on generate-provenance"

        # Verify it runs always (even on failure)
        if_condition = summary_job.get("if")
        assert if_condition and "always()" in if_condition, \
            "summary job should run always to report results"

        # Verify it only runs for releases
        assert "is_release == 'true'" in if_condition, \
            "summary should only run for releases"

        # Verify it checks provenance results
        steps = summary_job.get("steps", [])
        found_result_check = False
        for step in steps:
            run = step.get("run", "")
            if "generate-provenance.result" in run or "failure" in run:
                found_result_check = True

        assert found_result_check, "summary job should check provenance results"
