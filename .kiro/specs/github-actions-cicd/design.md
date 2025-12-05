# Design Document

## Overview

This design document specifies a comprehensive GitHub Actions CI/CD pipeline for the VibeVoice Realtime Agent platform. The pipeline will provide automated testing, multi-architecture container builds, Kubernetes validation, performance benchmarking, and security scanning. The design follows a modular workflow approach where each workflow is independently triggered and can be composed together for complete CI/CD coverage.

The system will consist of five primary workflow categories:
1. **Core CI Workflows** - Testing, linting, and code quality checks
2. **Container Build Workflows** - Multi-architecture Docker image builds and smoke tests
3. **Kubernetes Validation Workflows** - Helm chart linting and deployment validation
4. **Performance Workflows** - Automated benchmarking and regression detection
5. **Security Workflows** - Dependency scanning, container security, and provenance

## Architecture

### Workflow Organization

The CI/CD pipeline will be organized as separate GitHub Actions workflow files in `.github/workflows/`:

```
.github/
└── workflows/
    ├── ci-test.yml              # Core testing and linting
    ├── ci-build-push.yml        # Multi-arch Docker builds
    ├── ci-smoke-test.yml        # Container smoke tests
    ├── ci-helm-validate.yml     # Kubernetes/Helm validation
    ├── ci-performance.yml       # Performance benchmarks
    ├── ci-security-deps.yml     # Dependency scanning
    ├── ci-security-images.yml   # Container image scanning
    └── ci-provenance.yml        # Build provenance and signing
```

### Trigger Strategy

Each workflow will have specific triggers optimized for its purpose:

- **ci-test.yml**: Push to main/feature branches, pull requests
- **ci-build-push.yml**: Tags (releases), merges to main
- **ci-smoke-test.yml**: Completion of build workflow
- **ci-helm-validate.yml**: Changes to `infra/k8s/**` paths
- **ci-performance.yml**: Scheduled (nightly), manual dispatch
- **ci-security-deps.yml**: Push to main, pull requests, scheduled weekly
- **ci-security-images.yml**: Completion of build workflow
- **ci-provenance.yml**: Completion of build workflow

### Caching Strategy

To optimize CI/CD performance, the following caching strategies will be employed:

1. **Python Dependencies**: Cache pip packages using `actions/cache` with key based on `pyproject.toml` hash
2. **Docker Layers**: Use Docker BuildKit cache mounts and GitHub Actions cache
3. **Model Artifacts**: Optionally cache HuggingFace model downloads (with size limits)
4. **Helm Dependencies**: Cache Helm chart dependencies

## Components and Interfaces

### 1. Core CI Workflow (ci-test.yml)

**Purpose**: Execute Python tests, linting, and type checking on every code change.

**Triggers**:
- Push to `main` and feature branches
- Pull requests to `main`

**Jobs**:
1. **lint**: Run ruff and mypy
   - Set up Python 3.11
   - Cache pip dependencies
   - Install dev dependencies
   - Run `ruff check src/ tests/`
   - Run `mypy src/`

2. **test**: Run pytest with coverage
   - Set up Python 3.11
   - Cache pip dependencies
   - Install project with dev extras
   - Run `pytest tests/ -v --cov=src --cov-report=xml`
   - Upload coverage to artifacts

**Outputs**: Test results, coverage reports, lint results

### 2. Container Build Workflow (ci-build-push.yml)

**Purpose**: Build multi-architecture Docker images and push to container registry.

**Triggers**:
- Tags matching `v*.*.*`
- Push to `main` branch

**Strategy Matrix**:
```yaml
matrix:
  service: [tts, agent]
  platform: [linux/amd64, linux/arm64]
```

**Jobs**:
1. **build-and-push**:
   - Set up Docker Buildx
   - Log in to GitHub Container Registry
   - Extract metadata (tags, labels)
   - Build and push images with:
     - Git SHA tag
     - Semantic version tag (for releases)
     - `latest` tag (for main branch)
   - Generate SBOM (Software Bill of Materials)

**Outputs**: Image digests, tags, SBOM artifacts

### 3. Smoke Test Workflow (ci-smoke-test.yml)

**Purpose**: Verify basic functionality of newly built container images.

**Triggers**:
- Completion of `ci-build-push.yml` workflow

**Jobs**:
1. **smoke-test**:
   - Pull newly built images
   - Start services using `docker compose`
   - Wait for health checks to pass
   - Test health endpoints (`/health`, `/ready`)
   - Execute minimal TTS synthesis request
   - Verify metrics endpoint (`/metrics`)
   - Collect logs on failure

**Outputs**: Test results, service logs (on failure)

### 4. Helm Validation Workflow (ci-helm-validate.yml)

**Purpose**: Validate Kubernetes manifests and Helm charts.

**Triggers**:
- Changes to `infra/k8s/**` paths
- Pull requests affecting Kubernetes configs

**Jobs**:
1. **helm-lint**:
   - Set up Helm
   - Run `helm lint` on all charts
   - Check for deprecated API versions

2. **helm-template**:
   - Render templates with test values
   - Validate rendered YAML syntax
   - Check for required resources

3. **kind-test** (optional):
   - Create ephemeral Kind cluster
   - Install cert-manager and ingress-nginx
   - Deploy rendered manifests
   - Verify pod startup
   - Test basic connectivity

**Outputs**: Lint results, rendered templates, deployment logs

### 5. Performance Benchmark Workflow (ci-performance.yml)

**Purpose**: Track performance metrics and detect regressions.

**Triggers**:
- Scheduled (nightly at 2 AM UTC)
- Manual workflow dispatch

**Jobs**:
1. **benchmark**:
   - Set up Python environment
   - Start TTS service (CPU mode for consistency)
   - Run `scripts/load_test_tts.py`
   - Capture metrics:
     - Time-to-first-audio (TTFA)
     - End-to-end latency (p50, p95, p99)
     - Throughput (requests/second)
     - Error rate
   - Store results as JSON artifact

2. **regression-check**:
   - Download baseline metrics
   - Compare current vs baseline
   - Calculate percentage changes
   - Fail if thresholds exceeded:
     - TTFA > 10% increase
     - p95 latency > 15% increase
     - Error rate > 1%
   - Update baseline if no regressions
   - Post summary comment (for PRs)

**Outputs**: Performance metrics JSON, regression report

### 6. Dependency Security Workflow (ci-security-deps.yml)

**Purpose**: Scan Python dependencies for known vulnerabilities.

**Triggers**:
- Push to `main`
- Pull requests
- Scheduled weekly (Monday 9 AM UTC)

**Jobs**:
1. **scan-dependencies**:
   - Set up Python 3.11
   - Install pip-audit
   - Run `pip-audit --requirement pyproject.toml`
   - Generate vulnerability report
   - Fail on critical/high severity CVEs

2. **dependabot-config**:
   - Verify `.github/dependabot.yml` exists
   - Check configuration is valid

**Outputs**: Vulnerability report, SARIF file for GitHub Security tab

### 7. Container Security Workflow (ci-security-images.yml)

**Purpose**: Scan container images for security vulnerabilities.

**Triggers**:
- Completion of `ci-build-push.yml` workflow

**Jobs**:
1. **scan-images**:
   - Install Trivy scanner
   - Pull newly built images
   - Scan for vulnerabilities:
     - OS packages
     - Python dependencies
     - Known CVEs
   - Generate SARIF report
   - Fail on critical CVEs
   - Upload results to GitHub Security

**Outputs**: Trivy scan reports, SARIF files

### 8. Provenance Workflow (ci-provenance.yml)

**Purpose**: Generate build provenance and sign container images.

**Triggers**:
- Completion of `ci-build-push.yml` workflow (for releases only)

**Jobs**:
1. **generate-provenance**:
   - Use `slsa-github-generator`
   - Generate SLSA Level 3 provenance
   - Include build metadata:
     - Git SHA
     - Workflow run ID
     - Build matrix info
     - Builder identity

2. **sign-images**:
   - Use Cosign with keyless signing
   - Sign images with Sigstore
   - Attach provenance as attestation
   - Verify signatures

**Outputs**: Provenance attestations, signature bundles

## Data Models

### Performance Metrics Schema

```json
{
  "timestamp": "2024-12-04T12:00:00Z",
  "git_sha": "abc123...",
  "workflow_run_id": "123456789",
  "metrics": {
    "ttfa_ms": {
      "mean": 450.2,
      "p50": 445.0,
      "p95": 520.0,
      "p99": 580.0
    },
    "e2e_latency_ms": {
      "mean": 1250.5,
      "p50": 1200.0,
      "p95": 1450.0,
      "p99": 1600.0
    },
    "throughput_rps": 12.5,
    "error_rate": 0.002,
    "total_requests": 1000
  }
}
```

### Build Metadata Schema

```json
{
  "image": "ghcr.io/org/vibevoice-tts:v1.0.0",
  "digest": "sha256:abc123...",
  "platform": "linux/amd64",
  "git_sha": "abc123...",
  "git_ref": "refs/tags/v1.0.0",
  "workflow_run_id": "123456789",
  "build_timestamp": "2024-12-04T12:00:00Z",
  "builder": "github-actions"
}
```

### Vulnerability Report Schema

```json
{
  "scan_timestamp": "2024-12-04T12:00:00Z",
  "scanner": "trivy",
  "scanner_version": "0.48.0",
  "target": "ghcr.io/org/vibevoice-tts:v1.0.0",
  "vulnerabilities": [
    {
      "id": "CVE-2024-1234",
      "severity": "HIGH",
      "package": "openssl",
      "installed_version": "1.1.1",
      "fixed_version": "1.1.1w",
      "description": "..."
    }
  ],
  "summary": {
    "critical": 0,
    "high": 1,
    "medium": 5,
    "low": 12
  }
}
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Most of the acceptance criteria for this feature involve GitHub Actions workflow configuration, which is declarative YAML that GitHub's platform executes. These configurations are not amenable to property-based testing in the traditional sense, as they describe infrastructure-as-code rather than algorithmic behavior.

However, there are specific components where we can extract testable logic into standalone scripts that can be validated with property-based testing:

### Property 1: Regression detection correctly identifies threshold violations

*For any* pair of current and baseline performance metrics, when a metric exceeds its threshold percentage (e.g., latency increases by more than 10%, error rate exceeds 1%), the regression detection logic should correctly identify it as a regression.

**Validates: Requirements 6.1, 6.2, 6.3**

**Implementation Note**: This requires extracting the regression comparison logic into a Python script (e.g., `scripts/check_performance_regression.py`) that takes current metrics and baseline metrics as input and outputs whether a regression occurred.

### Property 2: Build metadata includes all required fields

*For any* build context (git SHA, workflow run ID, build matrix info), the metadata generation function should produce a metadata object that contains all required fields with valid values.

**Validates: Requirements 9.2, 9.3**

**Implementation Note**: This requires extracting metadata generation into a Python script (e.g., `scripts/generate_build_metadata.py`) that can be tested independently.

### Example Test Cases

While not properties, the following example-based tests should be included:

**Example 1: Baseline creation when none exists**
When no baseline metrics file exists, the regression check script should create a new baseline file with the current metrics and exit successfully without reporting a regression.
**Validates: Requirements 6.5**

**Example 2: Smoke test script validates health endpoints**
The smoke test script should successfully verify that `/health` and `/ready` endpoints return 200 status codes and that a minimal TTS synthesis request completes without error.
**Validates: Requirements 3.3, 3.4**

### Configuration Validation

Since most requirements involve workflow configuration, we should implement configuration validation tests that verify:

1. **Workflow YAML syntax**: All workflow files are valid YAML
2. **Required triggers**: Workflows have appropriate trigger configurations
3. **Job dependencies**: Workflows have correct job dependency chains
4. **Required steps**: Critical steps (caching, testing, building) are present
5. **Secret references**: All referenced secrets are documented

These validations should be implemented as unit tests that parse and validate the workflow YAML files.

## Error Handling

### Workflow Failures

Each workflow should implement proper error handling:

1. **Test Failures**: Fail fast on test failures, upload test results and logs as artifacts
2. **Build Failures**: Capture build logs, notify on persistent failures
3. **Timeout Handling**: Set reasonable timeouts for each job (tests: 30min, builds: 60min, benchmarks: 45min)
4. **Retry Logic**: Implement retry for transient failures (network issues, registry timeouts)

### Notification Strategy

- **Critical Failures**: Notify on Slack/email for main branch failures
- **PR Failures**: Comment on PR with failure details and logs
- **Security Issues**: Immediate notification for critical CVEs
- **Performance Regressions**: Create GitHub issue with regression details

### Artifact Retention

- **Test Results**: 30 days
- **Coverage Reports**: 30 days
- **Performance Metrics**: 90 days (for trend analysis)
- **Security Scan Reports**: 90 days
- **Build Logs**: 14 days

## Testing Strategy

### Unit Tests for Helper Scripts

Unit tests will validate the behavior of helper scripts used in workflows:

1. **Regression Detection Script** (`scripts/check_performance_regression.py`):
   - Test threshold calculations
   - Test baseline comparison logic
   - Test output formatting
   - Test error handling for malformed input

2. **Metadata Generation Script** (`scripts/generate_build_metadata.py`):
   - Test metadata field extraction
   - Test JSON schema validation
   - Test handling of missing environment variables

3. **Smoke Test Script** (`scripts/smoke_test.py`):
   - Test health check logic
   - Test TTS request validation
   - Test timeout handling

### Property-Based Tests

Property-based tests will use **Hypothesis** (already in dev dependencies) to validate universal properties:

1. **Regression Detection Properties**:
   - Property 1: Threshold violation detection
   - Test with randomly generated metric pairs
   - Verify correct regression identification
   - Minimum 100 iterations per test

2. **Metadata Generation Properties**:
   - Property 2: Required fields presence
   - Test with randomly generated build contexts
   - Verify all required fields are present and valid
   - Minimum 100 iterations per test

### Configuration Validation Tests

Unit tests will validate workflow YAML files:

1. **YAML Syntax Tests**: Parse all workflow files and verify valid YAML
2. **Schema Validation Tests**: Validate against GitHub Actions schema
3. **Trigger Configuration Tests**: Verify correct triggers for each workflow
4. **Job Dependency Tests**: Verify correct `needs` relationships
5. **Secret Reference Tests**: Verify all secrets are documented

### Integration Tests

Integration tests will validate end-to-end workflow behavior in a test environment:

1. **Workflow Execution Tests**: Trigger workflows via GitHub API and verify completion
2. **Artifact Generation Tests**: Verify expected artifacts are produced
3. **Multi-Arch Build Tests**: Verify both amd64 and arm64 images are built
4. **Smoke Test Integration**: Verify smoke tests catch actual failures

### Manual Testing Checklist

Before merging workflow changes:

1. Test workflow triggers (push, PR, tag, schedule)
2. Verify caching works correctly
3. Verify secrets are properly masked in logs
4. Test failure scenarios (failing tests, build errors)
5. Verify notifications work correctly
6. Test artifact uploads and downloads

## Implementation Phases

### Phase 1: Core CI Workflows (Week 1)
- Implement `ci-test.yml` with testing and linting
- Set up dependency caching
- Implement helper scripts with tests
- Validate on feature branch

### Phase 2: Container Build Workflows (Week 1-2)
- Implement `ci-build-push.yml` with multi-arch builds
- Set up GitHub Container Registry
- Implement `ci-smoke-test.yml`
- Test with actual image builds

### Phase 3: Kubernetes Validation (Week 2)
- Implement `ci-helm-validate.yml`
- Set up Kind-based testing
- Validate with existing Helm charts

### Phase 4: Performance & Security (Week 3)
- Implement `ci-performance.yml` with benchmarking
- Implement regression detection script
- Implement `ci-security-deps.yml` and `ci-security-images.yml`
- Set up Trivy scanning

### Phase 5: Provenance & Polish (Week 3-4)
- Implement `ci-provenance.yml` with Sigstore
- Add notification integrations
- Document all workflows
- Create runbook for common issues

## Dependencies

### External Services
- **GitHub Container Registry**: For storing Docker images
- **GitHub Actions**: Platform for running workflows
- **Sigstore**: For image signing and provenance

### GitHub Actions
- `actions/checkout@v4`: Repository checkout
- `actions/setup-python@v5`: Python environment setup
- `actions/cache@v4`: Dependency caching
- `docker/setup-buildx-action@v3`: Docker Buildx setup
- `docker/login-action@v3`: Registry authentication
- `docker/build-push-action@v5`: Multi-arch builds
- `docker/metadata-action@v5`: Image metadata extraction
- `helm/kind-action@v1`: Kind cluster setup
- `aquasecurity/trivy-action@master`: Container scanning
- `sigstore/cosign-installer@v3`: Image signing

### Required Secrets
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions
- `ANTHROPIC_API_KEY`: For agent service testing (optional)
- `SLACK_WEBHOOK_URL`: For notifications (optional)

## Monitoring and Observability

### Workflow Metrics

Track the following metrics for workflow health:

1. **Execution Time**: Duration of each workflow and job
2. **Success Rate**: Percentage of successful workflow runs
3. **Failure Patterns**: Common failure reasons and frequency
4. **Cache Hit Rate**: Effectiveness of dependency caching
5. **Build Time**: Time to build and push images
6. **Test Coverage**: Code coverage trends over time

### Dashboards

Create GitHub Actions dashboards to visualize:

1. **CI Health**: Success rates, execution times, failure trends
2. **Performance Trends**: TTFA, latency, throughput over time
3. **Security Posture**: Vulnerability counts, scan results
4. **Build Metrics**: Build times, cache efficiency, image sizes

### Alerts

Configure alerts for:

1. **Consecutive Failures**: 3+ consecutive failures on main branch
2. **Performance Degradation**: Regression detection failures
3. **Security Issues**: Critical CVEs detected
4. **Workflow Errors**: Workflow configuration errors
5. **Long Execution Times**: Workflows exceeding expected duration

## Documentation

### Workflow Documentation

Each workflow file should include:

1. **Header Comment**: Purpose, triggers, and outputs
2. **Job Descriptions**: What each job does
3. **Environment Variables**: Required and optional variables
4. **Secrets**: Required secrets and how to configure them

### Developer Guide

Create `docs/ci-cd.md` with:

1. **Workflow Overview**: Description of each workflow
2. **Local Testing**: How to test workflows locally with `act`
3. **Troubleshooting**: Common issues and solutions
4. **Adding New Workflows**: Guidelines for extending CI/CD
5. **Performance Benchmarking**: How to interpret results

### Runbook

Create `docs/ci-cd-runbook.md` with:

1. **Incident Response**: Steps for handling CI/CD failures
2. **Rollback Procedures**: How to rollback failed deployments
3. **Secret Rotation**: How to rotate secrets safely
4. **Cache Management**: When and how to clear caches
5. **Emergency Procedures**: Bypassing CI/CD for hotfixes
