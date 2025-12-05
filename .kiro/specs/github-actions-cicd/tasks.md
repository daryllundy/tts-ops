# Implementation Plan

- [x] 1. Set up project structure for CI/CD workflows and helper scripts
  - Create `.github/workflows/` directory structure
  - Create `scripts/` directory for helper scripts
  - Set up test structure for workflow validation
  - _Requirements: All_

- [x] 2. Implement regression detection helper script
  - Create `scripts/check_performance_regression.py` with baseline comparison logic
  - Implement threshold checking for latency and error rates
  - Add JSON input/output handling for metrics
  - Add baseline file creation when none exists
  - _Requirements: 6.1, 6.2, 6.3, 6.5_

- [x] 2.1 Write property test for regression detection
  - **Property 1: Regression detection correctly identifies threshold violations**
  - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 2.2 Write unit tests for regression detection edge cases
  - Test baseline creation when file doesn't exist
  - Test malformed JSON handling
  - Test missing metric fields
  - _Requirements: 6.5_

- [x] 3. Implement build metadata generation helper script
  - Create `scripts/generate_build_metadata.py` to extract build context
  - Implement metadata field extraction (git SHA, workflow run ID, build matrix)
  - Add JSON schema validation for output
  - Handle missing environment variables gracefully
  - _Requirements: 9.2, 9.3_

- [x] 3.1 Write property test for metadata generation
  - **Property 2: Build metadata includes all required fields**
  - **Validates: Requirements 9.2, 9.3**

- [x] 3.2 Write unit tests for metadata generation
  - Test with missing environment variables
  - Test field validation
  - Test JSON output format
  - _Requirements: 9.2, 9.3_

- [x] 4. Implement smoke test helper script
  - Create `scripts/smoke_test.py` for container validation
  - Implement health endpoint checking logic
  - Add minimal TTS synthesis request test
  - Add timeout and retry handling
  - _Requirements: 3.3, 3.4_

- [x] 4.1 Write unit tests for smoke test script
  - Test health check logic
  - Test TTS request validation
  - Test timeout handling
  - Test failure scenarios
  - _Requirements: 3.3, 3.4, 3.5_

- [x] 5. Implement core CI testing workflow
  - Create `.github/workflows/ci-test.yml`
  - Configure triggers for push and pull requests
  - Add Python setup with version 3.11
  - Implement pip dependency caching using actions/cache
  - Add lint job with ruff and mypy
  - Add test job with pytest and coverage
  - Configure artifact upload for test results and coverage
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 5.1 Write configuration validation tests for ci-test.yml
  - Validate YAML syntax
  - Verify trigger configuration
  - Verify caching configuration
  - Verify required steps are present
  - _Requirements: 1.1, 1.5_

- [x] 6. Implement container build and push workflow
  - Create `.github/workflows/ci-build-push.yml`
  - Configure triggers for tags and main branch pushes
  - Set up Docker Buildx with multi-platform support
  - Configure build matrix for amd64 and arm64
  - Add GitHub Container Registry authentication
  - Implement image metadata extraction (tags, labels)
  - Add build and push steps for TTS service
  - Add build and push steps for agent service
  - Generate and upload SBOM artifacts
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 6.1 Write configuration validation tests for ci-build-push.yml
  - Validate YAML syntax
  - Verify trigger configuration for tags and main
  - Verify build matrix includes both architectures
  - Verify registry authentication steps
  - _Requirements: 2.1, 2.2_

- [x] 7. Implement smoke test workflow
  - Create `.github/workflows/ci-smoke-test.yml`
  - Configure trigger on completion of build workflow
  - Add job to pull newly built images
  - Implement docker compose startup
  - Call smoke test helper script
  - Add failure handling and log collection
  - Configure workflow to fail on smoke test failure
  - _Requirements: 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 7.1 Write configuration validation tests for ci-smoke-test.yml
  - Validate YAML syntax
  - Verify workflow_run trigger configuration
  - Verify job dependencies
  - Verify failure handling
  - _Requirements: 2.5, 3.5_

- [x] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement Helm validation workflow
  - Create `.github/workflows/ci-helm-validate.yml`
  - Configure path-based triggers for infra/k8s/**
  - Add Helm setup action
  - Implement helm lint job for all charts
  - Implement helm template rendering job
  - Add optional Kind cluster creation job
  - Add manifest application and connectivity test
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 9.1 Write configuration validation tests for ci-helm-validate.yml
  - Validate YAML syntax
  - Verify path-based trigger configuration
  - Verify Helm setup steps
  - Verify optional Kind job configuration
  - _Requirements: 4.1_

- [x] 10. Implement performance benchmark workflow
  - Create `.github/workflows/ci-performance.yml`
  - Configure scheduled trigger (nightly at 2 AM UTC)
  - Add manual workflow_dispatch trigger
  - Implement job to start TTS service
  - Add step to run load test script (`scripts/load_test_tts.py`)
  - Implement metrics capture and JSON output
  - Add artifact upload for performance results
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 10.1 Add regression detection to performance workflow
  - Add regression check job that depends on benchmark job
  - Download baseline metrics artifact from previous runs
  - Call `scripts/check_performance_regression.py` with current and baseline metrics
  - Implement failure on threshold violations
  - Add baseline update on successful runs (upload new baseline artifact)
  - Optionally implement PR comment posting for regressions
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 10.2 Write configuration validation tests for ci-performance.yml
  - Validate YAML syntax
  - Verify scheduled trigger configuration
  - Verify job dependencies between benchmark and regression check
  - Verify artifact handling
  - _Requirements: 5.1, 6.1_

- [x] 11. Implement dependency security scanning workflow
  - Create `.github/workflows/ci-security-deps.yml`
  - Configure triggers for push, PR, and weekly schedule (Monday 9 AM UTC)
  - Add Python setup with version 3.11
  - Install and run pip-audit on pyproject.toml
  - Implement vulnerability report generation
  - Configure workflow to fail on critical/high CVEs
  - Upload SARIF report to GitHub Security tab
  - _Requirements: 7.1, 7.2, 7.4, 7.5_

- [x] 11.1 Write configuration validation tests for ci-security-deps.yml
  - Validate YAML syntax
  - Verify trigger configuration including schedule
  - Verify pip-audit execution steps
  - Verify SARIF upload configuration
  - _Requirements: 7.1_

- [x] 12. Implement container security scanning workflow
  - Create `.github/workflows/ci-security-images.yml`
  - Configure trigger on build workflow completion
  - Add Trivy scanner installation
  - Implement image pull for newly built images (TTS and agent)
  - Add vulnerability scanning for both images
  - Generate SARIF reports
  - Configure workflow to fail on critical CVEs
  - Upload scan results to GitHub Security tab
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 12.1 Write configuration validation tests for ci-security-images.yml
  - Validate YAML syntax
  - Verify workflow_run trigger configuration
  - Verify Trivy setup and execution
  - Verify SARIF upload steps
  - _Requirements: 8.1_

- [x] 13. Implement provenance and signing workflow
  - Create `.github/workflows/ci-provenance.yml`
  - Configure trigger on build workflow completion (releases only - tags matching v*.*.*)
  - Add SLSA provenance generation using slsa-github-generator
  - Include build metadata in provenance (git SHA, workflow run ID, build matrix)
  - Add Cosign installation for image signing
  - Implement keyless signing with Sigstore
  - Attach provenance as attestation to images
  - Add signature verification step
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 13.1 Write configuration validation tests for ci-provenance.yml
  - Validate YAML syntax
  - Verify conditional trigger for releases only
  - Verify Cosign and SLSA generator setup
  - Verify attestation attachment steps
  - _Requirements: 9.1_

- [x] 14. Set up Dependabot configuration
  - Create `.github/dependabot.yml`
  - Configure Python dependency updates (pip ecosystem)
  - Configure GitHub Actions dependency updates
  - Set update schedule (weekly)
  - Configure reviewers if needed
  - _Requirements: 7.3_

- [x] 15. Enhance CI/CD documentation
  - Update existing `CICD.md` with detailed workflow information
  - Document each workflow's purpose, triggers, and outputs
  - Add local testing guide using act (optional)
  - Document required secrets and configuration
  - Add troubleshooting section for common issues
  - _Requirements: All_

- [x] 16. Create CI/CD runbook
  - Create `docs/ci-cd-runbook.md`
  - Document incident response procedures for workflow failures
  - Add rollback procedures for failed deployments
  - Document secret rotation process
  - Add cache management procedures (clearing caches)
  - Document emergency bypass procedures
  - _Requirements: All_

- [x] 17. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
