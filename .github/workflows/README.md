# GitHub Actions CI/CD Workflows

This directory contains the GitHub Actions workflows for the VibeVoice Realtime Agent platform.

## Workflow Overview

### Core CI Workflows
- **ci-test.yml** - Testing and linting on every code change
- **ci-build-push.yml** - Multi-architecture Docker image builds
- **ci-smoke-test.yml** - Container smoke tests after builds

### Kubernetes Validation
- **ci-helm-validate.yml** - Helm chart linting and validation

### Performance & Monitoring
- **ci-performance.yml** - Automated performance benchmarks

### Security Workflows
- **ci-security-deps.yml** - Dependency vulnerability scanning
- **ci-security-images.yml** - Container image security scanning
- **ci-provenance.yml** - Build provenance and image signing

## Workflow Triggers

| Workflow | Triggers |
|----------|----------|
| ci-test.yml | Push to main/feature branches, Pull requests |
| ci-build-push.yml | Tags (v*.*.*), Push to main |
| ci-smoke-test.yml | Completion of ci-build-push.yml |
| ci-helm-validate.yml | Changes to infra/k8s/** |
| ci-performance.yml | Scheduled (nightly 2 AM UTC), Manual dispatch |
| ci-security-deps.yml | Push to main, Pull requests, Weekly schedule |
| ci-security-images.yml | Completion of ci-build-push.yml |
| ci-provenance.yml | Completion of ci-build-push.yml (releases only) |

## Required Secrets

- `GITHUB_TOKEN` - Automatically provided by GitHub Actions
- `ANTHROPIC_API_KEY` - For agent service testing (optional)
- `SLACK_WEBHOOK_URL` - For notifications (optional)

## Local Testing

Workflows can be tested locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act

# Test a specific workflow
act -W .github/workflows/ci-test.yml

# Test with specific event
act push -W .github/workflows/ci-test.yml
```

## Documentation

For detailed information about the CI/CD pipeline, see:
- [CI/CD Documentation](../../docs/ci-cd.md)
- [CI/CD Runbook](../../docs/ci-cd-runbook.md)
