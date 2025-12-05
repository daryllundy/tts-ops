# CI/CD Runbook

## Overview

This runbook provides operational procedures for managing the VibeVoice Realtime Agent CI/CD pipeline. It covers incident response, troubleshooting, maintenance procedures, and emergency protocols.

**Target Audience**: DevOps engineers, SREs, and developers responsible for maintaining the CI/CD infrastructure.

**Last Updated**: December 5, 2024

---

## Table of Contents

1. [Incident Response Procedures](#incident-response-procedures)
2. [Rollback Procedures](#rollback-procedures)
3. [Secret Rotation](#secret-rotation)
4. [Cache Management](#cache-management)
5. [Emergency Bypass Procedures](#emergency-bypass-procedures)
6. [Common Issues and Solutions](#common-issues-and-solutions)
7. [Monitoring and Alerts](#monitoring-and-alerts)
8. [Contact Information](#contact-information)

---

## Incident Response Procedures

### Workflow Failure Response

#### 1. Initial Assessment (5 minutes)

**Objective**: Quickly determine severity and impact.

```bash
# Check workflow status
gh run list --workflow=<workflow-name> --limit 5

# View specific run details
gh run view <run-id>

# Download logs for analysis
gh run download <run-id>
```

**Severity Classification**:
- **P0 (Critical)**: Main branch build failures, security vulnerabilities blocking releases
- **P1 (High)**: PR blocking failures, performance regressions
- **P2 (Medium)**: Intermittent failures, non-blocking warnings
- **P3 (Low)**: Documentation updates, minor improvements

#### 2. Triage Checklist

- [ ] Is this affecting main branch or releases? → Escalate to P0
- [ ] Is this blocking developer PRs? → P1
- [ ] Is this a known transient issue? → Check recent incidents
- [ ] Are multiple workflows affected? → Infrastructure issue
- [ ] Is this a new failure pattern? → Investigate root cause

#### 3. Common Failure Patterns

**Test Failures** (`ci-test.yml`)
```bash
# Download test artifacts
gh run download <run-id> --name test-results

# Review coverage report
gh run download <run-id> --name coverage-report

# Re-run failed jobs
gh run rerun <run-id> --failed
```

**Build Failures** (`ci-build-push.yml`)
```bash
# Check Docker build logs
gh run view <run-id> --log | grep "ERROR"

# Verify registry connectivity
docker login ghcr.io

# Check for rate limiting
gh api rate_limit
```

**Security Scan Failures** (`ci-security-*.yml`)
```bash
# Download vulnerability reports
gh run download <run-id> --name trivy-results
gh run download <run-id> --name pip-audit-report

# Review SARIF files
cat trivy-results.sarif | jq '.runs[0].results'
```

#### 4. Escalation Path

1. **Self-service** (0-15 min): Developer investigates and fixes
2. **Team lead** (15-30 min): Escalate if not resolved
3. **DevOps on-call** (30-60 min): Infrastructure or pipeline issues
4. **Engineering manager** (60+ min): Requires architectural decisions


#### 5. Communication Protocol

**During Incident**:
- Post in `#engineering-alerts` Slack channel
- Update incident tracking document
- Notify affected teams (if blocking PRs)

**Post-Incident**:
- Document root cause in incident log
- Create follow-up tasks for prevention
- Update runbook with lessons learned

---

## Rollback Procedures

### Container Image Rollback

#### Scenario: Newly deployed images are causing issues

**Step 1: Identify Last Known Good Version**

```bash
# List recent image tags
gh api /orgs/<org>/packages/container/vibevoice-tts/versions | jq '.[].metadata.container.tags'

# Check deployment history
kubectl rollout history deployment/tts-service -n voice-agent
```

**Step 2: Rollback Kubernetes Deployment**

```bash
# Rollback to previous revision
kubectl rollout undo deployment/tts-service -n voice-agent
kubectl rollout undo deployment/agent-service -n voice-agent

# Rollback to specific revision
kubectl rollout undo deployment/tts-service -n voice-agent --to-revision=<revision-number>

# Verify rollback
kubectl rollout status deployment/tts-service -n voice-agent
kubectl get pods -n voice-agent -l app=tts-service
```

**Step 3: Update Image Tags (if using Helm)**

```bash
# Rollback using Helm
helm rollback voice-agent -n voice-agent

# Or specify revision
helm rollback voice-agent <revision> -n voice-agent

# Verify
helm history voice-agent -n voice-agent
```

**Step 4: Verify Service Health**

```bash
# Check pod status
kubectl get pods -n voice-agent

# Check logs for errors
kubectl logs -n voice-agent -l app=tts-service --tail=100

# Test endpoints
kubectl port-forward -n voice-agent svc/tts-service 8000:8000
curl http://localhost:8000/health
```

### Workflow Configuration Rollback

#### Scenario: New workflow changes are causing failures

**Step 1: Identify Problematic Commit**

```bash
# View recent workflow changes
git log --oneline .github/workflows/

# Show specific workflow changes
git log -p .github/workflows/<workflow-name>.yml
```

**Step 2: Revert Workflow Changes**

```bash
# Revert specific commit
git revert <commit-sha>

# Or restore from previous version
git checkout <previous-commit-sha> -- .github/workflows/<workflow-name>.yml

# Commit and push
git commit -m "Revert workflow changes causing failures"
git push origin main
```

**Step 3: Disable Problematic Workflow (Temporary)**

Edit workflow file and add at the top:

```yaml
on:
  workflow_dispatch:  # Manual trigger only
# Temporarily disabled due to incident #<incident-number>
```

### Dependency Rollback

#### Scenario: Dependency update breaks builds or tests

**Step 1: Identify Problematic Dependency**

```bash
# Check recent dependency changes
git log -p pyproject.toml

# View Dependabot PRs
gh pr list --author app/dependabot
```

**Step 2: Pin or Downgrade Dependency**

Edit `pyproject.toml`:

```toml
[project.dependencies]
# Pin to last known good version
problematic-package = "==1.2.3"  # Was: ">=1.2.0"
```

**Step 3: Update Lock File and Test**

```bash
# Update lock file
uv lock

# Test locally
pytest tests/ -v

# Commit fix
git add pyproject.toml uv.lock
git commit -m "Pin <package> to version <version> due to compatibility issue"
git push origin main
```

---

## Secret Rotation

### GitHub Actions Secrets

#### Rotation Schedule
- **GITHUB_TOKEN**: Auto-rotated by GitHub (no action needed)
- **ANTHROPIC_API_KEY**: Rotate quarterly or on suspected compromise
- **SLACK_WEBHOOK_URL**: Rotate annually or on team changes
- **Container Registry Tokens**: Rotate semi-annually

#### Rotation Procedure

**Step 1: Generate New Secret**

```bash
# For API keys, generate through provider dashboard
# For webhooks, regenerate in Slack/notification service
# For tokens, use provider CLI or UI
```

**Step 2: Update GitHub Secrets**

```bash
# Using GitHub CLI
gh secret set ANTHROPIC_API_KEY --body "<new-key>"

# Or via UI: Settings → Secrets and variables → Actions → Update secret
```

**Step 3: Verify Workflows Still Function**

```bash
# Trigger test workflow manually
gh workflow run ci-test.yml

# Monitor execution
gh run watch

# Check for authentication errors
gh run view <run-id> --log | grep -i "auth\|unauthorized\|forbidden"
```

**Step 4: Revoke Old Secret**

- Revoke old API key in provider dashboard
- Disable old webhook URL
- Monitor for any failures indicating missed updates

#### Emergency Secret Rotation (Compromise Suspected)

**Immediate Actions** (within 15 minutes):

1. **Revoke compromised secret** in provider dashboard
2. **Generate new secret** immediately
3. **Update GitHub secret** using CLI or UI
4. **Notify security team** via `#security-incidents`
5. **Review recent workflow runs** for suspicious activity

```bash
# Check recent workflow runs for anomalies
gh run list --limit 50 --json conclusion,createdAt,headBranch

# Review audit logs
gh api /orgs/<org>/audit-log | jq '.[] | select(.action | contains("secret"))'
```

### Container Registry Credentials

#### GitHub Container Registry (GHCR)

**Using GITHUB_TOKEN** (Recommended):
- Auto-rotated by GitHub
- Scoped to repository
- No manual rotation needed

**Using Personal Access Token**:

```bash
# Create new PAT with write:packages scope
# Settings → Developer settings → Personal access tokens → Generate new token

# Update secret
gh secret set GHCR_TOKEN --body "<new-token>"

# Test authentication
echo "<new-token>" | docker login ghcr.io -u <username> --password-stdin
```

---

## Cache Management

### When to Clear Caches

**Indicators**:
- Build failures with "corrupted cache" errors
- Dependency resolution issues
- Stale test data causing failures
- Disk space issues in runners
- After major dependency updates

### Clearing GitHub Actions Caches

#### Method 1: Using GitHub CLI

```bash
# List all caches
gh cache list

# Delete specific cache
gh cache delete <cache-id>

# Delete all caches for a branch
gh cache delete --all --branch <branch-name>

# Delete caches matching pattern
gh cache list | grep "pip-" | awk '{print $1}' | xargs -I {} gh cache delete {}
```

#### Method 2: Using GitHub API

```bash
# List caches
gh api /repos/<owner>/<repo>/actions/caches

# Delete specific cache
gh api -X DELETE /repos/<owner>/<repo>/actions/caches/<cache-id>

# Delete all caches (script)
gh api /repos/<owner>/<repo>/actions/caches | \
  jq -r '.actions_caches[].id' | \
  xargs -I {} gh api -X DELETE /repos/<owner>/<repo>/actions/caches/{}
```

#### Method 3: Via Workflow

Add to workflow file:

```yaml
- name: Clear pip cache
  run: |
    rm -rf ~/.cache/pip
    pip cache purge
```

### Cache Key Strategy

**Current Cache Keys**:

```yaml
# Python dependencies
key: pip-${{ runner.os }}-${{ hashFiles('pyproject.toml') }}

# Docker layers
key: docker-${{ runner.os }}-${{ github.sha }}

# Helm dependencies
key: helm-${{ hashFiles('infra/k8s/helm/**/Chart.yaml') }}
```

**Best Practices**:
- Include file hashes for automatic invalidation
- Use restore-keys for fallback
- Set reasonable cache size limits
- Monitor cache hit rates

### Monitoring Cache Effectiveness

```bash
# Check cache hit rates in workflow logs
gh run view <run-id> --log | grep "Cache restored\|Cache not found"

# Calculate hit rate
HITS=$(gh run view <run-id> --log | grep -c "Cache restored")
MISSES=$(gh run view <run-id> --log | grep -c "Cache not found")
echo "Hit rate: $(($HITS * 100 / ($HITS + $MISSES)))%"
```

---

## Emergency Bypass Procedures

### Scenario 1: Critical Hotfix Needed (CI Failing)

**When to Use**: Production incident requires immediate fix, but CI is broken.

**Procedure**:

1. **Create hotfix branch from last known good commit**

```bash
# Find last successful main branch commit
gh run list --branch main --status success --limit 1

# Create hotfix branch
git checkout -b hotfix/<issue-number> <last-good-commit>
```

2. **Apply minimal fix**

```bash
# Make only essential changes
# Avoid refactoring or additional features
git add <changed-files>
git commit -m "hotfix: <description> [skip ci]"
```

3. **Manual verification**

```bash
# Run tests locally
pytest tests/ -v

# Build containers locally
docker build -f infra/docker/Dockerfile.tts -t vibevoice-tts:hotfix .
docker build -f infra/docker/Dockerfile.agent -t vibevoice-agent:hotfix .

# Run smoke tests
docker-compose up -d
python scripts/smoke_test.py
```

4. **Deploy with manual approval**

```bash
# Tag for deployment
git tag -a hotfix-v1.0.1 -m "Emergency hotfix for <issue>"
git push origin hotfix-v1.0.1

# Deploy manually (bypass CI)
kubectl set image deployment/tts-service \
  tts-service=ghcr.io/<org>/vibevoice-tts:hotfix-v1.0.1 \
  -n voice-agent
```

5. **Post-incident cleanup**

```bash
# Fix CI issues
# Re-run full CI on hotfix branch
# Merge hotfix to main once CI is fixed
# Document incident and bypass in post-mortem
```

### Scenario 2: Bypass Specific Workflow Check

**When to Use**: One workflow is failing due to external service issues, but code is verified safe.

**Procedure**:

1. **Verify code quality manually**

```bash
# Run linting locally
ruff check src/ tests/
mypy src/

# Run tests locally
pytest tests/ -v --cov=src

# Review changes carefully
git diff main...HEAD
```

2. **Use workflow_dispatch to skip checks**

Add to PR description:
```
## CI Bypass Justification
- Workflow: ci-security-deps.yml
- Reason: pip-audit service outage (confirmed on status page)
- Manual verification: All dependencies reviewed, no new CVEs
- Approver: @security-team-lead
```

3. **Merge with admin override** (requires permissions)

```bash
# Merge PR bypassing required checks (admin only)
gh pr merge <pr-number> --admin --squash
```

4. **Monitor deployment closely**

```bash
# Watch deployment
kubectl rollout status deployment/tts-service -n voice-agent

# Monitor logs for issues
kubectl logs -n voice-agent -l app=tts-service -f --tail=100

# Check metrics
kubectl port-forward -n voice-agent svc/tts-service 8000:8000
curl http://localhost:8000/metrics
```

### Scenario 3: Disable All CI Temporarily

**When to Use**: GitHub Actions outage or critical infrastructure issue.

**Procedure**:

1. **Disable workflows via repository settings**

```bash
# Via UI: Settings → Actions → Disable Actions

# Or add to each workflow file:
on:
  workflow_dispatch:  # Manual only
# Disabled during GitHub Actions outage - <date>
```

2. **Communicate to team**

```
🚨 CI/CD DISABLED 🚨
Reason: GitHub Actions outage
Duration: Until <time> or further notice
Process: Manual review and deployment only
Contact: @devops-oncall for urgent deployments
```

3. **Manual deployment process**

```bash
# Build locally
docker build -t vibevoice-tts:manual-$(date +%Y%m%d-%H%M) .

# Push manually
docker push ghcr.io/<org>/vibevoice-tts:manual-$(date +%Y%m%d-%H%M)

# Deploy with extra caution
helm upgrade voice-agent ./infra/k8s/helm/voice-agent \
  --set tts.image.tag=manual-$(date +%Y%m%d-%H%M) \
  --namespace voice-agent
```

4. **Re-enable when resolved**

```bash
# Remove workflow_dispatch-only restrictions
# Re-enable Actions in repository settings
# Run full CI on main branch to verify
gh workflow run ci-test.yml --ref main
```

---

## Common Issues and Solutions

### Issue: "Resource not accessible by integration" Error

**Symptoms**: Workflows fail with permission errors when accessing GitHub API.

**Solution**:

```yaml
# Add to workflow file
permissions:
  contents: read
  packages: write
  security-events: write
```

### Issue: Docker Build Timeout

**Symptoms**: Multi-arch builds exceed 60-minute timeout.

**Solution**:

```yaml
# Increase timeout
jobs:
  build:
    timeout-minutes: 120
    
# Or split builds
strategy:
  matrix:
    platform: [linux/amd64, linux/arm64]
```

### Issue: Flaky Tests

**Symptoms**: Tests pass/fail intermittently.

**Solution**:

```bash
# Run tests multiple times locally
for i in {1..10}; do pytest tests/test_flaky.py || break; done

# Add retries to workflow
- name: Run tests with retry
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 30
    max_attempts: 3
    command: pytest tests/ -v
```

### Issue: Cache Corruption

**Symptoms**: "Failed to restore cache" or dependency resolution errors.

**Solution**:

```bash
# Clear all caches
gh cache delete --all

# Update cache key in workflow
key: pip-v2-${{ runner.os }}-${{ hashFiles('pyproject.toml') }}
```

### Issue: Rate Limiting (Docker Hub, GitHub API)

**Symptoms**: "Too many requests" or "Rate limit exceeded" errors.

**Solution**:

```yaml
# Use GHCR instead of Docker Hub
- name: Login to GHCR
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    
# Add delays between API calls
- name: Wait before API call
  run: sleep 5
```

---

## Monitoring and Alerts

### Key Metrics to Monitor

1. **Workflow Success Rate**
   - Target: >95% for main branch
   - Alert: <90% over 24 hours

2. **Build Duration**
   - Target: <30 minutes for tests, <60 minutes for builds
   - Alert: >2x baseline

3. **Cache Hit Rate**
   - Target: >80%
   - Alert: <50% over 7 days

4. **Security Scan Results**
   - Target: 0 critical CVEs
   - Alert: Any critical CVE detected

### Setting Up Alerts

**GitHub Actions Status Badge**:

```markdown
![CI Status](https://github.com/<org>/<repo>/workflows/CI%20Test/badge.svg)
```

**Slack Notifications**:

```yaml
- name: Notify on failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
    payload: |
      {
        "text": "❌ Workflow failed: ${{ github.workflow }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Workflow*: ${{ github.workflow }}\n*Branch*: ${{ github.ref }}\n*Run*: <${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View>"
            }
          }
        ]
      }
```

### Dashboard Queries

**Workflow Success Rate** (GitHub API):

```bash
# Get last 100 runs
gh api /repos/<org>/<repo>/actions/runs?per_page=100 | \
  jq '[.workflow_runs[] | select(.conclusion != null)] | 
      group_by(.conclusion) | 
      map({conclusion: .[0].conclusion, count: length})'
```

**Average Build Time**:

```bash
gh api /repos/<org>/<repo>/actions/runs?per_page=50 | \
  jq '[.workflow_runs[] | select(.conclusion == "success")] | 
      map(.run_duration_ms / 1000 / 60) | 
      add / length'
```

---

## Contact Information

### Escalation Contacts

- **DevOps On-Call**: `@devops-oncall` in Slack
- **Security Team**: `security@company.com`
- **Engineering Manager**: `@eng-manager`

### External Support

- **GitHub Support**: https://support.github.com
- **Docker Hub Support**: https://hub.docker.com/support
- **Sigstore Community**: https://sigstore.dev/community

### Documentation Links

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Buildx Documentation](https://docs.docker.com/buildx/)
- [Helm Documentation](https://helm.sh/docs/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)

---

## Appendix: Useful Commands

### GitHub CLI Quick Reference

```bash
# Workflow management
gh workflow list
gh workflow view <workflow-name>
gh workflow run <workflow-name>
gh workflow disable <workflow-name>
gh workflow enable <workflow-name>

# Run management
gh run list --limit 20
gh run view <run-id>
gh run watch <run-id>
gh run rerun <run-id>
gh run download <run-id>

# Secret management
gh secret list
gh secret set <name>
gh secret remove <name>

# Cache management
gh cache list
gh cache delete <cache-id>
```

### Kubernetes Quick Reference

```bash
# Deployment management
kubectl get deployments -n voice-agent
kubectl describe deployment <name> -n voice-agent
kubectl rollout status deployment/<name> -n voice-agent
kubectl rollout history deployment/<name> -n voice-agent
kubectl rollout undo deployment/<name> -n voice-agent

# Pod management
kubectl get pods -n voice-agent
kubectl logs -n voice-agent <pod-name> -f
kubectl exec -it -n voice-agent <pod-name> -- /bin/bash

# Service testing
kubectl port-forward -n voice-agent svc/<service-name> 8000:8000
```

### Docker Quick Reference

```bash
# Image management
docker images
docker pull ghcr.io/<org>/<image>:<tag>
docker tag <source> <target>
docker push ghcr.io/<org>/<image>:<tag>

# Container management
docker ps
docker logs <container-id> -f
docker exec -it <container-id> /bin/bash

# Cleanup
docker system prune -a
docker volume prune
```

---

## Revision History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2024-12-05 | 1.0 | Initial runbook creation | DevOps Team |

---

**End of Runbook**

For questions or suggestions, please contact the DevOps team or submit a PR to update this document.
