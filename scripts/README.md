# Helper Scripts

This directory contains utility scripts for development, testing, and CI/CD automation.

## Development Scripts

### run_local_tts.sh
Starts the TTS service locally for development.

```bash
./scripts/run_local_tts.sh
```

### run_local_agent.sh
Starts the agent service locally for development.

```bash
./scripts/run_local_agent.sh
```

### load_test_tts.py
Performance testing script for the TTS service.

```bash
python scripts/load_test_tts.py
```

## CI/CD Helper Scripts

### check_performance_regression.py
Compares current performance metrics against baseline and detects regressions.

**Usage:**
```bash
python scripts/check_performance_regression.py \
  --current metrics.json \
  --baseline baseline.json \
  --output regression_report.json
```

**Input Format:**
```json
{
  "timestamp": "2024-12-04T12:00:00Z",
  "metrics": {
    "ttfa_ms": {"mean": 450.2, "p50": 445.0, "p95": 520.0, "p99": 580.0},
    "e2e_latency_ms": {"mean": 1250.5, "p50": 1200.0, "p95": 1450.0, "p99": 1600.0},
    "throughput_rps": 12.5,
    "error_rate": 0.002
  }
}
```

### generate_build_metadata.py
Extracts build context and generates metadata for container images.

**Usage:**
```bash
python scripts/generate_build_metadata.py --output metadata.json
```

**Output Format:**
```json
{
  "git_sha": "abc123...",
  "git_ref": "refs/tags/v1.0.0",
  "workflow_run_id": "123456789",
  "build_timestamp": "2024-12-04T12:00:00Z",
  "builder": "github-actions"
}
```

### smoke_test.py
Validates container functionality after builds.

**Usage:**
```bash
python scripts/smoke_test.py \
  --tts-url http://localhost:8001 \
  --agent-url http://localhost:8000 \
  --timeout 60
```

**Tests:**
- Health endpoint checks (`/health`, `/ready`)
- Minimal TTS synthesis request
- Metrics endpoint validation (`/metrics`)
