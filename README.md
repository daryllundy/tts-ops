# tts-ops: production-ready realtime TTS infrastructure built on VibeVoice

Production-grade realtime AI voice agent platform using Microsoft's VibeVoice-Realtime-0.5B model.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Kubernetes Cluster                           │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Ingress    │───▶│ Agent Service│───▶│     TTS Service (GPU)    │  │
│  │   (NGINX)    │    │  (FastAPI)   │    │  (VibeVoice-Realtime)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│         │                   │                        │                 │
│         ▼                   ▼                        ▼                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Observability Stack                           │  │
│  │  Prometheus ─────▶ Grafana ─────▶ AlertManager                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

## Features

- **Low-latency TTS**: Sub-500ms time-to-first-audio using VibeVoice-Realtime-0.5B
- **Streaming support**: WebSocket and HTTP streaming for real-time audio delivery
- **Multi-platform GPU support**: CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends
- **Automatic device detection**: Intelligently selects best available hardware acceleration
- **Production observability**: Prometheus metrics, structured logging, distributed tracing
- **Kubernetes-native**: Helm charts with HPA, PDB, and resource management
- **Health monitoring**: Liveness/readiness probes with circuit breaker patterns
- **Multi-architecture Docker**: Native support for x86_64 and ARM64 platforms

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (with NVIDIA Container Toolkit for x86_64 GPU support)
- Kubernetes cluster with GPU nodes (for production)
- GPU: NVIDIA CUDA 12.1+ compatible GPU OR Apple Silicon (M1/M2/M3/M4) with MPS support

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start TTS service (requires GPU)
./scripts/run_local_tts.sh

# Start agent service (separate terminal)
./scripts/run_local_agent.sh

# Test the endpoint
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?"}'
```

### Docker Compose (Development)

```bash
# For x86_64 with NVIDIA GPU
docker-compose up -d

# For Apple Silicon (ARM64)
docker-compose -f docker-compose.yml -f docker-compose.arm64.yml up -d
```

See [infra/docker/QUICKSTART.md](infra/docker/QUICKSTART.md) for detailed Docker instructions.

### Running on Apple Silicon

The service provides native GPU acceleration on Apple Silicon (M1/M2/M3/M4) via Metal Performance Shaders (MPS), enabling local development without NVIDIA hardware.

#### Prerequisites

- **macOS**: Version 12.3 (Monterey) or later
- **Python**: Version 3.11 or later
- **PyTorch**: Version 2.0+ with MPS support (installed automatically)
- **Memory**: Minimum 8GB RAM (16GB recommended for optimal performance)

#### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vibevoice-realtime-agent.git
cd vibevoice-realtime-agent

# Install dependencies (includes PyTorch with MPS support)
pip install -e ".[dev]"
```

#### Verifying MPS Availability

Before running the service, verify that MPS is available on your system:

```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'MPS Built: {torch.backends.mps.is_built()}')"
```

Expected output:
```
MPS Available: True
MPS Built: True
```

If MPS is not available, ensure you're running macOS 12.3+ and have the latest PyTorch installed.

#### Device Configuration

The service supports three device configuration modes:

**1. Automatic Detection (Recommended)**

The default configuration automatically selects the best available device:

```bash
# No configuration needed - uses auto detection
./scripts/run_local_tts.sh
```

Device priority order: `cuda` → `mps` → `cpu`

**2. Explicit MPS Selection**

Force MPS usage even if CUDA is available:

```bash
export TTS_DEVICE=mps
./scripts/run_local_tts.sh
```

**3. CPU Fallback**

Use CPU-only inference (useful for debugging):

```bash
export TTS_DEVICE=cpu
./scripts/run_local_tts.sh
```

#### Verifying Device Selection

Check the startup logs to confirm which device is being used:

```
INFO:     Resolved device configuration configured_device='auto' resolved_device='mps' dtype='torch.float32'
INFO:     Loading model microsoft/VibeVoice-Realtime-0.5B on device mps
INFO:     Model loaded successfully in 3.45s
```

You can also check device usage programmatically:

```python
from common.device_utils import detect_best_device, is_mps_available

print(f"Best device: {detect_best_device()}")
print(f"MPS available: {is_mps_available()}")
```

#### Performance Expectations

Typical time-to-first-audio latencies for VibeVoice-Realtime-0.5B:

| Device | Latency | Relative Speed |
|--------|---------|----------------|
| NVIDIA RTX 4090 (CUDA) | 200-300ms | 1.0x (baseline) |
| NVIDIA RTX 3080 (CUDA) | 300-400ms | 0.75x |
| Apple M3 Max (MPS) | 400-600ms | 0.5x |
| Apple M2 Pro (MPS) | 500-700ms | 0.4x |
| Apple M1 (MPS) | 600-800ms | 0.35x |
| CPU (16-core) | 1500-2500ms | 0.15x |

**Notes:**
- MPS provides 2-5x speedup over CPU inference
- High-end NVIDIA GPUs are typically 1.5-3x faster than MPS
- Performance varies based on model size and batch size
- MPS uses unified memory architecture (shared with system RAM)

#### Docker on Apple Silicon

The Docker images support native ARM64 architecture:

```bash
# Build ARM64 image locally
docker build -f infra/docker/Dockerfile.tts \
  --platform linux/arm64 \
  -t vibevoice-tts:arm64 .

# Run with Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.arm64.yml up -d

# Verify device selection in logs
docker-compose logs tts-service | grep "Resolved device"
```

**Docker Desktop for Mac** automatically provides MPS access to containers. No additional configuration is required.

#### Troubleshooting

**Issue: MPS not available**

```
RuntimeError: MPS device requested but not available
```

**Solutions:**
- Verify macOS version: `sw_vers` (must be 12.3+)
- Update PyTorch: `pip install --upgrade torch`
- Check system compatibility: Some older M1 Macs may need macOS updates
- Fallback to CPU: `export TTS_DEVICE=cpu`

**Issue: MPS out of memory**

```
RuntimeError: MPS backend out of memory
```

**Solutions:**
- Reduce batch size: `export TTS_MAX_BATCH_SIZE=1`
- Close other applications to free memory
- MPS shares memory with system - ensure sufficient free RAM
- Monitor memory: Activity Monitor → Memory tab

**Issue: Unsupported MPS operations**

```
NotImplementedError: The operator 'aten::some_op' is not currently implemented for the MPS device
```

**Solutions:**
- Update PyTorch to latest version (operations are continuously added)
- Fallback to CPU for specific operations (handled automatically)
- Check PyTorch MPS status: https://github.com/pytorch/pytorch/issues?q=is%3Aissue+mps
- Use CPU mode if persistent: `export TTS_DEVICE=cpu`

**Issue: Slow performance on MPS**

**Solutions:**
- Verify MPS is actually being used (check logs)
- Ensure no other GPU-intensive apps are running
- Check Activity Monitor → GPU History for utilization
- Consider using float32 instead of float16: `export TTS_DTYPE=float32`
- Update to latest macOS for MPS performance improvements

**Issue: Model loading fails**

```
RuntimeError: Failed to load model on mps device
```

**Solutions:**
- Check available disk space (models are ~2GB)
- Verify HuggingFace cache: `~/.cache/huggingface/`
- Clear cache and retry: `rm -rf ~/.cache/huggingface/hub/models--microsoft--*`
- Try CPU first to isolate device issues: `export TTS_DEVICE=cpu`

#### Monitoring MPS Usage

**Activity Monitor:**
1. Open Activity Monitor (Applications → Utilities)
2. Select "Window" → "GPU History"
3. Monitor GPU utilization while running inference

**Command Line:**

```bash
# Monitor GPU usage
sudo powermetrics --samplers gpu_power -i 1000

# Check memory pressure
memory_pressure

# View process GPU usage
sudo iotop -C
```

#### Platform-Specific Notes

**M1/M2/M3 Differences:**
- M3 series has enhanced GPU cores and memory bandwidth
- M2 Pro/Max have more GPU cores than base M2
- All Apple Silicon Macs support MPS, but performance scales with GPU core count

**macOS Version Compatibility:**
- macOS 12.3+: Basic MPS support
- macOS 13.0+: Improved MPS performance and operation coverage
- macOS 14.0+: Additional optimizations and stability improvements

**PyTorch Version:**
- PyTorch 2.0+: Required for MPS support
- PyTorch 2.1+: Recommended for best MPS compatibility
- PyTorch 2.2+: Latest MPS optimizations and bug fixes

#### Development Tips

1. **Use automatic device detection** during development to ensure code works across platforms
2. **Test on CPU occasionally** to verify fallback behavior works correctly
3. **Monitor memory usage** - MPS shares system memory, unlike discrete GPUs
4. **Keep PyTorch updated** - MPS support is actively improving with each release
5. **Profile performance** - Use `torch.profiler` to identify bottlenecks

#### Example: Complete Setup

```bash
# 1. Verify prerequisites
sw_vers  # Check macOS version (12.3+)
python --version  # Check Python version (3.11+)

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Verify MPS
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# 4. Run TTS service (auto-detects MPS)
./scripts/run_local_tts.sh

# 5. In another terminal, test inference
curl -X POST http://localhost:8001/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from Apple Silicon!", "voice_id": "default"}' \
  --output test.wav

# 6. Verify audio file
afplay test.wav
```

### Kubernetes Deployment

```bash
# Add Helm repo and install
helm upgrade --install voice-agent ./infra/k8s/helm/voice-agent \
  --namespace voice-agent \
  --create-namespace \
  -f infra/k8s/helm/voice-agent/values-production.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_MODEL_NAME` | HuggingFace model identifier | `microsoft/VibeVoice-Realtime-0.5B` |
| `TTS_DEVICE` | Inference device (`auto`, `cuda:N`, `mps`, `cpu`) | `auto` |
| `TTS_MAX_BATCH_SIZE` | Maximum batch size for inference | `4` |
| `LLM_PROVIDER` | LLM backend (`openai`, `anthropic`, `local`) | `anthropic` |
| `LLM_MODEL` | Model name for LLM | `claude-sonnet-4-20250514` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `METRICS_PORT` | Prometheus metrics port | `9090` |

### Helm Values

See `infra/k8s/helm/voice-agent/values.yaml` for full configuration options.

## API Reference

### POST /chat

Generate speech from text input.

**Request:**
```json
{
  "text": "Hello, world!",
  "voice_id": "default",
  "stream": true
}
```

**Response (streaming):** Audio chunks as `audio/wav` stream

### POST /synthesize

Direct TTS synthesis without LLM processing.

**Request:**
```json
{
  "text": "Text to synthesize",
  "voice_id": "default",
  "output_format": "wav"
}
```

### GET /health

Health check endpoint returning service status.

### GET /metrics

Prometheus-formatted metrics endpoint.

## Observability

### Key Metrics

- `tts_request_duration_seconds` - TTS request latency histogram
- `tts_time_to_first_audio_seconds` - Time to first audio byte
- `tts_requests_total` - Total request count by status
- `tts_gpu_utilization_percent` - GPU utilization
- `tts_model_load_time_seconds` - Model initialization time

### Grafana Dashboards

Import dashboards from `infra/monitoring/dashboards/`:
- `voice-agent-overview.json` - High-level service health
- `tts-performance.json` - Detailed TTS metrics

## Testing

```bash
# Unit tests
pytest tests/ -v

# Integration tests (requires running services)
pytest tests/integration/ -v --integration

# Load testing
python scripts/load_test_tts.py --rps 10 --duration 60
```

## Project Structure

```
vibevoice-realtime-agent/
├── src/
│   ├── agent_app/          # Voice agent service
│   ├── tts_service/        # TTS microservice
│   └── common/             # Shared utilities
├── infra/
│   ├── docker/             # Dockerfiles
│   ├── k8s/helm/           # Helm charts
│   ├── terraform/          # Infrastructure as code
│   └── monitoring/         # Prometheus/Grafana configs
├── scripts/                # Development scripts
├── tests/                  # Test suites
└── docs/                   # Additional documentation
```

## License

MIT License - see [LICENSE](LICENSE) for details.
