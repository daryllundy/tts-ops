# Docker Multi-Architecture Quick Start

## For Apple Silicon Users (M1/M2/M3/M4)

### Option 1: Using Docker Compose (Recommended)

```bash
# Use the ARM64 override configuration
docker-compose -f docker-compose.yml -f docker-compose.arm64.yml up -d

# View logs
docker-compose logs -f tts

# Stop services
docker-compose down
```

### Option 2: Building Directly

```bash
# Build for ARM64
docker build \
  --build-arg TARGETARCH=arm64 \
  -f infra/docker/Dockerfile.tts \
  -t vibevoice-tts:latest \
  .

# Run the container
docker run --rm -p 8000:8000 vibevoice-tts:latest
```

The service will automatically detect and use MPS (Metal Performance Shaders) for GPU acceleration.

## For x86_64 Users (NVIDIA GPU)

### Option 1: Using Docker Compose (Recommended)

```bash
# Standard docker-compose (defaults to x86_64 with CUDA)
docker-compose up -d

# View logs
docker-compose logs -f tts

# Stop services
docker-compose down
```

### Option 2: Building Directly

```bash
# Build for x86_64 with CUDA
docker build \
  --build-arg TARGETARCH=amd64 \
  -f infra/docker/Dockerfile.tts \
  -t vibevoice-tts:latest \
  .

# Run with GPU access
docker run --rm --gpus all -p 8000:8000 vibevoice-tts:latest
```

## Building Multi-Architecture Images

For maintainers who want to build images for both architectures:

```bash
# Build for both platforms
./infra/docker/build-multiarch.sh

# Build and push to registry
IMAGE_NAME=myregistry/vibevoice-tts \
PUSH=true \
./infra/docker/build-multiarch.sh
```

## Verifying GPU Usage

### Check Device Selection

```bash
# View container logs to see which device was selected
docker logs vibevoice-tts 2>&1 | grep -i "device"
```

You should see log messages indicating:
- **CUDA**: "Using device: cuda:0"
- **MPS**: "Using device: mps"
- **CPU**: "Using device: cpu"

### Test Inference

```bash
# Health check
curl http://localhost:8000/health

# Test synthesis
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "default"}' \
  --output test.wav
```

## Troubleshooting

### Apple Silicon: MPS Not Detected

If the service falls back to CPU on Apple Silicon:

1. Check macOS version (requires 12.3+)
2. Verify PyTorch MPS support:
   ```bash
   docker exec vibevoice-tts python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
   ```
3. Update Docker Desktop to latest version

### x86_64: CUDA Not Available

If CUDA is not detected:

1. Install NVIDIA Container Toolkit:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. Test GPU access:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

### Build Failures

If the build fails:

1. Ensure Docker Buildx is available:
   ```bash
   docker buildx version
   ```

2. Create and use a builder:
   ```bash
   docker buildx create --name multiarch-builder --use
   docker buildx inspect --bootstrap
   ```

3. Clear build cache:
   ```bash
   docker buildx prune -af
   ```

## Performance Expectations

Typical time-to-first-audio latencies:

| Platform | Device | Expected Latency |
|----------|--------|------------------|
| x86_64 | NVIDIA GPU (CUDA) | <300ms |
| ARM64 | Apple Silicon (MPS) | <500ms |
| Any | CPU | <2000ms |

## Environment Variables

Key configuration options:

| Variable | Default | Options |
|----------|---------|---------|
| `TTS_DEVICE` | `auto` | `auto`, `cuda:0`, `mps`, `cpu` |
| `TTS_MODEL_NAME` | `microsoft/VibeVoice-Realtime-0.5B` | Any compatible model |
| `TTS_DTYPE` | `float16` | `float16`, `bfloat16`, `float32` |
| `TTS_WARMUP_ON_START` | `true` | `true`, `false` |

## Next Steps

- See [README.md](./README.md) for detailed documentation
- Check [../k8s/helm/](../k8s/helm/) for Kubernetes deployment
- Review [../../README.md](../../README.md) for application usage
