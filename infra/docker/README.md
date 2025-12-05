# Docker Multi-Architecture Support

This directory contains Docker configurations for building the TTS service with multi-architecture support.

## Supported Architectures

- **linux/amd64** (x86_64): NVIDIA CUDA GPU acceleration
- **linux/arm64** (Apple Silicon): Metal Performance Shaders (MPS) GPU acceleration

## Quick Start

### Building for Current Architecture

```bash
# Build for your current platform
docker build -f infra/docker/Dockerfile.tts -t vibevoice-tts:latest .
```

### Building Multi-Architecture Images

```bash
# Build for both x86_64 and ARM64
./infra/docker/build-multiarch.sh

# Build and push to registry
IMAGE_NAME=myregistry/vibevoice-tts PUSH=true ./infra/docker/build-multiarch.sh

# Build for specific platforms only
PLATFORMS=linux/arm64 ./infra/docker/build-multiarch.sh
```

## Running the Container

### On x86_64 with NVIDIA GPU

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e TTS_DEVICE=auto \
  vibevoice-tts:latest
```

The service will automatically detect and use CUDA.

### On Apple Silicon (ARM64)

```bash
docker run --rm \
  -p 8000:8000 \
  -e TTS_DEVICE=auto \
  vibevoice-tts:latest
```

The service will automatically detect and use MPS if available, or fall back to CPU.

### CPU-Only Mode

```bash
docker run --rm \
  -p 8000:8000 \
  -e TTS_DEVICE=cpu \
  vibevoice-tts:latest
```

## Architecture-Specific Details

### x86_64 (CUDA)

- Base image: `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
- PyTorch installed with CUDA 12.1 support
- Requires NVIDIA Container Toolkit for GPU access
- Default device: `auto` (will select CUDA if available)

### ARM64 (Apple Silicon)

- Base image: `python:3.11-slim`
- PyTorch installed with MPS support
- Works on macOS with Docker Desktop
- Default device: `auto` (will select MPS if available)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_DEVICE` | `auto` | Device selection: `auto`, `cuda:0`, `mps`, or `cpu` |
| `TTS_HOST` | `0.0.0.0` | Server bind address |
| `TTS_PORT` | `8000` | Server port |
| `TTS_MODEL_ID` | `microsoft/VibeVoice-Realtime-0.5B` | Model identifier |
| `TTS_DTYPE` | `float16` | Model dtype (float16, bfloat16, float32) |

## Docker Compose

Update your `docker-compose.yml` to use the multi-arch image:

```yaml
services:
  tts:
    image: vibevoice-tts:latest
    platform: linux/amd64  # or linux/arm64
    ports:
      - "8000:8000"
    environment:
      - TTS_DEVICE=auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia  # Only for x86_64 with NVIDIA GPU
              count: 1
              capabilities: [gpu]
```

## Troubleshooting

### CUDA Not Available on x86_64

1. Ensure NVIDIA Container Toolkit is installed
2. Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
3. Check Docker daemon configuration includes `nvidia` runtime

### MPS Not Available on Apple Silicon

1. Verify macOS version is 12.3 or later
2. Check PyTorch MPS support: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Update to latest Docker Desktop for Mac

### Build Failures

1. Ensure Docker Buildx is installed: `docker buildx version`
2. Create builder: `docker buildx create --use`
3. Bootstrap builder: `docker buildx inspect --bootstrap`

## Performance Expectations

Approximate time-to-first-audio latencies:

- **CUDA (x86_64)**: <300ms
- **MPS (ARM64)**: <500ms
- **CPU**: <2000ms

Actual performance depends on specific hardware and model configuration.

## Development

### Testing Multi-Arch Builds Locally

```bash
# Build for ARM64 on x86_64 (or vice versa)
docker buildx build \
  --platform linux/arm64 \
  -f infra/docker/Dockerfile.tts \
  -t vibevoice-tts:arm64 \
  --load \
  .

# Note: --load only works for single platform builds
# For multi-platform, use --push to registry
```

### Inspecting Built Images

```bash
# View image manifest
docker buildx imagetools inspect vibevoice-tts:latest

# Check architecture
docker inspect vibevoice-tts:latest | jq '.[0].Architecture'
```

## References

- [Docker Buildx Documentation](https://docs.docker.com/buildx/working-with-buildx/)
- [Multi-platform Images](https://docs.docker.com/build/building/multi-platform/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
