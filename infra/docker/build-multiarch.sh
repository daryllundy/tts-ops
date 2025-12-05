#!/bin/bash
# =============================================================================
# Multi-Architecture Docker Build Script
# Builds TTS service images for both x86_64 (CUDA) and ARM64 (Apple Silicon)
# =============================================================================

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-vibevoice-tts}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Multi-Architecture Docker Build${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""
echo -e "Image Name:  ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo -e "Platforms:   ${YELLOW}${PLATFORMS}${NC}"
echo -e "Push:        ${YELLOW}${PUSH}${NC}"
echo ""

# Check if buildx is available
if ! docker buildx version &> /dev/null; then
    echo -e "${RED}Error: docker buildx is not available${NC}"
    echo "Please install Docker Buildx: https://docs.docker.com/buildx/working-with-buildx/"
    exit 1
fi

# Create builder if it doesn't exist
BUILDER_NAME="vibevoice-builder"
if ! docker buildx inspect ${BUILDER_NAME} &> /dev/null; then
    echo -e "${YELLOW}Creating buildx builder: ${BUILDER_NAME}${NC}"
    docker buildx create --name ${BUILDER_NAME} --use
else
    echo -e "${YELLOW}Using existing builder: ${BUILDER_NAME}${NC}"
    docker buildx use ${BUILDER_NAME}
fi

# Bootstrap builder
docker buildx inspect --bootstrap

# Build command
BUILD_CMD="docker buildx build"
BUILD_CMD="${BUILD_CMD} --platform ${PLATFORMS}"
BUILD_CMD="${BUILD_CMD} -t ${IMAGE_NAME}:${IMAGE_TAG}"
BUILD_CMD="${BUILD_CMD} -f infra/docker/Dockerfile.tts"
BUILD_CMD="${BUILD_CMD} --build-arg BUILDKIT_INLINE_CACHE=1"

if [ "${PUSH}" = "true" ]; then
    BUILD_CMD="${BUILD_CMD} --push"
else
    BUILD_CMD="${BUILD_CMD} --load"
fi

BUILD_CMD="${BUILD_CMD} ."

# Execute build
echo -e "${GREEN}Building image...${NC}"
echo -e "${YELLOW}${BUILD_CMD}${NC}"
echo ""

eval ${BUILD_CMD}

echo ""
echo -e "${GREEN}==============================================================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}==============================================================================${NC}"
echo ""

# Show image info
if [ "${PUSH}" = "false" ]; then
    echo -e "${YELLOW}Image loaded locally. To inspect:${NC}"
    echo "  docker images ${IMAGE_NAME}"
    echo ""
    echo -e "${YELLOW}To run on current architecture:${NC}"
    echo "  docker run --rm -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo -e "${YELLOW}To run with GPU (x86_64 only):${NC}"
    echo "  docker run --rm --gpus all -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo -e "${YELLOW}Image pushed to registry. To pull:${NC}"
    echo "  docker pull ${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo ""
