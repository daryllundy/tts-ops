# Requirements Document

## Introduction

This document specifies requirements for adding Apple Silicon (M1/M2/M3/M4) GPU acceleration support to the VibeVoice Realtime Agent platform. Currently, the system only supports NVIDIA CUDA GPUs for TTS inference. This feature will enable the platform to leverage Metal Performance Shaders (MPS) backend in PyTorch, allowing developers with Apple Silicon Macs to run GPU-accelerated inference locally without requiring NVIDIA hardware.

## Glossary

- **Apple Silicon**: Apple's ARM-based system-on-chip processors (M1, M2, M3, M4 series) with integrated GPU
- **MPS**: Metal Performance Shaders - Apple's GPU acceleration framework for machine learning on macOS
- **CUDA**: NVIDIA's parallel computing platform and API for GPU acceleration
- **TTS Service**: The text-to-speech microservice that performs model inference
- **Device Backend**: The hardware acceleration framework used by PyTorch (cuda, mps, or cpu)
- **Model Manager**: The TTSModelManager class responsible for loading and managing the TTS model
- **Inference Context**: The execution environment for running model predictions

## Requirements

### Requirement 1

**User Story:** As a developer with an Apple Silicon Mac, I want to run GPU-accelerated TTS inference locally, so that I can develop and test the voice agent without requiring NVIDIA hardware.

#### Acceptance Criteria

1. WHEN the TTS Service starts on an Apple Silicon Mac THEN the system SHALL detect MPS availability and use it as the default device
2. WHEN MPS is not available THEN the system SHALL fall back to CPU inference
3. WHEN the device configuration is set to "mps" THEN the system SHALL validate that MPS is available before loading the model
4. WHEN the device is set to "cuda" on a non-CUDA system THEN the system SHALL provide a clear error message indicating CUDA is unavailable
5. WHEN the device is set to "auto" THEN the system SHALL automatically select the best available device in order: cuda, mps, cpu

### Requirement 2

**User Story:** As a system administrator, I want to configure which GPU backend to use, so that I can optimize performance for different hardware environments.

#### Acceptance Criteria

1. WHEN the TTS_DEVICE environment variable is set to "mps" THEN the system SHALL use Metal Performance Shaders for inference
2. WHEN the TTS_DEVICE environment variable is set to "auto" THEN the system SHALL automatically detect and use the best available device
3. WHEN the TTS_DEVICE environment variable is not set THEN the system SHALL default to automatic device selection
4. WHEN an invalid device string is provided THEN the system SHALL reject the configuration with a validation error
5. WHEN the configuration is loaded THEN the system SHALL log the selected device backend and availability status

### Requirement 3

**User Story:** As a developer, I want the model loading process to handle device-specific optimizations, so that inference performance is maximized on each platform.

#### Acceptance Criteria

1. WHEN loading a model on MPS THEN the system SHALL use float32 dtype if float16 is not fully supported
2. WHEN loading a model on CUDA THEN the system SHALL continue using the configured dtype (float16, bfloat16, or float32)
3. WHEN clearing GPU memory after inference THEN the system SHALL use the appropriate cache clearing method for the device backend
4. WHEN running warmup inference THEN the system SHALL complete successfully on all supported device backends
5. WHEN synchronizing device operations THEN the system SHALL use backend-specific synchronization methods

### Requirement 4

**User Story:** As a DevOps engineer, I want to run the TTS service in Docker on Apple Silicon, so that I can maintain consistent deployment workflows across different hardware.

#### Acceptance Criteria

1. WHEN building the Docker image on Apple Silicon THEN the system SHALL create an ARM64-compatible image
2. WHEN the Docker container starts on Apple Silicon THEN the system SHALL detect and use MPS if available
3. WHEN running in Docker without GPU access THEN the system SHALL fall back to CPU inference gracefully
4. WHEN the Dockerfile is built THEN the system SHALL support multi-architecture builds for both x86_64 and ARM64
5. WHEN GPU metrics are collected THEN the system SHALL handle MPS-specific metrics appropriately

### Requirement 5

**User Story:** As a developer, I want comprehensive documentation on running the platform on Apple Silicon, so that I can quickly set up my development environment.

#### Acceptance Criteria

1. WHEN a developer reads the README THEN the system documentation SHALL include Apple Silicon setup instructions
2. WHEN a developer encounters device selection issues THEN the documentation SHALL provide troubleshooting guidance
3. WHEN a developer wants to verify GPU usage THEN the documentation SHALL explain how to check MPS utilization
4. WHEN comparing performance across devices THEN the documentation SHALL provide expected latency benchmarks for MPS vs CUDA vs CPU
5. WHEN installing dependencies on Apple Silicon THEN the documentation SHALL specify any platform-specific requirements

### Requirement 6

**User Story:** As a quality assurance engineer, I want automated tests to verify device backend selection, so that I can ensure the system works correctly across different hardware configurations.

#### Acceptance Criteria

1. WHEN tests run on a system with CUDA available THEN the automatic device selection SHALL choose CUDA
2. WHEN tests run on Apple Silicon with MPS available THEN the automatic device selection SHALL choose MPS
3. WHEN tests run on a system with only CPU available THEN the automatic device selection SHALL choose CPU
4. WHEN device validation is tested THEN the system SHALL correctly reject invalid device strings
5. WHEN device fallback is tested THEN the system SHALL gracefully handle unavailable devices
