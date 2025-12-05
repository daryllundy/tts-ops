# Design Document: Apple Silicon GPU Support

## Overview

This design adds Metal Performance Shaders (MPS) backend support to the VibeVoice Realtime Agent platform, enabling GPU-accelerated inference on Apple Silicon Macs. The implementation focuses on automatic device detection, graceful fallback mechanisms, and maintaining compatibility with existing NVIDIA CUDA deployments.

The key design principle is transparent device abstraction - the system should automatically select the best available hardware acceleration without requiring manual configuration in most cases, while still allowing explicit device selection for advanced use cases.

## Architecture

### Device Selection Strategy

The system implements a three-tier device selection hierarchy:

1. **Explicit Configuration**: User-specified device via `TTS_DEVICE` environment variable
2. **Automatic Detection**: When set to "auto" or unspecified, detect best available device
3. **Graceful Fallback**: If selected device unavailable, fall back to next best option

Device priority order for automatic selection:
- CUDA (highest performance for NVIDIA GPUs)
- MPS (Apple Silicon GPU acceleration)
- CPU (universal fallback)

### Component Changes

The implementation requires modifications to three core components:

1. **Configuration Layer** (`src/common/config.py`)
   - Extend device validation to accept "mps" and "auto"
   - Add device detection utility functions
   - Update default device from "cuda:0" to "auto"

2. **Model Manager** (`src/tts_service/model_loader.py`)
   - Implement device detection logic
   - Add MPS-specific memory management
   - Handle dtype compatibility for MPS
   - Update synchronization for different backends

3. **Metrics Collection** (`src/common/metrics.py`)
   - Add MPS GPU metrics collection
   - Handle backend-specific metric gathering

## Components and Interfaces

### Device Detection Module

```python
def detect_best_device() -> str:
    """
    Detect the best available device for inference.
    
    Returns:
        Device string in format: "cuda:0", "mps", or "cpu"
    """
    
def is_mps_available() -> bool:
    """Check if MPS backend is available and functional."""
    
def resolve_device(device_str: str) -> str:
    """
    Resolve device string to actual device.
    
    Args:
        device_str: User-specified device ("auto", "cuda:0", "mps", "cpu")
        
    Returns:
        Resolved device string
        
    Raises:
        RuntimeError: If specified device is unavailable
    """
```

### Updated Configuration Schema

```python
class TTSServiceSettings(BaseSettings):
    device: str = Field(
        default="auto",
        description="Inference device (auto, cuda:N, mps, or cpu)"
    )
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device string format."""
        # Accept: auto, cpu, cuda:N, mps
```

### Model Manager Extensions

```python
class TTSModelManager:
    def _resolve_device(self) -> str:
        """Resolve configured device to actual hardware device."""
        
    def _get_optimal_dtype(self, device: str) -> torch.dtype:
        """Get optimal dtype for the target device."""
        
    def _clear_device_cache(self) -> None:
        """Clear GPU cache using backend-specific method."""
        
    def _synchronize_device(self) -> None:
        """Synchronize device operations using backend-specific method."""
```

## Data Models

No new data models required. Existing `ModelInfo` dataclass already captures device information.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Device detection determinism
*For any* system configuration, calling device detection multiple times in the same environment should return the same device
**Validates: Requirements 1.1**

### Property 2: Automatic device selection priority
*For any* mocked environment with available devices, automatic device selection should choose devices in priority order: cuda, then mps, then cpu
**Validates: Requirements 1.5, 6.1, 6.2, 6.3**

### Property 3: Fallback to CPU when GPU unavailable
*For any* environment where neither CUDA nor MPS is available, the system should fall back to CPU inference
**Validates: Requirements 1.2, 6.5**

### Property 4: Device availability validation
*For any* explicitly requested device, the system should validate availability before model loading and raise a clear error if unavailable
**Validates: Requirements 1.3, 1.4**

### Property 5: Configuration validation consistency
*For any* invalid device string, the configuration validation should reject it with a validation error before attempting model loading
**Validates: Requirements 2.4, 6.4**

### Property 6: Environment variable device selection
*For any* valid device string in TTS_DEVICE environment variable, the system should use that device for inference
**Validates: Requirements 2.1**

### Property 7: Default configuration auto-selection
*For any* configuration where TTS_DEVICE is not set, the system should default to automatic device selection
**Validates: Requirements 2.3**

### Property 8: Device selection logging
*For any* device selection, the system should log the selected device backend and availability status
**Validates: Requirements 2.5**

### Property 9: MPS dtype compatibility
*For any* model loading on MPS device, the system should use float32 dtype when float16 is not fully supported
**Validates: Requirements 3.1**

### Property 10: CUDA dtype preservation
*For any* configured dtype on CUDA device, the system should use that dtype without modification
**Validates: Requirements 3.2**

### Property 11: Backend-specific cache clearing
*For any* device backend, calling cache clear should use the appropriate backend-specific method without errors
**Validates: Requirements 3.3**

### Property 12: Warmup cross-platform success
*For any* available device backend, warmup inference should complete successfully without errors
**Validates: Requirements 3.4**

### Property 13: Backend-specific synchronization
*For any* device backend, synchronization operations should use the correct backend-specific method
**Validates: Requirements 3.5**

### Property 14: MPS metrics collection
*For any* metrics collection on MPS device, the system should gather appropriate metrics without errors
**Validates: Requirements 4.5**

## Error Handling

### Device Unavailability

When a requested device is unavailable:
- **Explicit device request**: Raise `RuntimeError` with clear message indicating which device was requested and why it's unavailable
- **Auto mode**: Log warning and fall back to next available device
- **No devices available**: Should never occur (CPU always available)

### MPS-Specific Errors

MPS backend may have limitations:
- Some operations not yet implemented → Fall back to CPU for those operations
- Memory pressure → Clear MPS cache and retry
- Driver issues → Log detailed error and suggest updating macOS

### Configuration Errors

Invalid device strings caught at configuration validation:
- Provide clear error message with valid options
- Include examples of correct device strings
- Suggest "auto" for automatic detection

## Testing Strategy

### Unit Tests

1. **Device Detection Tests**
   - Test `detect_best_device()` returns valid device string
   - Test `is_mps_available()` on different platforms
   - Test `resolve_device()` with various inputs

2. **Configuration Validation Tests**
   - Test valid device strings pass validation
   - Test invalid device strings fail validation
   - Test default value is "auto"

3. **Model Manager Tests**
   - Test device resolution logic
   - Test dtype selection for different devices
   - Test cache clearing for different backends
   - Test synchronization for different backends

4. **Error Handling Tests**
   - Test unavailable device raises appropriate error
   - Test invalid device string rejected at config level
   - Test fallback behavior in auto mode

### Property-Based Tests

Property-based testing will use **Hypothesis** library for Python. Each test will run a minimum of 100 iterations.

1. **Property Test: Device Detection Determinism**
   - **Feature: apple-silicon-gpu-support, Property 1: Device detection determinism**
   - Generate: Multiple calls to device detection in same environment
   - Verify: All calls return identical device string

2. **Property Test: Automatic Device Selection Priority**
   - **Feature: apple-silicon-gpu-support, Property 2: Automatic device selection priority**
   - Generate: Mock environments with different combinations of available devices
   - Verify: Device selection follows priority order (cuda > mps > cpu)

3. **Property Test: Fallback to CPU**
   - **Feature: apple-silicon-gpu-support, Property 3: Fallback to CPU when GPU unavailable**
   - Generate: Mock environments with no GPU available
   - Verify: System selects CPU device

4. **Property Test: Device Availability Validation**
   - **Feature: apple-silicon-gpu-support, Property 4: Device availability validation**
   - Generate: Explicit device requests in environments where device is unavailable
   - Verify: Clear error raised before model loading

5. **Property Test: Configuration Validation**
   - **Feature: apple-silicon-gpu-support, Property 5: Configuration validation consistency**
   - Generate: Random invalid device strings
   - Verify: All invalid strings rejected at configuration validation

6. **Property Test: Environment Variable Device Selection**
   - **Feature: apple-silicon-gpu-support, Property 6: Environment variable device selection**
   - Generate: Valid device strings in TTS_DEVICE environment variable
   - Verify: System uses specified device

7. **Property Test: Default Auto-Selection**
   - **Feature: apple-silicon-gpu-support, Property 7: Default configuration auto-selection**
   - Generate: Configurations with unset TTS_DEVICE
   - Verify: System defaults to automatic device selection

8. **Property Test: Device Selection Logging**
   - **Feature: apple-silicon-gpu-support, Property 8: Device selection logging**
   - Generate: Various device selections
   - Verify: Logs contain device backend and availability status

9. **Property Test: MPS Dtype Compatibility**
   - **Feature: apple-silicon-gpu-support, Property 9: MPS dtype compatibility**
   - Generate: Model loading on MPS with different dtype configurations
   - Verify: float32 used when float16 unsupported

10. **Property Test: CUDA Dtype Preservation**
    - **Feature: apple-silicon-gpu-support, Property 10: CUDA dtype preservation**
    - Generate: Different dtype configurations on CUDA
    - Verify: Configured dtype is used without modification

11. **Property Test: Backend-Specific Cache Clearing**
    - **Feature: apple-silicon-gpu-support, Property 11: Backend-specific cache clearing**
    - Generate: Cache clear calls on different backends
    - Verify: Correct method called without errors

12. **Property Test: Warmup Cross-Platform**
    - **Feature: apple-silicon-gpu-support, Property 12: Warmup cross-platform success**
    - Generate: Warmup on different available backends
    - Verify: Warmup completes successfully

13. **Property Test: Backend-Specific Synchronization**
    - **Feature: apple-silicon-gpu-support, Property 13: Backend-specific synchronization**
    - Generate: Synchronization calls on different backends
    - Verify: Correct synchronization method used

14. **Property Test: MPS Metrics Collection**
    - **Feature: apple-silicon-gpu-support, Property 14: MPS metrics collection**
    - Generate: Metrics collection on MPS device
    - Verify: Metrics gathered without errors

### Integration Tests

1. **End-to-End Device Selection**
   - Start TTS service with different device configurations
   - Verify correct device selected and logged
   - Verify inference works on selected device

2. **Docker Multi-Architecture**
   - Build Docker image on ARM64
   - Run container and verify device detection
   - Test inference performance

3. **Metrics Collection**
   - Verify GPU metrics collected for CUDA
   - Verify MPS metrics collected on Apple Silicon
   - Verify CPU metrics when no GPU available

## Implementation Notes

### PyTorch MPS Considerations

1. **Dtype Support**: MPS has limited float16 support in some PyTorch versions. The implementation should:
   - Check PyTorch version
   - Default to float32 on MPS if float16 causes issues
   - Log dtype selection reasoning

2. **Memory Management**: MPS uses unified memory architecture:
   - `torch.mps.empty_cache()` for cache clearing
   - `torch.mps.synchronize()` for operation synchronization
   - No separate device memory tracking like CUDA

3. **Operation Coverage**: Some operations may not be implemented:
   - Catch `NotImplementedError` from MPS operations
   - Fall back to CPU for unsupported operations
   - Log warnings when fallback occurs

### Docker Considerations

1. **Multi-Architecture Builds**:
   - Use `docker buildx` for multi-platform images
   - Separate base images for x86_64 (CUDA) and ARM64 (MPS)
   - Conditional CUDA installation based on architecture

2. **GPU Access**:
   - CUDA requires `--gpus all` flag
   - MPS works automatically on macOS Docker Desktop
   - Linux ARM64 may not have MPS support

### Performance Expectations

Based on PyTorch MPS benchmarks:
- MPS typically 2-5x faster than CPU on Apple Silicon
- CUDA typically 1.5-3x faster than MPS (depending on GPU)
- Latency targets:
  - CUDA: <300ms time-to-first-audio
  - MPS: <500ms time-to-first-audio
  - CPU: <2000ms time-to-first-audio

## Documentation Updates

### README.md

Add section: "Running on Apple Silicon"
- Prerequisites (macOS 12.3+, PyTorch 2.0+)
- Device configuration options
- Performance expectations
- Troubleshooting common issues

### Configuration Guide

Update device configuration documentation:
- Explain "auto" mode (recommended)
- Show explicit device selection examples
- Document device priority order
- Explain fallback behavior

### Troubleshooting Guide

Add MPS-specific troubleshooting:
- Verifying MPS availability
- Checking PyTorch MPS support
- Monitoring GPU utilization on macOS
- Common MPS errors and solutions

## Migration Path

For existing deployments:

1. **No Breaking Changes**: Default "auto" mode maintains backward compatibility
2. **Explicit CUDA Users**: Existing `TTS_DEVICE=cuda:0` continues working
3. **New Apple Silicon Users**: Works out-of-box with automatic MPS detection
4. **Docker Users**: Rebuild images to get multi-architecture support

## Future Enhancements

Potential future improvements:
1. AMD ROCm support for AMD GPUs
2. Intel GPU support via oneAPI
3. Multi-GPU support for MPS (when available)
4. Automatic batch size tuning per device
5. Device-specific model optimization (quantization, pruning)
