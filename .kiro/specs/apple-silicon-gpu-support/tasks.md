# Implementation Plan

- [x] 1. Add device detection and resolution utilities
  - Create device detection module in `src/common/device_utils.py`
  - Implement `detect_best_device()` function that checks availability in priority order (cuda, mps, cpu)
  - Implement `is_mps_available()` function to check MPS backend availability
  - Implement `resolve_device(device_str)` function to handle "auto" and validate explicit devices
  - Add comprehensive error messages for unavailable devices
  - _Requirements: 1.1, 1.2, 1.5, 2.2_

- [x] 2. Update configuration to support MPS and auto device selection
  - Modify `TTSServiceSettings` in `src/common/config.py` to accept "mps" and "auto" device values
  - Change default device from "cuda:0" to "auto"
  - Update `validate_device()` validator to accept new device formats
  - Add validation for device string format (auto, cpu, cuda:N, mps)
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Enhance model manager with device resolution and backend-specific operations
  - Add `_resolve_device()` method to `TTSModelManager` that uses device detection utilities
  - Implement `_get_optimal_dtype(device)` method to handle MPS float16 limitations
  - Update `load()` method to call `_resolve_device()` and `_get_optimal_dtype()`
  - Add logging for device selection with backend and availability status
  - _Requirements: 1.1, 1.2, 1.5, 2.5, 3.1, 3.2_

- [x] 4. Implement backend-specific memory management
  - Add `_clear_device_cache()` method that handles CUDA, MPS, and CPU backends
  - Add `_synchronize_device()` method for backend-specific synchronization
  - Update `unload()` method to use `_clear_device_cache()`
  - Update `inference_context()` to use `_synchronize_device()`
  - Handle MPS cache clearing with `torch.mps.empty_cache()`
  - Handle MPS synchronization with `torch.mps.synchronize()`
  - _Requirements: 3.3, 3.5_

- [x] 5. Update warmup to work across all device backends
  - Ensure `_warmup()` method works on CUDA, MPS, and CPU
  - Add error handling for backend-specific warmup issues
  - Log warmup completion with device information
  - _Requirements: 3.4_

- [x] 6. Update metrics collection for MPS
  - Modify `record_gpu_metrics()` in `src/common/metrics.py` to handle MPS devices
  - Add MPS-specific metric collection (if available via PyTorch)
  - Handle cases where GPU metrics are not available for MPS
  - Ensure metrics collection doesn't fail on MPS devices
  - _Requirements: 4.5_

- [x] 7. Write unit tests for all implemented functionality
  - Unit tests for device detection utilities (7 tests)
  - Unit tests for configuration validation (4 tests)
  - Unit tests for model manager device handling (6 tests)
  - Unit tests for metrics collection (3 tests)
  - _Requirements: All_

- [ ] 8. Add Hypothesis property-based tests
  - Install hypothesis library as dev dependency
  - Configure hypothesis to run minimum 100 iterations per property test
  - _Requirements: All_

- [ ] 8.1 Write property-based test for device detection determinism
  - **Property 1: Device detection determinism**
  - **Validates: Requirements 1.1**
  - Use Hypothesis to generate multiple calls in same environment
  - Verify all calls return identical device string

- [ ] 8.2 Write property-based test for automatic device selection priority
  - **Property 2: Automatic device selection priority**
  - **Validates: Requirements 1.5, 6.1, 6.2, 6.3**
  - Use Hypothesis to generate mock environments with different device combinations
  - Verify device selection follows priority order (cuda > mps > cpu)

- [ ] 8.3 Write property-based test for CPU fallback
  - **Property 3: Fallback to CPU when GPU unavailable**
  - **Validates: Requirements 1.2, 6.5**
  - Use Hypothesis to generate environments with no GPU
  - Verify system selects CPU device

- [ ] 8.4 Write property-based test for device availability validation
  - **Property 4: Device availability validation**
  - **Validates: Requirements 1.3, 1.4**
  - Use Hypothesis to generate explicit device requests in unavailable environments
  - Verify clear error raised before model loading

- [ ] 8.5 Write property-based test for configuration validation
  - **Property 5: Configuration validation consistency**
  - **Validates: Requirements 2.4, 6.4**
  - Use Hypothesis to generate random invalid device strings
  - Verify all invalid strings rejected at configuration validation

- [ ] 8.6 Write property-based test for environment variable device selection
  - **Property 6: Environment variable device selection**
  - **Validates: Requirements 2.1**
  - Use Hypothesis to generate valid device strings in TTS_DEVICE
  - Verify system uses specified device

- [ ] 8.7 Write property-based test for default auto-selection
  - **Property 7: Default configuration auto-selection**
  - **Validates: Requirements 2.3**
  - Use Hypothesis to generate configurations with unset TTS_DEVICE
  - Verify system defaults to automatic device selection

- [ ] 8.8 Write property-based test for device selection logging
  - **Property 8: Device selection logging**
  - **Validates: Requirements 2.5**
  - Use Hypothesis to generate various device selections
  - Verify logs contain device backend and availability status

- [ ] 8.9 Write property-based test for MPS dtype compatibility
  - **Property 9: MPS dtype compatibility**
  - **Validates: Requirements 3.1**
  - Use Hypothesis to generate model loading on MPS with different dtypes
  - Verify float32 used when float16 unsupported

- [ ] 8.10 Write property-based test for CUDA dtype preservation
  - **Property 10: CUDA dtype preservation**
  - **Validates: Requirements 3.2**
  - Use Hypothesis to generate different dtype configurations on CUDA
  - Verify configured dtype is used without modification

- [ ] 8.11 Write property-based test for backend-specific cache clearing
  - **Property 11: Backend-specific cache clearing**
  - **Validates: Requirements 3.3**
  - Use Hypothesis to generate cache clear calls on different backends
  - Verify correct method called without errors

- [ ] 8.12 Write property-based test for warmup cross-platform success
  - **Property 12: Warmup cross-platform success**
  - **Validates: Requirements 3.4**
  - Use Hypothesis to generate warmup on different available backends
  - Verify warmup completes successfully

- [ ] 8.13 Write property-based test for backend-specific synchronization
  - **Property 13: Backend-specific synchronization**
  - **Validates: Requirements 3.5**
  - Use Hypothesis to generate synchronization calls on different backends
  - Verify correct synchronization method used

- [ ] 8.14 Write property-based test for MPS metrics collection
  - **Property 14: MPS metrics collection**
  - **Validates: Requirements 4.5**
  - Use Hypothesis to generate metrics collection on MPS device
  - Verify metrics gathered without errors

- [ ] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Update Docker configuration for multi-architecture support
  - Modify `infra/docker/Dockerfile.tts` to support ARM64 architecture
  - Add conditional CUDA installation based on target architecture
  - Use multi-stage build with architecture-specific base images
  - Update base image selection to use ARM64-compatible images for Apple Silicon
  - Add TARGETARCH build argument to detect platform
  - Install PyTorch with appropriate backend for each architecture
  - _Requirements: 4.1, 4.4_

- [ ] 11. Enhance documentation with comprehensive Apple Silicon instructions
  - Expand "Running on Apple Silicon" section in README.md with detailed setup steps
  - Add troubleshooting guide for MPS-specific issues (driver updates, memory pressure, unsupported ops)
  - Document how to verify MPS availability using Python commands
  - Include performance expectations and benchmarks for MPS vs CUDA vs CPU
  - Document platform-specific requirements (macOS version, PyTorch version)
  - Add examples of device configuration for different scenarios
  - Document Docker usage on Apple Silicon
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
