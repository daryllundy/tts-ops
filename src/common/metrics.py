"""Prometheus metrics definitions and utilities."""

from prometheus_client import Counter, Gauge, Histogram, Info
from common.device_utils import is_mps_available

# TTS Service Metrics
TTS_REQUEST_DURATION = Histogram(
    "tts_request_duration_seconds",
    "Time spent processing TTS requests",
    ["endpoint", "status"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0),
)

TTS_TIME_TO_FIRST_AUDIO = Histogram(
    "tts_time_to_first_audio_seconds",
    "Time to first audio byte in streaming responses",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0),
)

TTS_REQUESTS_TOTAL = Counter(
    "tts_requests_total",
    "Total number of TTS requests",
    ["endpoint", "status"],
)

TTS_INPUT_LENGTH = Histogram(
    "tts_input_text_length_chars",
    "Length of input text in characters",
    buckets=(50, 100, 250, 500, 1000, 2000, 4000),
)

TTS_OUTPUT_DURATION = Histogram(
    "tts_output_audio_duration_seconds",
    "Duration of generated audio",
    buckets=(1, 2, 5, 10, 20, 30, 60),
)

TTS_GPU_UTILIZATION = Gauge(
    "tts_gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"],
)

TTS_GPU_MEMORY_USED = Gauge(
    "tts_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["device"],
)

TTS_MODEL_LOADED = Gauge(
    "tts_model_loaded",
    "Whether the TTS model is loaded and ready",
)

TTS_MODEL_INFO = Info(
    "tts_model",
    "TTS model information",
)

# Agent Service Metrics
AGENT_CHAT_DURATION = Histogram(
    "agent_chat_duration_seconds",
    "Total time for chat request processing",
    ["status"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
)

AGENT_LLM_DURATION = Histogram(
    "agent_llm_duration_seconds",
    "Time spent on LLM inference",
    ["provider", "model"],
    buckets=(0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

AGENT_REQUESTS_TOTAL = Counter(
    "agent_requests_total",
    "Total number of agent requests",
    ["endpoint", "status"],
)

AGENT_ACTIVE_CONNECTIONS = Gauge(
    "agent_active_websocket_connections",
    "Number of active WebSocket connections",
)


def record_gpu_metrics(device: str = "cuda:0") -> None:
    """Record current GPU metrics."""
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            memory_allocated = torch.cuda.memory_allocated(device_idx)
            TTS_GPU_MEMORY_USED.labels(device=device).set(memory_allocated)

            # GPU utilization requires pynvml, so we handle import gracefully
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                TTS_GPU_UTILIZATION.labels(device=device).set(util.gpu)
            except ImportError:
                pass
        
        elif device == "mps" and is_mps_available():
            # MPS metrics are limited, but we can get memory usage
            try:
                memory_allocated = torch.mps.current_allocated_memory()
                TTS_GPU_MEMORY_USED.labels(device=device).set(memory_allocated)
                # MPS utilization is not easily available via PyTorch
            except Exception:
                pass

    except Exception:
        pass
