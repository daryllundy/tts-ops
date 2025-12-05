# Product Overview

VibeVoice Realtime Agent is a production-grade AI voice agent platform that combines LLM-powered conversation with low-latency text-to-speech synthesis.

## Core Capabilities

- Real-time voice synthesis using Microsoft's VibeVoice-Realtime-0.5B model
- Multi-provider LLM integration (Anthropic Claude, OpenAI, local models)
- Sub-500ms time-to-first-audio latency
- Streaming audio delivery via WebSocket and HTTP
- GPU-accelerated inference with CUDA optimization

## Architecture

The system consists of two microservices:

1. **Agent Service** - Handles chat requests, orchestrates LLM responses, and coordinates with TTS
2. **TTS Service** - Dedicated GPU-accelerated text-to-speech inference service

Both services expose FastAPI REST endpoints with health checks, metrics, and streaming support.

## Target Deployment

Kubernetes-native with Helm charts, designed for production environments with GPU nodes. Includes comprehensive observability (Prometheus metrics, structured logging, distributed tracing).
