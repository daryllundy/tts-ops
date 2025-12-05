#!/usr/bin/env python3
"""
Smoke test script for validating TTS service health and functionality.

Checks the health endpoint and performs a minimal TTS synthesis request.
Used in CI/CD pipelines and for local validation.

Usage:
    python smoke_test.py --url http://localhost:8000
    python smoke_test.py --url http://localhost:8000 --retries 5 --timeout 10
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def check_health(url: str, timeout: int = 5) -> bool:
    """Check if the service health endpoint is up."""
    health_url = f"{url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("status") == "healthy"
            return False
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        return False


def test_tts_synthesis(url: str, text: str = "Hello world", timeout: int = 10) -> bool:
    """Perform a minimal TTS synthesis request."""
    synthesize_url = f"{url.rstrip('/')}/v1/audio/speech"
    payload = {
        "input": text,
        "voice": "alloy",  # Default voice
        "model": "tts-1",
        "response_format": "mp3"
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            synthesize_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                # Verify we got some audio data back
                content = response.read()
                return len(content) > 0
            return False

    except (urllib.error.URLError, OSError) as e:
        print(f"Synthesis test failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run smoke tests against TTS service")
    parser.add_argument("--url", required=True, help="Base URL of the TTS service")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for health check")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout in seconds for requests")
    parser.add_argument("--delay", type=int, default=2, help="Delay between retries in seconds")

    args = parser.parse_args()

    print(f"Starting smoke tests against {args.url}...")

    # 1. Health Check with retries
    print("Checking service health...")
    health_passed = False
    for i in range(args.retries):
        if check_health(args.url, args.timeout):
            print("✅ Health check passed")
            health_passed = True
            break
        print(f"Health check attempt {i+1}/{args.retries} failed. Retrying in {args.delay}s...")
        time.sleep(args.delay)

    if not health_passed:
        print("❌ Health check failed after all retries")
        sys.exit(1)

    # 2. Functional Test (Synthesis)
    print("Testing TTS synthesis...")
    if test_tts_synthesis(args.url, timeout=args.timeout):
        print("✅ TTS synthesis passed")
    else:
        print("❌ TTS synthesis failed")
        sys.exit(1)

    print("\nAll smoke tests passed! 🚀")
    sys.exit(0)


if __name__ == "__main__":
    main()
