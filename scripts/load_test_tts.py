#!/usr/bin/env python3
"""Load testing script for TTS service using locust patterns."""

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class LoadTestResults:
    """Results from load test run."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    @property
    def p50_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]


TEST_TEXTS = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to our voice agent platform. How can I assist you?",
    "Please hold while I process your request.",
    "Thank you for your patience. Your order has been confirmed.",
]


async def make_request(
    client: httpx.AsyncClient,
    base_url: str,
    text: str,
    results: LoadTestResults,
) -> None:
    """Make a single TTS request and record results."""
    start_time = time.perf_counter()

    try:
        response = await client.post(
            f"{base_url}/synthesize",
            json={"text": text, "output_format": "wav"},
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        results.total_requests += 1

        if response.status_code == 200:
            results.successful_requests += 1
            results.latencies_ms.append(latency_ms)
        else:
            results.failed_requests += 1
            results.errors.append(f"HTTP {response.status_code}: {response.text[:100]}")

    except Exception as e:
        results.total_requests += 1
        results.failed_requests += 1
        results.errors.append(str(e)[:100])


async def run_load_test(
    base_url: str,
    rps: float,
    duration_seconds: int,
    concurrency: int = 10,
) -> LoadTestResults:
    """Run load test against TTS service."""
    results = LoadTestResults()
    interval = 1.0 / rps
    end_time = time.time() + duration_seconds
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=60.0) as client:
        tasks: list[asyncio.Task] = []
        text_idx = 0

        print(f"Starting load test: {rps} RPS for {duration_seconds}s")
        print(f"Target URL: {base_url}")
        print("-" * 50)

        while time.time() < end_time:
            text = TEST_TEXTS[text_idx % len(TEST_TEXTS)]
            text_idx += 1

            async def bounded_request(t: str) -> None:
                async with semaphore:
                    await make_request(client, base_url, t, results)

            task = asyncio.create_task(bounded_request(text))
            tasks.append(task)

            await asyncio.sleep(interval)

            # Progress update every 10 seconds
            if text_idx % int(rps * 10) == 0:
                print(
                    f"  Progress: {results.total_requests} requests, "
                    f"{results.success_rate:.1f}% success"
                )

        # Wait for remaining requests
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return results


def print_results(results: LoadTestResults) -> None:
    """Print load test results."""
    print("\n" + "=" * 50)
    print("LOAD TEST RESULTS")
    print("=" * 50)
    print(f"Total Requests:     {results.total_requests}")
    print(f"Successful:         {results.successful_requests}")
    print(f"Failed:             {results.failed_requests}")
    print(f"Success Rate:       {results.success_rate:.2f}%")
    print("-" * 50)
    print("LATENCY (successful requests)")
    print(f"  P50:              {results.p50_latency:.2f} ms")
    print(f"  P95:              {results.p95_latency:.2f} ms")
    print(f"  P99:              {results.p99_latency:.2f} ms")

    if results.errors:
        print("-" * 50)
        print(f"SAMPLE ERRORS ({min(5, len(results.errors))} of {len(results.errors)}):")
        for err in results.errors[:5]:
            print(f"  - {err}")

    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load test TTS service")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="TTS service base URL",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=5.0,
        help="Requests per second",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum concurrent requests",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run_load_test(
            base_url=args.url,
            rps=args.rps,
            duration_seconds=args.duration,
            concurrency=args.concurrency,
        )
    )

    print_results(results)


if __name__ == "__main__":
    main()
