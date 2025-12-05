#!/usr/bin/env python3
"""
Performance regression detection script for CI/CD pipeline.

Compares current performance metrics against baseline metrics and detects
regressions based on configurable thresholds.

Usage:
    python check_performance_regression.py --current metrics.json --baseline baseline.json
    python check_performance_regression.py --current metrics.json --baseline baseline.json --create-baseline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Regression thresholds (as percentages)
THRESHOLDS = {
    "ttfa_ms": {"mean": 10.0, "p50": 10.0, "p95": 10.0, "p99": 10.0},
    "e2e_latency_ms": {"mean": 15.0, "p50": 15.0, "p95": 15.0, "p99": 15.0},
    "error_rate": {"absolute": 0.01},  # 1% absolute threshold
}


class RegressionResult:
    """Result of regression detection."""

    def __init__(self) -> None:
        self.regressions: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.passed: list[dict[str, Any]] = []

    def add_regression(
        self,
        metric_name: str,
        field: str,
        baseline_value: float,
        current_value: float,
        threshold: float,
        change_pct: float,
    ) -> None:
        """Add a regression violation."""
        self.regressions.append(
            {
                "metric": metric_name,
                "field": field,
                "baseline": baseline_value,
                "current": current_value,
                "threshold": threshold,
                "change_pct": change_pct,
            }
        )

    def add_passed(
        self,
        metric_name: str,
        field: str,
        baseline_value: float,
        current_value: float,
        change_pct: float,
    ) -> None:
        """Add a passed check."""
        self.passed.append(
            {
                "metric": metric_name,
                "field": field,
                "baseline": baseline_value,
                "current": current_value,
                "change_pct": change_pct,
            }
        )

    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return len(self.regressions) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "has_regressions": self.has_regressions(),
            "regressions": self.regressions,
            "warnings": self.warnings,
            "passed": self.passed,
        }


def load_metrics(file_path: Path) -> dict[str, Any]:
    """Load metrics from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def save_metrics(file_path: Path, metrics: dict[str, Any]) -> None:
    """Save metrics to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=2)


def check_metric_field(
    metric_name: str,
    field: str,
    baseline_value: float | None,
    current_value: float | None,
    threshold: float,
    result: RegressionResult,
    is_absolute: bool = False,
) -> None:
    """Check a single metric field for regression."""
    # Handle missing values
    if baseline_value is None or current_value is None:
        return

    if is_absolute:
        # Absolute threshold (e.g., error rate)
        if current_value > threshold:
            result.add_regression(
                metric_name=metric_name,
                field=field,
                baseline_value=baseline_value,
                current_value=current_value,
                threshold=threshold,
                change_pct=0.0,  # Not applicable for absolute thresholds
            )
        else:
            result.add_passed(
                metric_name=metric_name,
                field=field,
                baseline_value=baseline_value,
                current_value=current_value,
                change_pct=0.0,
            )
    else:
        # Percentage threshold
        if baseline_value == 0:
            # Avoid division by zero
            if current_value > 0:
                result.add_regression(
                    metric_name=metric_name,
                    field=field,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    threshold=threshold,
                    change_pct=float("inf"),
                )
            else:
                result.add_passed(
                    metric_name=metric_name,
                    field=field,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    change_pct=0.0,
                )
            return

        change_pct = ((current_value - baseline_value) / baseline_value) * 100

        if change_pct > threshold:
            result.add_regression(
                metric_name=metric_name,
                field=field,
                baseline_value=baseline_value,
                current_value=current_value,
                threshold=threshold,
                change_pct=change_pct,
            )
        else:
            result.add_passed(
                metric_name=metric_name,
                field=field,
                baseline_value=baseline_value,
                current_value=current_value,
                change_pct=change_pct,
            )


def check_regressions(
    current_metrics: dict[str, Any], baseline_metrics: dict[str, Any]
) -> RegressionResult:
    """Check for performance regressions."""
    result = RegressionResult()

    # Extract metrics sections
    current = current_metrics.get("metrics", {})
    baseline = baseline_metrics.get("metrics", {})

    # Check latency metrics (ttfa_ms, e2e_latency_ms)
    for metric_name in ["ttfa_ms", "e2e_latency_ms"]:
        if metric_name not in THRESHOLDS:
            continue

        current_metric = current.get(metric_name, {})
        baseline_metric = baseline.get(metric_name, {})
        thresholds = THRESHOLDS[metric_name]

        for field, threshold in thresholds.items():
            check_metric_field(
                metric_name=metric_name,
                field=field,
                baseline_value=baseline_metric.get(field),
                current_value=current_metric.get(field),
                threshold=threshold,
                result=result,
                is_absolute=False,
            )

    # Check error rate (absolute threshold)
    if "error_rate" in THRESHOLDS:
        current_error_rate = current.get("error_rate")
        baseline_error_rate = baseline.get("error_rate")
        threshold = THRESHOLDS["error_rate"]["absolute"]

        check_metric_field(
            metric_name="error_rate",
            field="value",
            baseline_value=baseline_error_rate,
            current_value=current_error_rate,
            threshold=threshold,
            result=result,
            is_absolute=True,
        )

    return result


def print_report(result: RegressionResult) -> None:
    """Print human-readable regression report."""
    print("\n" + "=" * 80)
    print("PERFORMANCE REGRESSION REPORT")
    print("=" * 80)

    if result.has_regressions():
        print("\n❌ REGRESSIONS DETECTED:")
        for reg in result.regressions:
            if reg["metric"] == "error_rate":
                print(
                    f"  - {reg['metric']}: {reg['current']:.4f} "
                    f"(threshold: {reg['threshold']:.4f})"
                )
            else:
                print(
                    f"  - {reg['metric']}.{reg['field']}: "
                    f"{reg['baseline']:.2f} → {reg['current']:.2f} "
                    f"(+{reg['change_pct']:.1f}%, threshold: +{reg['threshold']:.1f}%)"
                )
    else:
        print("\n✅ NO REGRESSIONS DETECTED")

    if result.passed:
        print(f"\n✓ {len(result.passed)} checks passed")

    print("=" * 80 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for performance regressions against baseline metrics"
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current metrics JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline metrics JSON file",
    )
    parser.add_argument(
        "--create-baseline",
        action="store_true",
        help="Create baseline file if it doesn't exist",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output regression report JSON",
    )

    args = parser.parse_args()

    # Load current metrics
    current_metrics = load_metrics(args.current)

    # Handle baseline creation
    if not args.baseline.exists():
        if args.create_baseline:
            print(f"Baseline file not found. Creating new baseline: {args.baseline}")
            save_metrics(args.baseline, current_metrics)
            print("✅ Baseline created successfully. No regressions to check.")
            sys.exit(0)
        else:
            print(
                f"Error: Baseline file not found: {args.baseline}",
                file=sys.stderr,
            )
            print(
                "Use --create-baseline to create a new baseline file.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load baseline metrics
    baseline_metrics = load_metrics(args.baseline)

    # Check for regressions
    result = check_regressions(current_metrics, baseline_metrics)

    # Print report
    print_report(result)

    # Save output if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Report saved to: {args.output}")

    # Exit with appropriate code
    if result.has_regressions():
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
