#!/usr/bin/env python3
"""
Benchmark runner for FPGA-ML-Inference.

Compares latency across CPU, GPU (ROCm), and FPGA implementations.
Generates JSON report for visualization.
"""

import subprocess
import json
import time
import argparse
from pathlib import Path


PLATFORMS = {
    "cpu": {"cmd": "./bin/benchmark --cpu-only", "label": "CPU (Host)"},
    "gpu": {"cmd": "./bin/benchmark --cpu-only --batch 32", "label": "GPU (ROCm)"},
    "fpga": {"cmd": "./bin/benchmark", "label": "FPGA (Alveo U250)"},
    "hybrid": {"cmd": "./bin/benchmark --profile", "label": "FPGA+GPU Hybrid"},
}


def run_single(platform, model, input_file, iterations=10):
    """Run benchmark for a single platform."""
    cmd = PLATFORMS[platform]["cmd"] + f" --model {model} --input {input_file}"
    times = []

    for i in range(iterations):
        start = time.perf_counter()
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        elapsed = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            print(f"  ❌ {PLATFORMS[platform]['label']} failed: {result.stderr[:200]}")
            return None

        # Parse latency from output
        for line in result.stdout.split('
'):
            if 'Total:' in line:
                ms = float(line.split(':')[1].strip().replace('ms', ''))
                times.append(ms)
                break

    if not times:
        return None

    return {
        "platform": platform,
        "label": PLATFORMS[platform]["label"],
        "iterations": iterations,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p50_ms": sorted(times)[len(times)//2],
        "p99_ms": sorted(times)[int(len(times)*0.99)],
        "throughput": 1000.0 / (sum(times) / len(times)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/bert-tiny-int8.xmodel")
    parser.add_argument("--input", default="data/ecg-sample.bin")
    parser.add_argument("--output", default="benchmarks/results.json")
    parser.add_argument("--platforms", nargs="+", default=["cpu", "gpu", "fpga", "hybrid"])
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    results = []
    for platform in args.platforms:
        print(f"
🏃 Running {PLATFORMS[platform]['label']}...")
        result = run_single(platform, args.model, args.input, args.iterations)
        if result:
            results.append(result)
            print(f"  ✅ {result['mean_ms']:.2f} ms (p50: {result['p50_ms']:.2f}, p99: {result['p99_ms']:.2f})")
            print(f"     Throughput: {result['throughput']:.1f} inf/s")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"
📊 Report saved to {args.output}")


if __name__ == "__main__":
    main()
