#!/usr/bin/env python3
"""
Benchmark script for comparing Newton-Schulz orthogonalization variants.

Usage:
    python benchmarks/benchmark_newton_schulz.py --M 2048 --N 5464 --batch-size 32 --warmup 10 --repeats 100
"""

import argparse
import sys
import time
from datetime import datetime

import torch
from triton.testing import do_bench

from gram_newton_schulz import StandardNewtonSchulz, GramNewtonSchulz, YOU_COEFFICIENTS

def benchmark_ns_variant(callable_fn, X, warmup=5, repeats=30, desc=""):
    print(f"\n  {desc}")
    timing_ms = do_bench(lambda: callable_fn(X), warmup=warmup, rep=repeats)
    print(f"    Time: {timing_ms:8.3f} ms")

    return timing_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Newton-Schulz orthogonalization variants"
    )
    parser.add_argument(
        "--M",
        type=int,
        required=True,
        help="Matrix dimension M (rows)",
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Matrix dimension N (columns)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (number of matrices to orthogonalize)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler",
    )
    parser.add_argument(
        "--profile-trace",
        type=str,
        default=None,
        help="Output trace filename for profiler (default: ns_profile_<timestamp>.json)",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    compute_capability = capability[0] * 10 + capability[1]

    print("=" * 80)
    print("Newton-Schulz Orthogonalization Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Compute Capability: {capability[0]}.{capability[1]} (SM{compute_capability})")
    print(f"Batch size: {args.batch_size}")
    print(f"Input dtype: bfloat16")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.repeats}")
    print("=" * 80)

    can_use_kernels = compute_capability >= 90

    if can_use_kernels:
        print("Custom kernels available (H100/B200)")
    else:
        print(f"Custom kernels not available (requires SM90+, found SM{compute_capability})")
        print("Will only benchmark PyTorch implementations")

    torch_dtype = torch.bfloat16

    M, N = args.M, args.N
    print(f"\n{'=' * 80}")
    print(f"Shape: {M}x{N} | Batch size: {args.batch_size}")
    print(f"{'=' * 80}")

    X = torch.randn(args.batch_size, M, N, dtype=torch_dtype, device="cuda")

    print("\n[1] Standard Newton-Schulz (PyTorch)")
    standard_torch = StandardNewtonSchulz(
        ns_epsilon=1e-7,
        ns_use_kernels=False,
        ns_coefficients=YOU_COEFFICIENTS,
    )

    _ = standard_torch(X)
    torch.cuda.synchronize()
    time.sleep(1.0)

    timing_standard_torch = benchmark_ns_variant(
        standard_torch,
        X,
        warmup=args.warmup,
        repeats=args.repeats,
        desc="Standard Newton-Schulz (PyTorch)",
    )
    torch.cuda.synchronize()
    time.sleep(1.0)

    timing_standard_kernels = None
    if can_use_kernels:
        print("\n[2] Standard Newton-Schulz (Kernels)")
        standard_kernels = StandardNewtonSchulz(
            ns_epsilon=1e-7,
            ns_use_kernels=True,
            ns_coefficients=YOU_COEFFICIENTS,
        )

        _ = standard_kernels(X)
        torch.cuda.synchronize()
        time.sleep(1.0)

        timing_standard_kernels = benchmark_ns_variant(
            standard_kernels,
            X,
            warmup=args.warmup,
            repeats=args.repeats,
            desc="Standard Newton-Schulz (Kernels)",
        )
        torch.cuda.synchronize()
        time.sleep(1.0)

    print("\n[3] Gram Newton-Schulz (PyTorch)")
    gram_torch = GramNewtonSchulz(
        ns_epsilon=1e-7,
        ns_use_kernels=False,
        ns_coefficients=YOU_COEFFICIENTS,
        gram_newton_schulz_reset_iterations=[2],
    )

    _ = gram_torch(X)
    torch.cuda.synchronize()
    time.sleep(1.0)

    timing_gram_torch = benchmark_ns_variant(
        gram_torch,
        X,
        warmup=args.warmup,
        repeats=args.repeats,
        desc="Gram Newton-Schulz (PyTorch)",
    )
    torch.cuda.synchronize()
    time.sleep(1.0)

    timing_gram_kernels = None
    if can_use_kernels:
        print("\n[4] Gram Newton-Schulz (Kernels)")
        gram_kernels = GramNewtonSchulz(
            ns_epsilon=1e-7,
            ns_use_kernels=True,
            ns_coefficients=YOU_COEFFICIENTS,
            gram_newton_schulz_reset_iterations=[2],
        )

        _ = gram_kernels(X)
        torch.cuda.synchronize()
        time.sleep(1.0)

        timing_gram_kernels = benchmark_ns_variant(
            gram_kernels,
            X,
            warmup=args.warmup,
            repeats=args.repeats,
            desc="Gram Newton-Schulz (Kernels)",
        )
        torch.cuda.synchronize()
        time.sleep(1.0)

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"{'Variant':<50} | {'Time (ms)':>12}")
    print("-" * 80)

    print(f"{'Standard Newton-Schulz (PyTorch)':<50} | {timing_standard_torch:12.3f}")
    if can_use_kernels:
        print(f"{'Standard Newton-Schulz (Kernels)':<50} | {timing_standard_kernels:12.3f}")
    print(f"{'Gram Newton-Schulz (PyTorch)':<50} | {timing_gram_torch:12.3f}")
    if can_use_kernels:
        print(f"{'Gram Newton-Schulz (Kernels)':<50} | {timing_gram_kernels:12.3f}")

    print("-" * 80)

    if args.profile:
        if args.profile_trace:
            trace_filename = args.profile_trace
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trace_filename = f'ns_profile_{timestamp}.json'

        print(f"\n{'=' * 80}")
        print("Running Profiler")
        print(f"{'=' * 80}")
        print(f"Output trace: {trace_filename}")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            _ = standard_torch(X)
            torch.cuda.synchronize()

            if can_use_kernels:
                _ = standard_kernels(X)
                torch.cuda.synchronize()

            _ = gram_torch(X)
            torch.cuda.synchronize()

            if can_use_kernels:
                _ = gram_kernels(X)
                torch.cuda.synchronize()

        prof.export_chrome_trace(trace_filename)
        print(f"Trace saved to: {trace_filename}")
        print(f"View at: chrome://tracing")
        print("-" * 80)


if __name__ == "__main__":
    main()
