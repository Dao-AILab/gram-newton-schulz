#!/usr/bin/env python3
"""
Benchmark Newton-Schulz algorithms on Gemma-4 E2B model shapes.
Benchmarks NS variants on fused parameter shapes from Gemma-4 E2B (15 layers: 12 sliding + 3 full).
"""

import argparse
import csv
import sys
import time
import torch
from datetime import datetime
from triton.testing import do_bench

torch._dynamo.config.cache_size_limit = 256

from gram_newton_schulz import StandardNewtonSchulz, GramNewtonSchulz, YOU_COEFFICIENTS

GEMMA_4_CONFIG = {
    'n_embd': 1536,
    'n_head': 8,
    'n_layer': 15,
    'n_inner': 6144,
    'vocab_size': 262144,
    'sliding': {
        'n_head_kv': 1,
        'head_dim': 256,
    },
    'full': {
        'n_head_kv': 1,
        'head_dim': 512,
        'k_eq_v': False,
    },
}


def get_gemma4_shapes():
    cfg = GEMMA_4_CONFIG
    n_embd = cfg['n_embd']
    n_head = cfg['n_head']
    n_inner = cfg['n_inner']
    n_layer = cfg['n_layer']
    sliding_cfg = cfg['sliding']
    full_cfg = cfg['full']

    num_sliding = 12
    num_full = 3

    q_dim_sliding = n_head * sliding_cfg['head_dim']
    kv_dim_sliding = sliding_cfg['n_head_kv'] * sliding_cfg['head_dim']
    q_dim_full = n_head * full_cfg['head_dim']
    kv_dim_full = full_cfg['n_head_kv'] * full_cfg['head_dim']

    return [
        ((q_dim_sliding, n_embd), num_sliding, "Wqkv_Q_sliding"),
        ((kv_dim_sliding, n_embd), num_sliding * 2, "Wqkv_KV_sliding"),
        ((q_dim_full, n_embd), num_full, "Wqkv_Q_full"),
        ((kv_dim_full, n_embd), num_full * 2, "Wqkv_KV_full"),
        ((n_embd, q_dim_sliding), num_sliding, "out_proj_sliding"),
        ((n_embd, q_dim_full), num_full, "out_proj_full"),
        ((n_inner, n_embd), n_layer * 2, "gate_up"),
        ((n_embd, n_inner), n_layer, "down_proj"),
    ]


def benchmark_ns_variant(callable_fn, X, warmup, repeats, desc):
    timing_ms = do_bench(lambda: callable_fn(X), warmup=warmup, rep=repeats)
    print(f"  {desc:50s} | {timing_ms:8.3f} ms")
    return timing_ms


def main():
    parser = argparse.ArgumentParser(description='Benchmark Newton-Schulz on Gemma-4 shapes')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (default: gemma4_ns_results_<timestamp>.csv)')
    parser.add_argument('--warmup', type=int, default=5,
                        help='Number of warmup iterations (default: 5)')
    parser.add_argument('--repeats', type=int, default=10,
                        help='Number of benchmark iterations (default: 10)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling and save trace')
    parser.add_argument('--profile-dir', type=str, default='gemma4_profiles',
                        help='Directory to save profile traces (default: gemma4_profiles)')
    args = parser.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device_idx = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_idx)
    compute_capability = capability[0] * 10 + capability[1]
    can_use_kernels = compute_capability >= 90

    if args.profile:
        import os
        os.makedirs(args.profile_dir, exist_ok=True)

    print("=" * 100)
    print("Gemma-4 Newton-Schulz Benchmark")
    print("=" * 100)
    print(f"Device: {torch.cuda.get_device_name(device_idx)}")
    print(f"Compute Capability: {capability[0]}.{capability[1]} (SM{compute_capability})")
    print(f"Custom kernels available: {can_use_kernels}")
    print("=" * 100)

    cfg = GEMMA_4_CONFIG
    print(f"  n_embd: {cfg['n_embd']}")
    print(f"  n_head: {cfg['n_head']}")
    print(f"  n_layer: {cfg['n_layer']} (12 sliding + 3 full)")
    print(f"  n_inner: {cfg['n_inner']}")
    print(f"  vocab_size: {cfg['vocab_size']}")
    print(f"  Sliding attention: n_head_kv={cfg['sliding']['n_head_kv']}, head_dim={cfg['sliding']['head_dim']}")
    print(f"  Full attention: n_head_kv={cfg['full']['n_head_kv']}, head_dim={cfg['full']['head_dim']}")
    print("=" * 100)

    variants = []

    if can_use_kernels:
        variants.extend([
            {
                'name': 'Gram Newton-Schulz (Kernels)',
                'ns': GramNewtonSchulz(
                    ns_use_kernels=True,
                    ns_coefficients=YOU_COEFFICIENTS,
                    gram_newton_schulz_reset_iterations=[2]
                ),
            },
            {
                'name': 'Standard Newton-Schulz (Kernels)',
                'ns': StandardNewtonSchulz(ns_use_kernels=True, ns_coefficients=YOU_COEFFICIENTS),
            },
        ])

    variants.append({
        'name': 'Standard Newton-Schulz (PyTorch)',
        'ns': StandardNewtonSchulz(ns_use_kernels=False, ns_coefficients=YOU_COEFFICIENTS),
    })

    all_results = []
    shapes = get_gemma4_shapes()

    for variant in variants:
        print(f"\nVariant: {variant['name']}")
        print("-" * 100)

        variant_timings = {}
        total_time = 0.0

        for shape, count, description in shapes:
            M, N = shape
            print(f"\n{description:30s} | Shape: {M:5d}x{N:5d} | Count: {count:3d}")

            X = torch.randn(count, M, N, device=device, dtype=dtype)

            ns_class = variant['ns'].__class__
            ns_kwargs = {
                'ns_epsilon': variant['ns'].ns_epsilon,
                'ns_use_kernels': variant['ns'].ns_use_kernels,
                'ns_coefficients': variant['ns'].ns_coefficients,
            }
            if hasattr(variant['ns'], 'gram_newton_schulz_reset_iterations'):
                ns_kwargs['gram_newton_schulz_reset_iterations'] = variant['ns'].gram_newton_schulz_reset_iterations

            ns_instance = ns_class(**ns_kwargs)

            if args.profile and description in ["Sliding_gate_up", "Full_Wqkv_Q"]:
                safe_variant_name = variant['name'].replace(' ', '_').replace('(', '').replace(')', '')
                profile_trace = f"{args.profile_dir}/gemma4_{description}_{safe_variant_name}.json"

                for _ in range(10):
                    _ = ns_instance(X)
                torch.cuda.synchronize()

                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    with_stack=True,
                ) as prof:
                    _ = ns_instance(X)

                prof.export_chrome_trace(profile_trace)
                print(f"    Profile saved to: {profile_trace}")

            timing = benchmark_ns_variant(
                ns_instance, X,
                warmup=args.warmup,
                repeats=args.repeats,
                desc=variant['name']
            )

            torch.cuda.synchronize()
            time.sleep(2.0)

            key = f"{description}_{M}x{N}"
            variant_timings[key] = timing
            total_time += timing

            torch.cuda.empty_cache()

        print(f"\n  Total time: {total_time:.3f} ms")

        all_results.append({
            'variant': variant['name'],
            'total_time': total_time,
            'timings': variant_timings,
        })

    # Write CSV
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'gemma4_ns_results_{timestamp}.csv'

    all_keys = set()
    for result in all_results:
        all_keys.update(result['timings'].keys())
    all_keys = sorted(all_keys)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['variant', 'total_time_ms'] + all_keys
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_results:
            row = {
                'variant': result['variant'],
                'total_time_ms': result['total_time'],
            }
            for key in all_keys:
                row[key] = result['timings'].get(key, 'N/A')
            writer.writerow(row)

    print(f"\n{'=' * 100}")
    print(f"Results written to: {output_file}")
    print(f"{'=' * 100}")

    print("\nSummary:")
    print("-" * 80)
    print(f"{'Variant':<50s} | {'Total Time (ms)':>15s}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['variant']:<50s} | {result['total_time']:>15.3f}")
    print("-" * 80)


if __name__ == '__main__':
    main()
