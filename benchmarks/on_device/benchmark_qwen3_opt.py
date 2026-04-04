#!/usr/bin/env python3
"""
Benchmark for optimizer.step() on Qwen3-1.7B architecture.

Compares 4 variants:
  1. AdamW (torch.optim.AdamW)
  2. Gram Newton-Schulz Muon (this package, kernels)
  3. Standard Newton-Schulz Muon (this package, kernels)
  4. PyTorch Muon (torch.optim.Muon)

Each variant runs in a subprocess to guarantee full GPU memory isolation.

Usage:
    python benchmarks/on_device/benchmark_qwen3_opt.py
    python benchmarks/on_device/benchmark_qwen3_opt.py --num-layers 14
    python benchmarks/on_device/benchmark_qwen3_opt.py --profile
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
#  Qwen3-1.7B config
# ──────────────────────────────────────────────────────────────────────
QWEN3_1_7B_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}


# ──────────────────────────────────────────────────────────────────────
#  Split / recombine for Qwen3 GQA (Q+K+V fused) and SwiGLU (gate+up)
# ──────────────────────────────────────────────────────────────────────
def qkv_split_fn(param: torch.Tensor) -> List[torch.Tensor]:
    cfg = QWEN3_1_7B_CONFIG
    q_dim = cfg["num_attention_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_key_value_heads"] * cfg["head_dim"]
    return [param[:q_dim], param[q_dim:q_dim + kv_dim], param[q_dim + kv_dim:]]


def qkv_recombine_fn(splits: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(splits, dim=0)


def swiglu_split_fn(param: torch.Tensor) -> List[torch.Tensor]:
    half = param.shape[0] // 2
    return [param[:half], param[half:]]


def swiglu_recombine_fn(splits: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(splits, dim=0)


# ──────────────────────────────────────────────────────────────────────
#  Dummy Qwen3 model
# ──────────────────────────────────────────────────────────────────────
class DummyQwen3Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size):
        super().__init__()
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class DummyQwen3Model(nn.Module):
    def __init__(self, config: Dict, num_layers: int = None):
        super().__init__()
        n_layers = num_layers or config["num_hidden_layers"]
        self.layers = nn.ModuleList([
            DummyQwen3Layer(
                config["hidden_size"],
                config["num_attention_heads"],
                config["num_key_value_heads"],
                config["head_dim"],
                config["intermediate_size"],
            )
            for _ in range(n_layers)
        ])

    def get_muon_param_groups(self) -> List[Dict]:
        qkv_params, gate_up_params, no_split_params = [], [], []
        for layer in self.layers:
            qkv_params.append(layer.qkv_proj.weight)
            gate_up_params.append(layer.gate_up_proj.weight)
            no_split_params.append(layer.o_proj.weight)
            no_split_params.append(layer.down_proj.weight)
        return [
            {"params": qkv_params, "param_split_fn": qkv_split_fn, "param_recombine_fn": qkv_recombine_fn},
            {"params": gate_up_params, "param_split_fn": swiglu_split_fn, "param_recombine_fn": swiglu_recombine_fn},
            {"params": no_split_params},
        ]

    def get_all_2d_params(self) -> List[torch.Tensor]:
        return [p for p in self.parameters() if p.ndim == 2]


def create_fake_gradients(model: nn.Module):
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)


# ──────────────────────────────────────────────────────────────────────
#  Single-variant runner (called in subprocess)
# ──────────────────────────────────────────────────────────────────────
def run_single_variant(variant, num_layers, warmup, repeats, do_profile, trace_file, ns_max_batch_size=None):
    """Run one optimizer variant. Prints JSON result to stdout."""
    from triton.testing import do_bench

    lr, momentum, weight_decay = 3e-4, 0.95, 0.1
    cfg = QWEN3_1_7B_CONFIG
    model = DummyQwen3Model(cfg, num_layers=num_layers).cuda()

    if variant == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        def step():
            create_fake_gradients(model)
            opt.step()
            opt.zero_grad()

    elif variant == "pt_muon":
        opt = torch.optim.Muon(
            model.get_all_2d_params(),
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=True, ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
        def step():
            create_fake_gradients(model)
            opt.step()
            opt.zero_grad()

    elif variant == "std_ns":
        from gram_newton_schulz.muon import Muon
        opt = Muon(
            model.get_muon_param_groups(),
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=True,
            ns_algorithm="standard_newton_schulz",
            ns_use_kernels=True,
            adjust_lr="rms_norm",
            ns_max_batch_size=ns_max_batch_size,
        )
        def step():
            create_fake_gradients(model)
            opt.step()
            opt.zero_grad()

    elif variant == "gram_ns":
        from gram_newton_schulz.muon import Muon
        opt = Muon(
            model.get_muon_param_groups(),
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=True,
            ns_algorithm="gram_newton_schulz",
            ns_use_kernels=True,
            adjust_lr="rms_norm",
            ns_max_batch_size=ns_max_batch_size,
        )
        def step():
            create_fake_gradients(model)
            opt.step()
            opt.zero_grad()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    compiled_step = torch.compile(step, fullgraph=False)
    time.sleep(0.5)
    ms = do_bench(compiled_step, warmup=warmup, rep=repeats)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    if do_profile and trace_file:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(5):
                compiled_step()
                torch.cuda.synchronize()
        prof.export_chrome_trace(trace_file)

    # Output result as JSON on a dedicated line
    result = {"variant": variant, "ms": ms, "peak_gb": peak_gb}
    print(f"RESULT:{json.dumps(result)}")


# ──────────────────────────────────────────────────────────────────────
#  Main (orchestrator)
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Benchmark optimizers on Qwen3-1.7B")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of layers (default: all 28). Reduce if OOM.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--ns-max-batch-size", type=int, default=None,
                        help="Max matrices per NS call. Reduces peak memory at cost of speed.")
    parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
    parser.add_argument("--profile-trace", type=str, default=None)
    # Internal: run a single variant in subprocess mode
    parser.add_argument("--_run_variant", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_trace_file", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = QWEN3_1_7B_CONFIG
    num_layers = args.num_layers or cfg["num_hidden_layers"]

    # ── Subprocess mode: run single variant and exit ──
    if args._run_variant:
        run_single_variant(
            args._run_variant, num_layers, args.warmup, args.repeats,
            args.profile, args._trace_file,
            ns_max_batch_size=args.ns_max_batch_size,
        )
        return

    # ── Orchestrator mode ──
    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"\n{'=' * 80}")
    print(f"Optimizer Benchmark -- Qwen3-1.7B ({num_layers} layers)")
    print(f"{'=' * 80}")
    print(f"GPU:          {torch.cuda.get_device_name(device)} ({gpu_total_gb:.0f} GB)")
    print(f"Compute:      SM{capability[0]}{capability[1]}")
    print(f"hidden_size:  {cfg['hidden_size']}")
    print(f"num_layers:   {num_layers} / {cfg['num_hidden_layers']}")
    print(f"intermediate: {cfg['intermediate_size']}")
    print(f"num_heads:    {cfg['num_attention_heads']}  (kv_heads: {cfg['num_key_value_heads']})")
    print(f"head_dim:     {cfg['head_dim']}")
    if args.ns_max_batch_size:
        print(f"ns_max_batch_size: {args.ns_max_batch_size}")

    # Quick param count
    model_tmp = DummyQwen3Model(cfg, num_layers=num_layers)
    total_params = sum(p.numel() for p in model_tmp.parameters())
    del model_tmp
    print(f"\nTotal parameters: {total_params:,}")

    trace_base = args.profile_trace or f"qwen3_opt_{datetime.now():%Y%m%d_%H%M%S}"

    variants = [
        ("adamw",   "AdamW (torch.optim.AdamW)"),
        ("gram_ns", "Gram NS Muon (this package, kernels)"),
        ("std_ns",  "Standard NS Muon (this package, kernels)"),
        ("pt_muon", "PyTorch Muon (torch.optim.Muon)"),
    ]

    results = []
    for i, (tag, label) in enumerate(variants):
        print(f"\n{'=' * 80}")
        print(f"[{i+1}] {label}")
        print(f"{'=' * 80}")

        trace_file = f"{trace_base}_{tag}.json" if args.profile else ""
        cmd = [
            sys.executable, __file__,
            "--_run_variant", tag,
            "--num-layers", str(num_layers),
            "--warmup", str(args.warmup),
            "--repeats", str(args.repeats),
        ]
        if args.ns_max_batch_size is not None:
            cmd += ["--ns-max-batch-size", str(args.ns_max_batch_size)]
        if args.profile:
            cmd += ["--profile", "--_trace_file", trace_file]

        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

        # Print subprocess stderr (compilation logs, warnings)
        for line in proc.stderr.splitlines():
            if "Profiler record function" not in line:
                print(f"  {line}")

        # Parse result from stdout
        result_line = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                result_line = line[7:]
            else:
                print(f"  {line}")

        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            if "OutOfMemoryError" in proc.stderr:
                print(f"  OOM -- try --num-layers {max(1, num_layers - 4)}")
            results.append((label, float("inf")))
        elif result_line:
            data = json.loads(result_line)
            ms = data["ms"]
            peak = data["peak_gb"]
            print(f"  Median time: {ms:.3f} ms  (peak GPU: {peak:.1f} GB)")
            results.append((label, ms))
            if args.profile and trace_file:
                print(f"  Trace: {trace_file}")
        else:
            print(f"  ERROR: no result parsed")
            results.append((label, float("inf")))

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"SUMMARY -- Qwen3-1.7B ({num_layers} layers, {total_params:,} params)")
    print(f"{'=' * 80}")
    print(f"{'Variant':<45} | {'Time (ms)':>10} | {'vs AdamW':>10}")
    print("-" * 75)
    baseline = results[0][1]
    for name, ms in results:
        if ms == float("inf"):
            print(f"{name:<45} |        OOM |        N/A")
        else:
            ratio = f"{baseline / ms:.2f}x" if ms > 0 else "N/A"
            print(f"{name:<45} | {ms:10.3f} | {ratio:>10}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
