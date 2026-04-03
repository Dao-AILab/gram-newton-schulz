#!/usr/bin/env python3
"""
Standalone benchmark for Muon optimizer on Gemma-4 E2B architecture.
Creates dummy parameters fitting as many layers of Gemma-4 E2B on RTX 5090 as possible (15 layers: 12 sliding + 3 full)
and benchmarks optimizer.step() with fake gradients.

Automatically runs 3 variants:
- Standard Newton-Schulz (PyTorch)
- Standard Newton-Schulz (Kernels)
- Gram Newton-Schulz (Kernels)
"""

import torch
import torch.nn as nn
from typing import List, Dict
from triton.testing import do_bench

from gram_newton_schulz.muon import Muon

# Split/recombine functions for Gemma-4 E2B (trainstation-style)
def qkv_split_sliding(param: torch.Tensor) -> List[torch.Tensor]:
    """Split Wqkv for sliding_attention: [Q, K, V] layout."""
    q_dim = 2048
    kv_dim = 256
    Wq = param[:q_dim, :]
    Wk = param[q_dim:q_dim + kv_dim, :]
    Wv = param[q_dim + kv_dim:, :]
    return [Wq, Wk, Wv]


def qkv_recombine_sliding(splits: List[torch.Tensor]) -> torch.Tensor:
    """Recombine [Q, K, V] back into Wqkv."""
    return torch.cat(splits, dim=0)


def qkv_split_full(param: torch.Tensor) -> List[torch.Tensor]:
    """Split Wqkv for full_attention: [Q, K, V] layout."""
    q_dim = 4096
    kv_dim = 512
    Wq = param[:q_dim, :]
    Wk = param[q_dim:q_dim + kv_dim, :]
    Wv = param[q_dim + kv_dim:, :]
    return [Wq, Wk, Wv]


def qkv_recombine_full(splits: List[torch.Tensor]) -> torch.Tensor:
    """Recombine [Q, K, V] back into Wqkv."""
    return torch.cat(splits, dim=0)


def geglu_split_fn(param: torch.Tensor) -> List[torch.Tensor]:
    """Split gate_up_proj into [up, gate] with INTERLEAVED layout."""
    gate_matrix = param[::2, :]
    up_matrix = param[1::2, :]
    return [up_matrix, gate_matrix]


def geglu_recombine_fn(splits: List[torch.Tensor]) -> torch.Tensor:
    """Recombine [up, gate] back into interleaved gate_up_proj."""
    up_matrix, gate_matrix = splits
    out_features, in_features = up_matrix.shape
    total_out = 2 * out_features
    result = torch.empty((total_out, in_features), dtype=up_matrix.dtype, device=up_matrix.device)
    result[::2, :] = gate_matrix
    result[1::2, :] = up_matrix
    return result


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


class DummyGemma4Layer(nn.Module):
    """Simplified Gemma-4 transformer layer with fused QKV and fused gate/up (trainstation-style)."""
    def __init__(self, n_embd: int, n_head: int, n_head_kv: int, head_dim: int, n_inner: int, k_eq_v: bool = False):
        super().__init__()
        q_dim = n_head * head_dim
        kv_dim = n_head_kv * head_dim

        if k_eq_v:
            self.Wqkv = nn.Linear(n_embd, q_dim + kv_dim, bias=False)
        else:
            self.Wqkv = nn.Linear(n_embd, q_dim + 2 * kv_dim, bias=False)

        self.out_proj = nn.Linear(q_dim, n_embd, bias=False)
        self.gate_up_proj = nn.Linear(n_embd, 2 * n_inner, bias=False)
        self.down_proj = nn.Linear(n_inner, n_embd, bias=False)


class DummyGemma4Model(nn.Module):
    """Gemma-4 E2B model for benchmarking with 15 layers (12 sliding + 3 full)."""
    def __init__(self, config: Dict):
        super().__init__()
        full_indices = {4, 9, 14}

        self.layers = nn.ModuleList()
        for i in range(config['n_layer']):
            is_full = i in full_indices
            layer_config = config['full'] if is_full else config['sliding']
            self.layers.append(DummyGemma4Layer(
                n_embd=config['n_embd'],
                n_head=config['n_head'],
                n_head_kv=layer_config['n_head_kv'],
                head_dim=layer_config['head_dim'],
                n_inner=config['n_inner'],
                k_eq_v=layer_config.get('k_eq_v', False),
            ))

    def get_muon_param_groups(self) -> List[Dict]:
        sliding_wqkv = []
        full_wqkv = []
        all_gate_up = []
        no_split_params = []
        full_indices = {4, 9, 14}

        for i, layer in enumerate(self.layers):
            if i in full_indices:
                full_wqkv.append(layer.Wqkv.weight)
            else:
                sliding_wqkv.append(layer.Wqkv.weight)
            all_gate_up.append(layer.gate_up_proj.weight)
            no_split_params.append(layer.out_proj.weight)
            no_split_params.append(layer.down_proj.weight)

        return [
            {
                'params': sliding_wqkv,
                'param_split_fn': qkv_split_sliding,
                'param_recombine_fn': qkv_recombine_sliding,
            },
            {
                'params': full_wqkv,
                'param_split_fn': qkv_split_full,
                'param_recombine_fn': qkv_recombine_full,
            },
            {
                'params': all_gate_up,
                'param_split_fn': geglu_split_fn,
                'param_recombine_fn': geglu_recombine_fn,
            },
            {'params': no_split_params},
        ]


def create_fake_gradients(model: nn.Module):
    for param in model.parameters():
        if param.requires_grad:
            param.grad = torch.randn_like(param)


def benchmark_optimizer_step(optimizer, model: nn.Module, num_warmup: int, num_iterations: int):
    def step_fn():
        create_fake_gradients(model)
        optimizer.step()
        optimizer.zero_grad()

    compiled_step_fn = torch.compile(step_fn, fullgraph=False)
    return do_bench(compiled_step_fn, warmup=num_warmup, rep=num_iterations)


def main():
    lr = 3e-4
    momentum = 0.95
    weight_decay = 0.1
    num_warmup = 5
    num_iterations = 15

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available! This benchmark requires a GPU.")

    print(f"\n{'='*80}")
    print(f"Muon Optimizer Benchmark - Gemma-4 E2B (15 layers: 12 sliding + 3 full)")
    print(f"{'='*80}")
    print(f"Device: cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LR: {lr}")
    print(f"Momentum: {momentum}")
    print(f"Weight Decay: {weight_decay}")
    print(f"{'='*80}\n")

    print(f"Gemma-4 E2B Model Config (15 layers: 12 sliding + 3 full):")
    print(f"  n_embd: {GEMMA_4_CONFIG['n_embd']}")
    print(f"  n_head: {GEMMA_4_CONFIG['n_head']}")
    print(f"  n_layer: {GEMMA_4_CONFIG['n_layer']}")
    print(f"  n_inner: {GEMMA_4_CONFIG['n_inner']}")
    print(f"  vocab_size: {GEMMA_4_CONFIG['vocab_size']}")
    print(f"  Sliding: {GEMMA_4_CONFIG['sliding']}")
    print(f"  Full: {GEMMA_4_CONFIG['full']}")

    model = DummyGemma4Model(GEMMA_4_CONFIG).to('cuda')
    param_groups = model.get_muon_param_groups()

    total_params = sum(sum(p.numel() for p in g['params']) for g in param_groups)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Number of param groups: {len(param_groups)}")

    print(f"\nLayer 0 (sliding_attention) FUSED weight shapes:")
    sliding_layer = model.layers[0]
    print(f"  Wqkv:         {sliding_layer.Wqkv.weight.shape} (Q+K+V fused)")
    print(f"  out_proj:     {sliding_layer.out_proj.weight.shape}")
    print(f"  gate_up_proj: {sliding_layer.gate_up_proj.weight.shape} (gate+up interleaved)")
    print(f"  down_proj:    {sliding_layer.down_proj.weight.shape}")

    print(f"\nLayer 4 (full_attention) FUSED weight shapes:")
    full_layer = model.layers[4]
    print(f"  Wqkv:         {full_layer.Wqkv.weight.shape} (Q+K+V fused)")
    print(f"  out_proj:     {full_layer.out_proj.weight.shape}")
    print(f"  gate_up_proj: {full_layer.gate_up_proj.weight.shape} (gate+up interleaved)")
    print(f"  down_proj:    {full_layer.down_proj.weight.shape}")

    variants = [
        ('standard_newton_schulz', False, 'Standard NS (PyTorch)'),
        ('standard_newton_schulz', True, 'Standard NS (Kernels)'),
        ('gram_newton_schulz', True, 'Gram NS (Kernels)'),
    ]

    results = []

    for ns_algorithm, use_kernels, variant_name in variants:
        print(f"\n{'='*80}")
        print(f"Variant: {variant_name}")
        print(f"{'='*80}")

        optimizer = Muon(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
            ns_algorithm=ns_algorithm,
            ns_use_kernels=use_kernels,
            adjust_lr='rms_norm',
        )

        print(f"Warmup iterations: {num_warmup}")
        print(f"Timed iterations: {num_iterations}")

        median_ms = benchmark_optimizer_step(
            optimizer,
            model,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
        )

        print(f"Median time: {median_ms:.3f} ms")
        results.append((variant_name, median_ms))

    print(f"\n{'='*80}")
    print(f"SUMMARY - Gemma-4 E2B (15 layers: 12 sliding + 3 full)")
    print(f"{'='*80}")
    for variant_name, median_ms in results:
        print(f"{variant_name:30s} | {median_ms:8.3f} ms")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
