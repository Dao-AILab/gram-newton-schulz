# Gram Newton-Schulz: A Fast, Hardware-Aware Muon Optimizer

Authors: Jack Zhang, Noah Amsel, Berlin Chen, Tri Dao\
Blogpost: 

Achieve up to 2x faster Newton-Schulz with Gram Newton-Schulz and symmetric CuTeDSL GEMM kernels!

What you're probably here for:
1. Gram Newton-Schulz: https://github.com/Dao-AILab/gram-newton-schulz/blob/main/newton_schulz/gram_newton_schulz.py
2. Gram Newton-Schulz Restart Autotune: https://github.com/Dao-AILab/gram-newton-schulz/blob/main/stability/restart_autotune.py
3. Symmetric GEMMs for Hopper and Blackwell in CuTeDSL: https://github.com/Dao-AILab/quack/blob/main/quack/gemm_symmetric.py

**Gram Newton-Schulz**

> Input: $X \in \mathbb{R}^{n \times m}$ with $n \leq m$, coefficients $\{(a_t, b_t, c_t)\}_{t=1}^5$
> 1. $X \gets X \,/\, (\|X\|_{F} + \epsilon)$ &emsp; // Normalize sing vals to $[0, 1]$. &emsp; $\epsilon = 10^{-7}$
> 2. $X \gets \texttt{float16}(X)$ &emsp; // Cast to half precision for speed
> 3. If $m < n$: &emsp; $X \gets X^\top$ &emsp; // Trick to make $XX^\top$ cheaper
> 4. $R_0 \gets XX^\top$
> 5. $Q_0 \gets I$
> 6. For $t = 1, \ldots, 5$:
>    - If $t = 3$: &emsp; // Restart to stabilize
>      - $X \gets Q_2 X$
>      - $R_2 \gets XX^\top$
>      - $Q_2 \gets I$
>    - $Z_t \gets b_t R_{t-1} + c_t R_{t-1}^2$
>    - $Q_t \gets Q_{t-1} Z_t + a_t Q_{t-1}$
>    - $RZ_t \gets R_{t-1} Z_t + a_t R_{t-1}$
>    - $R_t \gets Z_t (RZ_t) + a_t (RZ_t)$
> 7. $X \gets Q_4 X$
> 8. If $m < n$: &emsp; $X \gets X^\top$ &emsp; // Undo trick
> 9. Return $X$

<img width="2799" height="1126" alt="kimi (2)" src="https://github.com/user-attachments/assets/861b3e7d-bae9-4f84-8a9c-5aa3396e02c8" />

## About

Gram Newton-Schulz is a hardware-aware algorithm for polar decomposition that is mathematically equivalent to and faster than Newton-Schulz. 
Polar decomposition is most commonly used in Muon, and Gram Newton-Schulz serves as a direct drop-in for standard Newton-Schulz with no training accuracy tradeoff.

## Installation

Requirements:
- NVIDIA Hopper (H100) or Blackwell (B200/B300) GPU
- PyTorch 2.7.1+
- CUDA 12.9+

Install PyTorch first, then:
```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required so that pip uses your existing CUDA-enabled PyTorch instead of installing torch-cpu in an isolated build environment.

This will install:
- gram-newton-schulz (this package)
- nvidia-cutlass-dsl 4.4.1
- quack-kernels @ git+https://github.com/Dao-AILab/quack.git@490a300b09981fe9565c82ff64d5448a6bc1bb7d

## Usage

### Gram Newton-Schulz

```
from newton_schulz.coefficients import POLAR_EXPRESS_COEFFICIENT

gram_NS = GramNewtonSchulz(
            ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
            gram_newton_schulz_reset_iterations=[2]
        )
result = gram_NS(X)
```

GramNewtonSchulz is a callable function that is initialized with `ns_coefficients` (List of List of floats) and a list of `gram_newton_schulz_reset_iterations` immediately after which to restart Gram Newton-Schulz's iterative loop (List of ints) for stability. For example, [2] means a restart occurs after the 2nd iteration and [2,4] means a restart occurs after the 2nd iteration and then after the 4th.

To find the best `num-restarts` restart location(s) for a set of coefficients, run
```bash
python stability/restart_autotune.py --num-restarts 1 --coefs "4.0848,-6.8946,2.9270;3.9505,-6.3029,2.6377;3.7418,-5.5913,2.3037;2.8769,-3.1427,1.2046;2.8366,-3.0525,1.2012"
```

For 5 steps of Newton-Schulz, we recommend `num-restarts = 1` for maximum speed while maintaining numerical stability. However, users who experience numerical instability or use more than 5 steps should consider using more restarts.

### Muon

The Muon class supports an auxiliary scalar optimizer that updates all non-Muon parameters, custom functions that split model weights for orthogonalization, and Gram Newton-Schulz with autotuned restart locations.

```
from newton_schulz.coefficients import YOU_COEFFICIENTS

qkv_params = []
regular_2d_params = []
scalar_params = []

for name, param in model.named_parameters():
    if 'qkv_weight' in name:
      qkv_params.append(param)
    elif param.ndim >= 2:
      regular_2d_params.append(param)
    else:
      scalar_params.append(param)

scalar_optimizer = AdamW(
    scalar_params,
    lr=1e-3,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

def qkv_split_fn(param: torch.Tensor):
    """
    Split Wqkv into [Wq, Wk, Wv].

    Assumes param has shape (3*hidden_dim, hidden_dim) where the first dimension
    is concatenated [Q, K, V] weights.
    """
    hidden_dim = param.size(1)
    Wq = param[:hidden_dim, :]
    Wk = param[hidden_dim:2*hidden_dim, :]
    Wv = param[2*hidden_dim:, :]
    return [Wq, Wk, Wv]

def qkv_recombine_fn(splits):
    """Recombine [Wq, Wk, Wv] back into Wqkv."""
    return torch.cat(splits, dim=0)

muon_param_groups = []

muon_param_groups.append({
    'params': qkv_params,
    'param_split_fn': qkv_split_fn,
    'param_recombine_fn': qkv_recombine_fn,
    'lr': 3e-3,
    'weight_decay': 0.1,
    'momentum': 0.95,
})

muon_param_groups.append({
    'params': regular_2d_params,
    'lr': 3e-3,
    'weight_decay': 0.1,
    'momentum': 0.95,
})

optimizer = Muon(
    params=muon_param_groups,
    scalar_optimizer=scalar_optimizer,
    lr=3e-3,
    weight_decay=0.1,
    momentum=0.95,
    nesterov=True,
    adjust_lr='rms_norm',
    ns_algorithm='gram_newton_schulz',
    ns_use_kernels=True,
    ns_coefficients=YOU_COEFFICIENTS,
    num_reset_iterations=1,
)
```

See `examples/example.py` for a full training example.

## Citation

If you use this codebase, or otherwise find our work valuable, please cite Gram Newton-Schulz:
```bibtex
@misc{GramNewtonSchulz,
  title   = {Gram Newton-Schulz},
  author  = {Jack Zhang and Noah Amsel and Berlin Chen and Tri Dao},
  year    = {2026},
  url = {TODO}
}
```
