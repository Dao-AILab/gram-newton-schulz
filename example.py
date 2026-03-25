"""
Basic usage example for the Muon optimizer.

Shows how to:
1. Create a simple model with 2D parameters (for Muon) and 1D parameters (for scalar optimizer)
2. Set up parameter groups with custom split/recombine functions for QKV and SwiGLU
3. Integrate with AdamW for scalar parameters (embeddings, norms)
4. Use with an LR scheduler
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from gram_newton_schulz.muon import Muon
from gram_newton_schulz.coefficients import YOU_COEFFICIENTS

import torch._dynamo
torch._dynamo.config.cache_size_limit = 128 

# Example split/recombine functions for QKV attention weights
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


def swiglu_split_fn(param: torch.Tensor):
    """
    Split SwiGLU weight into [up, gate].

    Assumes INTERLEAVED layout [gate0, up0, gate1, up1, ...]
    """
    gate_matrix = param[::2, :]
    up_matrix = param[1::2, :]

    return [up_matrix, gate_matrix]


def swiglu_recombine_fn(splits):
    """Recombine [up, gate] back into SwiGLU weight."""
    up_matrix, gate_matrix = splits
    out_features = gate_matrix.size(0) + up_matrix.size(0)
    in_features = gate_matrix.size(1)
    result = torch.empty(out_features, in_features, dtype=gate_matrix.dtype, device=gate_matrix.device)
    result[::2, :] = gate_matrix
    result[1::2, :] = up_matrix
    return result


class SimpleTransformerLayer(nn.Module):
    """
    Simplified transformer layer for demonstration.
    Contains 2D params (matrices) and 1D params (norms, embeddings).
    """
    def __init__(self, hidden_dim=512, intermediate_dim=2048):
        super().__init__()

        # 2D parameters (will use Muon)
        self.qkv_weight = nn.Parameter(torch.randn(3 * hidden_dim, hidden_dim))  # Combined QKV
        self.out_proj = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.fc1_weight = nn.Parameter(torch.randn(2 * intermediate_dim, hidden_dim))  # SwiGLU
        self.fc2_weight = nn.Parameter(torch.randn(hidden_dim, intermediate_dim))

        # 1D parameters (will use scalar optimizer like AdamW)
        self.ln1_weight = nn.Parameter(torch.ones(hidden_dim))
        self.ln1_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.ln2_weight = nn.Parameter(torch.ones(hidden_dim))
        self.ln2_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        """
        Simple forward pass that actually uses the weights.
        x: (batch_size, seq_len, hidden_dim)
        """
        # Layer norm 1
        x = x * self.ln1_weight + self.ln1_bias

        # Attention
        qkv = x @ self.qkv_weight.mT
        q, k, v = qkv.chunk(3, dim=-1)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
        ).transpose(0, 1)

        x = x + (attn_output @ self.out_proj.mT)

        # Layer norm 2
        x = x * self.ln2_weight + self.ln2_bias

        # FFN with SwiGLU
        fc1_out = x @ self.fc1_weight.mT
        gate = fc1_out[:, :, ::2]  # Even indices
        up = fc1_out[:, :, 1::2]   # Odd indices
        swiglu_out = gate * torch.nn.functional.silu(up)  # SwiGLU activation
        x = x + (swiglu_out @ self.fc2_weight.mT)

        return x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = SimpleTransformerLayer(hidden_dim=512, intermediate_dim=2048).to(device)

    # Separate parameters into param groups
    qkv_params = []
    swiglu_params = []
    regular_2d_params = []
    scalar_params = []

    for name, param in model.named_parameters():
        if 'qkv_weight' in name:
            qkv_params.append(param)
        elif 'fc1_weight' in name:
            swiglu_params.append(param)
        elif param.ndim >= 2:
            regular_2d_params.append(param)
        else:
            scalar_params.append(param)

    # Create scalar optimizer for 1D parameters (embeddings, norms)
    scalar_optimizer = AdamW(
        scalar_params,
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    ) if scalar_params else None

    # Create Muon optimizer with parameter groups
    muon_param_groups = []

    # Group 1: QKV weights with custom split/recombine
    if qkv_params:
        muon_param_groups.append({
            'params': qkv_params,
            'param_split_fn': qkv_split_fn,
            'param_recombine_fn': qkv_recombine_fn,
            'lr': 3e-3,
            'weight_decay': 0.1,
            'momentum': 0.95,
        })

    # Group 2: SwiGLU weights with custom split/recombine
    if swiglu_params:
        muon_param_groups.append({
            'params': swiglu_params,
            'param_split_fn': swiglu_split_fn,
            'param_recombine_fn': swiglu_recombine_fn,
            'lr': 3e-3,
            'weight_decay': 0.1,
            'momentum': 0.95,
        })

    # Group 3: Regular 2D parameters (no splitting)
    if regular_2d_params:
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
        gram_newton_schulz_num_restarts=1,
    )

    # Optional: Add LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)

    print("Starting training...")

    batch_size, seq_len, hidden_dim = 4, 64, 512
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device) * 0.01
    target = torch.ones(batch_size, seq_len, hidden_dim, device=device) * 0.01

    for step in range(10):
        output = model(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Step {step}: loss={loss.item():.6f}, lr={current_lr:.6f}")

    print("\nTraining complete!")
    muon_groups = len(optimizer._muon_param_groups)
    adamw_groups = len(optimizer.scalar_optimizer.param_groups) if optimizer.scalar_optimizer else 0
    print(f"Muon groups: {muon_groups} (QKV, SwiGLU, Regular 2D)")
    print(f"AdamW groups: {adamw_groups} (Layer norms)")


if __name__ == '__main__':
    main()
