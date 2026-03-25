from typing import List
from collections import defaultdict
import math
import torch
from torch import Tensor

def adjust_lr_rms_norm(lr, param_shape):
    """
    Adjust learning rate for constant element-wise RMS norm.
    https://arxiv.org/abs/2502.16982
    """
    fan_out, fan_in = param_shape[-2:]
    adjusted_ratio = 0.2 * math.sqrt(max(fan_out, fan_in))
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr

def adjust_lr_spectral_norm(lr, param_shape):
    """
    Adjust from spectral norm 1 to RMS operator norm 1.
    https://arxiv.org/abs/2310.17813
    """
    fan_out, fan_in = param_shape[-2:]
    adjusted_lr = lr * math.sqrt(fan_out / fan_in)
    return adjusted_lr

@torch.compile(fullgraph=True)
def muon_update_pre_orthogonalize(
    G: List[Tensor],
    M: List[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> List[Tensor]:
    """
    Adapted from Dion: https://github.com/microsoft/dion/blob/main/dion/muon.py

    Update momentum with gradient and compute the input to orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().
    """
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U

def muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: Tensor,
    weight_decay: Tensor,
):
    """
    Adapted from Dion: https://github.com/microsoft/dion/blob/main/dion/muon.py

    Apply weight decay and weight update after orthogonalization.
    Inputs and outputs should be lists of regular Tensor, not DTensor.
    This is a separate function for compatibility with torch.compile().

    Note: U should already be scaled by adjusted learning rate before calling this function.
          Batches of X/U have varying lengths, which breaks torch.compile.
    """
    decay_factor = 1 - base_lr * weight_decay
    for x in X:
        x.mul_(decay_factor)

    for x, u in zip(X, U):
        x.sub_(u)

def create_param_batches(
    params: List[Tensor]
) -> List[List[Tensor]]:
    """
    Batch parameters into identical shape and dtype.
    """
    # Group parameters by shape and dtype
    groups = defaultdict(list)
    for p in params:
        groups[(p.shape, p.dtype)].append(p)

    batches = []
    for (shape, dtype), group in groups.items():
        group.sort(key=lambda p: p.data_ptr())
        batches.append(group)

    return batches

@torch._dynamo.disable
def get_or_initialize_muon_state(optimizer_state_dict, param: Tensor) -> dict:
    """
    Get optimizer state for the given parameter tensor,
    or lazy-initialize it if it doesn't exist.

    Args:
        optimizer_state_dict: The optimizer's state dict (optimizer.state)
        param: The parameter tensor

    Returns:
        State dict with 'momentum' buffer
    """
    state = optimizer_state_dict[param]
    if not state:
        state["momentum"] = torch.zeros_like(param)
    return state
