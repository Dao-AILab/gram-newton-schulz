from typing import List, Optional, Tuple, Callable, Dict
from collections import defaultdict
import torch
from torch import Tensor

def get_newton_schulz_inputs_from_gradients(
    ns_inputs_for_rank: List[Tensor],
    param_split_fn: Optional[Callable],
) -> Tuple[Dict[Tuple[int, ...], List[Tensor]], List[Tuple], Dict]:
    """
    Apply 3D -> 2D splitting and custom parameter splitting if provided (e.g., for QKV, SwiGLU).

    Args:
        ns_inputs_for_rank: List of gradient tensors for this rank (2D or 3D)
        param_split_fn: Optional function to split parameters

    Returns:
        Tuple of:
            - ns_inputs_by_shape: Dict mapping shape -> list of matrices with that shape
            - shape_indices: List of (shape, index_within_shape_group) tuples for reconstruction
            - metadata: Dict with 'is_3d', 'tensor_dim_0_if_3d', 'num_submatrices_per_input'
    """
    num_submatrices_per_input = None
    is_3d = ns_inputs_for_rank[0].ndim == 3
    tensor_dim_0_if_3d = ns_inputs_for_rank[0].shape[0] if is_3d else None

    if param_split_fn is not None:
        # Apply custom split function and 3D -> 2D splitting
        split_ns_inputs = []
        for idx, ns_input in enumerate(ns_inputs_for_rank):
            sub_matrices = param_split_fn(ns_input)
            if idx == 0:
                validate_param_split_fn(param_split_fn, ns_input, sub_matrices)
                num_submatrices_per_input = len(sub_matrices) * tensor_dim_0_if_3d if is_3d else len(sub_matrices)
            split_ns_inputs.extend([sub_matrix[i] for sub_matrix in sub_matrices for i in range(sub_matrix.shape[0])] if is_3d else sub_matrices)
        ns_inputs_for_rank = split_ns_inputs
    elif is_3d:
        # Apply 3D -> 2D splitting without custom splitting
        split_ns_inputs = []
        for ns_input in ns_inputs_for_rank:
            sub_matrices = [ns_input[i] for i in range(tensor_dim_0_if_3d)]
            split_ns_inputs.extend(sub_matrices)
        ns_inputs_for_rank = split_ns_inputs

    # Group by shape and track original order for reconstruction
    ns_inputs_by_shape = defaultdict(list)
    shape_indices = []
    for matrix in ns_inputs_for_rank:
        cur_shape = matrix.shape
        shape_indices.append((cur_shape, len(ns_inputs_by_shape[cur_shape])))
        ns_inputs_by_shape[cur_shape].append(matrix)

    # Store metadata for reconstruction
    metadata = {
        'is_3d': is_3d,
        'tensor_dim_0_if_3d': tensor_dim_0_if_3d,
        'num_submatrices_per_input': num_submatrices_per_input,
    }

    return ns_inputs_by_shape, shape_indices, metadata

def scale_newton_schulz_outputs_with_adjusted_lr(
    orthogonalized_by_shape: Dict[Tuple[int, ...], Tensor],
    lr: Tensor,
    adjust_lr_fn: Optional[Callable],
) -> Dict[Tuple[int, ...], Tensor]:
    """
    Apply learning rate adjustment to each orthogonalized split section based on the split section's shape.
    If there is no adjust_lr_fn, scale by base LR.

    Args:
        orthogonalized_by_shape: Dict mapping shape -> batched tensor (batch, M, N)
        lr: Base learning rate
        adjust_lr_fn: Optional function to compute adjusted LR per split section shape

    Returns:
        Dict with each batched tensor scaled by its shape-specific adjusted learning rate
    """
    return {
        shape: batched_tensor.mul_(lr if adjust_lr_fn is None else adjust_lr_fn(lr, shape))
        for shape, batched_tensor in orthogonalized_by_shape.items()
    }

def reconstruct_update_from_newton_schulz_outputs(
    orthogonalized_by_shape: Dict[Tuple[int, ...], Tensor],
    shape_indices: List[Tuple],
    metadata: Dict,
    param_recombine_fn: Optional[Callable],
) -> List[Tensor]:
    """
    Reconstruct and recombine orthogonalized matrices to original gradient shape.

    Args:
        orthogonalized_by_shape: Dict mapping shape -> batched tensor (batch, M, N)
        shape_indices: List of (shape, index_within_shape_group) tuples for reconstruction
        metadata: Dict with 'is_3d', 'tensor_dim_0_if_3d', 'num_submatrices_per_input'
        param_recombine_fn: Optional function to recombine split parameters

    Returns:
        List of recombined tensors in original order
    """
    is_3d = metadata['is_3d']
    tensor_dim_0_if_3d = metadata['tensor_dim_0_if_3d']
    num_submatrices_per_input = metadata['num_submatrices_per_input']

    orthogonalized_submatrices = [orthogonalized_by_shape[shape][idx] for shape, idx in shape_indices]

    if param_recombine_fn is not None:
        orthogonalized = [
            param_recombine_fn(
                [torch.stack(orthogonalized_submatrices[i + j:i + j + tensor_dim_0_if_3d])
                 for j in range(0, num_submatrices_per_input, tensor_dim_0_if_3d)]
                if is_3d else orthogonalized_submatrices[i:i + num_submatrices_per_input]
            )
            for i in range(0, len(orthogonalized_submatrices), num_submatrices_per_input)
        ]
    elif is_3d:
        orthogonalized = [
            torch.stack(orthogonalized_submatrices[i:i + tensor_dim_0_if_3d])
            for i in range(0, len(orthogonalized_submatrices), tensor_dim_0_if_3d)
        ]
    else:
        orthogonalized = orthogonalized_submatrices

    return orthogonalized

@torch.compiler.disable
def validate_param_split_fn(
    param_split_fn: Callable,
    ns_input: Tensor,
    sub_matrices: List[Tensor],
) -> None:
    fn_name = getattr(param_split_fn, '__name__', repr(param_split_fn))

    assert all(sub.ndim == ns_input.ndim for sub in sub_matrices), \
        f"param_split_fn ({fn_name}) must preserve ndim. Input: {ns_input.ndim}D, Output: {[sub.ndim for sub in sub_matrices]}"

    # For 3D tensors, enforce that only last 2 dims can be split (first dim must be preserved)
    if ns_input.ndim == 3:
        tensor_dim_0_if_3d = ns_input.shape[0]
        assert all(sub.shape[0] == tensor_dim_0_if_3d for sub in sub_matrices), \
            f"param_split_fn ({fn_name}) for 3D tensors must preserve first dimension (number of batched 2D tensors, e.g. number of experts). " \
            f"Input shape: {ns_input.shape} (dim_0={tensor_dim_0_if_3d}), " \
            f"Output shapes: {[sub.shape for sub in sub_matrices]}. " \
            f"All outputs must have shape[0] == {tensor_dim_0_if_3d}."
    
    return