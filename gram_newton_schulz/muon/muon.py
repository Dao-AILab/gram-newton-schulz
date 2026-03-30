from typing import List, Tuple, Union, Optional, Callable
import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from .muon_utils.muon_opt_utils import (
    adjust_lr_rms_norm,
    adjust_lr_spectral_norm,
    muon_update_pre_orthogonalize,
    muon_update_post_orthogonalize,
    create_param_batches,
    get_or_initialize_muon_state,
)
from .muon_utils.muon_matrix_split_utils import (
    get_newton_schulz_inputs_from_gradients,
    scale_newton_schulz_outputs_with_adjusted_lr,
    reconstruct_update_from_newton_schulz_outputs,
)
from ..gram_newton_schulz import GramNewtonSchulz, StandardNewtonSchulz
from ..coefficients import POLAR_EXPRESS_COEFFICIENTS, YOU_COEFFICIENTS
from ..restart_autotune import find_best_restarts

class Muon(Optimizer):
    """
    Fast Muon implementation for Gram Newton-Schulz and standard Newton-Schulz.
    Supports:
        - Custom Hopper and Blackwell Symmetric GEMM and standard GEMM kernels for accelerated Newton-Schulz
        - Auxiliary scalar optimizer for non-Muon weight updates, supporting LR scheduling.
        - Custom NS coefficients, with default POLAR_EXPRESS_COEFFICIENTS from newton-schulz/coefficients.py
        - Custom weight splitting logic via lambda functions during preprocessing before Newton-Schulz
        - Custom Muon LR adjustment function
        - Single GPU training only

    Args:
        params: Parameter groups. Each group can specify:
            - param_split_fn: Function to split a parameter into submatrices before orthogonalization (e.g., split Wqkv into Wq, Wk, Wv)
            - param_recombine_fn: Function to recombine submatrices after orthogonalization back into original parameter shape
            - 3D weights are by default treated as batched 2D weights, with the first dimension being the batch dimension
            - See example.py for example usage
        lr: Learning rate (default: 1e-3)
        weight_decay: Weight decay coefficient (default: 0.1)
        momentum: Momentum factor (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        adjust_lr: Learning rate adjustment method. Options:
            - "rms_norm": Scale by sqrt(max(fan_out, fan_in)) for constant element-wise RMS norm
            - "spectral_norm": Scale from spectral norm 1 to RMS operator norm 1
            - Callable: Custom function taking (lr, param_shape) -> adjusted_lr
            - None: No adjustment
            (default: "rms_norm")
        ns_coefficients: List of 3-coefficient tuples for each Newton-Schulz iteration.
            Each tuple contains [a, b, c] coefficients for the iteration formula.
            If ns_coefficients_preset is provided, this parameter is ignored.
            (default: POLAR_EXPRESS_COEFFICIENTS from newton-schulz/coefficients.py)
        ns_coefficients_preset: Select a coefficient preset by name. Options:
            - "POLAR_EXPRESS_COEFFICIENTS": Polar Express coefficients with /1.01 scaling
            - "YOU_COEFFICIENTS": YOU coefficients
            - None: Use ns_coefficients parameter or default
            (default: None)
        ns_algorithm: Newton-Schulz algorithm variant. Options:
            - "gram_newton_schulz": Gram Newton-Schulz iteration with optional resets
            - "standard_newton_schulz": Standard Newton-Schulz iteration
            (default: "gram_newton_schulz")
        ns_epsilon: Epsilon for Frobenius normalization before orthogonalization (default: 1e-7)
        ns_use_kernels: Use custom CUDA kernels if available (requires compute capability 9.0+) (default: True)
        gram_newton_schulz_num_restarts: Number of restarts for Gram Newton-Schulz. Restart positions are automatically tuned during initialization. Ignored if gram_newton_schulz_restart_iterations is provided. (default: 1)
        gram_newton_schulz_restart_iterations: Manual restart positions for Gram Newton-Schulz. If "2" is an entry, the user wants a restart after the 2nd iteration. If provided, auto-tuning is skipped. (default: None)
        scalar_optimizer: Optional secondary optimizer for non-matrix parameters (default: None)

    Example: examples/example.py
    """
    def __init__(
        self,
        # Muon
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        adjust_lr: Union[str, Callable[[float, Tuple[int, ...]], float], None] = "rms_norm",
        # Newton-Schulz
        ns_coefficients: List[Union[Tuple[float, float, float], List[float]]] = None,
        ns_coefficients_preset: Optional[str] = None,
        ns_algorithm: str = "gram_newton_schulz",
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        gram_newton_schulz_num_restarts: int = 1,
        gram_newton_schulz_restart_iterations: Optional[Union[List[int], Tuple[int, ...]]] = None,
        # Scalar optimizer
        scalar_optimizer: Optional[Optimizer] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Learning rate must be positive: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay can't be negative: {weight_decay}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Momentum must be in [0, 1): {momentum}")
        if ns_epsilon <= 0.0:
            raise ValueError(f"Newton-Schulz epsilon for normalization must be positive: {ns_epsilon}")

        if ns_coefficients_preset is not None:
            preset_map = {
                "POLAR_EXPRESS_COEFFICIENTS": POLAR_EXPRESS_COEFFICIENTS,
                "YOU_COEFFICIENTS": YOU_COEFFICIENTS,
            }
            if ns_coefficients_preset not in preset_map:
                raise ValueError(f"Invalid ns_coefficients_preset: {ns_coefficients_preset}. Must be one of: {list(preset_map.keys())}")
            ns_coefficients = preset_map[ns_coefficients_preset]
        elif ns_coefficients is None:
            ns_coefficients = POLAR_EXPRESS_COEFFICIENTS

        if ns_algorithm not in ("gram_newton_schulz", "standard_newton_schulz"):
            raise ValueError(
                f"Invalid ns_algorithm: {ns_algorithm}. Must be 'gram_newton_schulz' or 'standard_newton_schulz'."
            )

        if not isinstance(gram_newton_schulz_num_restarts, int) or gram_newton_schulz_num_restarts < 0:
            raise ValueError(f"gram_newton_schulz_num_restarts must be a non-negative integer, got {gram_newton_schulz_num_restarts}")

        ns_coefficients = [list(coef) if hasattr(coef, '__iter__') and not isinstance(coef, str) else coef
                          for coef in ns_coefficients]

        for i, coef in enumerate(ns_coefficients):
            if len(coef) != 3:
                raise ValueError(
                    f"Each iteration must have exactly 3 Newton-Schulz coefficients, got {len(coef)} at iteration {i}"
                )

        self.ns_coefficients = ns_coefficients
        self.ns_algorithm = ns_algorithm
        self.ns_epsilon = ns_epsilon

        if gram_newton_schulz_restart_iterations is not None:
            self.gram_newton_schulz_reset_iterations = gram_newton_schulz_restart_iterations
        elif ns_algorithm == "gram_newton_schulz" and gram_newton_schulz_num_restarts > 0:
            x_eigenvalues = np.logspace(0, -10, 10000)
            most_negative_gram_eigenvalue = -4e-4

            print(f"Auto-tuning {gram_newton_schulz_num_restarts} restart position(s) for Gram Newton-Schulz...")
            self.gram_newton_schulz_reset_iterations = find_best_restarts(
                x_eigenvalues, ns_coefficients, most_negative_gram_eigenvalue, num_restarts=gram_newton_schulz_num_restarts, high_precision=False
            )
            print(f"Selected restart positions: {self.gram_newton_schulz_reset_iterations}")
        else:
            self.gram_newton_schulz_reset_iterations = []

        if ns_use_kernels and torch.cuda.is_available():
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            compute_capability = capability[0] * 10 + capability[1]

            if compute_capability < 90:
                print(f"Warning: Custom kernels require compute capability 9.0+ (H100/B200), but found {capability[0]}.{capability[1]}. "
                      f"Falling back to PyTorch operations (ns_use_kernels=False).")
                self.ns_use_kernels = False
            else:
                self.ns_use_kernels = ns_use_kernels
        else:
            self.ns_use_kernels = ns_use_kernels


        self.scalar_optimizer = scalar_optimizer
        self._muon_param_groups = None
        self._combined_param_groups = None  # Combined list of muon + scalar param groups, to be exposed for e.g. LR schedulers

        if self.scalar_optimizer is not None:
            @torch.compile(fullgraph=False)
            def _compiled_scalar_step():
                self.scalar_optimizer.step()

            self._compiled_scalar_step = _compiled_scalar_step
        else:
            self._compiled_scalar_step = None

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults)
        if self.scalar_optimizer is not None:
            self._combined_param_groups = self._muon_param_groups + self.scalar_optimizer.param_groups
        else:
            self._combined_param_groups = self._muon_param_groups

        if self.ns_algorithm == "gram_newton_schulz":
            self.newton_schulz = GramNewtonSchulz(
                ns_epsilon=self.ns_epsilon,
                ns_use_kernels=self.ns_use_kernels,
                ns_coefficients=self.ns_coefficients,
                gram_newton_schulz_reset_iterations=self.gram_newton_schulz_reset_iterations,
            )
        elif self.ns_algorithm == "standard_newton_schulz":
            self.newton_schulz = StandardNewtonSchulz(
                ns_epsilon=self.ns_epsilon,
                ns_use_kernels=self.ns_use_kernels,
                ns_coefficients=self.ns_coefficients,
            )
        else:
            raise ValueError(f"Invalid ns_algorithm: {self.ns_algorithm}. Must be 'gram_newton_schulz' or 'standard_newton_schulz'.")

    @property
    def param_groups(self):
        """
        Expose both muon and scalar optimizer param groups for LR schedulers.
        LR's for both optimizers will be scheduled equivalently.
        """
        # During __init__, combined list might not be built yet - return _muon_param_groups
        if self._combined_param_groups is None:
            return self._muon_param_groups if self._muon_param_groups is not None else []
        return self._combined_param_groups

    @param_groups.setter
    def param_groups(self, value):
        """
        Set param_groups during initialization. Called by PyTorch's Optimizer.__init__.
        After init, schedulers modify param group dicts in-place, so this setter is not called.
        """
        self._muon_param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._muon_step(self._muon_param_groups)

        if self._compiled_scalar_step is not None:
            self._compiled_scalar_step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients for both Muon parameters and scalar optimizer parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero
        """
        super().zero_grad(set_to_none=set_to_none)

        if self.scalar_optimizer is not None:
            self.scalar_optimizer.zero_grad(set_to_none=set_to_none)

    def _muon_step(self, param_groups: List[dict]):
        for group in param_groups:
            assert all(
                p.ndim >= 2 for p in group["params"]
            ), "Muon optimizer only supports 2D matrix parameters or 3D parameters that will be treated as a batch of 2D matrices."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            adjust_lr = group["adjust_lr"]
            if isinstance(adjust_lr, str):
                if adjust_lr == "rms_norm":
                    adjust_lr_fn = adjust_lr_rms_norm
                elif adjust_lr == "spectral_norm":
                    adjust_lr_fn = adjust_lr_spectral_norm
                else:
                    raise ValueError(
                        f"Invalid adjust_lr: {adjust_lr}. Must be 'rms_norm', 'spectral_norm', or a callable."
                    )
            elif callable(adjust_lr) or adjust_lr is None:
                adjust_lr_fn = adjust_lr
            else:
                raise TypeError(f"Invalid adjust_lr type: {type(adjust_lr)}")

            muon_batch_update_args = dict(
                lr=group["lr"],
                momentum=group["momentum"],
                weight_decay=group["weight_decay"],
                nesterov=group["nesterov"],
                adjust_lr_fn=adjust_lr_fn,
                param_split_fn=group.get("param_split_fn", None),
                param_recombine_fn=group.get("param_recombine_fn", None),
            )

            for params in create_param_batches(group_params):
                gradients = [p.grad for p in params]
                states = [get_or_initialize_muon_state(self.state, p) for p in params]
                momentums = [s["momentum"] for s in states]
                self._muon_batch_update(params, gradients, momentums, **muon_batch_update_args)

    def _muon_batch_update(
        self,
        params: List[Tensor],
        gradients: List[Tensor],
        momentums: List[Tensor],
        lr: Tensor,
        momentum: Tensor,
        weight_decay: Tensor,
        nesterov: bool,
        adjust_lr_fn: Optional[Callable],
        param_split_fn: Optional[Callable],
        param_recombine_fn: Optional[Callable],
    ):
        assert len(params) == len(gradients) == len(momentums), "Number of parameters, gradients, and momentums for Muon must match"

        # Validate split/recombine are both provided or both None
        if (param_split_fn is None) != (param_recombine_fn is None):
            raise ValueError("param_split_fn and param_recombine_fn must both be provided or both be None")

        # Update momentum and compute the inputs for orthogonalization
        ns_inputs = muon_update_pre_orthogonalize(
            G=gradients,
            M=momentums,
            momentum=momentum,
            nesterov=nesterov,
        )

        if len(ns_inputs) > 0:
            # Split and group gradient submatrices for orthogonalization
            ns_inputs_by_shape, shape_indices, split_metadata = get_newton_schulz_inputs_from_gradients(
                ns_inputs, param_split_fn
            )

            # Orthogonalize each submatrix shape group
            orthogonalized_by_shape = {}
            for shape, ns_inputs_for_shape in ns_inputs_by_shape.items():
                batched_input = torch.stack(ns_inputs_for_shape, dim=0)

                orthogonalized_batched = self.newton_schulz(batched_input)
                orthogonalized_by_shape[shape] = orthogonalized_batched.clone()

            # Apply LR to each split section based on its shape
            orthogonalized_by_shape = scale_newton_schulz_outputs_with_adjusted_lr(
                orthogonalized_by_shape, lr, adjust_lr_fn
            )

            # Reconstruct orthogonalized matrices
            orthogonalized = reconstruct_update_from_newton_schulz_outputs(
                orthogonalized_by_shape, shape_indices, split_metadata, param_recombine_fn
            )
        else:
            orthogonalized = []

        # Apply weight decay and update parameters
        muon_update_post_orthogonalize(params, orthogonalized, lr, weight_decay)
    
  
