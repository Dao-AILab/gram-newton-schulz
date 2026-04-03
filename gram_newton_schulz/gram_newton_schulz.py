from types import SimpleNamespace
from typing import Dict, List, Optional
import torch
from torch import Tensor
from .coefficients import POLAR_EXPRESS_COEFFICIENTS

SYMMETRIC_KERNEL_TILE_SIZE = 256


_TORCH_BACKEND = SimpleNamespace(
    sym_mm=lambda A, B: A @ B,
    sym_baddbmm=lambda A, B, C, alpha=1, beta=1: torch.baddbmm(C, A, B, alpha=alpha, beta=beta),
    mm=lambda A, B: A @ B,
    mm_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)


def _make_kernel_backend():
    from quack.gemm_interface import gemm_symmetric, gemm, gemm_add
    return SimpleNamespace(
        sym_mm=gemm_symmetric,
        sym_baddbmm=lambda A, B, C, alpha=1, beta=1: gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta),
        mm=gemm,
        mm_add=lambda A, B, C, beta: gemm_add(A, B, C=C, beta=beta),
    )


class GramNewtonSchulz:
    """
    Gram Newton-Schulz orthogonalization.

    Example:
        from newton_schulz.coefficients import POLAR_EXPRESS_COEFFICIENTS
        gram_NS = GramNewtonSchulz(
            ns_coefficients=POLAR_EXPRESS_COEFFICIENTS,
            gram_newton_schulz_reset_iterations=[2]
        )
        result = gram_NS(X)
    """

    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Optional[List[List[float]]] = None,
        use_gram_newton_schulz: bool = True,
        gram_newton_schulz_reset_iterations: List[int] = None,
        compile_kwargs: Optional[Dict] = {"fullgraph": True, "mode": "reduce-overhead"},
    ):
        """
        Initialize GramNewtonSchulz orthogonalizer.

        Args:
            ns_epsilon: Epsilon for normalization
            ns_use_kernels: Whether to use custom CuTeDSL kernels
            ns_coefficients: Coefficients for each iteration. Defaults to POLAR_EXPRESS_COEFFICIENTS.
            gram_newton_schulz_reset_iterations: Iterations at which to reset. Defaults to [2].
            compile_kwargs: Keyword arguments forwarded to torch.compile for __call__.
                Defaults to {"fullgraph": True, "mode": "reduce-overhead"}. Pass None to disable compilation.
        """
        self.ns_epsilon = ns_epsilon
        self.ns_use_kernels = ns_use_kernels
        self.ns_coefficients = ns_coefficients if ns_coefficients is not None else POLAR_EXPRESS_COEFFICIENTS
        self.use_gram_newton_schulz = use_gram_newton_schulz
        if use_gram_newton_schulz:
            self.gram_newton_schulz_reset_iterations = gram_newton_schulz_reset_iterations if gram_newton_schulz_reset_iterations is not None else [2]

        self._kernel_backend = _make_kernel_backend() if self.ns_use_kernels else None

        if compile_kwargs is not None:
            self.__call__ = torch.compile(self.__call__, **compile_kwargs)

    def _select_backend(self, X: Tensor):
        if self._kernel_backend is not None and min(X.size(-2), X.size(-1)) > SYMMETRIC_KERNEL_TILE_SIZE:
            return self._kernel_backend
        return _TORCH_BACKEND

    def __call__(self, X: Tensor) -> Tensor:
        """
        Orthogonalize a batch of matrices using Gram Newton-Schulz iteration.

        Args:
            X: Input tensor of shape (batch, M, N) or (M, N)
               Will be treated as a batch of 2D matrices

        Returns:
            Orthogonalized tensor with same shape as input
        """
        original_shape = X.shape
        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.view(-1, *X.shape[-2:])

        original_dtype = X.dtype
        X = X.to(torch.float32)

        if should_transpose := (X.size(-2) > X.size(-1)):
            X = X.mT

        X /= X.norm(dim=(-2, -1), keepdim=True) + self.ns_epsilon
        X = X.to(torch.float16)

        if self.use_gram_newton_schulz and max(X.shape[-2:]) > min(X.shape[-2:]):
            X = self._gram_newton_schulz(X)
        else:
            X = self._standard_newton_schulz(X)

        if should_transpose:
            X = X.mT

        return X.to(original_dtype).view(original_shape)

    def _gram_newton_schulz(self, X: Tensor) -> Tensor:
        ops = self._select_backend(X)
        R = ops.sym_mm(X, X.mT)

        batch_size = R.size(0)
        I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        Q = None

        for i, (a, b, c) in enumerate(self.ns_coefficients):
            if i in self.gram_newton_schulz_reset_iterations and i != 0:
                X = ops.mm(Q, X)
                R = ops.sym_mm(X, X.mT)
                Q = None

            Z = ops.sym_baddbmm(R, R, C=R, alpha=c, beta=b)
            if i == 0 or i in self.gram_newton_schulz_reset_iterations:
                Q = Z + a * I
            else:
                Q = ops.sym_baddbmm(Q, Z, C=Q, beta=a)
            if i < len(self.ns_coefficients) - 1 and i + 1 not in self.gram_newton_schulz_reset_iterations:
                RZ = ops.sym_baddbmm(R, Z, C=R, beta=a)
                R = ops.sym_baddbmm(Z, RZ, C=RZ, beta=a)

        X = ops.mm(Q, X)

        return X

    def _standard_newton_schulz(self, X: Tensor) -> Tensor:
        ops = self._select_backend(X)
        for a, b, c in self.ns_coefficients:
            A = ops.sym_mm(X, X.mT)
            B = ops.sym_baddbmm(A, A, C=A, alpha=c, beta=b)
            X = ops.mm_add(B, X, C=X, beta=a)

        return X


class StandardNewtonSchulz(GramNewtonSchulz):
    """
    Standard Newton-Schulz orthogonalization.

    Equivalent to GramNewtonSchulz with use_gram_newton_schulz=False.

    Example:
        from gram_newton_schulz import StandardNewtonSchulz, POLAR_EXPRESS_COEFFICIENTS
        standard_NS = StandardNewtonSchulz(ns_coefficients=POLAR_EXPRESS_COEFFICIENTS)
        result = standard_NS(X)
    """

    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Optional[List[List[float]]] = None,
        compile_kwargs: Optional[Dict] = {"fullgraph": True, "mode": "reduce-overhead"},
    ):
        super().__init__(
            ns_epsilon=ns_epsilon,
            ns_use_kernels=ns_use_kernels,
            ns_coefficients=ns_coefficients,
            use_gram_newton_schulz=False,
            compile_kwargs=compile_kwargs,
        )