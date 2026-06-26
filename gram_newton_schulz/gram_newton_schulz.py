from types import SimpleNamespace
from typing import Dict, List, Optional
import torch
from torch import Tensor
from .coefficients import POLAR_EXPRESS_COEFFICIENTS

SYMMETRIC_KERNEL_TILE_SIZE = 256


_TORCH_BACKEND = SimpleNamespace(
    sym_mm=lambda A, B: A @ B,
    sym_baddbmm=lambda A, B, C, alpha=1., beta=1.: torch.baddbmm(C, A, B, alpha=alpha, beta=beta),
    mm=lambda A, B: A @ B,
    mm_add=lambda A, B, C, beta: torch.baddbmm(C, A, B, beta=beta),
)


def _make_kernel_backend():
    from quack.gemm_interface import gemm_symmetric, gemm, gemm_add
    return SimpleNamespace(
        sym_mm=gemm_symmetric,
        sym_baddbmm=lambda A, B, C, alpha=1., beta=1.: gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta),
        mm=gemm,
        mm_add=lambda A, B, C, beta: gemm_add(A, B, C=C, beta=beta),
    )


def _make_compiled_gram(ops, ns_coefficients, gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs):
    """Build a compiled closure for gram Newton-Schulz with a fixed backend."""
    ns_coefficients = list(ns_coefficients)
    gram_newton_schulz_reset_iterations = set(gram_newton_schulz_reset_iterations)

    def _gram_newton_schulz(X: Tensor) -> Tensor:
        tall_skinny = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
        X = X.to(torch.float16)

        if tall_skinny:
            R = ops.sym_mm(X.mT, X)
        else:
            R = ops.sym_mm(X, X.mT)

        batch_size = R.size(0)
        I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        Q = None

        for i, (a, b, c) in enumerate(ns_coefficients):
            if i in gram_newton_schulz_reset_iterations and i != 0:
                if tall_skinny:
                    X = ops.mm(X, Q)
                    R = ops.sym_mm(X.mT, X)
                else:
                    X = ops.mm(Q, X)
                    R = ops.sym_mm(X, X.mT)
                Q = None

            Z = ops.sym_baddbmm(R, R, C=R, alpha=c, beta=b)
            if i == 0 or i in gram_newton_schulz_reset_iterations:
                Q = Z + a * I
            else:
                Q = ops.sym_baddbmm(Q, Z, C=Q, beta=a)
            if i < len(ns_coefficients) - 1 and i + 1 not in gram_newton_schulz_reset_iterations:
                RZ = ops.sym_baddbmm(R, Z, C=R, beta=a)
                R = ops.sym_baddbmm(Z, RZ, C=RZ, beta=a)

        if tall_skinny:
            X = ops.mm(X, Q)
        else:
            X = ops.mm(Q, X)
        return X

    if compile_kwargs is not None:
        _gram_newton_schulz = torch.compile(_gram_newton_schulz, **compile_kwargs)
    return _gram_newton_schulz


def _make_compiled_standard(ops, ns_coefficients, ns_epsilon, compile_kwargs):
    """Build a compiled closure for standard Newton-Schulz with a fixed backend."""
    ns_coefficients = list(ns_coefficients)

    def _standard_newton_schulz(X: Tensor) -> Tensor:
        tall_skinny = X.size(-2) > X.size(-1)
        X = X.to(torch.float32)
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + ns_epsilon)
        X = X.to(torch.float16)

        for a, b, c in ns_coefficients:
            if tall_skinny:
                A = ops.sym_mm(X.mT, X)
            else:
                A = ops.sym_mm(X, X.mT)
            B = ops.sym_baddbmm(A, A, C=A, alpha=c, beta=b)
            if tall_skinny:
                X = ops.mm_add(X, B, C=X, beta=a)
            else:
                X = ops.mm_add(B, X, C=X, beta=a)
        return X

    if compile_kwargs is not None:
        _standard_newton_schulz = torch.compile(_standard_newton_schulz, **compile_kwargs)
    return _standard_newton_schulz


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
            compile_kwargs: Keyword arguments forwarded to torch.compile for the inner Newton-Schulz functions.
                Defaults to {"fullgraph": True, "mode": "reduce-overhead"}. Pass None to disable compilation.
        """
        self.ns_epsilon = ns_epsilon
        self.ns_use_kernels = ns_use_kernels
        self.ns_coefficients = ns_coefficients if ns_coefficients is not None else POLAR_EXPRESS_COEFFICIENTS
        self.use_gram_newton_schulz = use_gram_newton_schulz
        if use_gram_newton_schulz:
            self.gram_newton_schulz_reset_iterations = gram_newton_schulz_reset_iterations if gram_newton_schulz_reset_iterations is not None else [2]

        kernel_backend = _make_kernel_backend() if self.ns_use_kernels else None

        # Build compiled closures for each (backend × algorithm) combination.
        # This avoids compiling bound methods, which torch.compile handles poorly.
        if use_gram_newton_schulz:
            self._gram_torch = _make_compiled_gram(
                _TORCH_BACKEND, self.ns_coefficients, self.gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs)
            if kernel_backend is not None:
                self._gram_kernel = _make_compiled_gram(
                    kernel_backend, self.ns_coefficients, self.gram_newton_schulz_reset_iterations, ns_epsilon, compile_kwargs)

        self._standard_torch = _make_compiled_standard(
            _TORCH_BACKEND, self.ns_coefficients, ns_epsilon, compile_kwargs)
        if kernel_backend is not None:
            self._standard_kernel = _make_compiled_standard(
                kernel_backend, self.ns_coefficients, ns_epsilon, compile_kwargs)

        self._kernel_backend = kernel_backend

    def _select_gram(self, X: Tensor) -> Tensor:
        if self._kernel_backend is not None and min(X.size(-2), X.size(-1)) >= SYMMETRIC_KERNEL_TILE_SIZE:
            return self._gram_kernel(X)
        return self._gram_torch(X)

    def _select_standard(self, X: Tensor) -> Tensor:
        if self._kernel_backend is not None and min(X.size(-2), X.size(-1)) >= SYMMETRIC_KERNEL_TILE_SIZE:
            return self._standard_kernel(X)
        return self._standard_torch(X)

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

        if self.use_gram_newton_schulz and max(X.shape[-2:]) > min(X.shape[-2:]):
            X = self._select_gram(X)
        else:
            X = self._select_standard(X)

        return X.to(original_dtype).view(original_shape)


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
