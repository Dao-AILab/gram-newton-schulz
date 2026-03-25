from typing import List, Optional
import torch
from torch import Tensor
from .coefficients import POLAR_EXPRESS_COEFFICIENTS

SYMMETRIC_KERNEL_TILE_SIZE = 256

class StandardNewtonSchulz:
    """
    Standard Newton-Schulz orthogonalization.

    Example:
        from newton_schulz.coefficients import POLAR_EXPRESS_COEFFICIENTS
        standard_NS = StandardNewtonSchulz(ns_coefficients=POLAR_EXPRESS_COEFFICIENTS)
        result = standard_NS(X)
    """

    def __init__(
        self,
        ns_epsilon: float = 1e-7,
        ns_use_kernels: bool = True,
        ns_coefficients: Optional[List[List[float]]] = None,
    ):
        """
        Initialize StandardNewtonSchulz orthogonalizer.

        Args:
            ns_epsilon: Epsilon for normalization
            ns_use_kernels: Whether to use custom CuTeDSL kernels
            ns_coefficients: Coefficients for each iteration. Defaults to POLAR_EXPRESS_COEFFICIENTS.
        """
        self.ns_epsilon = ns_epsilon
        self.ns_use_kernels = ns_use_kernels
        self.ns_coefficients = ns_coefficients if ns_coefficients is not None else POLAR_EXPRESS_COEFFICIENTS

        if self.ns_use_kernels:
            from quack.gemm_interface import gemm_symmetric, gemm_add
            self.gemm_symmetric = gemm_symmetric
            self.gemm_add = gemm_add
        else:
            self.gemm_symmetric = None
            self.gemm_add = None

    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def __call__(
        self,
        X: Tensor,
    ) -> Tensor:
        """
        Orthogonalize a batch of matrices using standard Newton-Schulz iteration.

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

        dtype = X.dtype
        X = X.to(torch.float32)

        if should_transpose := X.size(-2) > X.size(-1):
            X = X.mT

        X /= X.norm(dim=(-2, -1), keepdim=True) + self.ns_epsilon
        X = X.to(torch.float16)

        for i, (a, b, c) in enumerate(self.ns_coefficients):
            if not self.ns_use_kernels or X.size(-2) <= SYMMETRIC_KERNEL_TILE_SIZE:
                A = X @ X.mT
                B = torch.baddbmm(A, A, A, beta=b, alpha=c)
                X = torch.baddbmm(X, B, X, beta=a)
            else:
                A = self.gemm_symmetric(X, X.mT)
                B = self.gemm_symmetric(A, A, C=A, alpha=c, beta=b)
                X = self.gemm_add(B, X, C=X, beta=a)

        if should_transpose:
            X = X.mT
        X = X.to(dtype)

        return X.view(original_shape)
