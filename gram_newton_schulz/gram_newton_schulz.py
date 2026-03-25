from typing import List, Optional
import torch
from torch import Tensor
from .coefficients import POLAR_EXPRESS_COEFFICIENTS

SYMMETRIC_KERNEL_TILE_SIZE = 256

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
        gram_newton_schulz_reset_iterations: List[int] = None,
    ):
        """
        Initialize GramNewtonSchulz orthogonalizer.

        Args:
            ns_epsilon: Epsilon for normalization
            ns_use_kernels: Whether to use custom CuTeDSL kernels
            ns_coefficients: Coefficients for each iteration. Defaults to POLAR_EXPRESS_COEFFICIENTS.
            gram_newton_schulz_reset_iterations: Iterations at which to reset. Defaults to [2].
        """
        self.ns_epsilon = ns_epsilon
        self.ns_use_kernels = ns_use_kernels
        self.ns_coefficients = ns_coefficients if ns_coefficients is not None else POLAR_EXPRESS_COEFFICIENTS
        self.gram_newton_schulz_reset_iterations = gram_newton_schulz_reset_iterations if gram_newton_schulz_reset_iterations is not None else [2]

        if self.ns_use_kernels:
            from quack.gemm_interface import gemm_symmetric, gemm, gemm_add
            self.gemm_symmetric = gemm_symmetric
            self.gemm = gemm
            self.gemm_add = gemm_add
        else:
            self.gemm_symmetric = None
            self.gemm = None
            self.gemm_add = None

    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def __call__(
        self,
        X: Tensor,
    ) -> Tensor:
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

        dtype, device = X.dtype, X.device
        X = X.to(torch.float32)

        if should_transpose := X.size(-2) > X.size(-1):
            X = X.mT

        X /= X.norm(dim=(-2, -1), keepdim=True) + self.ns_epsilon
        X = X.to(torch.float16)

        if X.size(-2) != X.size(-1):
            if not self.ns_use_kernels or X.size(-2) <= SYMMETRIC_KERNEL_TILE_SIZE:
                R = X @ X.mT
            else:
                R = self.gemm_symmetric(X, X.mT)

            batch_size = R.size(0)
            I = torch.eye(R.size(-1), device=device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            Q = None

            for i, (a, b, c) in enumerate(self.ns_coefficients):
                if i in self.gram_newton_schulz_reset_iterations and i != 0:
                    if not self.ns_use_kernels or X.size(-2) <= SYMMETRIC_KERNEL_TILE_SIZE:
                        X = Q @ X
                        R = X @ X.mT
                    else:
                        X = self.gemm(Q, X)
                        R = self.gemm_symmetric(X, X.mT)
                    Q = None

                if not self.ns_use_kernels or X.size(-2) <= SYMMETRIC_KERNEL_TILE_SIZE:
                    Z = torch.baddbmm(R, R, R, beta=b, alpha=c)
                    Q = torch.baddbmm(Q, Q, Z, beta=a) if i != 0 and i not in self.gram_newton_schulz_reset_iterations else Z + a * I
                    if i < len(self.ns_coefficients) - 1 and i + 1 not in self.gram_newton_schulz_reset_iterations:
                        RZ = torch.baddbmm(R, R, Z, beta=a)
                        R = torch.baddbmm(RZ, Z, RZ, beta=a)
                else:
                    Z = self.gemm_symmetric(R, R, C=R, alpha=c, beta=b) 
                    if i == 0 or i in self.gram_newton_schulz_reset_iterations:
                        Q = Z + a * I
                    else:
                        Q = self.gemm_symmetric(Q, Z, C=Q, beta=a)
                    if i < len(self.ns_coefficients) - 1 and i + 1 not in self.gram_newton_schulz_reset_iterations:
                        RZ = self.gemm_symmetric(R, Z, C=R, beta=a)
                        R = self.gemm_symmetric(Z, RZ, C=RZ, beta=a)

            if not should_transpose:
                if not self.ns_use_kernels or X.size(-2) <= SYMMETRIC_KERNEL_TILE_SIZE:
                    X = (Q @ X).to(dtype)
                else:
                    X = self.gemm(Q, X).to(dtype)
            else:
                if not self.ns_use_kernels or X.size(-2) <= SYMMETRIC_KERNEL_TILE_SIZE:
                    X = (X.mT @ Q).to(dtype)
                else:
                    X = self.gemm(X.mT, Q).to(dtype)
        else:
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
