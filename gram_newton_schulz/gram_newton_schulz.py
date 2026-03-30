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
        use_gram_newton_schulz: bool = True,
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
        if use_gram_newton_schulz:
            self.aspect_ratio_to_use_gram_newton_schulz = 1
            self.gram_newton_schulz_reset_iterations = gram_newton_schulz_reset_iterations if gram_newton_schulz_reset_iterations is not None else [2]
        else:
            self.aspect_ratio_to_use_gram_newton_schulz = float('inf')

        if self.ns_use_kernels:
            from quack.gemm_interface import gemm_symmetric, gemm, gemm_add
            self._gemm_symmetric = gemm_symmetric
            self._gemm = gemm
            self._gemm_add = gemm_add

    def _use_kernels(self, A: Tensor, B: Tensor) -> bool:
        return self.ns_use_kernels and min(A.size(-2), B.size(-1)) > SYMMETRIC_KERNEL_TILE_SIZE

    def _sym_mm(self, A: Tensor, B: Tensor) -> Tensor:
        if self._use_kernels(A, B):
            return self._gemm_symmetric(A, B)
        else:
            return A @ B

    def _sym_baddbmm(self, A: Tensor, B: Tensor, C: Tensor, alpha: float = 1, beta: float = 1) -> Tensor:
        if self._use_kernels(A, B):
            return self._gemm_symmetric(A, B, C=C, alpha=alpha, beta=beta)
        else:
            return torch.baddbmm(C, A, B, alpha=alpha, beta=beta)

    def _mm(self, A: Tensor, B: Tensor) -> Tensor:
        if self._use_kernels(A, B):
            return self._gemm(A, B)
        else:
            return A @ B

    def _mm_add(self, A: Tensor, B: Tensor, C: Tensor, beta: float) -> Tensor:
        if self._use_kernels(A, B):
            return self._gemm_add(A, B, C=C, beta=beta)
        else:
            return torch.baddbmm(C, A, B, beta=beta)

    @torch.compile(fullgraph=True, mode="reduce-overhead")
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

        if max(X.shape[-2:]) > self.aspect_ratio_to_use_gram_newton_schulz * min(X.shape[-2:]):
            X = self._gram_newton_schulz(X)
        else:
            X = self._standard_newton_schulz(X)

        if should_transpose:
            X = X.mT

        return X.to(original_dtype).view(original_shape)

    def _gram_newton_schulz(self, X: Tensor) -> Tensor:
        R = self._sym_mm(X, X.mT)

        batch_size = R.size(0)
        I = torch.eye(R.size(-1), device=X.device, dtype=X.dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        Q = None

        for i, (a, b, c) in enumerate(self.ns_coefficients):
            if i in self.gram_newton_schulz_reset_iterations and i != 0:
                X = self._mm(Q, X)
                R = self._sym_mm(X, X.mT)
                Q = None

            Z = self._sym_baddbmm(R, R, C=R, alpha=c, beta=b)
            if i == 0 or i in self.gram_newton_schulz_reset_iterations:
                Q = Z + a * I
            else:
                Q = self._sym_baddbmm(Q, Z, C=Q, beta=a)
            if i < len(self.ns_coefficients) - 1 and i + 1 not in self.gram_newton_schulz_reset_iterations:
                RZ = self._sym_baddbmm(R, Z, C=R, beta=a)
                R = self._sym_baddbmm(Z, RZ, C=RZ, beta=a)

        X = self._mm(Q, X)

        return X

    def _standard_newton_schulz(self, X: Tensor) -> Tensor:
        for a, b, c in self.ns_coefficients:
            A = self._sym_mm(X, X.mT)
            B = self._sym_baddbmm(A, A, C=A, alpha=c, beta=b)
            X = self._mm_add(B, X, C=X, beta=a)

        return X