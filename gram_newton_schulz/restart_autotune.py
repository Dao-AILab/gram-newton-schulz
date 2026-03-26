#!/usr/bin/env python3
"""
Find locations of restarts.
"""
from itertools import combinations
import numpy as np


def _init_high_precision():
    try:
        import gmpy2
        import flamp
    except ImportError:
        raise ImportError(
            "high_precision=True requires the 'flamp' package (which provides gmpy2-backed "
            "arbitrary-precision arrays). Install it with: pip install flamp"
        ) from None
    flamp.set_dps(100)
    return gmpy2, flamp


def run_gram_newton_schulz(x_eigenvalues, coefs, most_negative_gram_eigenvalue, high_precision=False, reset_indices=None):
    if reset_indices is None:
        reset_indices = []

    assert most_negative_gram_eigenvalue < 0, "Most negative Gram eigenvalue must be negative"

    if high_precision:
        gmpy2, flamp = _init_high_precision()
        x_eigenvalues = flamp.to_mp(x_eigenvalues)
        most_negative_gram_eigenvalue = gmpy2.mpfr(most_negative_gram_eigenvalue)
        coefs = [(gmpy2.mpfr(a), gmpy2.mpfr(b), gmpy2.mpfr(c)) for a, b, c in coefs]
        ones = flamp.ones
    else:
        ones = np.ones

    q_values = {}
    q = ones(len(x_eigenvalues))
    r = x_eigenvalues * x_eigenvalues + most_negative_gram_eigenvalue

    for iter_idx, (a, b, c) in enumerate(coefs):
        if (iter_idx == 0) or (iter_idx in reset_indices):
            if iter_idx != 0:
                x_eigenvalues = q * x_eigenvalues
                r = x_eigenvalues * x_eigenvalues + most_negative_gram_eigenvalue
            q = ones(len(x_eigenvalues))

        z = c * r * r + b * r + a
        q *= z
        r *= z * z
        q_values[f'Q_{iter_idx}'] = q.astype(np.float64)

    return q_values


def find_best_restarts(x_eigenvalues, coefs, most_negative_gram_eigenvalue, num_restarts=1, high_precision=False):
    possible_positions = list(range(1, len(coefs)))
    if num_restarts == 0:
        return []
    if num_restarts > len(possible_positions):
        raise ValueError(f"Cannot have {num_restarts} restarts with only {len(coefs)} iterations")

    best_restarts = None
    best_max_q = float('inf')

    total_combinations = len(list(combinations(possible_positions, num_restarts)))
    print(f"Testing {total_combinations} combinations of {num_restarts} restart position(s)...")

    for i, restart_combo in enumerate(combinations(possible_positions, num_restarts)):
        test_restarts = list(restart_combo)
        q_results = run_gram_newton_schulz(x_eigenvalues, coefs, most_negative_gram_eigenvalue, high_precision=high_precision, reset_indices=test_restarts)
        max_q = max(np.max(np.abs(vals)) for vals in q_results.values())

        if max_q < best_max_q or (best_max_q == float('inf') and max_q != float('inf')):
            best_max_q = max_q
            best_restarts = test_restarts

        if (i + 1) % max(1, total_combinations // 10) == 0 or i == 0:
            print(f"  [{i+1}/{total_combinations}] Best so far: {best_restarts} with max Q = {best_max_q:.3}")

    if best_max_q == float('inf'):
        raise ValueError(
            f"All {num_restarts} restart combinations resulted in infinite Q values. "
            f"Need more restarts to achieve numerical stability. Try increasing num_restarts."
        )

    print(f"\nBest restart locations (set `gram_newton_schulz_reset_iterations` in newton_schulz/gram_newton_schulz.py to this): {best_restarts} with max Q = {best_max_q:.3}")
    return best_restarts
