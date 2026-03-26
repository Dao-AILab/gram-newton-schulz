#!/usr/bin/env python3
"""
Find locations of restarts.
"""
from collections import defaultdict
from itertools import combinations
import mpmath
from .coefficients import POLAR_EXPRESS_COEFFICIENTS

mpmath.mp.dps = 100

def run_gram_newton_schulz(x, coefs, most_negative_gram_eigenvalue, reset_indices=None):
    if reset_indices is None:
        reset_indices = []

    q_values = {}
    q = mpmath.mpf(1)
    x = mpmath.mpf(x)
    most_negative_gram_eigenvalue = mpmath.mpf(most_negative_gram_eigenvalue)
    r = x * x + most_negative_gram_eigenvalue

    for iter_idx, (a, b, c) in enumerate(coefs):
        if (iter_idx == 0) or (iter_idx in reset_indices):
            if iter_idx != 0:
                x = q * x
                r = x * x + most_negative_gram_eigenvalue
            q = mpmath.mpf(1)

        a, b, c = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(c)
        z = c * r * r + b * r + a
        q *= z
        r *= z * z
        q_values[f'Q_{iter_idx}'] = float(q)

    return q_values


def q_polynomials(x_eigenvalues, coefs, most_negative_gram_eigenvalue, reset_indices=None):
    q_polynomial_at_t = defaultdict(list)
    for x_eigenval in x_eigenvalues:
        q_values = run_gram_newton_schulz(x_eigenval, coefs, most_negative_gram_eigenvalue, reset_indices)
        for key, value in q_values.items():
            q_polynomial_at_t[key].append(value)
    return q_polynomial_at_t


def find_best_restarts(x_eigenvalues, coefs, most_negative_gram_eigenvalue, num_restarts=1):
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
        q_results = q_polynomials(x_eigenvalues, coefs, most_negative_gram_eigenvalue, reset_indices=test_restarts)
        max_q = max(abs(v) for vals in q_results.values() for v in vals)

        if max_q < best_max_q or (best_max_q == float('inf') and max_q != float('inf')):
            best_max_q = max_q
            best_restarts = test_restarts

        if (i + 1) % max(1, total_combinations // 10) == 0 or i == 0:
            print(f"  [{i+1}/{total_combinations}] Best so far: {best_restarts} with max Q = {best_max_q:.6f}")

    if best_max_q == float('inf'):
        raise ValueError(
            f"All {num_restarts} restart combinations resulted in infinite Q values. "
            f"Need more restarts to achieve numerical stability. Try increasing num_restarts."
        )

    print(f"\nBest restart locations (set `gram_newton_schulz_reset_iterations` in newton_schulz/gram_newton_schulz.py to this): {best_restarts} with max Q = {best_max_q:.6f}")
    return best_restarts
