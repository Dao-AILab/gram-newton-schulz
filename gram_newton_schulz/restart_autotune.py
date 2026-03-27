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


def simulate_perturbed_gram_newton_schulz(x_eigenvalues, coefs, perturbation, high_precision=False, reset_indices=None):
    x_eigenvalues = x_eigenvalues.copy()

    if reset_indices is None:
        reset_indices = []

    assert perturbation < 0, "Perturbation should be negative"

    if high_precision:
        gmpy2, flamp = _init_high_precision()
        x_eigenvalues = flamp.to_mp(x_eigenvalues)
        perturbation = gmpy2.mpfr(perturbation)
        coefs = [(gmpy2.mpfr(a), gmpy2.mpfr(b), gmpy2.mpfr(c)) for a, b, c in coefs]
        ones = flamp.ones
    else:
        ones = np.ones

    q_values = {}

    with np.errstate(over='ignore', invalid='ignore'):
        for iter_idx, (a, b, c) in enumerate(coefs):
            if (iter_idx == 0) or (iter_idx in reset_indices):
                if iter_idx != 0:
                    x_eigenvalues *= q
                r = x_eigenvalues**2 + perturbation
                q = ones(len(x_eigenvalues))

            z = a + r*(b + r*c)
            q *= z
            r *= z**2
            q_values[f'Q_{iter_idx}'] = q.astype(np.float64)

    return q_values


def stability_metric(q_values):
    def condition(vals):
        abs_vals = np.abs(vals)
        return abs_vals.max() / abs_vals.min()
    return max(condition(vals) for vals in q_values.values())


def find_best_restarts(x_eigenvalues, coefs, most_negative_gram_eigenvalue, num_restarts=1, high_precision=False):
    possible_positions = list(range(1, len(coefs)))
    if num_restarts == 0:
        q_results = simulate_perturbed_gram_newton_schulz(x_eigenvalues, coefs, most_negative_gram_eigenvalue, high_precision=high_precision, reset_indices=[])
        max_q = stability_metric(q_results)
        print(f"With no restarts, condition number of Q = {max_q:.3f}")
        return []

    if num_restarts > len(possible_positions):
        raise ValueError(f"Cannot have {num_restarts} restarts with only {len(coefs)} iterations")

    best_restarts = None
    best_max_q = float('inf')

    total_combinations = len(list(combinations(possible_positions, num_restarts)))
    print(f"Testing {total_combinations} combinations of {num_restarts} restart position(s)...")

    for i, restart_combo in enumerate(combinations(possible_positions, num_restarts)):
        test_restarts = list(restart_combo)
        q_results = simulate_perturbed_gram_newton_schulz(x_eigenvalues, coefs, most_negative_gram_eigenvalue, high_precision=high_precision, reset_indices=test_restarts)
        max_q = stability_metric(q_results)

        if max_q < best_max_q:
            best_max_q = max_q
            best_restarts = test_restarts

        if (i + 1) % max(1, total_combinations // 10) == 0 or i == 0:
            print(f"  [{i+1}/{total_combinations}] Best so far: {best_restarts} with condition number of Q = {best_max_q:.3f}")

    if not np.isfinite(best_max_q) or best_max_q >= 1e8:
        raise ValueError(
            f"All {num_restarts} restart combinations resulted in blowups. "
            f"Need more restarts to achieve numerical stability. Try increasing num_restarts."
        )

    print(f"\nBest restart locations (set `gram_newton_schulz_reset_iterations` in newton_schulz/gram_newton_schulz.py to this): {best_restarts} with condition number of Q = {best_max_q:.3f}")
    return best_restarts
