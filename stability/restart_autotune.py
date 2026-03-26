#!/usr/bin/env python3
"""
Find locations of restarts.

python restart_autotune.py --num-restarts 1 --coefs "4.0848,-6.8946,2.9270;3.9505,-6.3029,2.6377;3.7418,-5.5913,2.3037;2.8769,-3.1427,1.2046;2.8366,-3.0525,1.2012"
"""
from collections import defaultdict
from itertools import combinations
from decimal import Decimal, getcontext, Overflow
from newton_schulz.coefficients import POLAR_EXPRESS_COEFFICIENTS

getcontext().prec = 100

def run_gram_newton_schulz(x, coefs, most_negative_gram_eigenvalue, reset_indices=None):
    if reset_indices is None:
        reset_indices = []

    q_values = {}
    q = Decimal(1)
    r = Decimal(str(x)) * Decimal(str(x)) + Decimal(str(most_negative_gram_eigenvalue))

    try:
        for iter_idx, (a, b, c) in enumerate(coefs):
            if (iter_idx == 0) or (iter_idx in reset_indices):
                if iter_idx != 0:
                    x = q * Decimal(str(x))
                    r = x * x + Decimal(str(most_negative_gram_eigenvalue))
                q = Decimal(1)

            z = Decimal(str(c)) * r * r + Decimal(str(b)) * r + Decimal(str(a))
            q *= z
            r *= z * z
            q_values[f'Q_{iter_idx}'] = float(q)
    except Overflow:
        for remaining_idx in range(iter_idx, len(coefs)):
            q_values[f'Q_{remaining_idx}'] = float('inf')

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

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(
        description='Find best restart locations for Gram Newton-Schulz given coefficients and number of restarts user wants. More restarts are more stable but slower. We recommend 1 restart.'
    )
    parser.add_argument(
        '--most-negative-gram-eigenvalue',
        type=float,
        default=-4e-4,
        help='Most negative Gram eigenvalue hypothetically formed from XX^T (default: -4e-4)'
    )
    parser.add_argument(
        '--coefs',
        type=str,
        default=None,
        help='Comma-separated list of coefficient float tuples (a,b,c;a,b,c;...). If not provided, uses default POLAR_EXPRESS_COEFFICIENTS.'
    )
    parser.add_argument(
        '--num-restarts',
        type=int,
        default=1,
        help='Number of restart positions to use (default: 1). We empirically determine that 1 restart is sufficient for 5 steps, but users experiencing instability or using more than 5 steps should use more restarts.'
    )

    args = parser.parse_args()

    if args.coefs:
        coefs = []
        for coef_str in args.coefs.split(';'):
            a, b, c = map(float, coef_str.split(','))
            coefs.append((a, b, c))
    else:
        coefs = POLAR_EXPRESS_COEFFICIENTS + [POLAR_EXPRESS_COEFFICIENTS[-1]] * 10

    x_eigenvalues = np.logspace(0, -10, 10000)

    if args.num_restarts == 1:
        print("Finding best restart position for Gram Newton-Schulz...")
    else:
        print(f"Finding best {args.num_restarts} restart positions...")

    best_restarts = find_best_restarts(
        x_eigenvalues, coefs, args.most_negative_gram_eigenvalue, num_restarts=args.num_restarts
    )

    if args.num_restarts == 1:
        print(f"\nBest restart position: {best_restarts}")
    else:
        print(f"\nBest restart positions: {best_restarts}")