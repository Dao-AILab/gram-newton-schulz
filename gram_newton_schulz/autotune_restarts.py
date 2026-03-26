#!/usr/bin/env python3
"""
CLI for analyzing numerical stability of Gram Newton-Schulz iteration.

Usage:
    python -m gram_newton_schulz.autotune_restarts --num-restarts 1 --coefs "4.0848,-6.8946,2.9270;3.9505,-6.3029,2.6377;3.7418,-5.5913,2.3037;2.8769,-3.1427,1.2046;2.8366,-3.0525,1.2012"
"""
import argparse

import numpy as np

from .coefficients import POLAR_EXPRESS_COEFFICIENTS
from .restart_autotune import find_best_restarts


def main():
    parser = argparse.ArgumentParser(
        description='Analyze numerical stability of Gram Newton-Schulz iteration'
    )
    parser.add_argument(
        '--most-negative-gram-eigenvalue',
        type=float,
        default=-4e-4,
        help='Most negative Gram eigenvalue to add to XX^T (default: -4e-4)'
    )
    parser.add_argument(
        '--coefs',
        type=str,
        default=None,
        help='Comma-separated list of coefficient tuples (a,b,c;a,b,c;...). '
             'If not provided, uses default POLAR_EXPRESS_COEFFICIENTS.'
    )
    parser.add_argument(
        '--num-restarts',
        type=int,
        default=1,
        help='Number of restart positions to find (default: 1)'
    )

    parser.add_argument(
        '--high-precision',
        action='store_true',
        default=False,
        help='Use high-precision (100 decimal digit) arithmetic via gmpy2/flamp (default: off)'
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
        x_eigenvalues, coefs, args.most_negative_gram_eigenvalue, num_restarts=args.num_restarts, high_precision=args.high_precision
    )

    if args.num_restarts == 1:
        print(f"\nBest restart position: {best_restarts}")
    else:
        print(f"\nBest restart positions: {best_restarts}")


if __name__ == "__main__":
    main()
