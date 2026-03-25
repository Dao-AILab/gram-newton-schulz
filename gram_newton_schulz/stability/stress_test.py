from itertools import repeat
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from .. import StandardNewtonSchulz, GramNewtonSchulz
import argparse

PE_coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
PE_coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5)
                for (a, b, c) in PE_coeffs_list[:-1]] + [PE_coeffs_list[-1]]



def polar_accuracy_metrics(A: torch.Tensor, estimated_polar_A: torch.Tensor) -> dict[str, float]:
    metrics = dict(
        orth_error=float('inf'),
        residual_error=float('inf'),
        psd_error=float('inf'),
        dual_obj=float('inf'),
        bound_violation=float('inf'),
    )
    if not estimated_polar_A.isfinite().all():
        return metrics

    estimated_polar_A = estimated_polar_A.to(A.dtype)
    H = estimated_polar_A.mT @ A
    H = (H + H.mT) / 2
    Heigs = torch.linalg.eigvalsh(H)
    nuc = torch.linalg.matrix_norm(A, ord='nuc')
    estimate_spectral_norm = torch.linalg.matrix_norm(estimated_polar_A, ord=2)
    I = torch.eye(estimated_polar_A.shape[0], device=estimated_polar_A.device, dtype=estimated_polar_A.dtype)

    metrics["orth_error"] = ((estimated_polar_A @ estimated_polar_A.mT - I).norm() / I.norm()).item()
    metrics["residual_error"] = ((estimated_polar_A @ H - A).norm() / A.norm()).item()
    metrics["psd_error"] = ((Heigs[Heigs < 0]).norm() / (Heigs[Heigs > 0]).norm()).item()
    metrics["dual_obj"] = ((nuc - torch.inner(A.flatten(), estimated_polar_A.flatten()))/nuc).item()
    metrics["bound_violation"] = max((estimate_spectral_norm - 1).item(), 0)
    return metrics


def facet_plot(df, outpath, title=None):
    # Reshape data for seaborn: one row per (sample, metric) with ref and ours columns
    plot_df = df.stack(level='metric', future_stack=True).reset_index(level='metric')

    g = sns.FacetGrid(plot_df, col='metric', col_wrap=3, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x='reference', y='ours')

    # Add y=x reference line to each facet
    for ax in g.axes.flat:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

    g.set_axis_labels('reference', 'ours')
    if title is not None:
        # set the title on the FacetGrid's Figure so it appears above all facets
        g.figure.suptitle(title)
        # move the subplot area down to make room for the suptitle
        g.figure.subplots_adjust(top=0.9)
    # use the Figure's tight_layout to arrange subplots
    g.figure.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    g.figure.savefig(outpath)


def spectrum2matrix(spectrum, aspect_ratio):
    n = len(spectrum)
    m = int(n * aspect_ratio)
    U, _, Vh = torch.linalg.svd(torch.randn(m, n, device=spectrum.device, dtype=spectrum.dtype), full_matrices=False)
    return U @ torch.diag(spectrum) @ Vh


def run_comparison(our_method, reference_method, matrix_iter, iterlen=None):
    results = []
    for matrix in tqdm(matrix_iter, total=iterlen):
        ref = reference_method(matrix)
        reference_metrics = polar_accuracy_metrics(matrix, ref)
        ours = our_method(matrix)
        ours_metrics = polar_accuracy_metrics(matrix, ours)
        results.append(list(reference_metrics.values()) + list(ours_metrics.values()))
    cols = pd.MultiIndex.from_tuples(
        [(k, "reference") for k in reference_metrics.keys()] + [(k, "ours") for k in ours_metrics.keys()],
        names=["metric", "method"],
    )
    return pd.DataFrame(results, columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the accuracy of Gram Newton-Schulz compared to the standard version")
    parser.add_argument("--steps", "-s", type=int, default=5, help="Number of Newton-Schulz iterations to run")
    parser.add_argument("--test-set", "-t", type=str, default="synthetic", help="Set of matrices to test on. Use 'synthetic' for the built-in set of synthetically generated matrices, or path to a folder of .pt files. For example: http://drive.google.com/file/d/14psG-YFnrXCwaQ82acnVByB_QVmmsI-X")
    parser.add_argument("--pure-pytorch", action="store_true", help="Whether to use built-in PyTorch functions instead of our custom kernels")
    parser.add_argument("--restarts", type=int, nargs="+", default=[2], help="List of positive integers for reset iterations")
    parser.add_argument("--normalize-max-output", action="store_true", help="Some series of polynomials (e.g. Polar Express) map singular values to slightly above 1, which may make some metrics misleading. This option adjusts the polynomials to ensure the final output is at most 1 in exact arithmetic.")
    parser.add_argument("--output-dir", "-o", type=str, default="plots", help="Output directory for results and plots")

    args = parser.parse_args()

    num_limit = None

    if args.test_set == "synthetic":
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        n = 768
        aspect_ratio = 4
        spectra = [
            torch.linspace(0, 0.5, steps=n),
            torch.logspace(0, -3, steps=n),
            torch.logspace(0, -5, steps=n),
            torch.logspace(0, -7, steps=n),
            torch.logspace(0, -9, steps=n),
            torch.cat((
                torch.logspace(0, -2, steps=n//2),
                torch.zeros(n - n//2),
            )),
            torch.cat((
                torch.logspace(0, -1, steps=10),
                torch.zeros(n - 10),
            )),
            torch.cat((
                torch.logspace(0, -0.5, steps=2),
                torch.zeros(n - 2),
            )),
            torch.cat((
                torch.logspace(0, -0.5, steps=2),
                torch.logspace(-4, -7, steps=n - 2),
            )),
            torch.concat((
                torch.logspace(0, -10, steps=1000, device=DEVICE, dtype=torch.float64),
                torch.logspace(-4, -10, steps=100, device=DEVICE, dtype=torch.float64),
            )),
        ] * 5 + [
            torch.distributions.Gamma(concentration=3, rate=0.5).sample((n,))
            for _ in range(5)
        ]
        spectra = [s.to(DEVICE) for s in spectra]
        matrix_iter = map(lambda s: spectrum2matrix(s, aspect_ratio), spectra)
        num = len(spectra)
    else:
        matrix_files = sorted(
            [os.path.join(args.test_set, f) for f in os.listdir(args.test_set) if f.endswith(".pt")],
            key=lambda x: os.path.getmtime(x)
        )
        matrix_iter = map(torch.load, matrix_files[:num_limit])
        num = len(matrix_files)
        if num_limit: num = min(num, num_limit)

    coeffs_steps = PE_coeffs_list[:args.steps] + list( 
        repeat(PE_coeffs_list[-1], args.steps - len(PE_coeffs_list)))
    if args.normalize_max_output:
        final_upper_bound = 1
        for (a, b, c) in coeffs_steps:
            final_upper_bound = a * final_upper_bound + b * final_upper_bound**3 + c * final_upper_bound**5
        print(f"Adjusting final by 1/{final_upper_bound:.3f}")
        coeffs_steps[-1] = (coeffs_steps[-1][0] / final_upper_bound, coeffs_steps[-1][1] / final_upper_bound, coeffs_steps[-1][2] / final_upper_bound)

    our_method = GramNewtonSchulz(ns_use_kernels=not args.pure_pytorch, ns_coefficients=coeffs_steps, gram_newton_schulz_reset_iterations=args.restarts)
    df = run_comparison(our_method, StandardNewtonSchulz(ns_use_kernels=not args.pure_pytorch, ns_coefficients=coeffs_steps), matrix_iter, iterlen=num)
    test_set_name = os.path.basename(os.path.normpath(args.test_set))
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_pickle(os.path.join(args.output_dir, f"appF_{test_set_name}_{args.steps}_steps.pkl"))
    print(len(df))
    print(df.head(5))
    blowups = df.loc[(~np.isfinite(df)).any(axis=1)]
    if len(blowups) == 0:
        print("No infinite blowups detected.")
    else:
        print("Blowups:")
        print(blowups)

    facet_plot(df, os.path.join(args.output_dir, f'appF_{test_set_name}_{args.steps}_steps.png'), title=f'Appendix F on {test_set_name} set ({args.steps} steps)')

