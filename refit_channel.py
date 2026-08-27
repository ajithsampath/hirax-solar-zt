#!/usr/bin/env python3
"""Manually override the Zernike N_full and/or sigma for one channel in
dataset.h5, or let this script search for the joint (N_full, sigma) minimum
itself.

N_full and sigma already stored in dataset.h5 came from an automated
two-stage search: optimize a single sigma_0 scale factor at a fixed probe
N_full, then scan N_full at that fixed sigma. If that choice doesn't look
right on a given channel, use this script to inspect other candidates, fit
with any (N_full, sigma) you pick, look at the plot, and only write it back
once you're happy -- everything else in the channel's group
(mean/var/valid/x0) is reused unchanged, so this never touches the raw
visibility data.

Usage:
    # see chi2_red for each candidate N_full at the channel's stored sigma
    python refit_channel.py dataset.h5 --freq-idx 40 --list

    # ... or at a sigma you choose
    python refit_channel.py dataset.h5 --freq-idx 40 --list --sigma 0.5

    # fit with a chosen N_full (and optionally sigma) and plot it
    # (dataset.h5 is not modified)
    python refit_channel.py dataset.h5 --freq-idx 40 --n 600
    python refit_channel.py dataset.h5 --freq-idx 40 --n 600 --sigma 0.5

    # once satisfied, write the fit back into dataset.h5
    python refit_channel.py dataset.h5 --freq-idx 40 --n 600 --sigma 0.5 --write

    # ... or write it into a different file, leaving dataset.h5 untouched
    # (dataset_refit.h5 is created as a full copy of dataset.h5 the first
    # time it's written to, then just that channel is updated on it)
    python refit_channel.py dataset.h5 --freq-idx 40 --n 600 --sigma 0.5 --write --out dataset_refit.h5

    # [beta] search the (N_full, sigma) plane for the joint chi2_red minimum:
    # for each candidate N_full, optimize sigma continuously, then take the
    # best N_full/sigma pair over all candidates
    python refit_channel.py dataset.h5 --freq-idx 40 --joint
    python refit_channel.py dataset.h5 --freq-idx 40 --joint --write
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import lib

# Same candidate list the original fit scanned over.
DEFAULT_NS = [100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]


def load_channel(h5_path, freq_idx):
    with h5py.File(h5_path, "r") as h5:
        group_name = f"freq_{freq_idx:03d}"
        if group_name not in h5:
            raise SystemExit(f"{group_name} not present in {h5_path}")
        g = h5[group_name]
        chan = dict(
            group_name=group_name,
            dt_grid=h5["dt_grid"][:],
            freq_ghz=float(h5["freqs_ghz"][freq_idx]),
            mean=g["mean"][:],
            var=g["var"][:],
            valid=g["valid"][:],
            x0=float(g.attrs["x0"]),
            sigma=float(g.attrs["sigma"]),
            sigma_gauss=float(g.attrs["sigma_gauss"]),
            sigma_0=float(g.attrs["sigma_0"]),
            current_N=int(g.attrs["N_full"]),
            current_chi2=float(g.attrs["chi2_red"]),
            sigma0_bounds=tuple(h5.attrs["sigma0_bounds"]) if "sigma0_bounds" in h5.attrs else (0.05, 2.0),
        )
    return chan


def fit_at(chan, N_full, sigma):
    x = chan["dt_grid"][chan["valid"]]
    data = chan["mean"][chan["valid"]]
    error = np.sqrt(chan["var"][chan["valid"]])
    coef, _, chi2_red = lib.fit_zernike_1d_theta(
        x=x, data=data, error=error, x0=chan["x0"], sigma=sigma, N_full=N_full,
    )
    model_full = lib.zernike_eval_1d_theta(chan["dt_grid"], coef, chan["x0"], sigma, N_full)
    return coef, model_full, chi2_red


def joint_search(chan, candidates, sigma0_bounds):
    """[beta] For each candidate N_full, optimize sigma (via its sigma_0
    scale factor) continuously to minimize chi2_red, then return the best
    (N_full, sigma) pair over all candidates. This is a nested/profile
    search of the 2D (N_full, sigma) chi2_red surface -- exact in sigma for
    each N_full, but still a discrete scan over N_full since it isn't a
    continuous parameter.
    """
    n_valid = int(chan["valid"].sum())
    x = chan["dt_grid"][chan["valid"]]
    data = chan["mean"][chan["valid"]]
    error = np.sqrt(chan["var"][chan["valid"]])

    rows = []
    for N_full in candidates:
        if N_full >= n_valid:
            continue

        def chi2_for_sigma0(sigma_0, N_full=N_full):
            _, _, chi2 = lib.fit_zernike_1d_theta(
                x=x, data=data, error=error, x0=chan["x0"],
                sigma=sigma_0 * chan["sigma_gauss"], N_full=N_full,
            )
            return chi2

        res = minimize_scalar(chi2_for_sigma0, bounds=sigma0_bounds, method="bounded")
        sigma_0_opt = res.x
        sigma_opt = sigma_0_opt * chan["sigma_gauss"]
        rows.append((N_full, sigma_0_opt, sigma_opt, res.fun))

    if not rows:
        raise SystemExit("No candidate N_full is below n_valid; nothing to search")

    rows.sort(key=lambda r: r[3])
    return rows


def plot_fit(chan, N_full, sigma, model_full, chi2_red):
    dt_grid, valid, mean, var = chan["dt_grid"], chan["valid"], chan["mean"], chan["var"]

    fig, (ax_fit, ax_resid) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_fit.errorbar(
        dt_grid[valid][::20], mean[valid][::20], yerr=np.sqrt(var[valid])[::20],
        fmt="o", ms=3, c="k", capsize=2, label="mean +/- day scatter",
    )
    ax_fit.plot(dt_grid, model_full, c="tab:red", lw=2, label=f"ZT fit (N={N_full}, sigma={sigma:.4f})")
    ax_fit.set_ylabel("log10(smoothed |vis|)")
    ax_fit.set_title(f"{chan['freq_ghz'] * 1e3:.0f} MHz -- {chan['group_name']}  chi2_red={chi2_red:.4g}")
    ax_fit.legend()

    resid = np.where(valid, mean - model_full, np.nan)
    ax_resid.axhline(0, c="k", lw=0.5)
    ax_resid.plot(dt_grid, resid, c="tab:blue", lw=0.8)
    ax_resid.set_xlabel("hours from calculated transit")
    ax_resid.set_ylabel("residual [dex]")

    plt.tight_layout()
    plt.show()


def write_channel(h5_path, chan, N_full, sigma, coef, model_full, chi2_red):
    with h5py.File(h5_path, "a") as h5:
        g = h5[chan["group_name"]]
        g.attrs["N_full"] = N_full
        g.attrs["sigma"] = sigma
        g.attrs["sigma_0"] = sigma / chan["sigma_gauss"]
        g.attrs["chi2_red"] = chi2_red
        del g["coef"]
        del g["model"]
        g.create_dataset("coef", data=coef)
        g.create_dataset("model", data=model_full)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5_path")
    p.add_argument("--freq-idx", type=int, required=True)
    p.add_argument("--n", type=int, default=None, help="N_full to fit with and plot")
    p.add_argument("--sigma", type=float, default=None,
                    help="sigma to fit with and plot (default: this channel's stored sigma)")
    p.add_argument("--list", action="store_true",
                    help="Print chi2_red for each candidate N_full at the chosen sigma, and exit (no plot, no write)")
    p.add_argument("--candidates", default=None,
                    help="Comma-separated N_full values for --list/--joint (default: the original fit's scan list)")
    p.add_argument("--joint", action="store_true",
                    help="[beta] Search the (N_full, sigma) plane: optimize sigma per candidate N_full and pick "
                         "the best pair overall, instead of fitting a single --n/--sigma you specify")
    p.add_argument("--sigma0-min", type=float, default=None,
                    help="Lower bound on the sigma_0 scale factor for --joint (default: this dataset's stored bounds, or 0.05)")
    p.add_argument("--sigma0-max", type=float, default=None,
                    help="Upper bound on the sigma_0 scale factor for --joint (default: this dataset's stored bounds, or 2.0)")
    p.add_argument("--write", action="store_true",
                    help="Write the resulting fit for this channel (coef/model/N_full/sigma/sigma_0/chi2_red). "
                         "Goes into h5_path in place unless --out names a different file")
    p.add_argument("--out", default=None,
                    help="Write to this HDF5 path instead of h5_path (default: same as h5_path, i.e. overwrite "
                         "in place). If this file doesn't exist yet, h5_path is copied here first, so h5_path "
                         "itself is never modified")
    p.add_argument("--no-plot", action="store_true",
                    help="Skip the plot, e.g. when scripting --write after you've already decided on N/sigma")
    args = p.parse_args()

    chan = load_channel(args.h5_path, args.freq_idx)
    n_valid = int(chan["valid"].sum())
    print(f"{chan['group_name']}: freq={chan['freq_ghz'] * 1e3:.0f} MHz  "
          f"x0={chan['x0']:.4f}  sigma_gauss={chan['sigma_gauss']:.4f}  n_valid={n_valid}")
    print(f"  current N_full={chan['current_N']}  current sigma={chan['sigma']:.4f}  "
          f"current chi2_red={chan['current_chi2']:.4g}")

    candidates = [int(s) for s in args.candidates.split(",")] if args.candidates else DEFAULT_NS

    if args.joint:
        sigma0_min = args.sigma0_min if args.sigma0_min is not None else chan["sigma0_bounds"][0]
        sigma0_max = args.sigma0_max if args.sigma0_max is not None else chan["sigma0_bounds"][1]
        print(f"[beta] joint search over N_full in {candidates}, "
              f"sigma_0 in [{sigma0_min}, {sigma0_max}] (sigma = sigma_0 * sigma_gauss)")

        rows = joint_search(chan, candidates, (sigma0_min, sigma0_max))
        print(f"{'N_full':>8}  {'sigma_0':>10}  {'sigma':>10}  {'chi2_red':>10}")
        for N_full, sigma_0, sigma, chi2_red in rows:
            print(f"{N_full:>8}  {sigma_0:>10.4f}  {sigma:>10.4f}  {chi2_red:>10.4g}")

        best_N, best_sigma_0, best_sigma, best_chi2 = rows[0]
        print(f"best: N_full={best_N}  sigma={best_sigma:.4f} (sigma_0={best_sigma_0:.4f})  chi2_red={best_chi2:.4g}")

        coef, model_full, chi2_red = fit_at(chan, best_N, best_sigma)
        N_full, sigma = best_N, best_sigma

    elif args.list:
        sigma = args.sigma if args.sigma is not None else chan["sigma"]
        print(f"chi2_red vs N_full at sigma={sigma:.4f}")
        print(f"{'N_full':>8}  {'chi2_red':>10}")
        for N_full in candidates:
            if N_full >= n_valid:
                print(f"{N_full:>8}  (skipped, >= n_valid={n_valid})")
                continue
            _, _, chi2_red = fit_at(chan, N_full, sigma)
            print(f"{N_full:>8}  {chi2_red:>10.4g}")
        return

    else:
        if args.n is None:
            raise SystemExit("Provide --n (or --list / --joint to explore candidates first)")
        if args.n >= n_valid:
            raise SystemExit(f"--n {args.n} >= n_valid ({n_valid}); dof would be <= 0")

        N_full = args.n
        sigma = args.sigma if args.sigma is not None else chan["sigma"]
        coef, model_full, chi2_red = fit_at(chan, N_full, sigma)
        print(f"N_full={N_full}  sigma={sigma:.4f}  chi2_red={chi2_red:.4g}")

    if not args.no_plot:
        plot_fit(chan, N_full, sigma, model_full, chi2_red)

    if args.write:
        out_path = args.out if args.out else args.h5_path
        if Path(out_path) != Path(args.h5_path) and not Path(out_path).exists():
            shutil.copyfile(args.h5_path, out_path)
            print(f"Copied {args.h5_path} -> {out_path}")
        write_channel(out_path, chan, N_full, sigma, coef, model_full, chi2_red)
        print(f"Wrote N_full={N_full}  sigma={sigma:.4f} fit to {chan['group_name']} in {out_path}")


if __name__ == "__main__":
    main()
