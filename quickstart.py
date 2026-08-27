#!/usr/bin/env python3
"""Quickstart: load dataset.h5 and plot one channel's data, fit, and residuals.

Usage:
    python quickstart.py dataset.h5                # picks the middle channel
    python quickstart.py dataset.h5 --freq-idx 40   # a specific channel
"""

from __future__ import annotations

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("h5_path")
    p.add_argument("--freq-idx", type=int, default=None, help="Channel index (default: middle channel present)")
    args = p.parse_args()

    with h5py.File(args.h5_path, "r") as h5:
        dt_grid = h5["dt_grid"][:]
        freqs_ghz = h5["freqs_ghz"][:]

        channel_names = sorted(k for k in h5.keys() if k.startswith("freq_"))
        if not channel_names:
            raise SystemExit("No freq_* groups found in this file -- is this the right dataset.h5?")

        if args.freq_idx is not None:
            group_name = f"freq_{args.freq_idx:03d}"
            if group_name not in h5:
                raise SystemExit(f"{group_name} not present. Available: {channel_names}")
        else:
            group_name = channel_names[len(channel_names) // 2]

        g = h5[group_name]
        freq_idx = int(group_name.split("_")[1])
        mean = g["mean"][:]
        var = g["var"][:]
        valid = g["valid"][:]
        model = g["model"][:]
        attrs = dict(g.attrs)

        print(f"{group_name}: freq={attrs['freq_ghz'] * 1e3:.0f} MHz")
        print(f"  x0={attrs['x0']:.4f}  sigma={attrs['sigma']:.4f}  N_full={attrs['N_full']}")
        print(f"  chi2_red={attrs['chi2_red']:.4g}  n_valid={attrs['n_valid']}")

    fig, (ax_fit, ax_resid) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_fit.errorbar(
        dt_grid[valid][::20], mean[valid][::20], yerr=np.sqrt(var[valid])[::20],
        fmt="o", ms=3, c="k", capsize=2, label="mean +/- day scatter",
    )
    ax_fit.plot(dt_grid, model, c="tab:red", lw=2, label=f"ZT fit (N={attrs['N_full']})")
    ax_fit.set_ylabel("log10(smoothed |vis|)")
    ax_fit.set_title(f"{freqs_ghz[freq_idx] * 1e3:.0f} MHz -- {group_name}")
    ax_fit.legend()

    resid = np.where(valid, mean - model, np.nan)
    ax_resid.axhline(0, c="k", lw=0.5)
    ax_resid.plot(dt_grid, resid, c="tab:blue", lw=0.8)
    ax_resid.set_xlabel("hours from calculated transit")
    ax_resid.set_ylabel("residual [dex]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
