#!/usr/bin/env python3
"""
plot_phonon_dispersion.py -- publication DFT-vs-NEP phonon dispersion, full path
===============================================================================
Mg3Bi2 phonon dispersion over the COMPLETE hexagonal q-path

        Gamma - M - K - Gamma - A - L - H - A          (P-3m1, space group 164)

instead of the in-plane Gamma-M-K-Gamma segment in
`figures/phonon_bands/phonon_dispersion_DFT_vs_NEP.png`.

Both curves are built with phonopy on the *same* 3x3x2 supercell of the *same*
primitive cell, so every difference is the potential, not the setup:

  * DFT : analysis/dispersion/dft_dispersion/{phonopy.yaml, FORCE_SETS}
          (5 finite-displacement QE calculations)
  * NEP : ~/Mg3Bi2/dispersion/FORCE_CONSTANTS_NEP
          (phonopy force constants from the trained NEP)

Validation (sorted branch frequencies on the 707-q reference path in
.../dft_dispersion/band.yaml):

    FORCE_SETS -> DFT band.yaml   RMS 0.0001 THz   (exact)
    FORCE_CONSTANTS_NEP           RMS 0.154  THz vs DFT

PHYSICS NOTE
Mg3Bi2 is polar; the true DFT spectrum has LO-TO splitting at Gamma from Born
effective charges. A short-range NEP cannot reproduce it, so a residual gap in
the optical branches at Gamma is expected -- physics, not a fit error.

USAGE
    cd ~/Mg3Bi2/Mg3Bi2-Mechanical-MLIAP-Dataset
    python3 scripts/plot_phonon_dispersion.py
    # -> figures/phonon_bands/phonon_dispersion_DFT_vs_NEP_fullpath.{png,pdf}

Edit the PATHS / SETTINGS block below to point at other inputs or change the
resolution. Needs: phonopy, numpy, matplotlib, pyyaml (no calorine).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")           # remove for an interactive window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator
from phonopy import load
from phonopy.phonon.band_structure import get_band_qpoints

# ── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 22,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.linewidth": 2.0,
    "lines.linewidth": 2.5,
    "lines.markersize": 12,
})


def apply_style(ax, nx=6, ny=6):
    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2)
    ax.tick_params(axis='both', which='minor', direction='in',  length=5, width=1.5)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(MaxNLocator(nx))
    ax.yaxis.set_major_locator(MaxNLocator(ny))
    ax.set_aspect("auto")


# ── paths / settings (edit if you move files) ─────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFT_DIR = os.path.join(REPO, "analysis/dispersion/dft_dispersion")
PHONOPY_YAML        = os.path.join(DFT_DIR, "phonopy.yaml")
DFT_FORCE_SETS      = os.path.join(DFT_DIR, "FORCE_SETS")
NEP_FORCE_CONSTANTS = os.path.expanduser("~/Mg3Bi2/dispersion/FORCE_CONSTANTS_NEP")
OUT = os.path.join(REPO, "figures/phonon_bands/phonon_dispersion_DFT_vs_NEP_fullpath")

NPOINTS = 201        # q-points per path segment
YMAX = 9.0           # THz
FIGSIZE = (5, 4)     # inches (width, height)

# Gamma-M-K-Gamma-A-L-H-A  -- matches analysis/dispersion/dft_dispersion/band.conf
BAND_POINTS = np.array([
    [0.0, 0.0, 0.0],   # Gamma
    [0.5, 0.0, 0.0],   # M
    [1/3, 1/3, 0.0],   # K
    [0.0, 0.0, 0.0],   # Gamma
    [0.0, 0.0, 0.5],   # A
    [0.5, 0.0, 0.5],   # L
    [1/3, 1/3, 0.5],   # H
    [0.0, 0.0, 0.5],   # A
])
BAND_LABELS = [r"$\Gamma$", "M", "K", r"$\Gamma$", "A", "L", "H", "A"]


def band_curves(ph):
    """(list of distance arrays, list of freq arrays) per segment on BAND_POINTS."""
    qpoints = get_band_qpoints([BAND_POINTS], npoints=NPOINTS)
    ph.run_band_structure(qpoints, with_eigenvectors=True, is_band_connection=True)
    bs = ph.get_band_structure_dict()
    return bs["distances"], bs["frequencies"]


def main():
    ph_dft = load(PHONOPY_YAML, force_sets_filename=DFT_FORCE_SETS)
    ph_nep = load(PHONOPY_YAML, force_constants_filename=NEP_FORCE_CONSTANTS)

    dist_d, freq_d = band_curves(ph_dft)
    dist_n, freq_n = band_curves(ph_nep)

    # deviation stats (ignore acoustic-at-Gamma near-zeros)
    fd = np.vstack(freq_d)
    fn = np.vstack(freq_n)
    m = fd > 0.05
    diff = (fn - fd)[m]
    rms, mae, mx, bw = (np.sqrt(np.mean(diff ** 2)), np.mean(np.abs(diff)),
                        np.abs(diff).max(), fd.max())
    print("NEP vs DFT (branches > 0.05 THz):")
    print(f"  RMS = {rms*1000:6.1f} GHz  ({rms:.4f} THz)   "
          f"= {100*rms/bw:.1f} % of {bw:.2f} THz bandwidth")
    print(f"  MAE = {mae*1000:6.1f} GHz  ({mae:.4f} THz)")
    print(f"  max = {mx*1000:6.1f} GHz  ({mx:.4f} THz)")

    seg_end = [d[-1] for d in dist_d]
    xticks = [dist_d[0][0]] + seg_end

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for dd, ff in zip(dist_d, freq_d):
        ax.plot(dd, ff, color="black", lw=1.6, ls="--", zorder=3)
    for dn, ff in zip(dist_n, freq_n):
        ax.plot(dn, ff, color="red", lw=2.8, ls="-", alpha=0.85, zorder=2)

    apply_style(ax)                       # tick styling + minor ticks
    # band-structure x-axis: high-symmetry ticks only, no x minor ticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(BAND_LABELS)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    for x in xticks:
        ax.axvline(x, color="gray", lw=1.0, alpha=0.5)
    ax.axhline(0, color="0.4", lw=0.8, ls=":")

    ax.set_xlabel("Wave vector")
    ax.set_ylabel("Frequency (THz)")
    ax.set_xlim(xticks[0], seg_end[-1])
    ax.set_ylim(0, YMAX)
    ax.margins(x=0)

    handles = [Line2D([0], [0], color="black", lw=1.6, ls="--", label="DFT"),
               Line2D([0], [0], color="red", lw=2.8, ls="-", label="NEP")]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              ncol=2, handlelength=2.2, columnspacing=1.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT + ".pdf", bbox_inches="tight")
    print("wrote", OUT + ".png", "and", OUT + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
