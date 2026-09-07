#!/usr/bin/env python3
"""
plot_pdos_only.py  —  replot PDOS from saved pdos_data.npz
Usage:
    python3 plot_pdos_only.py
    python3 plot_pdos_only.py --npz pdos_data.npz --sigma 0.05 --out pdos_clean.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz',   default='pdos_data.npz')
    ap.add_argument('--sigma', type=float, default=0.0,
                    help='Extra Gaussian smoothing in freq points (0=off)')
    ap.add_argument('--out',   default='pdos_clean.png')
    ap.add_argument('--fmax',  type=float, default=9.0)
    args = ap.parse_args()

    d        = np.load(args.npz)
    freq     = d['frequencies']       # THz
    pdos_tot = d['pdos_total']
    pdos_mg  = d['pdos_mg']
    pdos_bi  = d['pdos_bi']

    # optional extra smoothing
    if args.sigma > 0:
        pdos_tot = gaussian_filter1d(pdos_tot, args.sigma)
        pdos_mg  = gaussian_filter1d(pdos_mg,  args.sigma)
        pdos_bi  = gaussian_filter1d(pdos_bi,  args.sigma)

    mask = freq <= args.fmax
    freq     = freq[mask]
    pdos_tot = pdos_tot[mask]
    pdos_mg  = pdos_mg[mask]
    pdos_bi  = pdos_bi[mask]

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # filled areas first (bottom layer)
    ax.fill_between(freq, 0, pdos_mg,  alpha=0.20, color='#1f77b4', zorder=1)
    ax.fill_between(freq, 0, pdos_bi,  alpha=0.20, color='#d62728', zorder=1)

    # lines on top
    ax.plot(freq, pdos_tot, 'k-',  lw=1.8, label='Total',      zorder=4)
    ax.plot(freq, pdos_mg,  lw=2.0, color='#1f77b4',
            label='Mg (3×)',  zorder=3)
    ax.plot(freq, pdos_bi,  lw=2.0, color='#d62728', ls='--',
            label='Bi (2×)',  zorder=3)

    # zero line
    ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6)

    # annotate peaks
    bi_mask = (freq > 0.3) & (freq < 3.0)
    mg_mask = freq > 3.5
    bi_peak_f = freq[bi_mask][np.argmax(pdos_bi[bi_mask])]
    mg_peak_f = freq[mg_mask][np.argmax(pdos_mg[mg_mask])]

    ymax = pdos_tot.max()
    ax.axvline(bi_peak_f, color='#d62728', lw=1.0, ls=':', alpha=0.8)
    ax.text(bi_peak_f + 0.07, ymax * 0.88,
            f'Bi  {bi_peak_f:.2f} THz',
            color='#d62728', fontsize=10, va='top')

    ax.axvline(mg_peak_f, color='#1f77b4', lw=1.0, ls=':', alpha=0.8)
    ax.text(mg_peak_f + 0.07, ymax * 0.70,
            f'Mg  {mg_peak_f:.2f} THz',
            color='#1f77b4', fontsize=10, va='top')

    ax.set_xlabel('Frequency (THz)', fontsize=13)
    ax.set_ylabel('PDOS (states THz$^{-1}$)', fontsize=13)
    ax.set_title('Mg$_3$Bi$_2$ Phonon DOS — NEP', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.2, args.fmax)
    ax.set_ylim(bottom=-0.05)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.25, ls='--')

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f"Saved: {args.out}")
    plt.close()

if __name__ == '__main__':
    main()
