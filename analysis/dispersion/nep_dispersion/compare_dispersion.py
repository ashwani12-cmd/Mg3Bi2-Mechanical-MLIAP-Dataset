#!/usr/bin/env python3
"""
compare_dispersion.py -- overlay NEP phonon dispersion on the DFT reference
===========================================================================
Reads the DFT band.yaml (phonopy) and computes the NEP dispersion on the
EXACT same q-path, then plots them together and quantifies the deviation
branch by branch. This is the payoff figure: NEP curves lying on DFT curves.

WHY IT MATCHES THE PATH EXACTLY
The DFT band.yaml already contains every q-point and its distance along the
path. This script pulls those q-points straight out and evaluates the NEP at
them -- so the two curves are sampled identically and overlay without any
interpolation mismatch.

FAIR-COMPARISON NOTES (both matter)
  * The NEP cell is relaxed with the NEP first (its minimum != the DFT
    minimum), then phonopy builds force constants on the same 3x3x2 supercell
    the DFT used -- so any difference is the potential, not the setup.
  * Mg3Bi2 is POLAR (P-3m1). True DFT phonons have LO-TO splitting from Born
    charges at Gamma; a short-range NEP cannot reproduce it. Expect a residual
    gap in the optical branches at Gamma. That is physics, not a fit failure,
    and it is flagged in the output rather than hidden.

USAGE
  python3 compare_dispersion.py --nep nep.txt --struct espresso.pwi \
        --dft-band band.yaml
  # writes dft_vs_nep.png and prents per-branch RMS/max deviation
"""
import argparse, sys
import numpy as np
import yaml


def read_dft_band(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    q = np.array([p['q-position'] for p in d['phonon']])
    dist = np.array([p['distance'] for p in d['phonon']])
    freqs = np.array([[b['frequency'] for b in p['band']]
                      for p in d['phonon']])          # (nq, nbands)
    labels = d.get('labels')
    # segment boundaries: where distance repeats or label changes
    seg_end = [dist[0]]
    for lab in (labels or []):
        pass
    return q, dist, freqs, labels, d['nqpoint']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nep', required=True)
    ap.add_argument('--struct', default='espresso.pwi')
    ap.add_argument('--dft-band', default='band.yaml')
    ap.add_argument('--dim', nargs=3, type=int, default=[3, 3, 2],
                    help='must match the DFT supercell (band.conf DIM)')
    ap.add_argument('--disp', type=float, default=0.0159)
    ap.add_argument('--fmax', type=float, default=1e-6)
    ap.add_argument('--out', default='dft_vs_nep.png')
    args = ap.parse_args()

    from ase.io import read
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from calorine.calculators import CPUNEP

    # ---- DFT reference ----
    q, dist, fdft, labels, nq = read_dft_band(args.dft_band)
    print(f'DFT band.yaml: {nq} q-points, {fdft.shape[1]} branches, '
          f'max {fdft.max():.3f} THz')

    # ---- NEP dispersion on the SAME q-points ----
    calc = CPUNEP(args.nep)
    atoms = read(args.struct); atoms.pbc = True; atoms.calc = calc
    print('relaxing cell at the NEP minimum...')
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=args.fmax, steps=500)
    L = np.linalg.norm(atoms.cell, axis=1)
    print(f'  NEP cell a={L[0]:.4f} c={L[2]:.4f}')

    ph = Phonopy(PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                              cell=atoms.get_cell(),
                              scaled_positions=atoms.get_scaled_positions()),
                 supercell_matrix=np.diag(args.dim), primitive_matrix='auto')
    ph.generate_displacements(distance=args.disp)
    from ase import Atoms
    F = []
    for sc in ph.supercells_with_displacements:
        a = Atoms(symbols=sc.symbols, cell=sc.cell,
                  scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        F.append(a.get_forces())
    ph.forces = np.array(F)
    ph.produce_force_constants()

    # evaluate NEP frequencies at the DFT q-points
    fnep = np.array([ph.get_frequencies(qq) for qq in q])   # (nq, nbands)

    # ---- deviation stats (skip the acoustic-at-Gamma zeros) ----
    mask = fdft > 0.05
    diff = fnep - fdft
    rms = np.sqrt(np.mean(diff[mask] ** 2))
    mae = np.mean(np.abs(diff[mask]))
    mx = np.abs(diff[mask]).max()
    print('\n' + '=' * 56)
    print('NEP vs DFT dispersion deviation (branches > 0.05 THz)')
    print('=' * 56)
    print(f'  RMS  = {rms*1000:.1f} GHz  ({rms:.4f} THz)')
    print(f'  MAE  = {mae*1000:.1f} GHz  ({mae:.4f} THz)')
    print(f'  max  = {mx*1000:.1f} GHz  ({mx:.4f} THz)')
    print(f'  relative to the {fdft.max():.1f} THz bandwidth: '
          f'RMS {100*rms/fdft.max():.1f}%')

    # per-branch, sorted
    print(f'\n  per-branch RMS (THz), low to high:')
    for b in range(fdft.shape[1]):
        m = fdft[:, b] > 0.05
        if m.any():
            r = np.sqrt(np.mean((fnep[m, b] - fdft[m, b]) ** 2))
            print(f'    branch {b+1:2d}: {r:.4f}', end='   ')
            if (b + 1) % 3 == 0:
                print()
    print()

    # Gamma optical gap (the LO-TO the NEP structurally cannot get)
    ig = int(np.argmin(dist))
    print(f'\n  at Gamma, optical branches (LO-TO not in NEP):')
    print(f'    DFT: {np.round(fdft[ig][3:], 2)}')
    print(f'    NEP: {np.round(fnep[ig][3:], 2)}')

    # ---- plot ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    for b in range(fdft.shape[1]):
        ax.plot(dist, fdft[:, b], '-', color='k', lw=1.4,
                label='DFT' if b == 0 else None, zorder=2)
        ax.plot(dist, fnep[:, b], '--', color='tab:red', lw=1.1,
                label='NEP' if b == 0 else None, zorder=3)
    # segment boundaries + labels
    seg = [dist[0]]
    flat = []
    if labels:
        for pair in labels:
            flat.append(pair[0])
        flat.append(labels[-1][1])
    # boundaries where q distance stalls (segment ends)
    bd = [0]
    for i in range(1, len(dist)):
        if dist[i] == dist[i-1]:
            bd.append(i)
    bd.append(len(dist) - 1)
    xt = [dist[i] for i in bd]
    for x in xt:
        ax.axvline(x, color='gray', lw=0.5, zorder=1)
    lbls = [l.replace('$\\Gamma$', r'$\Gamma$') for l in (flat or [])]
    if len(lbls) == len(xt):
        ax.set_xticks(xt); ax.set_xticklabels(lbls)
    ax.set_ylabel('Frequency (THz)')
    ax.set_xlim(dist[0], dist[-1]); ax.set_ylim(bottom=min(0, fdft.min()))
    ax.axhline(0, color='blue', ls=':', lw=0.6)
    ax.legend(loc='upper right')
    ax.set_title(f'Mg$_3$Bi$_2$ phonon dispersion: DFT vs NEP  '
                 f'(RMS {rms*1000:.0f} GHz, {100*rms/fdft.max():.1f}%)')
    fig.tight_layout(); fig.savefig(args.out, dpi=200)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
