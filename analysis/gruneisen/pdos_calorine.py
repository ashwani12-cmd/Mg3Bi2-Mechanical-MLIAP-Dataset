#!/usr/bin/env python3
"""
pdos_calorine.py
================
Computes phonon dispersion + atom-projected PDOS for Mg₃Bi₂
using NEP potential via calorine + phonopy.

Outputs:
    phonon_dispersion.png  — dispersion along Γ-M-K-Γ-A-L-H-A
    pdos.png               — total + Mg-projected + Bi-projected DOS
    pdos_data.npz          — raw data for replotting

Usage:
    python3 pdos_calorine.py --nep nep.txt --pwi espresso.pwi
    python3 pdos_calorine.py --nep nep.txt --pwi espresso.pwi --mesh 20 20 20
"""

import argparse, re, sys
import numpy as np

try:
    from calorine.calculators import CPUNEP
    from calorine.tools import get_force_constants
    from ase.optimize import BFGS
    from ase.filters import FrechetCellFilter
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


# ── PWI parser ────────────────────────────────────────────────────────────────
def parse_pwi(path):
    with open(path) as f: lines = f.readlines()
    cell_rows=[]; in_cell=False
    for line in lines:
        if re.match(r'\s*CELL_PARAMETERS\s*\(angstrom\)', line, re.IGNORECASE):
            in_cell=True; continue
        if in_cell:
            nums=re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if len(nums)>=3: cell_rows.append([float(x) for x in nums[:3]])
            if len(cell_rows)==3: break
    cell=np.array(cell_rows)
    syms=[]; pos=[]; in_atoms=False
    for line in lines:
        if re.match(r'\s*ATOMIC_POSITIONS\s*\(angstrom\)', line, re.IGNORECASE):
            in_atoms=True; continue
        if in_atoms:
            parts=line.split()
            if not parts: continue
            if len(parts)>=4:
                try:
                    syms.append(parts[0])
                    pos.append([float(parts[i]) for i in [1,2,3]])
                except: break
            elif parts: break
    from ase import Atoms
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)


# ── Build phonopy object ──────────────────────────────────────────────────────
def build_phonopy(atoms, nep_path, supercell_matrix, displacement=0.01):
    """Relax, compute force constants, return Phonopy object."""
    print("  Relaxing primitive cell...")
    calc=CPUNEP(nep_path)
    atoms=atoms.copy(); atoms.calc=calc
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-7, steps=500)
    L=np.linalg.norm(atoms.cell, axis=1)
    print(f"  a={L[0]:.5f} Å  c={L[2]:.5f} Å  V={atoms.get_volume():.4f} Å³")

    print("  Computing force constants...")
    calc2=CPUNEP(nep_path)
    ph=get_force_constants(atoms, calc2,
                           supercell_matrix=supercell_matrix,
                           kwargs_generate_displacements={"distance": displacement})
    print(f"  Force constants done  "
          f"({len(ph.supercell)} atom supercell)")
    return ph, atoms


# ── Phonon dispersion ─────────────────────────────────────────────────────────
def compute_dispersion(ph):
    """Compute phonon dispersion along Γ-M-K-Γ-A-L-H-A for hexagonal."""
    # High-symmetry path for hexagonal (P-3m1)
    path = [
        [[0,   0,   0  ], 'Γ'],
        [[0.5, 0,   0  ], 'M'],
        [[1/3, 1/3, 0  ], 'K'],
        [[0,   0,   0  ], 'Γ'],
        [[0,   0,   0.5], 'A'],
        [[0.5, 0,   0.5], 'L'],
        [[1/3, 1/3, 0.5], 'H'],
        [[0,   0,   0.5], 'A'],
    ]
    from phonopy.phonon.band_structure import get_band_qpoints
    points = [p[0] for p in path]
    labels = [p[1] for p in path]

    qpoints = get_band_qpoints([points], npoints=101)
    ph.run_band_structure(qpoints, with_eigenvectors=True,
                          is_band_connection=False)
    band = ph.get_band_structure_dict()
    return band, labels


# ── PDOS ──────────────────────────────────────────────────────────────────────
def compute_pdos(ph, mesh, sigma=0.1):
    """
    Compute atom-projected PDOS on a q-mesh.
    sigma: Gaussian smearing in THz
    """
    ph.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True)
    ph.run_projected_dos(sigma=sigma, freq_min=-0.5, freq_max=10.0,
                         freq_pitch=0.01)
    pdos_dict = ph.get_projected_dos_dict()
    # frequencies in THz, pdos shape: (n_freq, n_atoms)
    return pdos_dict


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_dispersion(band, labels, outpath):
    """Plot phonon dispersion."""
    freqs     = band['frequencies']    # list of arrays shape (npath, nband)
    distances = band['distances']      # list of distance arrays

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f9f9f9')

    # Flatten paths
    dist_flat  = np.concatenate(distances)
    freqs_flat = np.concatenate(freqs)    # shape (n_total_q, n_bands)

    for b in range(freqs_flat.shape[1]):
        ax.plot(dist_flat, freqs_flat[:,b], 'k-', lw=1.0, alpha=0.75)

    # High-symmetry point markers
    # Find segment boundaries
    seg_ends = [distances[0][0]]
    for d in distances:
        seg_ends.append(d[-1])
    for x in seg_ends:
        ax.axvline(x, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='red', lw=0.8, ls='--', alpha=0.5)

    ax.set_xticks(seg_ends)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel('Frequency (THz)', fontsize=13)
    ax.set_title('Mg₃Bi₂ Phonon Dispersion (NEP)', fontsize=13, fontweight='bold')
    ax.set_xlim(dist_flat.min(), dist_flat.max())
    ax.set_ylim(-0.3, None)
    ax.grid(False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def plot_pdos(pdos_dict, syms_prim, outpath):
    """Plot total + element-projected PDOS."""
    freqs    = pdos_dict['frequency_points']   # THz
    pdos_all = pdos_dict['projected_dos']      # shape (n_atoms, n_freq)

    # Identify Mg and Bi atom indices in primitive cell
    mg_idx = [i for i,s in enumerate(syms_prim) if s=='Mg']
    bi_idx = [i for i,s in enumerate(syms_prim) if s=='Bi']

    pdos_mg    = pdos_all[mg_idx].sum(axis=0)   # sum over Mg atoms
    pdos_bi    = pdos_all[bi_idx].sum(axis=0)   # sum over Bi atoms
    pdos_total = pdos_all.sum(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={'width_ratios':[2,1]})
    fig.patch.set_facecolor('white')

    # ── Left: dispersion placeholder (or total DOS rotated) ──────────────────
    # Right panel: PDOS
    ax = axes[1]
    ax.set_facecolor('#f9f9f9')
    ax.fill_betweenx(freqs, 0, pdos_total,
                     alpha=0.15, color='k', label='Total')
    ax.plot(pdos_total, freqs, 'k-',  lw=1.5, label='Total')
    ax.plot(pdos_mg,    freqs, color='#1f77b4', lw=2.0, label='Mg')
    ax.plot(pdos_bi,    freqs, color='#d62728', lw=2.0, ls='--', label='Bi')
    ax.fill_betweenx(freqs, 0, pdos_mg, alpha=0.25, color='#1f77b4')
    ax.fill_betweenx(freqs, 0, pdos_bi, alpha=0.25, color='#d62728')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('PDOS (states/THz)', fontsize=12)
    ax.set_ylabel('Frequency (THz)', fontsize=12)
    ax.set_title('Phonon DOS', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(-0.3, freqs.max())
    ax.grid(True, alpha=0.3, ls='--')

    # ── Left: stacked PDOS ───────────────────────────────────────────────────
    ax2 = axes[0]
    ax2.set_facecolor('#f9f9f9')
    ax2.plot(freqs, pdos_total, 'k-',  lw=1.5, label='Total')
    ax2.plot(freqs, pdos_mg,    color='#1f77b4', lw=2.0, label='Mg (54 at.)')
    ax2.plot(freqs, pdos_bi,    color='#d62728', lw=2.0, ls='--', label='Bi (36 at.)')
    ax2.fill_between(freqs, 0, pdos_mg, alpha=0.25, color='#1f77b4')
    ax2.fill_between(freqs, 0, pdos_bi, alpha=0.25, color='#d62728')
    ax2.axvline(0, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel('Frequency (THz)', fontsize=12)
    ax2.set_ylabel('PDOS (states/THz)', fontsize=12)
    ax2.set_title('Mg₃Bi₂ Phonon DOS — Mg vs Bi projections (NEP)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim(-0.3, freqs.max())
    ax2.grid(True, alpha=0.3, ls='--')

    # Annotate key features
    # Find Bi peak in low-frequency region
    bi_low  = pdos_bi[freqs<3.0]
    f_low   = freqs[freqs<3.0]
    if len(bi_low)>0:
        bi_peak_f = f_low[np.argmax(bi_low)]
        ax2.axvline(bi_peak_f, color='#d62728', lw=1.0,
                    ls=':', alpha=0.7)
        ax2.text(bi_peak_f+0.05, ax2.get_ylim()[1]*0.85,
                 f'Bi peak\n{bi_peak_f:.1f} THz',
                 color='#d62728', fontsize=9)

    mg_high = pdos_mg[freqs>4.0]
    f_high  = freqs[freqs>4.0]
    if len(mg_high)>0:
        mg_peak_f = f_high[np.argmax(mg_high)]
        ax2.axvline(mg_peak_f, color='#1f77b4', lw=1.0,
                    ls=':', alpha=0.7)
        ax2.text(mg_peak_f+0.05, ax2.get_ylim()[1]*0.65,
                 f'Mg peak\n{mg_peak_f:.1f} THz',
                 color='#1f77b4', fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def plot_combined(band, labels, pdos_dict, syms_prim, outpath):
    """Publication-style: dispersion (left) + PDOS (right) combined."""
    freqs_b   = band['frequencies']
    distances = band['distances']
    dist_flat = np.concatenate(distances)
    freq_flat = np.concatenate(freqs_b)

    freq_p    = pdos_dict['frequency_points']
    pdos_all  = pdos_dict['projected_dos']
    mg_idx    = [i for i,s in enumerate(syms_prim) if s=='Mg']
    bi_idx    = [i for i,s in enumerate(syms_prim) if s=='Bi']
    pdos_mg   = pdos_all[mg_idx].sum(axis=0)
    pdos_bi   = pdos_all[bi_idx].sum(axis=0)
    pdos_tot  = pdos_all.sum(axis=0)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6),
                                  gridspec_kw={'width_ratios':[3,1.5],
                                               'wspace':0.05})
    fig.patch.set_facecolor('white')
    ylim=(freq_flat.min()-0.2, freq_flat.max()+0.3)

    # ── Dispersion ────────────────────────────────────────────────────────────
    ax1.set_facecolor('#f9f9f9')
    for b in range(freq_flat.shape[1]):
        ax1.plot(dist_flat, freq_flat[:,b], 'k-', lw=1.0, alpha=0.8)

    seg_ends=[distances[0][0]]
    for d in distances: seg_ends.append(d[-1])
    for x in seg_ends:
        ax1.axvline(x, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax1.axhline(0, color='#d62728', lw=0.8, ls='--', alpha=0.6)

    ax1.set_xticks(seg_ends)
    ax1.set_xticklabels(labels, fontsize=13)
    ax1.set_ylabel('Frequency (THz)', fontsize=13)
    ax1.set_title('Phonon Dispersion', fontsize=12, fontweight='bold')
    ax1.set_xlim(dist_flat.min(), dist_flat.max())
    ax1.set_ylim(ylim)
    ax1.tick_params(labelsize=11)
    ax1.grid(False)

    # ── PDOS ─────────────────────────────────────────────────────────────────
    ax2.set_facecolor('#f9f9f9')
    ax2.plot(pdos_tot, freq_p, 'k-',  lw=1.5, label='Total')
    ax2.plot(pdos_mg,  freq_p, color='#1f77b4', lw=2.0, label='Mg')
    ax2.plot(pdos_bi,  freq_p, color='#d62728', lw=2.0, ls='--', label='Bi')
    ax2.fill_betweenx(freq_p, 0, pdos_mg, alpha=0.25, color='#1f77b4')
    ax2.fill_betweenx(freq_p, 0, pdos_bi, alpha=0.25, color='#d62728')
    ax2.axhline(0, color='#d62728', lw=0.8, ls='--', alpha=0.6)
    ax2.set_xlabel('PDOS\n(states/THz)', fontsize=11)
    ax2.set_title('PDOS', fontsize=12, fontweight='bold')
    ax2.set_ylim(ylim)
    ax2.yaxis.set_ticklabels([])
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.set_xlim(left=0)

    fig.suptitle('Mg₃Bi₂ Phonon Dispersion & PDOS (NEP)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(description=__doc__,
          formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nep',      required=True)
    ap.add_argument('--pwi',      default='espresso.pwi')
    ap.add_argument('--supercell',nargs=9, type=int,
                    default=[3,0,0, 0,3,0, 0,0,2],
                    help='Supercell for force constants (default: 3×3×2=90 atoms)')
    ap.add_argument('--mesh',     nargs=3, type=int, default=[20,20,20],
                    help='q-mesh for PDOS (default: 20 20 20)')
    ap.add_argument('--sigma',    type=float, default=0.05,
                    help='Gaussian smearing in THz (default: 0.05)')
    ap.add_argument('--disp',     type=float, default=0.01,
                    help='Displacement distance Å (default: 0.01)')
    args=ap.parse_args()

    print("="*62)
    print("PHONON DISPERSION + PDOS — Mg₃Bi₂ (NEP/calorine/phonopy)")
    print("="*62)
    print(f"NEP       : {args.nep}")
    print(f"Supercell : {np.array(args.supercell).reshape(3,3).tolist()}")
    print(f"PDOS mesh : {args.mesh}")
    print()

    # ── Step 1: Build phonopy ─────────────────────────────────────────────────
    print("Step 1: Relaxing + force constants")
    print("-"*45)
    prim=parse_pwi(args.pwi)
    syms_prim=prim.get_chemical_symbols()
    sc_mat=np.array(args.supercell).reshape(3,3)
    ph, prim_relaxed = build_phonopy(prim, args.nep, sc_mat, args.disp)

    # ── Step 2: Dispersion ────────────────────────────────────────────────────
    print("\nStep 2: Phonon dispersion (Γ-M-K-Γ-A-L-H-A)")
    print("-"*45)
    band, labels = compute_dispersion(ph)
    plot_dispersion(band, labels, 'phonon_dispersion_nep.png')

    # ── Step 3: PDOS ─────────────────────────────────────────────────────────
    print("\nStep 3: Projected DOS")
    print("-"*45)
    print(f"  Running on mesh {args.mesh}...")
    ph.run_mesh(args.mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True)
    ph.run_projected_dos(sigma=args.sigma,
                         freq_min=-0.5, freq_max=10.0, freq_pitch=0.01)
    pdos_dict=ph.get_projected_dos_dict()
    freqs_pdos=pdos_dict['frequency_points']
    pdos_data =pdos_dict['projected_dos']   # (n_atoms_prim, n_freq)

    mg_idx=[i for i,s in enumerate(syms_prim) if s=='Mg']
    bi_idx=[i for i,s in enumerate(syms_prim) if s=='Bi']
    pdos_mg  = pdos_data[mg_idx].sum(axis=0)
    pdos_bi  = pdos_data[bi_idx].sum(axis=0)
    pdos_tot = pdos_data.sum(axis=0)

    # Print key statistics
    mask_ac = (freqs_pdos>0.05) & (freqs_pdos<2.5)
    mask_op = freqs_pdos>2.5
    if mask_ac.sum()>0:
        bi_ac_frac = pdos_bi[mask_ac].sum()/pdos_tot[mask_ac].sum()*100
        mg_ac_frac = pdos_mg[mask_ac].sum()/pdos_tot[mask_ac].sum()*100
        print(f"\n  Acoustic region (0–2.5 THz):")
        print(f"    Bi contribution: {bi_ac_frac:.1f}%")
        print(f"    Mg contribution: {mg_ac_frac:.1f}%")
    if mask_op.sum()>0:
        bi_op_frac = pdos_bi[mask_op].sum()/pdos_tot[mask_op].sum()*100
        mg_op_frac = pdos_mg[mask_op].sum()/pdos_tot[mask_op].sum()*100
        print(f"  Optical region (>2.5 THz):")
        print(f"    Bi contribution: {bi_op_frac:.1f}%")
        print(f"    Mg contribution: {mg_op_frac:.1f}%")

    plot_pdos(pdos_dict, syms_prim, 'pdos_nep.png')

    # ── Step 4: Combined publication figure ───────────────────────────────────
    print("\nStep 4: Combined dispersion + PDOS figure")
    print("-"*45)
    plot_combined(band, labels, pdos_dict, syms_prim,
                  'phonon_dispersion_pdos_nep.png')

    # Save raw data
    np.savez('pdos_data.npz',
             frequencies=freqs_pdos,
             pdos_total=pdos_tot,
             pdos_mg=pdos_mg,
             pdos_bi=pdos_bi)
    print("  Saved: pdos_data.npz")

    print("\n"+"="*62)
    print("SUMMARY")
    print("="*62)
    print(f"  Supercell     : {sc_mat.diagonal().tolist()}")
    print(f"  n_disp        : {len(ph.displacements)} displacements")
    print(f"  PDOS mesh     : {args.mesh}")
    bi_peak=freqs_pdos[np.argmax(pdos_bi)]
    mg_peak=freqs_pdos[np.argmax(pdos_mg)]
    print(f"  Bi PDOS peak  : {bi_peak:.2f} THz")
    print(f"  Mg PDOS peak  : {mg_peak:.2f} THz")
    print(f"  Outputs       : phonon_dispersion_nep.png")
    print(f"                  pdos_nep.png")
    print(f"                  phonon_dispersion_pdos_nep.png  ← use this")
    print(f"                  pdos_data.npz")


if __name__=='__main__':
    main()
