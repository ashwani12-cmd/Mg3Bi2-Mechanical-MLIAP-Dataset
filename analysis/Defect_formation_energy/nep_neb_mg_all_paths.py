#!/usr/bin/env python3
"""
nep_neb_mg_all_paths.py
========================
NEP-NEB for ALL Mg vacancy migration paths in Mg3Bi2. Produces a single
MEP plot with four mechanisms:

  1. Oct→Oct   (direct, high-barrier)
  2. Tet→Tet   (dominant 1-hop, lowest barrier)
  3. Oct→Tet   (asymmetric single hop)
  4. Oct→Tet→Oct  (two-step, effective barrier stitched from legs 1+2)

DFT+SOC references (Assadi et al., Chem. Phys. Lett. 801, 2022):
  Oct→Oct  = 0.66 eV
  Tet→Tet  = 0.27 eV
  Oct→Tet  = 0.30 eV

Usage
-----
    python3 nep_neb_mg_all_paths.py \\
        --nep  nep_y2026_m09_d04_h13_m17_s19_generation300000.txt \\
        --pwi  perfect_1.pwo \\
        --n-images 7 --fmax 0.05

Options
-------
  --gpu          Use GPUNEP instead of CPUNEP
  --outdir DIR   Output directory (default: neb_all_paths)
  --relax-fmax   Force threshold for endpoint relaxation (default 1e-4)
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from ase.mep import NEB
try:
    from ase.filters import UnitCellFilter
except ImportError:
    from ase.constraints import UnitCellFilter
try:
    from calorine.calculators import CPUNEP, GPUNEP
except ImportError:
    raise ImportError("pip install calorine")

# ── Plot style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        16,
    "axes.labelsize":   18,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  12,
    "axes.linewidth":   2.0,
    "lines.linewidth":  2.0,
})

DFT_REF = {'Oct→Oct': 0.66, 'Tet→Tet': 0.27, 'Oct→Tet': 0.30}

STYLES = {
    'Oct→Oct':     ('#d62728', 's', '--'),   # red  square  dashed
    'Tet→Tet':     ('#1f77b4', 'o', '-'),    # blue circle  solid
    'Oct→Tet':     ('#2ca02c', '^', '-.'),   # green triangle dash-dot
    'Oct→Tet→Oct': ('#9467bd', 'D', ':'),    # purple diamond dotted
}

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def mic_distance(p1, p2, cell):
    ci   = np.linalg.inv(cell)
    diff = p2 - p1
    frac = diff @ ci
    frac -= np.round(frac)
    return np.linalg.norm(frac @ cell)


def classify_mg_sites(atoms, r_cut=3.60):
    """
    Classify Mg by Bi coordination number (CN).
      CN >= 6  →  Octahedral (1a site)
      CN <= 4  →  Tetrahedral (2d site)
    """
    syms = np.array(atoms.get_chemical_symbols())
    pos  = atoms.get_positions()
    cell = atoms.get_cell().array
    ci   = np.linalg.inv(cell)
    bi   = pos[syms == 'Bi']

    mg_oct, mg_tet = [], []
    for i in np.where(syms == 'Mg')[0]:
        d  = bi - pos[i]
        fr = d @ ci
        fr -= np.round(fr)
        cn = int((np.linalg.norm(fr @ cell, axis=-1) < r_cut).sum())
        if cn >= 6:
            mg_oct.append(int(i))
        else:
            mg_tet.append(int(i))
    return mg_oct, mg_tet


def find_nn_pairs(idx_a, idx_b, atoms, n=5, dmin=0.0):
    pos  = atoms.get_positions()
    cell = atoms.get_cell().array
    pairs = []
    for ia in idx_a:
        for ib in idx_b:
            if ia == ib:
                continue
            d = mic_distance(pos[ia], pos[ib], cell)
            if d > dmin:
                pairs.append((d, ia, ib))
    pairs.sort()
    return pairs[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# Vacancy pair builder (consistent atom ordering for ASE NEB)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ordered_vacancy_pair(perfect, remove_a, remove_b):
    """
    Build (init, final) vacancy cells with IDENTICAL atom ordering,
    as required by ASE NEB.

    init  : vacancy at A, Mg_B present
    final : vacancy at B, Mg_A placed at slot_B (matches init ordering)
    """
    N    = len(perfect)
    syms = perfect.get_chemical_symbols()
    pos  = perfect.get_positions()
    cell = perfect.get_cell()
    pbc  = perfect.get_pbc()

    init_indices = [i for i in range(N) if i != remove_a]
    slot_B       = sum(1 for i in range(remove_b) if i != remove_a)

    final_indices    = [i for i in range(N) if i != remove_b]
    slot_A_in_final  = sum(1 for i in range(remove_a) if i != remove_b)

    perm = [None] * (N - 1)
    perm[slot_B] = slot_A_in_final
    for k, orig in enumerate(init_indices):
        if k == slot_B:
            continue
        perm[k] = final_indices.index(orig)

    init_syms = [syms[i] for i in init_indices]
    init_pos  = np.array([pos[i]  for i in init_indices])

    fn_syms = [syms[i] for i in final_indices]
    fn_pos  = np.array([pos[i]  for i in final_indices])

    final_syms = [fn_syms[perm[k]] for k in range(N - 1)]
    final_pos  = np.array([fn_pos[perm[k]] for k in range(N - 1)])

    if init_syms != final_syms:
        raise RuntimeError("Symbol mismatch after reordering — check site types.")

    init_atoms  = Atoms(symbols=init_syms,  positions=init_pos,  cell=cell, pbc=pbc)
    final_atoms = Atoms(symbols=final_syms, positions=final_pos, cell=cell, pbc=pbc)
    return init_atoms, final_atoms


# ═══════════════════════════════════════════════════════════════════════════════
# Relaxation
# ═══════════════════════════════════════════════════════════════════════════════

def relax_atoms(atoms, calc, fmax=1e-4, label=''):
    atoms = atoms.copy()
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=500000)
    E    = atoms.get_potential_energy()
    Fmax = np.sqrt((atoms.get_forces()**2).sum(axis=1).max())
    print(f"  {label:<40}: E={E:.6f} eV  |F|max={Fmax:.2e} eV/Å")
    return atoms, E


def make_vacancy_endpoints(perfect, ra, rb, calc_fn, fmax=1e-4, same_type=True):
    init_raw, final_raw = build_ordered_vacancy_pair(perfect, ra, rb)
    init,  Ei = relax_atoms(init_raw,  calc_fn(), fmax=fmax, label=f'init  (vac@{ra})')
    final, Ef = relax_atoms(final_raw, calc_fn(), fmax=fmax, label=f'final (vac@{rb})')

    dE  = abs(Ei - Ef)
    thr = 0.010 if same_type else 0.150
    if dE > thr:
        print(f"  ⚠  |Ei−Ef| = {dE*1000:.1f} meV  (threshold {thr*1000:.0f} meV)")
    else:
        if not same_type:
            print(f"  Mixed hop: ΔE = {(Ef-Ei)*1000:+.1f} meV  (asymmetric, expected)")
    return init, final, Ei, Ef


# ═══════════════════════════════════════════════════════════════════════════════
# NEB
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_mic(images):
    """Linear interpolation in fractional coords with MIC."""
    init     = images[0]
    final    = images[-1]
    n_int    = len(images) - 2
    cell     = init.get_cell().array
    ci       = np.linalg.inv(cell)
    pos_i    = init.get_positions()
    pos_f    = final.get_positions()
    df       = (pos_f - pos_i) @ ci
    df      -= np.round(df)
    dc       = df @ cell
    for k, img in enumerate(images[1:-1], start=1):
        img.set_positions(pos_i + k / (n_int + 1) * dc)


def run_neb(init, final, calc_fn, n_images=7, fmax=0.05, label='',
            climb=True):
    print(f"\n  NEB [{label}]  n_images={n_images}  fmax={fmax}  CI={climb}")

    images = []
    for x in [init] + [None] * n_images + [final]:
        img = init.copy() if x is None else x.copy()
        img.calc = calc_fn()
        images.append(img)

    neb  = NEB(images, climb=climb, method='improvedtangent',
               allow_shared_calculator=False)
    interpolate_mic(images)

    conv = FIRE(neb, logfile=None).run(fmax=fmax, steps=3000)
    E    = np.array([img.get_potential_energy() for img in images])
    E   -= E[0]
    Ea   = E.max()
    i_ts = int(E.argmax())
    print(f"  → Ea = {Ea:.4f} eV  (TS @ image {i_ts})  "
          + ("converged" if conv else "⚠ NOT converged"))
    return E, Ea, images[i_ts]


# ═══════════════════════════════════════════════════════════════════════════════
# Plot — all 4 mechanisms on one axes
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all_mep(results, outdir):
    label_map = {
        'Oct→Oct':     r'Oct$\to$Oct',
        'Tet→Tet':     r'Tet$\to$Tet',
        'Oct→Tet':     r'Oct$\to$Tet',
        'Oct→Tet→Oct': r'Oct$\to$Tet$\to$Oct',
    }
    fig, ax = plt.subplots(figsize=(5, 4))

    for path, (E_rel, Ea, _) in results.items():
        col, mk, ls = STYLES[path]
        rxn = np.linspace(0, 1, len(E_rel))
        ax.plot(rxn, E_rel * 1000, ls + mk, color=col,
                markersize=7, lw=2.0,
                markeredgecolor='white', markeredgewidth=0.6)

    legend_entries = []
    for path in results:
        col, mk, ls = STYLES[path]
        legend_entries.append(
            Line2D([0],[0], color=col, ls=ls, marker=mk, markersize=6,
                   markeredgecolor='white', markeredgewidth=0.6,
                   lw=2.0, label=label_map[path])
        )

    ax.legend(handles=legend_entries,
              fontsize=10,
              frameon=True, framealpha=0.92, edgecolor='#aaaaaa',
              loc='lower center',
              bbox_to_anchor=(0.5, 1.02),   # ← above the axes
              ncol=2,                        # ← 2 columns
              handlelength=2.0,
              handletextpad=0.4,
              borderpad=0.5,
              columnspacing=1.0)
    ax.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.4)
    ax.set_xlabel('Reaction coordinate')
    ax.set_ylabel('Energy [meV]')
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2)
    ax.tick_params(axis='both', which='minor', direction='in',    length=5, width=1.5)
    ax.minorticks_on()
    fig.tight_layout(pad=1.5)

    for ext in ['png', 'pdf']:
        p = os.path.join(outdir, f'neb_mep_all_paths.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: neb_mep_all_paths.png/.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nep',        required=True)
    ap.add_argument('--pwi',        default='perfect_1.pwo')
    ap.add_argument('--n-images',   type=int,   default=7)
    ap.add_argument('--fmax',       type=float, default=0.05)
    ap.add_argument('--relax-fmax', type=float, default=1e-4)
    ap.add_argument('--gpu',        action='store_true')
    ap.add_argument('--outdir',     default='neb_all_paths')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    CalcClass = GPUNEP if args.gpu else CPUNEP
    def make_calc(): return CalcClass(args.nep)

    print("=" * 65)
    print("NEP-NEB: ALL Mg VACANCY PATHS — Mg₃Bi₂")
    print("=" * 65)
    print(f"NEP    : {args.nep}")
    print(f"Cell   : {args.pwi}")
    print(f"Images : {args.n_images}  |  fmax = {args.fmax} eV/Å\n")

    # ── Step 1: Load + relax perfect cell ─────────────────────────────────────
    fmt = 'espresso-out' if args.pwi.endswith('.pwo') else 'espresso-in'
    raw = read(args.pwi, index=-1, format=fmt)
    print(f"Loaded: {len(raw)} atoms")
    perfect, _ = relax_atoms(raw, make_calc(),
                             fmax=args.relax_fmax, label='perfect supercell')

    # ── Step 2: Classify Mg ───────────────────────────────────────────────────
    print("\nClassifying Mg sites...")
    mg_oct, mg_tet = classify_mg_sites(perfect)
    print(f"  Octahedral (CN=6, 1a): {len(mg_oct)} sites")
    print(f"  Tetrahedral (CN=4, 2d): {len(mg_tet)} sites")
    if len(mg_oct) < 2 or len(mg_tet) < 2:
        print("ERROR: Too few Mg sites classified — check structure.")
        sys.exit(1)

    pos  = perfect.get_positions()
    cell = perfect.get_cell().array

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # PATH 1: Oct → Oct  (direct single hop)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PATH 1: Oct → Oct  (direct)")
    pairs = find_nn_pairs(mg_oct, mg_oct, perfect, n=5)
    d, ia, ib = pairs[0]
    print(f"  Selected: #{ia} ↔ #{ib}  dist={d:.3f} Å")
    init, final, Ei, Ef = make_vacancy_endpoints(
        perfect, ia, ib, make_calc, fmax=args.relax_fmax, same_type=True)
    E_oo, Ea_oo, ts = run_neb(init, final, make_calc,
                               n_images=args.n_images, fmax=args.fmax,
                               label='Oct→Oct')
    results['Oct→Oct'] = (E_oo, Ea_oo, d)
    write(os.path.join(args.outdir, 'ts_Oct_Oct.xyz'), ts)

    # ══════════════════════════════════════════════════════════════════════════
    # PATH 2: Tet → Tet  (dominant 1-hop)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PATH 2: Tet → Tet  (dominant mechanism)")
    pairs = find_nn_pairs(mg_tet, mg_tet, perfect, n=5)
    d, ia, ib = pairs[0]
    print(f"  Selected: #{ia} ↔ #{ib}  dist={d:.3f} Å")
    init, final, Ei, Ef = make_vacancy_endpoints(
        perfect, ia, ib, make_calc, fmax=args.relax_fmax, same_type=True)
    E_tt, Ea_tt, ts = run_neb(init, final, make_calc,
                               n_images=args.n_images, fmax=args.fmax,
                               label='Tet→Tet')
    results['Tet→Tet'] = (E_tt, Ea_tt, d)
    write(os.path.join(args.outdir, 'ts_Tet_Tet.xyz'), ts)

    # ══════════════════════════════════════════════════════════════════════════
    # PATH 3: Oct → Tet  (asymmetric single hop)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("PATH 3: Oct → Tet  (asymmetric single hop)")
    pairs = find_nn_pairs(mg_oct, mg_tet, perfect, n=5)
    d, ia, ib = pairs[0]
    print(f"  Selected: #{ia} ↔ #{ib}  dist={d:.3f} Å")
    init, final, Ei, Ef = make_vacancy_endpoints(
        perfect, ia, ib, make_calc, fmax=args.relax_fmax, same_type=False)
    E_ot, Ea_ot, ts = run_neb(init, final, make_calc,
                               n_images=args.n_images, fmax=args.fmax,
                               label='Oct→Tet')
    results['Oct→Tet'] = (E_ot, Ea_ot, d)
    write(os.path.join(args.outdir, 'ts_Oct_Tet.xyz'), ts)

    # ══════════════════════════════════════════════════════════════════════════
    # PATH 4: Oct → Tet → Oct  (two-step; stitch legs from paths above)
    # ══════════════════════════════════════════════════════════════════════════
    # Strategy: pick Oct_A, find its nearest Tet_M, then find nearest Oct_B
    # to Tet_M (different from A). Run two NEB legs: A→M and M→B.
    # Stitch: E_stitched = [E_leg1 | offset + E_leg2], renormalised to 0.
    # Effective barrier = max of stitched profile.
    print("\n" + "─"*60)
    print("PATH 4: Oct → Tet → Oct  (two-step, stitched)")

    A   = mg_oct[0]
    M   = min(mg_tet, key=lambda m: mic_distance(pos[A], pos[m], cell))
    B   = min([o for o in mg_oct if o != A],
              key=lambda o: mic_distance(pos[M], pos[o], cell))
    dAM = mic_distance(pos[A], pos[M], cell)
    dMB = mic_distance(pos[M], pos[B], cell)
    print(f"  Oct_A=#{A}  Tet_M=#{M}  Oct_B=#{B}")
    print(f"  d(A,M)={dAM:.3f} Å   d(M,B)={dMB:.3f} Å")

    # Leg 1: A → M
    init_AM, final_AM, E_A, E_M = make_vacancy_endpoints(
        perfect, A, M, make_calc, fmax=args.relax_fmax, same_type=False)
    E_leg1, _, ts1 = run_neb(init_AM, final_AM, make_calc,
                              n_images=args.n_images, fmax=args.fmax,
                              label='Oct→Tet (leg 1)')
    write(os.path.join(args.outdir, 'ts_OctTetOct_leg1.xyz'), ts1)

    # Leg 2: M → B
    init_MB, final_MB, E_M2, E_B = make_vacancy_endpoints(
        perfect, M, B, make_calc, fmax=args.relax_fmax, same_type=False)
    E_leg2, _, ts2 = run_neb(init_MB, final_MB, make_calc,
                              n_images=args.n_images, fmax=args.fmax,
                              label='Tet→Oct (leg 2)')
    write(os.path.join(args.outdir, 'ts_OctTetOct_leg2.xyz'), ts2)

    # Site energy offset between Oct and Tet vacancies
    offset = E_M - E_A   # eV  (positive → Tet vac higher in energy)

    # Stitch: leg1 on [0, 0.5], leg2 (shifted by offset) on [0.5, 1.0]
    E1_half = E_leg1                        # already zero at start
    E2_half = offset + E_leg2              # shift so it starts at E_M level

    n1 = len(E1_half); n2 = len(E2_half)
    # Interpolate both onto equal length for clean plotting
    n_pts   = max(n1, n2)
    x1 = np.linspace(0.0, 0.5, n1)
    x2 = np.linspace(0.5, 1.0, n2)
    xi = np.linspace(0.0, 1.0, n_pts * 2 - 1)
    E_stitch = np.interp(xi,
                         np.concatenate([x1, x2[1:]]),
                         np.concatenate([E1_half, E2_half[1:]]))

    Ea_eff = E_stitch.max()
    print(f"\n  Stitched two-step barrier  Ea_eff = {Ea_eff:.4f} eV")
    print(f"  (direct Oct→Oct was {results['Oct→Oct'][1]:.4f} eV)")

    results['Oct→Tet→Oct'] = (E_stitch, Ea_eff, None)
    np.savez(os.path.join(args.outdir, 'oct_tet_oct_mep.npz'),
             E_leg1=E_leg1, E_leg2=E_leg2, E_stitch=E_stitch, offset=offset)

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\n  {'Path':<16} {'Ea NEP (eV)':>13} {'Ea DFT (eV)':>12}")
    print(f"  {'─' * 45}")
    for path, (E, Ea, _) in results.items():
        dft = DFT_REF.get(path, '—')
        dft_str = f"{dft:.2f}" if isinstance(dft, float) else dft
        print(f"  {path:<16} {Ea:>13.4f} {dft_str:>12}")
    print(f"\n  Ref: Assadi et al., Chem. Phys. Lett. 801 (2022) 139694")

    # CSV
    csv = os.path.join(args.outdir, 'neb_barriers_all.csv')
    with open(csv, 'w') as f:
        f.write('path,Ea_nep_eV,Ea_dft_eV\n')
        for path, (E, Ea, _) in results.items():
            dft = DFT_REF.get(path, '')
            f.write(f"{path},{Ea:.6f},{dft}\n")
    print(f"  CSV → {csv}")

    # Plot
    print("\nGenerating plot...")
    plot_all_mep(results, args.outdir)
    print("\nDone ✓")


if __name__ == '__main__':
    main()
