#!/usr/bin/env python3
"""
analyze_diffusion_final.py  —  V_Mg diffusion, Mg3Bi2, NEP-MD
==============================================================
Directory structures supported:
  T{T}/run*/msd.out          ← primary (2000000/T300/run1/msd.out ...)
  T{T}/msd.out               ← fallback single run
  T{T}_N/msd.out             ← old numbered style (T700_1, T700_3 ...)

Plots produced:
  msd_traces.png/.pdf        ← MSD vs time with linear fit
  arrhenius.png/.pdf         ← Arrhenius: black filled circles, error bars,
                                T labels, single legend entry
  D_vs_T.png/.pdf            ← D vs T (same style)

Usage
-----
    # From 2000000/ directory:
    python3 analyze_diffusion_final.py --temps 300 400 500 600 700 800

    # From Mg/ parent with extra directory:
    python3 analyze_diffusion_final.py \\
        --base-dir . --extra-dir 2000000 \\
        --temps 300 400 500 600 700 800 900 1100
"""
import argparse, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from scipy.stats import linregress

kB = 8.617333e-5   # eV/K

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        16,
    "axes.labelsize":   18,
    "xtick.labelsize":  16,   # ← reduced from 16
    "ytick.labelsize":  16,   # ← reduced from 16
    "legend.fontsize":  18,
    "axes.linewidth":   2.0,
    "lines.linewidth":  2.0,
})

BLACK = '#000000'
GREY  = '#666666'

# ── Utilities ──────────────────────────────────────────────────────────────────
def apply_style(ax):
    ax.tick_params(axis='both', which='major', direction='inout', length=8, width=2)
    ax.tick_params(axis='both', which='minor', direction='in',    length=5, width=1.5)
    ax.minorticks_on()
    ax.set_aspect("auto")

def load_msd(path):
    data = np.loadtxt(path)
    if data.ndim == 1 or len(data) < 5:
        raise ValueError(f"Too few rows in {path}")
    data = data[1:] if data[0,0] == 0 else data
    return (data[:,0],
            data[:,1], data[:,2], data[:,3],
            data[:,4], data[:,5], data[:,6])

def fit_D(t_ps, mx, my, mz, fs, fe):
    msd = mx + my + mz
    n   = len(t_ps)
    i0, i1 = int(n*fs), int(n*fe)
    sl, ic, r, _, se = linregress(t_ps[i0:i1]*1e-12, msd[i0:i1])
    return sl/6*1e-20, se/6*1e-20, r**2, sl, ic, msd

def log_errbar(D, sem):
    """Asymmetric error bars for log scale."""
    eu = D*(np.exp( sem/D) - 1)
    ed = D*(1 - np.exp(-sem/D))
    return ed, eu

def find_msd_files(search_dirs, T):
    """Find all msd.out for temperature T across all search dirs."""
    paths = []
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for d in sorted(glob.glob(os.path.join(base, f'T{T}*'))):
            dname = os.path.basename(d)
            if not (dname == f'T{T}' or dname.startswith(f'T{T}_')):
                continue
            # direct msd.out
            p = os.path.join(d, 'msd.out')
            if os.path.exists(p):
                paths.append(p)
            # nested run*/msd.out
            for sub in sorted(glob.glob(os.path.join(d, '*', 'msd.out'))):
                paths.append(sub)
    return paths

# ── Plots ──────────────────────────────────────────────────────────────────────
def save(fig, name, outdir):
    for ext in ['png', 'pdf']:
        p = os.path.join(outdir, f'{name}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}.png/.pdf")

def plot_msd(results, fstart, fend, outdir):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
    fig, ax = plt.subplots(figsize=(5, 4))
    for r, col in zip(results, colors):
        t_ps = r['t']
        # mean MSD — bold solid line
        ax.plot(t_ps, r['msd_mean'], color=col, lw=2.0,
                label=f"{r['T']} K (N={r['n']})")
        # linear fit — dashed black over colour
        n  = len(t_ps); i0=int(n*fstart); i1=int(n*fend)
        tf = t_ps[i0:i1]*1e-12
        ax.plot(t_ps[i0:i1], r['slope']*tf+r['icpt'],
                color='black', lw=1.5, ls='--', alpha=0.7)
    ax.set_xlabel('Time [ps]')
    ax.set_ylabel(r'MSD [Å$^2$]')
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    apply_style(ax)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(frameon=True, framealpha=0.9, edgecolor='gray',
              loc='upper left', ncol=2, fontsize=12)
    fig.tight_layout(pad=1.5)
    save(fig, 'msd_traces', outdir)

def plot_arrhenius(results, Ea, D0, R2a, outdir):
    T_arr  = np.array([r['T'] for r in results])
    D_arr  = np.array([r['D'] for r in results])
    T_fine = np.linspace(T_arr.min()*0.85, T_arr.max()*1.1, 300)
    D_fit  = D0 * np.exp(-Ea/(kB*T_fine))

    fig, ax = plt.subplots(figsize=(5, 4))

    # Fit line
    ax.plot(1/(kB*T_fine), D_fit, '--', color='#d62728', lw=2.5, zorder=1)

    # Data points — black filled circles + error bars
    for r in results:
        ed, eu = log_errbar(r['D'], r['err'])
        ax.errorbar(1/(kB*r['T']), r['D'],
                    yerr=[[ed],[eu]],
                    fmt='o',
                    color=BLACK,
                    markersize=10,
                    markerfacecolor=BLACK,
                    markeredgecolor=BLACK,
                    markeredgewidth=1.5,
                    capsize=6, capthick=2.5, elinewidth=2.0,
                    zorder=5)

    # Single clean legend — reduced fontsize
    handles = [
        Line2D([0],[0], color='#d62728', ls='--', lw=2.5,
               label=rf'Arrhenius fit'
                     ),
        Line2D([0],[0], marker='o', ls='none',
               color=BLACK, markersize=8, markerfacecolor=BLACK,
               label=r'$D_{\mathrm{V_{Mg}}}$'),
    ]
    ax.legend(handles=handles, fontsize=12, frameon=True,       # ← 13 → 9
              framealpha=0.9, edgecolor='gray', loc='best')

    ax.set_yscale('log')
    ax.set_xlabel(r'$1/k_BT$ [eV$^{-1}$]')
    ax.set_ylabel(r'$D$ [m$^2$ s$^{-1}$]')
    apply_style(ax)

    # Top T axis — reduced tick label fontsize
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([1/(kB*t) for t in T_arr])
    ax2.set_xticklabels([str(t) for t in T_arr], fontsize=16, rotation=45)  # ← 13 → 10
    ax2.set_xlabel('Temperature [K]', fontsize=18)
    ax2.tick_params(direction='in', length=6, width=1.5)

    fig.tight_layout(pad=1.5)
    save(fig, 'arrhenius', outdir)

def plot_D_vs_T(results, Ea, D0, outdir):
    T_arr  = np.array([r['T'] for r in results])
    D_arr  = np.array([r['D'] for r in results])
    T_fine = np.linspace(T_arr.min()*0.85, T_arr.max()*1.1, 300)
    D_fit  = D0 * np.exp(-Ea/(kB*T_fine))

    fig, ax = plt.subplots(figsize=(5, 4))

    # Arrhenius fit curve
    ax.plot(T_fine, D_fit, '--', color='#d62728', lw=2.5, zorder=1)

    # Data points — black filled circles + error bars
    for r in results:
        ed, eu = log_errbar(r['D'], r['err'])
        ax.errorbar(r['T'], r['D'],
                    yerr=[[ed],[eu]],
                    fmt='o',
                    color=BLACK,
                    markersize=10,
                    markerfacecolor=BLACK,
                    markeredgecolor=BLACK,
                    markeredgewidth=1.5,
                    capsize=6, capthick=2.5, elinewidth=2.0,
                    zorder=5)

    # Single legend
    handles = [
        Line2D([0],[0], color='#d62728', ls='--', lw=2.5,
               label=rf'Arrhenius fit'),
        Line2D([0],[0], marker='o', ls='none',
               color=BLACK, markersize=10, markerfacecolor=BLACK,
               label=r'$D_{\mathrm{V_{Mg}}}$'),
    ]
    ax.legend(handles=handles, fontsize=12, frameon=True,       # ← 10 → 9
              framealpha=0.9, edgecolor='gray', loc='best')

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(r'$D$ [m$^2$ s$^{-1}$]')
    ax.set_yscale('log')
    apply_style(ax)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout(pad=1.5)
    save(fig, 'D_vs_T', outdir)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-dir',  default='.',
                    help='Base directory (default: .)')
    ap.add_argument('--extra-dir', default=None,
                    help='Additional search directory (e.g. 2000000)')
    ap.add_argument('--temps',     nargs='+', type=int, required=True,
                    help='Temperatures to include e.g. --temps 300 400 500 600')
    ap.add_argument('--fit-start', type=float, default=0.3,
                    help='Start fraction for linear fit (default 0.3)')
    ap.add_argument('--fit-end',   type=float, default=0.85,
                    help='End fraction for linear fit (default 0.85)')
    ap.add_argument('--outdir',    default='.',
                    help='Output directory (default: .)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    search_dirs = [args.base_dir]
    if args.extra_dir:
        search_dirs.append(args.extra_dir)

    print("="*65)
    print("Mg VACANCY DIFFUSION — Mg₃Bi₂ (NEP-MD)")
    print("="*65)
    print(f"Search : {[os.path.abspath(d) for d in search_dirs]}")
    print(f"Temps  : {args.temps}")
    print(f"Fit    : {args.fit_start*100:.0f}%–{args.fit_end*100:.0f}%\n")

    results = []

    for T in args.temps:
        files = find_msd_files(search_dirs, T)
        if not files:
            print(f"  T={T:>5}K : no msd.out → skip"); continue

        print(f"  T={T:>5}K : {len(files)} file(s)")
        D_list, r2_list = [], []
        slope_list, icpt_list = [], []
        msd_list, mx_list, my_list, mz_list = [], [], [], []
        t_ref = None

        for fpath in files:
            tag = os.path.basename(os.path.dirname(fpath))
            try:
                t, mx, my, mz, sx, sy, sz = load_msd(fpath)
                D, De, r2, sl, ic, msd = fit_D(
                    t, mx, my, mz, args.fit_start, args.fit_end)
                if D > 0:
                    D_list.append(D); r2_list.append(r2)
                    slope_list.append(sl); icpt_list.append(ic)
                    msd_list.append(msd)
                    mx_list.append(mx); my_list.append(my); mz_list.append(mz)
                    t_ref = t
                    print(f"    {tag:<8}: D={D:.3e}  R²={r2:.4f}")
            except Exception as e:
                print(f"    {tag}: SKIP — {e}")

        if not D_list: continue

        D_arr = np.array(D_list)

        # Select best 3 runs: closest to median (reduces outlier effect)
        if len(D_arr) > 3:
            med   = np.median(D_arr)
            idx3  = np.argsort(np.abs(D_arr - med))[:3]
            D_arr = D_arr[idx3]
            print(f"    Best-3 selected (closest to median {med:.3e}):")
            for i in idx3:
                print(f"      run idx {i}: D={D_list[i]:.3e}")

        D_mean = D_arr.mean()
        D_std  = D_arr.std(ddof=1) if len(D_arr)>1 else abs(D_arr[0])*0.1
        D_sem  = D_std / np.sqrt(len(D_arr))

        print(f"    → D = {D_mean:.3e} ± {D_sem:.2e} m²/s  "
              f"(N={len(D_arr)}, CV={D_std/D_mean*100:.1f}%)\n")

        results.append({
            'T':        T,
            'D':        D_mean,
            'err':      D_sem,
            'std':      D_std,
            'n':        len(D_arr),
            'r2':       np.mean(r2_list),
            'slope':    np.mean(slope_list),
            'icpt':     np.mean(icpt_list),
            't':        t_ref,
            'msd_mean': np.mean(msd_list, axis=0),
            'msd_x':    np.mean(mx_list,  axis=0),
            'msd_y':    np.mean(my_list,  axis=0),
            'msd_z':    np.mean(mz_list,  axis=0),
            'all_msds': msd_list,
            'all_D':    list(D_arr),
        })

    if not results:
        print("No data found."); sys.exit(1)

    # Arrhenius fit
    T_arr  = np.array([r['T'] for r in results])
    D_arr  = np.array([r['D'] for r in results])
    x = 1/(kB*T_arr); y = np.log(D_arr)
    sl_a, int_a, r_a, _, _ = linregress(x, y)
    Ea = -sl_a; D0 = np.exp(int_a); R2a = r_a**2

    print("="*55)
    print(f"  Ea  = {Ea:.4f} eV")
    print(f"  D0  = {D0:.3e} m²/s")
    print(f"  R²  = {R2a:.5f}")
    print(f"  Expt (Liu 2017)  ≈ 0.19 eV")
    print(f"  DFT-NEB (Assadi) ≈ 0.27–0.30 eV")
    print("="*55)

    # CSV
    csv = os.path.join(args.outdir, 'diffusion_results.csv')
    with open(csv,'w') as f:
        f.write('T_K,D_m2s,D_err_m2s,D_std_m2s,N_runs,R2\n')
        for r in results:
            f.write(f"{r['T']},{r['D']:.6e},{r['err']:.6e},"
                    f"{r['std']:.6e},{r['n']},{r['r2']:.6f}\n")
        f.write(f'\nEa_eV,{Ea:.6f}\nD0_m2s,{D0:.6e}\nArrhenius_R2,{R2a:.6f}\n')
    print(f"\n  Saved: {csv}")

    print("\nGenerating plots...")
    plot_msd(results, args.fit_start, args.fit_end, args.outdir)
    plot_arrhenius(results, Ea, D0, R2a, args.outdir)
    plot_D_vs_T(results, Ea, D0, args.outdir)
    print("\nDone ✓")

if __name__ == '__main__':
    main()
