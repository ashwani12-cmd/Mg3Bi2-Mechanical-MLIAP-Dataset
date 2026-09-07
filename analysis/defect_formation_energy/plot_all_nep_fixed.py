# ── Drop-in replacement for plot_all_mep() in nep_neb_mg_all_paths.py ─────────

def plot_all_mep(results, outdir):
    """
    results dict:
      'Oct→Oct', 'Tet→Tet', 'Oct→Tet'  →  (E_rel, Ea, dist)
      'Oct→Tet→Oct'                     →  (E_stitched, Ea_eff, None)
    """
    from matplotlib.lines import Line2D

    DFT_REF = {'Oct→Oct': 0.66, 'Tet→Tet': 0.27, 'Oct→Tet': 0.30}
    STYLES  = {
        'Oct→Oct':     ('#d62728', 's', '--'),
        'Tet→Tet':     ('#1f77b4', 'o', '-'),
        'Oct→Tet':     ('#2ca02c', '^', '-.'),
        'Oct→Tet→Oct': ('#9467bd', 'D', ':'),
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for path, (E_rel, Ea, _) in results.items():
        col, mk, ls = STYLES[path]
        rxn = np.linspace(0, 1, len(E_rel))
        dft = DFT_REF.get(path)
        if dft:
            ax.axhline(dft * 1000, color=col, ls=':', lw=1.0, alpha=0.22)
        ax.plot(rxn, E_rel * 1000, ls + mk, color=col,
                markersize=7, lw=2.0,
                markeredgecolor='white', markeredgewidth=0.6)

    # ── 1-column legend, no line-wrapping, plain $\to$ arrows ──────────────
    legend_entries = []
    for path, (E_rel, Ea, _) in results.items():
        col, mk, ls = STYLES[path]
        dft = DFT_REF.get(path)
        arrow = path.replace('→', r'$\to$')
        if dft:
            lbl = (rf'{arrow}: $E_a^\mathrm{{NEP}}$={Ea:.3f} eV'
                   rf' ($E_a^\mathrm{{DFT}}$={dft:.2f} eV)')
        else:
            lbl = rf'{arrow}: $E_a^\mathrm{{eff}}$={Ea:.3f} eV (two-step)'
        legend_entries.append(
            Line2D([0],[0], color=col, ls=ls, marker=mk, markersize=6,
                   markeredgecolor='white', markeredgewidth=0.6,
                   lw=2.0, label=lbl)
        )

    ax.legend(handles=legend_entries,
              fontsize=8.5,
              frameon=True, framealpha=0.92, edgecolor='#aaaaaa',
              loc='upper right',
              ncol=1,
              handlelength=2.2,
              handletextpad=0.5,
              borderpad=0.6,
              labelspacing=0.45)

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
