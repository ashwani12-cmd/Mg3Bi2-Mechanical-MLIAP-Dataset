import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Global style — publication quality
# ============================================================
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        16,
    "axes.labelsize":   18,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "legend.fontsize":  18,
    "axes.linewidth":   2.0,
    "lines.linewidth":  2.0,
})

# ============================================================
# Mg3Bi2 electronic band structure
# QE bands.x output
# ============================================================
data = np.loadtxt("mg3bi2_bands.dat.gnu")
k = np.unique(data[:, 0])
bands = np.reshape(data[:, 1], (-1, len(k)))

# ============================================================
# Actual QE Fermi energy
# ============================================================
E_F = 8.0118
# Shift energy relative to Fermi level
bands = bands - E_F

# ============================================================
# High-symmetry points
# ============================================================
xticks = [
    0.0000,
    0.5774,
    0.9107,
    1.5773,
    1.8937,
    2.4711,
    2.8044,
    3.4711
]
xlabels = [
    r'$\Gamma$',
    'M',
    'K',
    r'$\Gamma$',
    'A',
    'L',
    'H',
    'A'
]

# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(
    figsize=(5.0, 4.0),
    dpi=300
)

# Plot all bands
for band in range(len(bands)):
    ax.plot(
        k,
        bands[band, :],
        linewidth=0.8,
        color='black'
    )

# ============================================================
# Fermi level
# ============================================================
ax.axhline(
    0.0,
    linestyle='--',
    linewidth=0.8,
    color='black'
)

# ============================================================
# High-symmetry vertical lines
# ============================================================
for x in xticks:
    ax.axvline(
        x,
        linewidth=0.5,
        color='0.75'
    )

# ============================================================
# Energy range
# ============================================================
ax.set_xlim(k.min(), k.max())
ax.set_ylim(-2.0, 2.0)

# ============================================================
# Axis labels
# ============================================================
ax.set_xticks(xticks)
ax.set_xticklabels(
    xlabels,
    fontsize=10
)
ax.set_ylabel(
    r'Energy $-$ $E_\mathrm{F}$ [eV]',
    fontsize=10
)
ax.set_xlabel("")

# ============================================================
# Tick style
# ============================================================
ax.tick_params(
    axis='both',
    direction='in',
    length=3,
    width=0.7,
    labelsize=9
)
ax.tick_params(
    top=True,
    right=True
)

# ============================================================
# Border
# ============================================================
for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# ============================================================
# No legend / no annotations
# ============================================================
plt.tight_layout()

# ============================================================
# Save
# ============================================================
plt.savefig(
    "Mg3Bi2_band_structure_EF.pdf",
    dpi=600,
    bbox_inches="tight"
)
plt.savefig(
    "Mg3Bi2_band_structure_EF.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

# ============================================================
# Print important values
# ============================================================
VBM = 8.0497
CBM = 8.0326
print("==============================================")
print("Mg3Bi2 Electronic Band Structure")
print("==============================================")
print(f"Fermi energy = {E_F:.4f} eV")
print(f"VBM          = {VBM:.4f} eV")
print(f"CBM          = {CBM:.4f} eV")
print(f"VBM - EF     = {VBM-E_F:+.4f} eV")
print(f"CBM - EF     = {CBM-E_F:+.4f} eV")
print(f"Band overlap = {VBM-CBM:.4f} eV")
print(f"             = {(VBM-CBM)*1000:.1f} meV")
print("==============================================")
