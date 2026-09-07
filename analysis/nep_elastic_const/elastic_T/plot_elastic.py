import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
base_dir = "."   # directories: 300/, 400/, ...
temps = [300, 400, 500, 600, 700, 800, 900,1000]

# Containers
C11, C12, C13 = [], [], []
C33, C44 = [], []
Bulk, Shear, Poisson = [], [], []

# ============================================================
# REGEX PATTERNS
# ============================================================
patterns = {
    "C11": r"C11all\s*=\s*([-\d.]+)",
    "C12": r"C12all\s*=\s*([-\d.]+)",
    "C13": r"C13all\s*=\s*([-\d.]+)",
    "C33": r"C33all\s*=\s*([-\d.]+)",
    "C44": r"C44all\s*=\s*([-\d.]+)",

    "Bulk": r"Bulk Modulus\s*=\s*([-\d.]+)",
    "Shear1": r"Shear Modulus 1\s*=\s*([-\d.]+)",
    "Shear2": r"Shear Modulus 2\s*=\s*([-\d.]+)",
    "Poisson": r"Poisson Ratio\s*=\s*([-\d.]+)",
}

# ============================================================
# READ FILES
# ============================================================
for T in temps:
    filepath = os.path.join(base_dir, str(T), "out.dat")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Missing out.dat in {T}")

    with open(filepath, "r") as f:
        text = f.read()

    # Elastic constants
    C11.append(float(re.search(patterns["C11"], text).group(1)))
    C12.append(float(re.search(patterns["C12"], text).group(1)))
    C13.append(float(re.search(patterns["C13"], text).group(1)))
    C33.append(float(re.search(patterns["C33"], text).group(1)))
    C44.append(float(re.search(patterns["C44"], text).group(1)))

    # Moduli
    Bulk.append(float(re.search(patterns["Bulk"], text).group(1)))

    shear1 = float(re.search(patterns["Shear1"], text).group(1))
    shear2 = float(re.search(patterns["Shear2"], text).group(1))
    Shear.append(0.5 * (shear1 + shear2))   # G

    Poisson.append(float(re.search(patterns["Poisson"], text).group(1)))  # ν

# Convert to arrays
temps = np.array(temps)
C11, C12, C13, C33, C44 = map(np.array, [C11, C12, C13, C33, C44])
Bulk, Shear, Poisson = map(np.array, [Bulk, Shear, Poisson])

# ============================================================
# SAVE DATA TO TEXT FILE
# ============================================================
output_file = "Mg3Bi2_elastic_B_G_nu_vs_T.dat"

header = (
    "# Temperature-dependent elastic properties of Mg3Bi2\n"
    "# Columns:\n"
    "# T(K)  C11(GPa)  C12(GPa)  C13(GPa)  C33(GPa)  C44(GPa)  "
    "B(GPa)  G(GPa)  nu\n"
)

data = np.column_stack((
    temps,
    C11, C12, C13, C33, C44,
    Bulk, Shear, Poisson
))

np.savetxt(
    output_file,
    data,
    header=header,
    fmt="%8.1f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %10.5f"
)

print(f"✅ Data saved to {output_file}")

# ============================================================
# PLOT STYLE
# ============================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.2,
})

# ============================================================
# PLOT 1: ELASTIC CONSTANTS
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(temps, C11, "o-", label="C11")
ax.plot(temps, C12, "s-", label="C12")
ax.plot(temps, C13, "^-", label="C13")
ax.plot(temps, C33, "o--", label="C33")
ax.plot(temps, C44, "s--", label="C44")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Elastic constants (GPa)")
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig("elastic_constants_vs_T.png", dpi=300)
plt.show()
plt.close()

# ============================================================
# PLOT 2: BULK & SHEAR MODULUS
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(temps, Bulk, "o-", label="Bulk modulus (B)")
ax.plot(temps, Shear, "s-", label="Shear modulus (G)")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Modulus (GPa)")
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig("bulk_shear_vs_T.png", dpi=300)
plt.show()
plt.close()

# ============================================================
# PLOT 3: POISSON RATIO
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(temps, Poisson, "o-", color="black")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Poisson ratio (ν)")

fig.tight_layout()
fig.savefig("poisson_ratio_vs_T.png", dpi=300)
plt.show()
plt.close()

print("✅ Mg3Bi2 elastic constants, B, G, and ν plotted vs temperature")

