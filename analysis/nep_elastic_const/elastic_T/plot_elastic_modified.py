import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
base_dir = "."
temps = [300, 400, 500, 600, 700, 800, 900, 1000]

# ============================================================
# ELASTIC CONSTANT KEYS (FULL 21)
# ============================================================
elastic_keys = [
    "C11","C22","C33",
    "C12","C13","C23",
    "C44","C55","C66",
    "C14","C15","C16",
    "C24","C25","C26",
    "C34","C35","C36",
    "C45","C46","C56",
]

# Containers
elastic_data = {key: [] for key in elastic_keys}

# ============================================================
# REGEX PATTERNS
# ============================================================
patterns = {
    key: rf"{key}all\s*=\s*([-\d\.Ee+]+)" for key in elastic_keys
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

    for key in elastic_keys:
        match = re.search(patterns[key], text)
        if match:
            elastic_data[key].append(float(match.group(1)))
        else:
            # If missing, assume zero (safe for symmetry-forbidden terms)
            elastic_data[key].append(0.0)

# Convert to arrays
temps = np.array(temps)
for key in elastic_keys:
    elastic_data[key] = np.array(elastic_data[key])

# ============================================================
# SAVE FULL ELASTIC TENSOR VS TEMPERATURE
# ============================================================
output_file = "Mg3Bi2_elastic_tensor_vs_T.dat"

header = (
    "# Full elastic stiffness tensor vs temperature for Mg3Bi2 (P-3m1)\n"
    "# Columns:\n"
    "# T(K)  "
    + "  ".join([f"{k}(GPa)" for k in elastic_keys])
)

data = np.column_stack([temps] + [elastic_data[k] for k in elastic_keys])

np.savetxt(
    output_file,
    data,
    header=header,
    fmt="%8.1f " + " %10.5f"*len(elastic_keys)
)

print(f"✅ Full elastic tensor saved to {output_file}")

# ============================================================
# OPTIONAL: PLOT MAIN ELASTIC CONSTANTS
# ============================================================
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.linewidth": 1.5,
    "lines.linewidth": 2.2,
})

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(temps, elastic_data["C11"], "o-", label="C11")
ax.plot(temps, elastic_data["C12"], "s-", label="C12")
ax.plot(temps, elastic_data["C13"], "^-", label="C13")
ax.plot(temps, elastic_data["C33"], "o--", label="C33")
ax.plot(temps, elastic_data["C44"], "s--", label="C44")

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Elastic constants (GPa)")
ax.legend(frameon=True)

fig.tight_layout()
fig.savefig("elastic_constants_vs_T.png", dpi=300)
plt.show()
plt.close()

print("✅ Elastic constants extracted, saved, and plotted")

