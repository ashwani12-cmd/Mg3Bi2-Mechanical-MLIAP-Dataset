# Mg3Bi2-Mechanical-MLIAP-Dataset

This repository contains a curated dataset of mechanically perturbed **Mg₃Bi₂** atomic configurations
designed for **machine-learning interatomic potential (MLIAP / SNAP) training** and for studying
mechanical response under **pressure, strain, and atomic disorder**.

The dataset primarily focuses on nonlinear mechanical regimes; however, it also includes equilibrium (0% strain) reference structures within the uniaxial strain datasets to ensure consistency and accurate learning of elastic responses.

## Data (`data/`)

This directory contains the **physics-organized raw Mg₃Bi₂ atomic configurations**, grouped according to distinct **mechanical deformation protocols**, including:

- Atomic displacement
- Lattice strain
- Hydrostatic pressure
- Combined pressure–strain loading
- Uniaxial deformation

Each subfolder preserves the **physical meaning and generation logic** of the configurations, enabling clear interpretation of how different mechanical perturbations affect atomic structures, forces, and stresses.

These raw datasets serve as the **foundational source** from which all ML-ready formats are derived. This organization ensures:

- Full **transparency** in data generation
- Clear **traceability** between physical deformation modes and ML training data
- Consistent linkage between **mechanical physics** and **machine-learning model inputs**


## Dataset (`dataset/`)

This directory contains **machine-learning–ready datasets** derived from curated **Mg₃Bi₂ atomic configurations**, formatted for direct use with multiple machine-learning interatomic potential frameworks, including:

- **MTP / MLIP**
- **DeepMD**
- **XYZ-based models** (e.g., GPUMD / NEP)

The datasets are provided in the following formats:

- `.cfg`
- Extended `.xyz`
- DeepMD-compatible `.json`

All formats contain **identical atomic environments**, including energies, forces, and stresses. This guarantees:

- Fair **cross-framework benchmarking**
- Fully **reproducible training**
- Consistent evaluation of **elastic and nonlinear mechanical properties**
- Elimination of **format-dependent bias**

---

## Scripts (`scripts/`)

This directory contains **conversion and preprocessing scripts and notebooks** used to transform physics-organized raw data into ML-ready datasets compatible with different training pipelines.

The tools in this folder:

- Enable **transparent and loss-free conversion** between:
  - CFG
  - XYZ
  - DeepMD JSON formats
- Document the **exact workflow** used to generate:
  - Training datasets
  - Validation datasets
  - Test datasets

These scripts support:

- Full **reproducibility**
- Future **dataset extension**
- Consistent reuse across different **machine-learning interatomic potential models**


---

## Important Note on Dataset Design


This dataset is **not intended for equation-of-state (EOS) fitting**.  
All configurations include one or more of the following perturbations:
hydrostatic pressure, lattice strain, and/or random atomic displacement.

---

## Dataset Categorization

### Category A: Pressure + Atomic Displacement (No Lattice Strain)

**Data location:** `data/pressure_disp/`

Files:  
Mg3Bi2_1_GPa_0.3_disp.json, Mg3Bi2_3_GPa_0.3_disp.json,  
Mg3Bi2_5_GPa_0.3_disp.json, Mg3Bi2_7_GPa_0.3_disp.json

Description:  
Hydrostatic pressure (1–7 GPa) is applied while keeping lattice vectors fixed, with a random atomic
displacement amplitude of 0.3 Å.

Role in ML training:  
Provides anchor force data under compression and stabilizes high-pressure molecular dynamics.

---

### Category B: Pressure + Atomic Displacement + Lattice Strain (Core Dataset)

**Data location:** `data/pressure_strain_disp/`

Files follow the naming convention:  
`Mg3Bi2_<pressure>_GPa_0.3_disp_<strain>_pct.json`

Coverage:
- 1 GPa with 1%  3%, 5%, 7%, 9% strain
- 3 GPa with 1%, 3%, 5%, 7%, 9% strain
- 5 GPa with 1%, 3%, 5%, 7%, 9% strain
- 7 GPa with 1%, 3%, 5%, 7%, 9% strain

Description:  
This category captures the full coupling between hydrostatic pressure, lattice deformation, and
atomic disorder, spanning elastic to near-failure regimes.

Role in ML training:  
This is the **most important category** for preventing extrapolation failures and ensuring robustness
under combined mechanical loading. Extreme cases should be moderately down-weighted.

---

### Category C: Lattice Strain + Atomic Displacement (No Pressure)

**Data location:** `data/strain_disp/`

Files follow the naming convention:  
`Mg3Bi2_<strain>_pct_<disp>_disp.json`

Coverage:
- Strain levels: 1%, 3%, 5%, 7%, 9%
- Atomic displacement amplitudes: 0.2 Å and 0.3 Å
- No applied pressure

Description:  
Uniform lattice deformation is applied at ambient pressure together with random atomic displacements.

Role in ML training:  
Defines elastic constants, nonlinear strain response, and low-pressure mechanical stability.
High importance for accurate elastic behavior.

---

### Category D: Atomic Displacement Only

**Data location:** `data/disp_only/`

Files:  
Mg3Bi2_pct_0.2_disp.json, Mg3Bi2_pct_0.3_disp.json

Description:  
Only atomic positions are perturbed within the ideal lattice, without pressure or strain.

Role in ML training:  
Provides clean force–displacement anchors and should be assigned high training weight.

---

### Category E: Directional (Uniaxial) Strain

**Data location:** `data/uniaxial_strain/`

Files:  
strain_x.json, strain_y.json, strain_z.json

Description:  
Uniaxial strain is applied along a single Cartesian direction (x, y, or z) without random atomic
displacement. For each direction, the strain spans from **−10% to +10%** in increments of **2%**,
covering both compressive and tensile regimes.

Role in ML training:  
These configurations are critical for learning directional elastic constants, mechanical anisotropy,
and stress–strain symmetry. They should be assigned **very high training weight**, especially for
accurate elastic tensor prediction.


---

## Data Management

All JSON files are tracked using **Git Large File Storage (Git LFS)**.  
Total dataset size is approximately **284 MB**, while the repository remains lightweight through LFS pointers.

---

## Intended Use

- SNAP / MLIAP training
- Mechanical deformation studies
- Pressure-dependent molecular dynamics
- Robust machine-learning interatomic potential development


