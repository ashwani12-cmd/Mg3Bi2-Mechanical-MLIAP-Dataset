# Mg3Bi2-Mechanical-MLIAP-Dataset

DFT reference data for **Mg₃Bi₂** (α phase, P-3m1, trigonal −3m) for training machine-learning
interatomic potentials, together with ML-ready exports for **NEP (GPUMD)**, **MTP (MLIP-2)**
and **DeepMD**.

The dataset emphasizes mechanical response — pressure, lattice strain, uniaxial deformation
and atomic disorder — and also includes elastic (pure Voigt strain), phonon-displacement and
point-defect configurations.

**DFT level:** Quantum ESPRESSO 7.5, LDA ONCV, ecutwfc = 160 Ry, ecutrho = 640 Ry, MP smearing (degauss 0.01)
**Reference cell:** a = 4.6043 Å, c = 7.2764 Å
**Supercell:** 3×3×2 (90 atoms), K-points 4×4×4 (vacancy sets also use 2×2×2)

> This dataset is **not** intended for equation-of-state fitting. Every configuration carries
> at least one of: hydrostatic pressure, lattice strain, or random atomic displacement.

---

## Repository layout

```
Mg3Bi2-Mechanical-MLIAP-Dataset/
├── README.md
├── LICENSE                      MIT
├── data/                        Raw DFT configurations, organized by deformation protocol
│   └── README.md                Per-category detail, JSON schema, sign conventions
├── dataset/
│   ├── gpumd/                   train.xyz, test.xyz  (extended XYZ, NEP)
│   ├── mtp/                     train.cfg, val.cfg   (MLIP-2)
│   └── deepmd/                  train/, val/         (per-category .npy systems)
├── scripts/                     MTP_xyz_format.ipynb — format conversion
├── train/                       NEP training run: nep.in, nep.txt, loss.out, submit.sh
├── figures/                     (empty — placeholder)
├── validation/                  (empty — placeholder)
├── docs/                        (empty — placeholder)
├── nep_training/                (empty — placeholder)
└── primitive_cell/              (empty — placeholder)
```

Directories marked *placeholder* are currently empty.

---

## Data (`data/`)

Raw, physics-organized configurations grouped by deformation protocol. **4128 configurations**
across 13 categories:

| Category | Configs | Description |
|---|---|---|
| `pressure_strain_disp/` | 1000 | pressure 1–7 GPa × strain 1–9 % × 0.3 Å displacement |
| `pressure_disp/` | 1000 | cell compressed to 1/3/5/7 GPa + 0.3 Å displacement |
| `strain_disp/` | 1000 | strain 1/3/5/7/9 % × displacement 0.2 Å and 0.3 Å |
| `disp_only/` | 500 | random displacement (rms 0.30 Å) at the equilibrium cell |
| `elastic_Cij/` | 206 | pure Voigt strains, 8 modes × 13 magnitudes × 2 signs |
| `phonon_disp/` | 155 | all-atom Gaussian displacement, rms 0.009–0.11 Å |
| `equib_perturb/` | 86 | strain ±1 % + displacement rms 0.03 Å |
| `vacancy/` | 84 | mono- and di-vacancy, 3×3×2 and 2×2×2 cells |
| `elastic_vacancy/` | 39 | mono-vacancy cell under Voigt strains, ±2 % |
| `uniaxial_strain/` | 22 | `strain_x`, `strain_z`; ±10 % in 2 % steps, ionically relaxed |
| `compression/` | 19 | uniaxial compression, ε₁ or ε₃ to −10 % |
| `tensile/` | 17 | uniaxial stretch, ε₁ or ε₃ to +10 % |

Each entry stores `energy` (eV), `forces` (eV/Å), `virial_stress` (GPa, Voigt) and `volume`
(Å³) alongside a pymatgen `Structure`.

**`virial_stress` is stored as −σ, positive under compression.** See
[`data/README.md`](data/README.md) for the full schema, per-category caveats, and the
conversion factors for each ML format.

---

## Dataset (`dataset/`)

ML-ready exports in three formats, all containing the **same 4033 configurations** with an
identical 3630 / 403 (90 % / 10 %) train–test split:

| Format | Path | Stress field |
|---|---|---|
| NEP / GPUMD | `dataset/gpumd/train.xyz`, `test.xyz` | `stress=` in eV/Å³, σ convention |
| MTP / MLIP-2 | `dataset/mtp/train.cfg`, `val.cfg` | `PlusStress` in eV |
| DeepMD | `dataset/deepmd/{train,val}/<category>/set.000/` | `virial.npy` in eV |

The three exports have been checked against each other and are mutually consistent — energies,
forces and stresses agree after unit and sign conversion, and the split is identical across all
three. This supports fair cross-framework benchmarking without format-dependent bias.

### Scope of the exports

The exports currently cover only five of the thirteen categories: `disp_only`,
`strain_disp`, `pressure_disp`, `pressure_strain_disp` and `uniaxial_strain`.

The remaining **606 configurations — `elastic_Cij`, `phonon_disp`, `equib_perturb`, `vacancy`,
`elastic_vacancy`, `tensile` and `compression` — are present in `data/` but were not exported
or trained on.** Anyone reproducing or extending the potential in `train/` should be aware
that it has seen no pure-Voigt elastic data, no vacancy data, and no clean uniaxial
tensile/compressive data. In particular the e4 (yz) and e1+e4 shear modes exist only in
`elastic_Cij`, so the fitted potential has no small-amplitude out-of-plane shear coverage.

The exports also contain 502 configurations with no counterpart in `data/` — 500 equilibrium-cell
displacement configurations (the missing `disp_only/Mg3Bi2_pct_0.2_disp.json`) and 2 a₂-axis
uniaxial configurations (the missing `uniaxial_strain/strain_y.json`). Restoring those two files
and re-merging with the excluded categories would make `data/` a true superset of `dataset/`.

---

## Scripts (`scripts/`)

`MTP_xyz_format.ipynb` documents the conversion between the raw JSON, extended XYZ (NEP), MTP
CFG and DeepMD `.npy` formats.

The configuration generators and the QE parser used to produce `data/` are not currently in
the repository.

---

## Training (`train/`)

NEP4 trained with the standalone GPUMD `nep` binary:

```bash
cd train
/path/to/GPUMD/src/nep > out.dat     # see train/submit.sh for the PBS wrapper
```

Key `nep.in` settings: `type 2 Mg Bi`, `cutoff 7 4`, `n_max 4 4`, `basis_size 8 8`,
`l_max 4 2 0`, `neuron 50`, `lambda_e 1.0`, `lambda_f 1.0`, `lambda_v 0.1`, `batch 1000`,
`zbl 2.5`, `population 50`, `generation 300000`.

Outputs (`loss.out`, `energy_train.out`, `force_train.out`, `virial_train.out`, and the
corresponding `_test` files) plus intermediate restarts at 100k / 200k / 300k generations are
included.

---

## Data management

Raw configurations are tracked with **Git LFS** (`.gitattributes` covers `*.json`). Total
on-disk size is approximately **475 MB**: `data/` ≈ 295 MB, `dataset/` ≈ 90 MB,
`train/` ≈ 91 MB.

Note that the ML exports (`train.xyz`, `train.cfg`, `*.npy`) and the training outputs are
**not** currently covered by the LFS filter despite being ~180 MB combined; extending
`.gitattributes` to `*.xyz`, `*.cfg` and `*.npy` is recommended.

---

## Intended use

- NEP / MTP / DeepMD interatomic potential training
- Cross-framework benchmarking on a fixed, identically split dataset
- Mechanical deformation and pressure-dependent molecular dynamics
- Elastic, phonon and point-defect reference data for Mg₃Bi₂

---

## License

MIT — see [LICENSE](LICENSE).
