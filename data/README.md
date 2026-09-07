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
│                                make_displaced.py    — configuration generator (QE)
├── train/                       NEP training run: nep.in, nep.txt, loss.out, submit.sh
├── analysis/                    Downstream property checks on the trained NEP potential
│   ├── band_gap/                QE band structure (DFT)
│   ├── dispersion/              Phonon dispersion, DFT (phonopy) vs. NEP
│   ├── nep_elastic_const/       Elastic tensor vs. temperature (LAMMPS + NEP)
│   ├── gruneisen/               Grüneisen parameter, phonon DOS
│   ├── thermal_expansion/       NPT runs at 0–9 GPa (LAMMPS + NEP)
│   └── Defect_formation_energy/ NEB diffusion barriers, Arrhenius/MSD analysis
├── figures/                     (empty — placeholder)
├── validation/                  (empty — placeholder; subfolders mirror analysis/ categories
│                                and are not currently populated)
├── docs/                        (empty — placeholder)
├── nep_training/                (empty — placeholder)
└── primitive_cell/              (empty — placeholder)
```

Directories marked *placeholder* are currently empty.

`analysis/` holds validation studies run **using** the fitted NEP potential (`train/nep.txt`),
not raw DFT training data — see [Analysis](#analysis-analysis) below.

---

## Data (`data/`)

Raw, physics-organized configurations grouped by deformation protocol. **5064 configurations**
across 12 categories:

| Category | Configs | Description |
|---|---|---|
| `strain_disp/` | 1475 | strain 1/3/5/7/9 % × displacement 0.2 Å and 0.3 Å (1000); plus a T/P-grid lattice set with random displacement (475) |
| `disp_only/` | 1000 | random displacement at the equilibrium cell, 0.2 Å and 0.3 Å caps |
| `pressure_disp/` | 1000 | cell compressed to 1/3/5/7 GPa + 0.3 Å displacement |
| `pressure_strain_disp/` | 1000 | pressure 1–7 GPa × strain 1–9 % × 0.3 Å displacement |
| `elastic_Cij/` | 206 | pure Voigt strains, 8 modes × 13 magnitudes × 2 signs |
| `phonon_disp/` | 155 | all-atom Gaussian displacement, rms 0.009–0.11 Å |
| `equib_perturb/` | 86 | strain ±1 % + displacement rms 0.03 Å |
| `vacancy/` | 84 | mono- and di-vacancy, 3×3×2 and 2×2×2 cells |
| `elastic_vacancy/` | 39 | mono-vacancy cell under Voigt strains, ±2 % |
| `uniaxial_strain/` | 33 | `strain_x`, `strain_y`, `strain_z`; ±10 % in 2 % steps, ionically relaxed |
| `compression/` | 19 | uniaxial compression, ε₁ or ε₃ to −10 % |
| `tensile/` | 17 | uniaxial stretch, ε₁ or ε₃ to +10 % |

Each entry stores `energy` (eV), `forces` (eV/Å), `virial_stress` (GPa, Voigt) and `volume`
(Å³) alongside a pymatgen `Structure`.

**`virial_stress` is stored as −σ, positive under compression.** See
[`data/README.md`](data/README.md) for the full schema, per-category caveats, and the
conversion factors for each ML format.

### Sampling conventions

For the ten `strain_disp/Mg3Bi2_<N>_pct_<d>_disp.json` files, the label `<N>` is the bound on
each **strain-tensor** component, sampled uniformly in [−N %, +N %]. Because engineering shear
is γ = 2ε, and because the hexagonal basis vectors mix ε_xx with ε_xy, the apparent strain
measured from lattice-vector lengths can exceed N — up to ≈ 2N for the shear components. The
label `<d>` is the **cap** on the random atomic displacement, not its typical value: amplitudes
are drawn uniformly in [0, d], so the mean displacement is ≈ d/2.

### Known redundancy

`uniaxial_strain/strain_y.json` is numerically identical to `strain_x.json` (energies agree to
< 1e-5 eV, fractional coordinates to < 1e-9, forces to < 1e-6 eV/Å). This is expected: in the
trigonal −3m point group, a₁ and a₂ uniaxial strain are symmetry-equivalent. The file is
retained so that `data/` is a superset of `dataset/`, but the two sets should not be treated as
independent when weighting a fit.

Across the full 5064 configurations there are 8 exact duplicate structures and 28 near
duplicates (matching volume, energy and force spectrum). These occur where categories meet at
zero strain — the undeformed reference cell appears in `elastic_Cij`, `tensile`, `compression`
and `elastic_vacancy`.

---

## Coverage gaps

Two gaps are worth stating explicitly, since both affect derived properties:

**No shear in the 425-configuration T/P-grid set.** Those cells are generated from (a, c) pairs
via `hex_cell()`, so all off-diagonal strain components are exactly zero by construction. The
lattices span −5.1 % to +1.3 % relative to the reference cell — predominantly compressive,
reflecting the 0–9 GPa pressure axis of the grid rather than thermal expansion.

**Limited small-amplitude out-of-plane shear overall.** The e4 (yz) and e1+e4 modes exist only
in `elastic_Cij`, which is not part of the current ML exports. Potentials trained on the
exported subset should be expected to reproduce in-plane elastic response better than C₄₄.

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

The exports cover five of the twelve categories: `disp_only`, `strain_disp`, `pressure_disp`,
`pressure_strain_disp` and `uniaxial_strain`.

The remaining **606 configurations — `elastic_Cij`, `phonon_disp`, `equib_perturb`, `vacancy`,
`elastic_vacancy`, `tensile` and `compression` — are present in `data/` but were not exported
or trained on.** Anyone reproducing or extending the potential in `train/` should be aware
that it has seen no pure-Voigt elastic data, no vacancy data, and no clean uniaxial
tensile/compressive data.

`data/` is now a superset of `dataset/`: the previously absent
`disp_only/Mg3Bi2_pct_0.2_disp.json` and `uniaxial_strain/strain_y.json` have been restored.
The 425-configuration `strain_disp/Mg3Bi2_lattice_strain_atom_displaced.json` set postdates the
exports and is not included in them.

---

## Scripts (`scripts/`)

`MTP_xyz_format.ipynb` documents the conversion between the raw JSON, extended XYZ (NEP), MTP
CFG and DeepMD `.npy` formats.

`make_displaced.py` generates the QE inputs: it reads a vc-relaxed unit cell, builds the
supercell, optionally strains it onto a 24-point (a, c) grid covering T = 300–600 K and
P = 0–9 GPa (`--cover-lattices`), and applies a random per-atom displacement with a per-folder
amplitude cap drawn uniformly in [`--min-disp`, `--max-disp`]. The random seed defaults to 12,
so the sets are reproducible.

---

## Analysis (`analysis/`)

Downstream property checks run **with the fitted NEP potential** (`train/nep.txt`), used to
validate it against DFT and, where available, against known physics rather than to generate
further training data.

| Folder | Content |
|---|---|
| `band_gap/` | QE band structure calculation (DFT reference, not NEP) |
| `dispersion/` | Phonon dispersion — DFT (phonopy, `dft_dispersion/`) vs. NEP (`nep_dispersion/`) |
| `nep_elastic_const/` | Elastic tensor vs. temperature, 300–1000 K, via LAMMPS + NEP (`elastic_T/`), plus a single-point LAMMPS elastic run (`nep_lammps/`). Everything here is NEP+LAMMPS — no DFT elastic-constant calculation exists; the `.pwi` files inside `nep_lammps/` are QE inputs used only to relax the starting cell before conversion to LAMMPS format |
| `gruneisen/` | Grüneisen parameter and phonon DOS from the NEP potential |
| `thermal_expansion/` | LAMMPS NPT runs at 0, 1, 3, 5, 7, 9 GPa |
| `Defect_formation_energy/` | NEB migration barriers (`neb_all_paths/`) and NEP-driven MD diffusion (Arrhenius, MSD) at T = 300–700 K, 5 runs each |

These folders are **not** part of the training/export pipeline described above and are not
covered by the LFS filter beyond the `*.json`/`*.xyz`/`*.cfg`/`*.npy` patterns already in
`.gitattributes` — check individual file sizes before committing (QE scratch output such as
`band_gap/tmp/*.wfc*` is gitignored and should never be committed).

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
on-disk size is approximately **565 MB**: `data/` ≈ 385 MB, `dataset/` ≈ 90 MB,
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
