#!/usr/bin/env python3
"""
build_nep_dataset.py — rebuild the GPUMD/NEP training set from data/.

Reads every configuration JSON under data/ (skipping the categories listed in
--exclude), converts to extended XYZ in the exact format the existing
dataset/gpumd/train.xyz uses, and writes a fresh stratified train/test split.

Stress conversion (verified against the existing train.xyz):

    data/  outputs.virial_stress   -> GPa, Voigt [xx, yy, zz, yz, xz, xy],
                                      stored as -sigma (positive in compression)
    xyz    stress="..."            -> eV/A^3, full 3x3, sigma convention

    stress_matrix = -voigt_to_matrix(virial_stress) / 160.2176

Vacancy upweighting
-------------------
Vacancy and elastic_vacancy categories are repeated N times (default 4×) so
they make up a healthier fraction of the training set without new DFT runs.
Override with --vacancy-repeat.

Usage
-----
    cd ~/Mg3Bi2/Mg3Bi2-Mechanical-MLIAP-Dataset
    python3 build_nep_dataset.py

    # keep pressure_strain_disp in as well
    python3 build_nep_dataset.py --exclude ""

    # change repeat factor
    python3 build_nep_dataset.py --vacancy-repeat 6

    # write somewhere else, different split, tag config_type
    python3 build_nep_dataset.py --outdir dataset/gpumd_v2 --test-frac 0.05 --config-type
"""

import argparse
import json
import os
import random
import sys
from glob import glob

GPA_PER_EV_A3 = 160.2176  # 1 eV/A^3 in GPa

# Categories that get repeated to boost vacancy representation.
# Keys are category directory names; values are the repeat multiplier.
# The same multiplier is applied to both train and test splits so the
# test-set fraction stays consistent within each category.
VACANCY_CATEGORIES = {
    "vacancy":          4,
    "elastic_vacancy":  4,
}


def voigt_to_matrix(v):
    """[xx, yy, zz, yz, xz, xy] -> row-major 3x3, flattened to 9."""
    xx, yy, zz, yz, xz, xy = v
    return [xx, xy, xz,
            xy, yy, yz,
            xz, yz, zz]


def frame_to_xyz(entry, config_type=None):
    """One data/ JSON entry -> one extended-XYZ frame as a string."""
    struct = entry["structure"]
    out = entry["outputs"]
    sites = struct["sites"]
    n = len(sites)

    lat = [c for row in struct["lattice"]["matrix"] for c in row]
    lat_s = " ".join(f"{c:.9f}" for c in lat)

    # stored virial_stress is -sigma in GPa; NEP wants sigma in eV/A^3
    sig = [-c / GPA_PER_EV_A3 for c in voigt_to_matrix(out["virial_stress"])]
    sig_s = " ".join(f"{c:.12f}" for c in sig)

    head = (f'lattice="{lat_s}" energy={out["energy"]!r} stress="{sig_s}" '
            f"properties=species:S:1:pos:R:3:forces:R:3")
    if config_type:
        head += f" config_type={config_type}"

    lines = [str(n), head]
    for site, f in zip(sites, out["forces"]):
        el = site["label"]
        x, y, z = site["xyz"]
        fx, fy, fz = f
        lines.append(f"{el:2s} {x:18.9f} {y:18.9f} {z:18.9f} "
                     f"{fx:18.9f} {fy:18.9f} {fz:18.9f}")
    return "\n".join(lines) + "\n"


def load_category(cat_dir):
    """All configurations in one data/<category>/ directory."""
    entries = []
    for path in sorted(glob(os.path.join(cat_dir, "*.json"))):
        with open(path) as fh:
            j = json.load(fh)
        if isinstance(j, dict):
            j = [j]
        for e in j:
            # skip anything without a complete DFT result
            o = e.get("outputs", {})
            if not all(k in o for k in ("energy", "forces", "virial_stress")):
                print(f"  ! skipping incomplete config in {path}", file=sys.stderr)
                continue
            entries.append(e)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--outdir", default="dataset/gpumd")
    ap.add_argument("--exclude", default="",
                    help="comma-separated category names to skip")
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-type", action="store_true",
                    help="tag each frame with config_type=<category>")
    ap.add_argument("--vacancy-repeat", type=int, default=None,
                    help="override repeat factor for all vacancy categories "
                         f"(default: {VACANCY_CATEGORIES})")
    args = ap.parse_args()

    excluded = {c.strip() for c in args.exclude.split(",") if c.strip()}
    rng = random.Random(args.seed)

    # Build effective repeat map (allow CLI override)
    repeat_map = dict(VACANCY_CATEGORIES)
    if args.vacancy_repeat is not None:
        repeat_map = {k: args.vacancy_repeat for k in repeat_map}

    cats = sorted(d for d in os.listdir(args.datadir)
                  if os.path.isdir(os.path.join(args.datadir, d)))

    train, test = [], []
    print(f"{'category':26s} {'orig':>6s} {'repeat':>7s} {'train':>7s} {'test':>6s}")
    print("-" * 56)
    n_tot = 0

    for cat in cats:
        if cat in excluded:
            print(f"{cat:26s} {'--':>6s} {'--':>7s} {'--':>7s} {'--':>6s}   (excluded)")
            continue
        entries = load_category(os.path.join(args.datadir, cat))
        if not entries:
            continue

        repeat = repeat_map.get(cat, 1)

        idx = list(range(len(entries)))
        rng.shuffle(idx)
        n_test = round(len(entries) * args.test_frac)
        test_idx = set(idx[:n_test])

        tag = cat if args.config_type else None

        n_train_frames = 0
        n_test_frames  = 0
        for _ in range(repeat):
            for i, e in enumerate(entries):
                frame = frame_to_xyz(e, tag)
                if i in test_idx:
                    test.append(frame)
                    n_test_frames += 1
                else:
                    train.append(frame)
                    n_train_frames += 1

        n_tot += len(entries)
        repeat_str = f"{repeat}×" if repeat > 1 else "-"
        print(f"{cat:26s} {len(entries):6d} {repeat_str:>7s} "
              f"{n_train_frames:7d} {n_test_frames:6d}")

    print("-" * 56)
    total_train = len(train)
    total_test  = len(test)
    print(f"{'TOTAL (unique configs)':26s} {n_tot:6d}")
    print(f"{'TOTAL (with repeats)':26s} {'':6s} {'':>7s} "
          f"{total_train:7d} {total_test:6d}")

    # Print vacancy fraction for quick sanity check
    vac_train = sum(repeat_map.get(c, 1) * len(load_category(
                    os.path.join(args.datadir, c)))
                    for c in repeat_map if c not in excluded
                    and os.path.isdir(os.path.join(args.datadir, c)))
    if total_train > 0:
        print(f"\nVacancy configs in train (approx): ~{int(vac_train*0.9)} "
              f"({100*vac_train*0.9/total_train:.1f}% of train set)")

    rng.shuffle(train)
    rng.shuffle(test)

    os.makedirs(args.outdir, exist_ok=True)
    for name, frames in (("train.xyz", train), ("test.xyz", test)):
        path = os.path.join(args.outdir, name)
        with open(path, "w") as fh:
            fh.writelines(frames)
        mb = os.path.getsize(path) / 1024**2
        print(f"wrote {path}  ({len(frames)} frames, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
