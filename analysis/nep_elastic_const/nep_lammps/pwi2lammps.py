#!/usr/bin/env python3
"""
Convert relaxed_unitcell.pwi (QE) -> LAMMPS data file
Usage:
    python3 pwi2lammps.py relaxed_unitcell.pwi
"""
import numpy as np, re, sys

def parse_pwi(path):
    text = open(path).read()
    m = re.search(r"CELL_PARAMETERS[^\n]*\n(.*?)(?=\n[A-Z])", text, re.S)
    lines = [l for l in m.group(1).strip().split("\n") if l.strip()]
    cell = np.array([[float(x) for x in l.split()] for l in lines])
    m = re.search(r"ATOMIC_POSITIONS[^\n]*\n(.*)", text, re.S)
    lines = [l for l in m.group(1).strip().split("\n") if l.strip()]
    species, coords = [], []
    for l in lines:
        p = l.split()
        species.append(p[0])
        coords.append([float(x) for x in p[1:4]])
    return cell, species, np.array(coords)

def cell_to_lammps(cell):
    a, b, c = cell[0], cell[1], cell[2]
    ax = np.linalg.norm(a)
    bx = np.dot(b, a/ax)
    by = np.sqrt(np.dot(b,b) - bx**2)
    cx = np.dot(c, a/ax)
    cy = (np.dot(b,c) - bx*cx) / by
    cz = np.sqrt(np.dot(c,c) - cx**2 - cy**2)
    return ax, bx, by, cx, cy, cz

def write_data(cell, species, coords, outfile="lammps.data"):
    ax, bx, by, cx, cy, cz = cell_to_lammps(cell)
    types = sorted(set(species))
    type_map = {s: i+1 for i, s in enumerate(types)}
    R = np.array([[ax, 0,  0 ],
                  [bx, by, 0 ],
                  [cx, cy, cz]])
    frac = coords @ np.linalg.inv(cell)
    cart = frac @ R
    masses = {"Mg": 24.305, "Bi": 208.9804}

    with open(outfile, "w") as f:
        f.write("# Mg3Bi2 relaxed unit cell — converted from QE pwi\n\n")
        f.write(f"{len(species)} atoms\n")
        f.write(f"{len(types)} atom types\n\n")
        f.write(f"0.0 {ax:.9f} xlo xhi\n")
        f.write(f"0.0 {by:.9f} ylo yhi\n")
        f.write(f"0.0 {cz:.9f} zlo zhi\n")
        f.write(f"{bx:.9f} {cx:.9f} {cy:.9f} xy xz yz\n\n")
        f.write("Masses\n\n")
        for s in types:
            f.write(f"  {type_map[s]}  {masses[s]}  # {s}\n")
        f.write("\nAtoms  # atomic\n\n")
        for i, (sp, r) in enumerate(zip(species, cart)):
            f.write(f"  {i+1}  {type_map[sp]}  {r[0]:.9f}  {r[1]:.9f}  {r[2]:.9f}\n")
    print(f"wrote {outfile}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "relaxed_unitcell.pwi"
    cell, species, coords = parse_pwi(path)
    write_data(cell, species, coords)
