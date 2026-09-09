#!/usr/bin/env python3
"""
elastic_from_phonons.py -- elastic moduli of Mg3Bi2 from ACOUSTIC PHONON velocities
==================================================================================
A route to K, G, E, nu that is INDEPENDENT of the finite-strain C_ij tensor:
take the acoustic-branch group velocities near Gamma along many directions,
feed rho*v^2 into the Christoffel equation, and solve for the elastic
constants.  Then Voigt-Reuss-Hill -> K, G, E, nu, and a Debye temperature
from the acoustic velocities AND from the phonon DOS.

Why bother
----------
`gruneisen_calorine.py` takes K and G from the finite-strain C_ij and then uses
them for theta_D and gamma_macro -- so theta_D(elastic) is NOT an independent
check, it shares the same source.  This script adds separate routes:

  * theta_D   (a) from elastic v_D  vs  (b) DIRECT acoustic-branch average v_D
              vs  (c) 2nd-moment of the phonon DOS  <-- the robust cross-check
  * v_l, v_t, v_D   measured directly from the acoustic group velocities
  * C_ij, K, G, E, nu   from the Christoffel eq. applied to the acoustic
              velocities -- an independent route; with a converged FC supercell
              (>=4x4x3; the default here is 5x5x4) it matches the finite-strain
              C_ij to a few GPa (C13 the least reliable, no non-analytic term).

What it does
------------
  1. relax the primitive cell at the NEP minimum (calorine)
  2. build NEP force constants on the supercell (calorine)
  3. acoustic group velocities near Gamma along [001],[100],[101],... + a
     Fibonacci sphere of extra directions
  4. high-symmetry extraction (hexagonal formulas) along [001],[100],[101]
  5. global least-squares Christoffel fit of the trigonal -3m constants
     {C11, C12, C13, C33, C44, C14}  (C66 = (C11-C12)/2)
  6. VRH averages, Debye (elastic v_D and DOS), comparison table

Usage
-----
    cd analysis/gruneisen
    python3 elastic_from_phonons.py --nep nep.txt --pwi espresso.pwi

    # give the finite-strain result for a side-by-side column
    python3 elastic_from_phonons.py --nep nep.txt --pwi espresso.pwi \
        --cij-strain 67.9,35.9,23.0,78.5,15.5,3.0     # C11,C12,C13,C33,C44,C14

    # reuse an existing phonopy FORCE_CONSTANTS (skips calorine)
    python3 elastic_from_phonons.py --fc FORCE_CONSTANTS --phonopy-yaml phonopy.yaml \
        --pwi espresso.pwi

Needs: phonopy, ase, numpy, scipy  (+ calorine unless --fc is used).
Run it where `nep` / calorine work.
"""
import argparse
import sys

import numpy as np

AMU = 1.66053906660e-27          # kg
HBAR = 1.054571817e-34           # J s
KB = 1.380649e-23                # J/K
GPA = 1.0e9

# DFT reference (same numbers as gruneisen_calorine.py)
DFT_CIJ = dict(C11=67.93, C12=35.93, C13=22.98, C33=78.50, C44=15.53, C14=3.00)


# ----------------------------------------------------------------------
# elastic-tensor helpers
# ----------------------------------------------------------------------
def voigt_matrix_3m(C11, C12, C13, C33, C44, C14):
    """6x6 Voigt matrix for trigonal -3m (C66 = (C11-C12)/2)."""
    C66 = 0.5 * (C11 - C12)
    return np.array([
        [C11, C12, C13,  C14, 0.0, 0.0],
        [C12, C11, C13, -C14, 0.0, 0.0],
        [C13, C13, C33,  0.0, 0.0, 0.0],
        [C14, -C14, 0.0, C44, 0.0, 0.0],
        [0.0, 0.0, 0.0,  0.0, C44, C14],
        [0.0, 0.0, 0.0,  0.0, C14, C66],
    ])


_VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def voigt_to_tensor(Cv):
    """6x6 -> 3x3x3x3."""
    C = np.zeros((3, 3, 3, 3))
    for p, (i, j) in enumerate(_VOIGT):
        for q, (k, l) in enumerate(_VOIGT):
            v = Cv[p, q]
            for (a, b) in {(i, j), (j, i)}:
                for (c, d) in {(k, l), (l, k)}:
                    C[a, b, c, d] = v
    return C


def christoffel_rho_v2(Cv, n):
    """Sorted eigenvalues (= rho v^2, GPa) of the Christoffel matrix for dir n."""
    C = voigt_to_tensor(Cv)
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    Gamma = np.einsum("ijkl,j,l->ik", C, n, n)
    return np.sort(np.linalg.eigvalsh(Gamma))


def vrh_moduli(Cv):
    S = np.linalg.inv(Cv)
    Kv = ((Cv[0, 0] + Cv[1, 1] + Cv[2, 2]) + 2 * (Cv[0, 1] + Cv[1, 2] + Cv[0, 2])) / 9
    Gv = ((Cv[0, 0] + Cv[1, 1] + Cv[2, 2]) - (Cv[0, 1] + Cv[1, 2] + Cv[0, 2])
          + 3 * (Cv[3, 3] + Cv[4, 4] + Cv[5, 5])) / 15
    Kr = 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[1, 2] + S[0, 2]))
    Gr = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[1, 2] + S[0, 2])
                 + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    K, G = 0.5 * (Kv + Kr), 0.5 * (Gv + Gr)
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    return dict(K=K, G=G, E=E, nu=nu, Kv=Kv, Kr=Kr, Gv=Gv, Gr=Gr)


def debye_from_vel(vl, vt, rho, n_atoms, V_A3):
    """theta_D from sound velocities (m/s)."""
    vD = (1 / 3 * (1 / vl**3 + 2 / vt**3)) ** (-1 / 3)
    n = n_atoms / (V_A3 * 1e-30)
    tD = HBAR / KB * vD * (6 * np.pi**2 * n) ** (1 / 3)
    return vD, tD


def debye_from_moduli(K, G, rho, n_atoms, V_A3):
    K, G = K * GPA, G * GPA
    vl = np.sqrt((K + 4 * G / 3) / rho)
    vt = np.sqrt(G / rho)
    vD, tD = debye_from_vel(vl, vt, rho, n_atoms, V_A3)
    return dict(vl=vl, vt=vt, vD=vD, theta_D=tD)


def debye_from_dos(freq_THz, dos):
    """2nd-moment (thermodynamic) theta_D from the phonon DOS."""
    w = 2 * np.pi * np.asarray(freq_THz) * 1e12          # rad/s
    g = np.clip(np.asarray(dos), 0, None)
    ok = w > 0
    w, g = w[ok], g[ok]
    w2 = np.trapz(g * w**2, w) / np.trapz(g, w)
    wD = np.sqrt(5.0 / 3.0 * w2)
    return HBAR * wD / KB


# ----------------------------------------------------------------------
# phonon side
# ----------------------------------------------------------------------
def get_phonon(args):
    """Return (phonopy_obj, unit_in_m, rho, n_atoms, V_A3).

    unit_in_m converts the phonopy cell's length unit to metres, so that
    group_velocities [THz * cell_unit] -> m/s is  |v_g| * 1e12 * unit_in_m.
    """
    from ase.io import read

    atoms = read(args.pwi)
    atoms.pbc = True
    a_ref = np.linalg.norm(atoms.cell.array[0])          # Angstrom

    if args.fc:
        from phonopy import load
        if not args.phonopy_yaml:
            sys.exit("--fc needs --phonopy-yaml (the phonopy.yaml that FC was made with)")
        ph = load(args.phonopy_yaml, force_constants_filename=args.fc)
    else:
        from ase.optimize import FIRE
        from ase.filters import FrechetCellFilter
        from calorine.calculators import CPUNEP
        from calorine.tools import get_force_constants
        calc = CPUNEP(args.nep)
        atoms.calc = calc
        print("relaxing primitive cell at the NEP minimum ...")
        FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-6, steps=800)
        a_ref = np.linalg.norm(atoms.cell.array[0])
        ph = get_force_constants(atoms, calc, supercell_matrix=np.diag(args.supercell))

    prim = ph.primitive
    P = np.array(prim.cell)
    unit_in_m = 1e-10 * a_ref / np.linalg.norm(P[0])     # cell-unit -> metre
    to_ang = unit_in_m / 1e-10

    L = np.linalg.norm(P, axis=1) * to_ang
    if abs(P[0, 1]) + abs(P[0, 2]) > 1e-3 * np.linalg.norm(P[0]) or \
       abs(P[2, 0]) + abs(P[2, 1]) > 1e-3 * np.linalg.norm(P[2]):
        print("  WARNING: primitive cell not in the a||x, c||z setting -- the "
              "high-symmetry C_ij will be unreliable (the LSQ fit is unaffected).")

    V_A3 = abs(np.linalg.det(P)) * to_ang**3
    rho = prim.masses.sum() * AMU / (V_A3 * 1e-30)
    print(f"  a={L[0]:.5f}  c={L[2]:.5f} A   c/a={L[2]/L[0]:.5f}   V={V_A3:.4f} A^3")
    print(f"  primitive: {len(prim)} atoms   rho = {rho:.1f} kg/m^3")
    return ph, unit_in_m, rho, len(prim), V_A3


def sound_velocities(ph, unit_in_m, direction, qmag=1e-3):
    """Three acoustic sound velocities (m/s) along a Cartesian direction,
    from phonopy group velocities just off Gamma."""
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    # q-space vector whose Cartesian direction is n:  q_frac ~ n . cell^T
    qdir = n @ np.array(ph.primitive.cell).T
    qdir = qdir / np.linalg.norm(qdir)
    qs = np.array([qdir * qmag, qdir * 2 * qmag])
    ph.run_qpoints(qs, with_group_velocities=True)
    gv = ph.qpoints.group_velocities[0]                  # (nbands, 3)  THz*cell_unit
    f = ph.qpoints.frequencies[0]
    idx = np.argsort(f)[:3]
    speeds = np.linalg.norm(gv[idx], axis=1) * 1e12 * unit_in_m
    return np.sort(speeds)                               # vt1, vt2, vL


def high_symmetry_cij(vel, rho):
    """vel: dict dir -> sorted (vt1, vt2, vL) m/s.  C11..C14 (GPa), hexagonal approx."""
    r = rho / GPA                                        # r*v^2 -> GPa
    v001, v100, v101 = vel["001"], vel["100"], vel["101"]

    C44 = r * np.mean(v001[:2] ** 2)
    C33 = r * v001[2] ** 2
    C11 = r * v100[2] ** 2

    t = r * v100[:2] ** 2                                # two transverse along [100]
    C66 = t[np.argmax(np.abs(t - C44))]                  # the one further from C44
    C12 = C11 - 2 * C66

    rv2 = r * v101 ** 2                                  # [101]/sqrt2
    yshear = 0.5 * (C66 + C44)
    j = int(np.argmin(np.abs(rv2 - yshear)))
    sag = np.delete(rv2, j)                              # sagittal pair: quasi-T, quasi-L
    Gxx, Gzz = 0.5 * (C11 + C44), 0.5 * (C33 + C44)
    Gxz = np.sqrt(max(Gxx * Gzz - sag[0] * sag[1], 0.0))
    C13 = 2 * Gxz - C44
    return dict(C11=C11, C12=C12, C13=C13, C33=C33, C44=C44, C14=0.0)


def lsq_cij(dirs, meas_rv2, x0):
    """Global Christoffel least-squares for {C11,C12,C13,C33,C44,C14} (GPa)."""
    from scipy.optimize import least_squares

    def resid(p):
        Cv = voigt_matrix_3m(*p)
        return np.concatenate([christoffel_rho_v2(Cv, n) - m
                               for n, m in zip(dirs, meas_rv2)])

    lo = [0, -np.inf, -np.inf, 0, 0, -np.inf]
    sol = least_squares(resid, x0, bounds=(lo, [np.inf] * 6),
                        method="trf", max_nfev=4000)
    keys = ["C11", "C12", "C13", "C33", "C44", "C14"]
    return dict(zip(keys, sol.x)), float(np.sqrt(np.mean(sol.fun ** 2)))


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    gold = np.pi * (1 + 5 ** 0.5)
    th = gold * i
    return np.c_[np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)]


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nep", default="nep.txt")
    ap.add_argument("--pwi", default="espresso.pwi")
    ap.add_argument("--fc", default=None, help="existing phonopy FORCE_CONSTANTS")
    ap.add_argument("--phonopy-yaml", default=None, help="phonopy.yaml matching --fc")
    ap.add_argument("--supercell", nargs=3, type=int, default=[5, 5, 4],
                    help="FC supercell -- the BASAL constants (C11,C12,C66) need "
                         ">= 4x4x3 to converge; 3x3x2 gives C66 ~2x too small")
    ap.add_argument("--qmag", type=float, default=1e-3,
                    help="|q| off Gamma for the group velocity (default 1e-3)")
    ap.add_argument("--ndir", type=int, default=48,
                    help="extra Fibonacci-sphere directions for the global fit")
    ap.add_argument("--mesh", nargs=3, type=int, default=[24, 24, 16],
                    help="q-mesh for the DOS theta_D")
    ap.add_argument("--cij-strain", default=None,
                    help="finite-strain C11,C12,C13,C33,C44,C14 (GPa) for comparison")
    args = ap.parse_args()

    try:
        ph, unit_in_m, rho, nat, Vp = get_phonon(args)
    except ImportError as e:
        sys.exit(f"ERROR: {e}\n(run where calorine + phonopy are installed, "
                 f"or pass --fc FORCE_CONSTANTS --phonopy-yaml phonopy.yaml)")

    # ---- 3) acoustic velocities: extraction dirs + random dirs ----
    base = {"001": [0, 0, 1], "100": [1, 0, 0], "101": [1, 0, 1],
            "010": [0, 1, 0], "110": [1, 1, 0], "011": [0, 1, 1], "111": [1, 1, 1]}
    vel = {}
    print(f"\nacoustic group velocities  (|q| = {args.qmag}) :")
    for tag, d in base.items():
        vel[tag] = sound_velocities(ph, unit_in_m, d, args.qmag)
        print(f"  [{tag}]  vt1={vel[tag][0]:7.1f}  vt2={vel[tag][1]:7.1f}  "
              f"vL={vel[tag][2]:7.1f}  m/s")

    dirs = [base[t] for t in base]
    meas = [rho / GPA * vel[t] ** 2 for t in base]
    for n in fib_sphere(args.ndir):
        v = sound_velocities(ph, unit_in_m, n, args.qmag)
        dirs.append(n)
        meas.append(rho / GPA * v ** 2)

    # ---- 4) high-symmetry extraction ----
    hs = high_symmetry_cij(vel, rho)

    # ---- 5) global Christoffel least-squares ----
    x0 = [hs["C11"], hs["C12"], hs["C13"], hs["C33"], hs["C44"], 3.0]
    lsq, rms = lsq_cij(dirs, meas, x0)
    print(f"\nglobal Christoffel fit residual RMS = {rms:.2f} GPa "
          f"({len(dirs)} directions)")
    print("NOTE: phonon-route C_ij from acoustic group velocities near Gamma "
          "(no non-analytic\n      correction). With a converged FC supercell "
          "(>=4x4x3) it matches the finite-\n      strain C_ij to a few GPa; "
          "C13 is the least reliable. theta_D below is the\n      headline "
          "independent cross-check.")

    # ---- moduli comparison table ----
    cols = [("phonon high-sym", hs), ("phonon LSQ", lsq), ("DFT", DFT_CIJ)]
    if args.cij_strain:
        keys = ["C11", "C12", "C13", "C33", "C44", "C14"]
        vals = [float(x) for x in args.cij_strain.split(",")]
        cols.insert(2, ("strain C_ij", dict(zip(keys, vals))))

    print("\n" + "=" * 76)
    print("ELASTIC CONSTANTS  (GPa)")
    print("=" * 76)
    print(f"{'':6s}" + "".join(f"{name:>17s}" for name, _ in cols))
    for k in ["C11", "C12", "C13", "C33", "C44", "C14"]:
        print(f"{k:6s}" + "".join(f"{c.get(k, float('nan')):17.2f}" for _, c in cols))
    print(f"{'C66':6s}" + "".join(
        f"{0.5*(c['C11']-c['C12']):17.2f}" for _, c in cols))

    print("\n" + "=" * 76)
    print("VRH MODULI + DEBYE  (theta_D from elastic v_D)")
    print("=" * 76)
    print(f"{'':10s}" + "".join(f"{name:>17s}" for name, _ in cols))
    mod, deb = {}, {}
    for name, c in cols:
        Cv = voigt_matrix_3m(c["C11"], c["C12"], c["C13"], c["C33"], c["C44"],
                             c.get("C14", 0.0))
        mod[name] = vrh_moduli(Cv)
        deb[name] = debye_from_moduli(mod[name]["K"], mod[name]["G"], rho, nat, Vp)
    for q in ["K", "G", "E", "nu"]:
        print(f"{q:10s}" + "".join(f"{mod[name][q]:17.4f}" for name, _ in cols))
    for q, lab in [("vl", "v_l m/s"), ("vt", "v_t m/s"),
                   ("vD", "v_D m/s"), ("theta_D", "theta_D K")]:
        print(f"{lab:10s}" + "".join(f"{deb[name][q]:17.1f}" for name, _ in cols))

    # ---- theta_D: direct acoustic average vs DOS 2nd moment ----
    allv = np.array([vel[t] for t in base])
    vt_meas, vl_meas = allv[:, :2].mean(), allv[:, 2].mean()
    vD_meas, tD_meas = debye_from_vel(vl_meas, vt_meas, rho, nat, Vp)

    ph.run_mesh(args.mesh, is_gamma_center=True)
    ph.run_total_dos()
    tD_dos = debye_from_dos(ph.total_dos.frequency_points, ph.total_dos.dos)

    print("\n" + "=" * 76)
    print("DEBYE TEMPERATURE -- independent routes")
    print("=" * 76)
    print(f"  from elastic v_D (phonon LSQ C_ij) : {deb['phonon LSQ']['theta_D']:8.1f} K")
    if args.cij_strain:
        print(f"  from elastic v_D (strain C_ij)     : {deb['strain C_ij']['theta_D']:8.1f} K")
    print(f"  from elastic v_D (DFT C_ij)        : {deb['DFT']['theta_D']:8.1f} K")
    print(f"  direct acoustic-branch average     : {tD_meas:8.1f} K   "
          f"(v_l={vl_meas:.0f}, v_t={vt_meas:.0f}, v_D={vD_meas:.0f} m/s)")
    print(f"  phonon-DOS 2nd moment (thermodynamic): {tD_dos:7.1f} K   "
          f"(includes optical modes -> higher)")


if __name__ == "__main__":
    main()
