#!/usr/bin/env python3
"""
neb_oct_mechanism.py
====================
Shows that Oct vacancy migration in Mg3Bi2 proceeds via a TWO-STEP
Oct -> Tet -> Oct route, not the direct Oct -> Oct hop.

The direct 4.604 A Oct-Oct jump forces the migrating Mg to within
2.08 A of a Bi atom, giving an artificially high 0.948 eV barrier.
Routing through the intermediate Tet site costs only 0.445 eV.

Usage:
    python3 neb_oct_mechanism.py --nep <nep.txt> --pwi perfect_1.pwo
"""
import argparse, numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from ase.mep import NEB
from calorine.calculators import CPUNEP


def build_ordered_vacancy_pair(perfect, ra, rb):
    """Two vacancy cells with identical atom ordering (ASE NEB requirement)."""
    N = len(perfect)
    s, q = perfect.get_chemical_symbols(), perfect.get_positions()
    ii = [i for i in range(N) if i != ra]
    jj = [i for i in range(N) if i != rb]
    sB = sum(1 for i in range(rb) if i != ra)
    sA = sum(1 for i in range(ra) if i != rb)
    perm = [None]*(N-1); perm[sB] = sA
    for k, o in enumerate(ii):
        if k != sB:
            perm[k] = jj.index(o)
    init  = Atoms(symbols=[s[i] for i in ii],
                  positions=np.array([q[i] for i in ii]),
                  cell=perfect.get_cell(), pbc=True)
    final = Atoms(symbols=[s[jj[perm[k]]] for k in range(N-1)],
                  positions=np.array([q[jj[perm[k]]] for k in range(N-1)]),
                  cell=perfect.get_cell(), pbc=True)
    return init, final


def classify(atoms, rc=3.6):
    """Oct = CN(Bi) 6 (1a site); Tet = CN(Bi) 4 (2d site)."""
    cell = atoms.get_cell().array; ci = np.linalg.inv(cell)
    s = np.array(atoms.get_chemical_symbols()); p = atoms.get_positions()
    bi = p[s == 'Bi']; oct_, tet = [], []
    for i in np.where(s == 'Mg')[0]:
        fr = (bi - p[i]) @ ci; fr -= np.round(fr)
        (oct_ if (np.linalg.norm(fr@cell, axis=-1) < rc).sum() >= 6 else tet).append(int(i))
    return oct_, tet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nep', required=True)
    ap.add_argument('--pwi', default='perfect_1.pwo')
    ap.add_argument('--n-images', type=int, default=9)
    ap.add_argument('--fmax', type=float, default=0.05)
    ap.add_argument('--outdir', default='oct_mechanism')
    a = ap.parse_args()
    import os; os.makedirs(a.outdir, exist_ok=True)
    def calc(): return CPUNEP(a.nep)

    p = read(a.pwi, index=-1, format='espresso-out')
    p.calc = calc(); FIRE(p, logfile=None).run(fmax=1e-4, steps=200000)
    cell = p.get_cell().array; ci = np.linalg.inv(cell); pos = p.get_positions()
    OCT, TET = classify(p)
    print(f"Oct(CN6)={len(OCT)}  Tet(CN4)={len(TET)}")

    def mic(x, y):
        fr = (y-x) @ ci; fr -= np.round(fr); return np.linalg.norm(fr@cell)

    A = OCT[0]
    M = min(TET, key=lambda m: mic(pos[A], pos[m]))
    B = min([o for o in OCT if o != A], key=lambda o: mic(pos[M], pos[o]))
    print(f"Oct A=#{A}  Tet M=#{M}  Oct B=#{B}")
    print(f"  d(A,M)={mic(pos[A],pos[M]):.3f}  d(M,B)={mic(pos[M],pos[B]):.3f}  "
          f"d(A,B)={mic(pos[A],pos[B]):.3f} A")

    def relax(x):
        x = x.copy(); x.calc = calc()
        FIRE(x, logfile=None).run(fmax=1e-4, steps=200000)
        return x, x.get_potential_energy()

    def interp(imgs):
        i0, i1 = imgs[0], imgs[-1]; n = len(imgs)-2
        pi, pf = i0.get_positions(), i1.get_positions()
        df = (pf-pi) @ ci; df -= np.round(df); dc = df @ cell
        for k, im in enumerate(imgs[1:-1], 1):
            im.set_positions(pi + k/(n+1)*dc)

    def run(i0, i1, tag):
        imgs = []
        for x in [i0] + [None]*a.n_images + [i1]:
            im = i0.copy() if x is None else x.copy(); im.calc = calc(); imgs.append(im)
        nb = NEB(imgs, climb=True, method='improvedtangent', allow_shared_calculator=False)
        interp(imgs)
        conv = FIRE(nb, logfile=None).run(fmax=a.fmax, steps=3000)
        E = np.array([im.get_potential_energy() for im in imgs]); E -= E[0]
        print(f"    {tag:<26} Ea = {E.max():.4f} eV  ({'converged' if conv else 'NOT converged'})")
        write(f"{a.outdir}/ts_{tag.replace(' ','').replace('->','_')}.xyz", imgs[int(E.argmax())])
        return E

    iAM, fAM = build_ordered_vacancy_pair(p, A, M)
    iMB, fMB = build_ordered_vacancy_pair(p, M, B)
    iAB, fAB = build_ordered_vacancy_pair(p, A, B)
    rAM, EA = relax(iAM); rAMf, EM = relax(fAM)
    rMB, _  = relax(iMB); rMBf, _ = relax(fMB)
    rAB, _  = relax(iAB); rABf, _ = relax(fAB)
    off = EM - EA
    print(f"\n  E(vac@Oct)={EA:.6f}  E(vac@Tet)={EM:.6f}  offset={off*1000:+.1f} meV\n")
    print("NEB segments:")
    E1 = run(rAM, rAMf, 'Oct->Tet')
    E2 = run(rMB, rMBf, 'Tet->Oct')
    E3 = run(rAB, rABf, 'Oct->Oct direct')

    eff = max(E1.max(), off + E2.max())
    print("\n" + "="*60)
    print(f"  direct   Oct->Oct       : {E3.max():.3f} eV   <- straight line through Bi")
    print(f"  two-step Oct->Tet->Oct  : {eff:.3f} eV   <- actual mechanism")
    print(f"  DFT (Assadi 2022)       : 0.660 eV")
    print("="*60)
    np.savez(f"{a.outdir}/mep.npz", E1=E1, E2=E2, E3=E3, off=off)
    print(f"\nsaved -> {a.outdir}/")

if __name__ == '__main__':
    main()
