#!/usr/bin/env python3
"""
gruneisen_calorine.py
=====================
Computes:
  - Elastic constants (Cij) via calorine get_elastic_stiffness_tensor
  - VRH bulk/shear/Young/Poisson moduli
  - Sound velocities (vl, vt, vD)
  - Debye temperature θ_D
  - Grüneisen parameter γ (macroscopic from α_V + K)
  - Cauchy pressure, Zener anisotropy, Vickers hardness
  - Mode Grüneisen (optional, via QHA phonons)

Usage:
    python3 gruneisen_calorine.py --nep nep.txt --pwi espresso.pwi
    python3 gruneisen_calorine.py --nep nep.txt --pwi espresso.pwi --skip-phonon
"""

import argparse, re, os, sys
import numpy as np

try:
    from calorine.calculators import CPUNEP
    from calorine.tools import get_elastic_stiffness_tensor
    from ase.optimize import FIRE, BFGS
    from ase.filters import FrechetCellFilter, UnitCellFilter
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ── DFT reference (for comparison) ───────────────────────────────────────────
DFT = dict(C11=67.93, C12=35.93, C13=22.98, C14=3.00,
           C33=78.50, C44=15.53, C66=15.87)


# ── PWI parser ────────────────────────────────────────────────────────────────
def parse_pwi(path):
    with open(path) as f: lines = f.readlines()
    cell_rows=[]; in_cell=False
    for line in lines:
        if re.match(r'\s*CELL_PARAMETERS\s*\(angstrom\)', line, re.IGNORECASE):
            in_cell=True; continue
        if in_cell:
            nums=re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if len(nums)>=3: cell_rows.append([float(x) for x in nums[:3]])
            if len(cell_rows)==3: break
    cell=np.array(cell_rows)
    syms=[]; pos=[]; in_atoms=False
    for line in lines:
        if re.match(r'\s*ATOMIC_POSITIONS\s*\(angstrom\)', line, re.IGNORECASE):
            in_atoms=True; continue
        if in_atoms:
            parts=line.split()
            if not parts: continue
            if len(parts)>=4:
                try:
                    syms.append(parts[0])
                    pos.append([float(parts[i]) for i in [1,2,3]])
                except: break
            elif parts: break
    from ase import Atoms
    return Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)


# ── VRH averages ─────────────────────────────────────────────────────────────
def vrh_moduli(C):
    S  = np.linalg.inv(C)
    Kv = ((C[0,0]+C[1,1]+C[2,2]) + 2*(C[0,1]+C[1,2]+C[0,2])) / 9
    Gv = ((C[0,0]+C[1,1]+C[2,2]) - (C[0,1]+C[1,2]+C[0,2])
          + 3*(C[3,3]+C[4,4]+C[5,5])) / 15
    Kr = 1.0/((S[0,0]+S[1,1]+S[2,2]) + 2*(S[0,1]+S[1,2]+S[0,2]))
    Gr = 15.0/(4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[1,2]+S[0,2])
               + 3*(S[3,3]+S[4,4]+S[5,5]))
    K=(Kv+Kr)/2; G=(Gv+Gr)/2
    E=9*K*G/(3*K+G); nu=(3*K-2*G)/(2*(3*K+G))
    return dict(Kv=Kv,Kr=Kr,K=K,Gv=Gv,Gr=Gr,G=G,E=E,nu=nu)


# ── Symmetry checks ───────────────────────────────────────────────────────────
def hexagonal_checks(C):
    rel = [("C11=C22",        C[0,0], C[1,1]),
           ("C13=C23",        C[0,2], C[1,2]),
           ("C44=C55",        C[3,3], C[4,4]),
           ("C66=(C11-C12)/2",C[5,5], 0.5*(C[0,0]-C[0,1]))]
    print("\ntrigonal -3m symmetry relations:")
    worst=0.0
    for name,a,b in rel:
        d=abs(a-b); sc=max(abs(a),abs(b),1.0); worst=max(worst,100*d/sc)
        print(f"  {name:<22} {a:8.2f}  vs  {b:8.2f}   diff {d:.2f} GPa ({100*d/sc:.2f}%)")
    status = "OK ✓" if worst<5 else "CHECK ✗"
    print(f"  worst deviation {worst:.2f}%  — symmetry {status}")


# ── Debye temperature ─────────────────────────────────────────────────────────
def debye_props(K_GPa, G_GPa, rho, n_atoms, V_A3):
    K=K_GPa*1e9; G=G_GPa*1e9
    vl=np.sqrt((K+4*G/3)/rho)
    vt=np.sqrt(G/rho)
    vD=(1/3*(1/vl**3+2/vt**3))**(-1/3)
    hbar=1.0545718e-34; kB=1.380649e-23
    n=n_atoms/(V_A3*1e-30)
    tD=hbar/kB*vD*(6*np.pi**2*n)**(1/3)
    return dict(vl=vl,vt=vt,vD=vD,theta_D=tD)


# ── Mode Grüneisen ────────────────────────────────────────────────────────────
def compute_mode_gruneisen(atoms0, nep_path, strains, supercell_matrix, T=300):
    """Compute mode Grüneisen via finite difference on phonon frequencies."""
    try:
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        from calorine.tools import get_force_constants
    except ImportError as e:
        print(f"  Phonopy not available: {e}")
        return None

    print(f"\n  Building strained cells and computing force constants...")
    ph_list=[]; vol_list=[]

    for eps in strains:
        calc=CPUNEP(nep_path)
        strained=atoms0.copy()
        factor=(1+eps)**(1/3)
        strained.set_cell(strained.cell.array*factor, scale_atoms=True)
        # Relax ions at fixed cell
        strained.calc=calc
        opt=FIRE(strained, logfile=None)
        opt.run(fmax=1e-6, steps=5000)
        vol=strained.get_volume()
        vol_list.append(vol)

        ph_atoms=PhonopyAtoms(
            symbols  =strained.get_chemical_symbols(),
            positions=strained.get_positions(),
            cell     =strained.get_cell().array)
        ph=Phonopy(ph_atoms, supercell_matrix=supercell_matrix,
                   primitive_matrix='auto', log_level=0)

        calc2=CPUNEP(nep_path)
        ph_tmp=get_force_constants(strained, calc2,
                                 supercell_matrix=supercell_matrix)
        ph.force_constants=ph_tmp.force_constants
        ph_list.append(ph)
        print(f"    ε={eps:+.3f}: V={vol:.4f} Å³  FC done")

    # Finite difference Grüneisen
    # Use central 3 strains
    n=len(ph_list); mid=n//2
    ph_m=ph_list[mid-1]; ph_0=ph_list[mid]; ph_p=ph_list[mid+1]
    dV=vol_list[mid+1]-vol_list[mid-1]; V0=vol_list[mid]

    mesh=[8,8,8]
    gammas_all=[]; freqs_all=[]

    for ph in [ph_m,ph_0,ph_p]:
        ph.run_mesh(mesh, is_mesh_symmetry=True, with_eigenvectors=False)

    freq_m = ph_m.get_mesh_dict()['frequencies']  # (nq, nbands) THz
    freq_0 = ph_0.get_mesh_dict()['frequencies']
    freq_p = ph_p.get_mesh_dict()['frequencies']

    # γ_i = -V/ω * dω/dV ≈ -V0/freq_0 * (freq_p-freq_m)/dV
    with np.errstate(divide='ignore',invalid='ignore'):
        gamma_mode = np.where(
            freq_0 > 0.1,
            -V0/freq_0 * (freq_p-freq_m)/dV,
            np.nan)

    # Thermodynamic average
    hbar=1.0545718e-34; kB=1.380649e-23
    omega=freq_0*2*np.pi*1e12
    mask=(freq_0>0.1)
    Cv_list=[]; g_list=[]
    for iq in range(omega.shape[0]):
        for ib in range(omega.shape[1]):
            if not mask[iq,ib] or np.isnan(gamma_mode[iq,ib]): continue
            w=omega[iq,ib]; x=hbar*w/(kB*T)
            if x>50: continue
            Cv=kB*x**2*np.exp(x)/(np.exp(x)-1)**2
            Cv_list.append(Cv); g_list.append(gamma_mode[iq,ib])

    if Cv_list:
        gamma_thermo=np.average(g_list,weights=Cv_list)
        return dict(gamma_thermo=gamma_thermo,
                    gamma_mean=np.nanmean(gamma_mode[mask]),
                    gamma_acoustic=np.nanmean(gamma_mode[freq_0<2.0]),
                    gamma_optical=np.nanmean(gamma_mode[freq_0>2.0]))
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser(description=__doc__,
          formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nep',   required=True)
    ap.add_argument('--pwi',   default='espresso.pwi')
    ap.add_argument('--eps',   type=float, default=0.001,
                    help='Strain magnitude for elastic constants (default: 0.001)')
    ap.add_argument('--rho',   type=float, default=7000,
                    help='Density kg/m³ (default: 7000 for Mg3Bi2)')
    ap.add_argument('--alpha-v', type=float, default=65.18e-6,
                    help='Volumetric thermal expansion at 0GPa (K⁻¹, default: 65.18e-6)')
    ap.add_argument('--temp',  type=float, default=300)
    ap.add_argument('--skip-phonon', action='store_true',
                    help='Skip mode Grüneisen (faster, just elastic+Debye+γ_macro)')
    ap.add_argument('--strains', nargs='+', type=float,
                    default=[-0.03,-0.015,0.0,0.015,0.03])
    ap.add_argument('--supercell', nargs=9, type=int,
                    default=[2,0,0,0,2,0,0,0,2])
    args=ap.parse_args()

    print("="*65)
    print("GRÜNEISEN + MECHANICAL PROPERTIES — Mg₃Bi₂ (NEP/calorine)")
    print("="*65)
    print(f"NEP    : {args.nep}")
    print(f"fmax   : 1e-6 eV/Å  |  eps : {args.eps}  |  T : {args.temp} K")
    print()

    # ── Step 1: Relax ────────────────────────────────────────────────────────
    print("Step 1: Relaxing primitive cell")
    print("-"*45)
    prim=parse_pwi(args.pwi)
    calc=CPUNEP(args.nep)
    prim.calc=calc
    BFGS(FrechetCellFilter(prim), logfile=None).run(fmax=1e-6, steps=500)
    L=np.linalg.norm(prim.cell,axis=1)
    V0=prim.get_volume()
    fmax_act=np.sqrt((prim.get_forces()**2).sum(axis=1).max())
    print(f"  a={L[0]:.5f} Å  c={L[2]:.5f} Å  c/a={L[2]/L[0]:.5f}")
    print(f"  V={V0:.5f} Å³  |F|max={fmax_act:.2e} eV/Å")

    # ── Step 2: Elastic constants ─────────────────────────────────────────────
    print(f"\nStep 2: Elastic stiffness tensor (ε={args.eps})")
    print("-"*45)
    calc2=CPUNEP(args.nep)
    prim.calc=calc2
    C  = get_elastic_stiffness_tensor(prim, clamped=False, epsilon=args.eps)
    Cc = get_elastic_stiffness_tensor(prim, clamped=True,  epsilon=args.eps)

    print("\nRelaxed-ion Cij (GPa), Voigt xx yy zz yz xz xy:")
    for row in C:
        print("   "+" ".join(f"{v:9.2f}" for v in row))

    print("\nIndependent constants (relaxed / clamped):")
    for name,i,j in [("C11",0,0),("C12",0,1),("C13",0,2),
                      ("C33",2,2),("C44",3,3),("C66",5,5)]:
        print(f"  {name}: {C[i,j]:8.2f}  /  {Cc[i,j]:8.2f}  GPa")

    hexagonal_checks(C)

    # Born stability
    c11,c12,c13,c33,c44=C[0,0],C[0,1],C[0,2],C[2,2],C[3,3]
    tests=[("C11>|C12|",c11>abs(c12)),
           ("2C13²<C33(C11+C12)",2*c13**2<c33*(c11+c12)),
           ("C44>0",c44>0),("C66>0",C[5,5]>0)]
    print("\nBorn stability criteria:")
    for name,ok in tests:
        print(f"  {name:<28} {'PASS ✓' if ok else 'FAIL ✗'}")

    # ── Step 3: VRH + Debye + Grüneisen ──────────────────────────────────────
    print("\nStep 3: VRH averages + Debye + Grüneisen")
    print("-"*45)
    vrh=vrh_moduli(C)
    deb=debye_props(vrh['K'],vrh['G'],args.rho,len(prim),V0)

    # Macroscopic Grüneisen
    kB_J=1.380649e-23; N_m3=len(prim)/(V0*1e-30)
    Cv_DP=3*N_m3*kB_J  # Dulong-Petit J/(m³·K)
    gamma_macro=args.alpha_v*vrh['K']*1e9/Cv_DP

    # Additional
    cauchy=C[0,1]-C[3,3]
    zener=2*C[3,3]/(C[0,0]-C[0,1])
    AU=5*vrh['Gv']/vrh['Gr']+vrh['Kv']/vrh['Kr']-6
    Hv=max(0, 2*(vrh['G']/vrh['K'])**2*vrh['G']-3)

    print(f"\n  VRH Moduli:")
    print(f"    K   = {vrh['K']:.3f} GPa  (V={vrh['Kv']:.2f}, R={vrh['Kr']:.2f})")
    print(f"    G   = {vrh['G']:.3f} GPa  (V={vrh['Gv']:.2f}, R={vrh['Gr']:.2f})")
    print(f"    E   = {vrh['E']:.3f} GPa")
    print(f"    ν   = {vrh['nu']:.4f}")
    print(f"    G/K = {vrh['G']/vrh['K']:.4f}  "
          f"→ {'ductile' if vrh['G']/vrh['K']<0.571 else 'brittle'} (Pugh)")

    print(f"\n  Sound velocities (ρ={args.rho} kg/m³):")
    print(f"    vl  = {deb['vl']:.1f} m/s")
    print(f"    vt  = {deb['vt']:.1f} m/s")
    print(f"    vD  = {deb['vD']:.1f} m/s")

    print(f"\n  Debye temperature:")
    print(f"    θ_D = {deb['theta_D']:.1f} K")

    print(f"\n  Grüneisen parameter (macroscopic, Dulong-Petit):")
    print(f"    α_V = {args.alpha_v*1e6:.2f} × 10⁻⁶ K⁻¹  (from MD thermal expansion)")
    print(f"    K   = {vrh['K']:.3f} GPa")
    print(f"    Cv  = {Cv_DP:.4e} J/(m³·K)  (Dulong-Petit)")
    print(f"    γ   = α_V·K/Cv = {gamma_macro:.4f}")

    print(f"\n  Additional mechanical properties:")
    print(f"    Cauchy pressure (C12-C44) = {cauchy:.2f} GPa"
          f"  → {'metallic/ductile' if cauchy>0 else 'covalent/brittle'}")
    print(f"    Zener anisotropy Az       = {zener:.4f}"
          f"  ({'~isotropic' if abs(zener-1)<0.1 else 'anisotropic'})")
    print(f"    Universal anisotropy AU   = {AU:.4f}")
    print(f"    Vickers hardness Hv (est) = {Hv:.2f} GPa")

    # ── Step 4: DFT comparison ────────────────────────────────────────────────
    print("\n"+"="*65)
    print("Comparison with DFT")
    print("="*65)
    print(f"{'Property':<10} {'DFT':>10} {'NEP':>10} {'|Δ|':>10} {'%err':>10}")
    print("-"*52)
    vals={'C11':C[0,0],'C12':C[0,1],'C13':C[0,2],'C14':C[0,3],
          'C33':C[2,2],'C44':C[3,3],'C66':C[5,5]}
    errs=[]
    for k,v in vals.items():
        d=DFT[k]; err=abs(v-d); pct=err/abs(d)*100; errs.append(pct)
        print(f"{k:<10} {d:>10.2f} {v:>10.2f} {err:>10.2f} {pct:>9.2f}%")
    print("-"*52)
    print(f"{'MAPE':<10} {'':>10} {'':>10} {'':>10} {np.mean(errs):>9.2f}%")

    # ── Step 5: Mode Grüneisen (optional) ────────────────────────────────────
    gamma_thermo=None
    if not args.skip_phonon:
        print(f"\nStep 4: Mode Grüneisen parameters (QHA)")
        print("-"*45)
        sc_mat=np.array(args.supercell).reshape(3,3)
        gr=compute_mode_gruneisen(prim,args.nep,args.strains,sc_mat,args.temp)
        if gr:
            gamma_thermo=gr['gamma_thermo']
            print(f"\n  Mode Grüneisen results at {args.temp}K:")
            print(f"    γ_thermo  (Cv-weighted) = {gr['gamma_thermo']:.4f}")
            print(f"    γ_mean    (all modes)   = {gr['gamma_mean']:.4f}")
            print(f"    γ_acoustic (<2 THz)     = {gr['gamma_acoustic']:.4f}")
            print(f"    γ_optical  (>2 THz)     = {gr['gamma_optical']:.4f}")
            print(f"\n  Cross-check:")
            print(f"    γ_macro  (α_V·K/Cv) = {gamma_macro:.4f}")
            print(f"    γ_thermo (phonons)  = {gamma_thermo:.4f}")

    # ── Summary + save ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL SUMMARY — Mg₃Bi₂ (NEP)")
    print(f"{'='*65}")
    print(f"  a₀={L[0]:.5f} Å  c₀={L[2]:.5f} Å  c/a={L[2]/L[0]:.4f}")
    print(f"  K={vrh['K']:.2f} GPa  G={vrh['G']:.2f} GPa  E={vrh['E']:.2f} GPa  ν={vrh['nu']:.4f}")
    print(f"  θ_D = {deb['theta_D']:.1f} K")
    print(f"  γ_macro = {gamma_macro:.3f}", end='')
    if gamma_thermo: print(f"  γ_thermo = {gamma_thermo:.3f}")
    else: print()
    print(f"  vl={deb['vl']:.0f} m/s  vt={deb['vt']:.0f} m/s  vD={deb['vD']:.0f} m/s")

    with open('gruneisen_results.txt','w') as f:
        f.write(f"a0={L[0]:.6f} A\nc0={L[2]:.6f} A\nV0={V0:.6f} A3\n")
        f.write(f"K={vrh['K']:.4f} GPa\nG={vrh['G']:.4f} GPa\n")
        f.write(f"E={vrh['E']:.4f} GPa\nnu={vrh['nu']:.6f}\n")
        f.write(f"theta_D={deb['theta_D']:.2f} K\n")
        f.write(f"vl={deb['vl']:.2f} m/s\nvt={deb['vt']:.2f} m/s\nvD={deb['vD']:.2f} m/s\n")
        f.write(f"gamma_macro={gamma_macro:.6f}\n")
        if gamma_thermo: f.write(f"gamma_thermo={gamma_thermo:.6f}\n")
        f.write(f"Cauchy={cauchy:.4f} GPa\nZener={zener:.6f}\nAU={AU:.6f}\nHv={Hv:.4f} GPa\n")
    print(f"\n  Saved: gruneisen_results.txt")


if __name__=='__main__':
    main()
