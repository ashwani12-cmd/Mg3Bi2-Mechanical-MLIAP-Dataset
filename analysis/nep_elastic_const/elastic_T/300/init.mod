# NOTE: This script can be modified for different atomic structures, 
# units, etc. See in.elastic for more info.
#

# Define the finite deformation size. Try several values of this
# variable to verify that results do not depend on it.
variable up equal 2.0e-2
 
# metal units, elastic constants in GPa
units		metal
variable cfac equal 1.0e-4
variable cunits string GPa

# Define MD parameters
variable nevery equal 10                  # sampling interval
variable nrepeat equal 10                 # number of samples
variable nfreq equal ${nevery}*${nrepeat} # length of one average
variable nthermo equal ${nfreq}           # interval for thermo output
variable nequil equal 10*${nthermo}       # length of equilibration run
variable nrun equal 3*${nthermo}          # length of equilibrated run


variable temp equal 300.0                # temperature of initial sample


variable timestep equal 0.001             # timestep
variable adiabatic equal 2                # adiabatic (1) or isothermal (2)
variable tdamp equal 0.01                 # time constant for thermostat
variable seed equal 123457                # seed for thermostat

# generate the box and atom positions using a diamond lattice
boundary	p p p

lattice custom 1 &
	       a1 4.604277901 0.0 0.0 & 
	       a2 -2.30213895 3.987421628 -0.0 &
               a3 0.0 0.0 7.276386111 &
               basis 0.3333333299 0.6666666700 0.6284199547 &
               basis 0.6666666700 0.3333333300 0.3715800453 & 
               basis 0.0000000000 0.0000000000 0.0000000000 &
               basis 0.3333333299 0.6666666700 0.2196782529 &
               basis 0.6666666700 0.3333333300 0.7803217471 


# ---------- triclinic region consistent with a1,a2,a3 ----------
variable ax equal 4.604277901
variable bx equal -2.30213895     # = a2.x  -> xy tilt
variable by equal 3.987421628     # = a2.y
variable cz equal 7.276386111     # = a3.z

region box prism 0 ${ax}  0 ${by}  0 ${cz}  ${bx}  0.0 0.0  units box
create_box 2 box
create_atoms 1 box  basis 4 2  basis 5 2   # Bi on basis indices 4 & 5 (type 2), Mg others (type 1)

# realistic masses (not required for minimization, but good practice)
mass 1 24.305     # Mg
mass 2 208.98040  # Bi

# --- make an n×m×p supercell (example: 2×3×1) ---
replicate 15 15 12


velocity	all create ${temp} 87287


