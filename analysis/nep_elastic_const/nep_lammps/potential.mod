# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

pair_style  nep 
pair_coeff  * * nep_y2026_m09_d04_h13_m17_s19_generation300000.txt  Mg Bi 


# Setup neighbor style
neighbor 1.0 nsq
neigh_modify once no every 1 delay 0 check yes

# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1
thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_modify norm no
