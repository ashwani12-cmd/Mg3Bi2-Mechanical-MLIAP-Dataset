#!/bin/bash
# ============================================================
# reorganize_repo.sh
# ============================================================
# Run from inside: /home/ashwani/Mg3Bi2/Mg3Bi2-Mechanical-MLIAP-Dataset/
# Reorganizes your existing layout into clean GitHub structure.
# ============================================================
set -e

REPO="$(pwd)"
echo "Working in: $REPO"
echo ""

# ─── 1. REORGANIZE data/ — move top-level JSONs into category folders ───
cd "$REPO/data"

echo ">>> Reorganizing data/ ..."

# New categories: move JSONs into their own folders
mkdir -p elastic_Cij equib_perturb phonon_disp tensile compression elastic_vacancy

mv Mg3Bi2_elastic_Cij.json      elastic_Cij/data.json       2>/dev/null && echo "  moved elastic_Cij" || echo "  skip elastic_Cij (already moved?)"
mv Mg3Bi2_equib_perturb.json    equib_perturb/data.json     2>/dev/null && echo "  moved equib_perturb" || echo "  skip equib_perturb"
mv Mg3Bi2_phonon_disp.json      phonon_disp/data.json       2>/dev/null && echo "  moved phonon_disp" || echo "  skip phonon_disp"
mv Mg3Bi2_lattice_tension.json  tensile/data.json           2>/dev/null && echo "  moved tensile" || echo "  skip tensile"
mv Mg3Bi2_lattice_compressed.json compression/data.json     2>/dev/null && echo "  moved compression" || echo "  skip compression"
mv Mg3Bi2_elastic_vacancy.json  elastic_vacancy/data.json   2>/dev/null && echo "  moved elastic_vacancy" || echo "  skip elastic_vacancy"

# ─── 2. CLEAN TOXIC / REDUNDANT FILES ───
echo ""
echo ">>> Cleaning toxic and redundant data ..."

# Drop eps_strain ±85% files (E/at up to +646 eV, P up to -24000 GPa)
if ls uniaxial_strain/*eps_strain* 1>/dev/null 2>&1; then
    rm -v uniaxial_strain/*eps_strain*
    echo "  DROPPED: eps_strain files (±85% strain — toxic)"
fi

# Drop strain_y (identical to strain_x by hexagonal symmetry)
if [ -f uniaxial_strain/strain_y.json ]; then
    rm -v uniaxial_strain/strain_y.json
    echo "  DROPPED: strain_y.json (= strain_x by symmetry)"
fi

# Drop disp_only 0.2 Å (fully covered by 0.3 Å)
if [ -f disp_only/Mg3Bi2_pct_0.2_disp.json ]; then
    rm -v disp_only/Mg3Bi2_pct_0.2_disp.json
    echo "  DROPPED: disp_only 0.2 Å (redundant with 0.3 Å)"
fi

# ─── 3. CREATE PLACEHOLDER FOLDERS for missing categories ───
echo ""
echo ">>> Creating placeholders for pending data ..."

mkdir -p vacancy
mkdir -p phonopy_fd
mkdir -p aimd_snapshots/{300K,400K,500K,600K}

for d in vacancy phonopy_fd aimd_snapshots; do
    if [ ! -f "$d/README.md" ]; then
        echo "# $d — pending DFT calculations" > "$d/README.md"
        echo "  created $d/README.md placeholder"
    fi
done

# ─── 4. SET UP TOP-LEVEL STRUCTURE ───
cd "$REPO"

echo ""
echo ">>> Setting up top-level folders ..."

# Rename 'script' -> 'scripts' (convention)
if [ -d script ] && [ ! -d scripts ]; then
    mv script scripts
    echo "  renamed script/ -> scripts/"
fi

# Create missing top-level folders
mkdir -p primitive_cell
mkdir -p nep_training
mkdir -p validation/{elastic_constants,phonon_dispersion,thermal_expansion,vacancy_formation}
mkdir -p figures/{parity_plots,phonon_bands,thermal_expansion,elastic_constants}
mkdir -p docs

# ─── 5. CREATE .gitignore ───
if [ ! -f .gitignore ]; then
cat > .gitignore << 'GITEOF'
# QE temporaries
outdir/
tmp/
*.wfc*
*.save/

# Large binary outputs (use git-lfs if needed)
*.pwo

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Editor
*.swp
*.swo
*~
GITEOF
echo "  created .gitignore"
fi

# ─── 6. CREATE LICENSE ───
if [ ! -f LICENSE ]; then
cat > LICENSE << 'LICEOF'
MIT License

Copyright (c) 2026 Ashwani Kushwaha, IIT Bombay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
LICEOF
echo "  created LICENSE"
fi

# ─── 7. PRINT FINAL TREE ───
echo ""
echo "============================================================"
echo "FINAL STRUCTURE"
echo "============================================================"
find . -maxdepth 3 -type d | grep -v ".git/" | grep -v "__pycache__" | sort | \
    sed 's|[^/]*/|  |g; s|  |├── |'

echo ""
echo "============================================================"
echo "DATA FILE COUNTS"
echo "============================================================"
for d in data/*/; do
    n=$(find "$d" -name "*.json" | wc -l)
    echo "  $(basename $d): $n JSON files"
done

echo ""
echo "DONE. Next steps:"
echo "  1. Copy relaxed_unitcell.pwi into primitive_cell/"
echo "  2. Copy nep.in, nep.txt, loss.out into nep_training/"
echo "  3. git add -A && git commit -m 'reorganize dataset structure'"
echo "  4. git push origin main"
