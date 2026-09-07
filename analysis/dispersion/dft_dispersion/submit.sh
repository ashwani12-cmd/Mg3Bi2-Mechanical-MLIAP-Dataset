#!/bin/bash
#SBATCH --job-name=basal_z_relax
#SBATCH --partition=hm
#SBATCH --exclude=rmhm025,rmhm026,rmhm027,rmhm034,rmhm035,rmhm036,rmcn156,rmcn157,rmcn158,rmcn159,rmcn160,rmcn340,rmcn341,rmcn342,rmcn343,rmcn344,rmcn441,rmcn442,rmcn443,rmcn444,rmcn445,rmcn129,rmcn284,rmcn345,rmcn361,rmcn388,rmcn412,rmcn447,rmcn049,rmcn050,rmcn051,rmcn075,rmcn076,rmcn111,rmcn112,rmcn176,rmcn211,rmcn283,rmcn408,rmcn453,rmcn476,rmcn067,rmcn209,rmcn210,rmcn239,rmcn240,rmcn131,rmcn132,rmcn133,rmcn134,rmcn368,rmcn369,rmcn370,rmcn371,rmcn372,rmcn286,rmcn287,rmcn393,rmcn407,rmcn434,rmcn100,rmcn101,rmcn102,rmcn103,rmcn104,rmcn002,rmcn289,rmcn290,rmcn291,rmcn450,rmcn002,rmcn289,rmcn290,rmcn291,rmcn450
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=basal-%j.out
#SBATCH --error=basal-%j.err
#SBATCH --mem-per-cpu=800

# ============================================================
# Environment
# ============================================================
module purge
module load nvhpc/24.11
module load intel-oneapi-mkl/2024.2.2-oneapi-2025.0.1-5u4sz3m

# ============================================================
# OpenMP (QE MPI-only run)
# ============================================================
export OMP_NUM_THREADS=1

# ============================================================
# Paths
# ============================================================

# Correct mpirun from HPC-X (nvhpc)
MPIRUN_PATH="/home/apps/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ompi/bin/mpirun"

# Quantum ESPRESSO pw.x
PWX_BIN="/home/IITB/multiscale-mechanics/amit.k.singh/software/builds/qe-7.1/build/bin/pw.x"

# ============================================================
# Go to submission directory
# ============================================================
cd "${SLURM_SUBMIT_DIR}" || exit 1

echo "=============================================="
echo " Job started on: $(date)"
echo " Job ID        : ${SLURM_JOB_ID}"
echo " Nodes         : ${SLURM_JOB_NUM_NODES}"
echo " MPI tasks     : ${SLURM_NTASKS}"
echo " Working dir   : $(pwd)"
echo "=============================================="

# ============================================================
# Run Quantum ESPRESSO
# ============================================================
time ${MPIRUN_PATH} -np ${SLURM_NTASKS} ${PWX_BIN} <  Mg3Bi2-001.in >  Mg3Bi2-001.out

echo "=============================================="
echo " Job finished on: $(date)"
echo "=============================================="


