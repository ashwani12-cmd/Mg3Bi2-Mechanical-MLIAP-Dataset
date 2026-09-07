#!/bin/sh

#SBATCH --job-name=gpumd-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --mem=800

nvidia-smi

# Clean up existing environment
#module purge

# Load modules
#module load zlib-ng/2.2.1-gcc-8.5.0-orhazg6
#module load zstd/1.5.6-gcc-8.5.0-fg5aduu
#module load binutils/2.43.1-gcc-8.5.0-kdsptcv
#module load gmp/6.3.0-gcc-8.5.0-4gikgah
#module load mpfr/4.2.1-gcc-8.5.0-wvjtu4c
#module load mpc/1.3.1-gcc-8.5.0-qiu6dv2
#module load gcc/14.2.0-gcc-8.5.0-777kyuf
module load gcc
module load cuda/12.0

module list
echo "Running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# Set working directory
#cd /home/IITB/multiscale-mechanics/ashwani12/ashwani/gpumd-pot

# Debug statements
echo "Starting job"
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -l
echo "PATH: $PATH"
echo "Running nep"

# Run with full path
srun /home/IITB/multiscale-mechanics/ashwani12/software/builds/GPUMD/src/gpumd > out.dat

#srun nep > out.dat

echo "Job completed"

