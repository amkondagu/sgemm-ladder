#!/bin/bash
#----------------------------------------------------------------------
# ECE 379K Lab 2 -- SLURM batch script for one Lonestar6 A100.
#
#     sbatch -A YOUR_TACC_ACCOUNT job.slurm
#     squeue -u $USER
#     cat lab2.o<jobid>
#
# This run records the device, counter check, correctness checks,
# sanitizers, ladder, sweep, both ncu tables, and the full reg2d profile.
# The Part 3 barrier trials run afterward on temporary source copies.
#
# The lab needs one GPU, so use gpu-a100-small. Pass the allocation
# account with sbatch -A.
#----------------------------------------------------------------------
#SBATCH -J lab2
#SBATCH -o lab2.o%j
#SBATCH -e lab2.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00

set -exo pipefail
module load gcc/13.2.0 cuda/12.8
module list 2>&1
nvidia-smi
nvcc --version
ncu --version
hostname

# Plain FP32 everywhere: no TF32 tensor cores behind cuBLAS's back.
export NVIDIA_TF32_OVERRIDE=0

make clean
make
ncu --metrics sm__cycles_elapsed.avg ./sgemm cublas 1024 --once 2>&1 | tee counter-check.txt
make check
make sanitize
make ladder
make sweep
make ncu
make ncu-sweep
make ncu-full RUNG=reg2d
make ptxas

barrier_dir=$(mktemp -d)
trap 'rm -rf -- "$barrier_dir"' EXIT
cp Makefile kernels.cuh parts.h sgemm.cu "$barrier_dir/"

for barrier in 1 2; do
    awk -v barrier="$barrier" '
        /^__global__ void sgemm_tiled_kernel\(/ { in_tiled = 1 }
        in_tiled && /^[[:space:]]*__syncthreads\(\);/ {
            if (++seen == barrier) sub(/__syncthreads/, "// __syncthreads")
        }
        /^}/ { in_tiled = 0 }
        { print }
        END { if (seen != 2) exit 1 }
    ' kernels.cu > "$barrier_dir/kernels.cu"
    make -B -C "$barrier_dir"

    # A numerical failure is expected to be possible in these trials.
    "$barrier_dir/sgemm" check > "barrier${barrier}-check.txt" 2>&1 || test "$?" -eq 1
    compute-sanitizer --tool racecheck "$barrier_dir/sgemm" tiled 256 --once \
        > "barrier${barrier}-racecheck.txt" 2>&1
done
