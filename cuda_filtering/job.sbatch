#!/usr/bin/env bash
#SBATCH --job-name=es2_wfilter
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "[info] host: $(hostname)"
echo "[info] date: $(date)"
echo "[info] jobid: ${SLURM_JOB_ID}"

# --- Modules: gcc provides cc1plus; nvhpc provides nvcc ---
module purge
module load gcc/12.4.0
module load nvhpc-hpcx-cuda12/25.1

echo "[info] gcc/g++:"
which gcc
gcc --version | head -n 1
which g++
g++ --version | head -n 1

echo "[info] nvcc:"
which nvcc
nvcc --version | sed -n '1,5p'

echo "[info] nvidia-smi:"
nvidia-smi || true

# --- Conda env: only needed for python+cv2 resizing in run.sh ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate es2
python -c "import cv2; print('cv2', cv2.__version__)"

# --- Build the CUDA program (NO OpenCV C++ linking) ---
# Ensure stb headers exist:
test -f stb_image.h || { echo "[error] stb_image.h missing"; exit 10; }
test -f stb_image_write.h || { echo "[error] stb_image_write.h missing"; exit 11; }

# Compile
nvcc -O3 -std=c++17 -arch=sm_86 filter.cu -o filter

# --- Run ---
IMGPATH="${1:-images}"
REPS="${2:-10}"

echo "[info] IMGPATH=${IMGPATH}"
echo "[info] REPS=${REPS}"

srun --ntasks=1 ./run.sh "${IMGPATH}" "${REPS}"


