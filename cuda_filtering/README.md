# ES2 – CUDA Image Filtering (HPC Assignment)

This project implements **Exercise 2** of the *High Performance Computing* assignment (2024–2025).  
The goal is to apply a **2D stencil-based filter** to high-resolution images using **CUDA**, analyzing performance and scalability on GPUs.

---

## Project Overview

The application applies a weighted **3×3 stencil filter** to RGB images in order to smooth noise (e.g., salt-and-pepper noise).  
The filter is executed in parallel on the GPU using CUDA, processing each color channel independently.

Target image resolutions:
- **4K** (3840 × 2160)
- **8K** (7680 × 4320)
- **16K** (15360 × 8640)

---

## Filter Definition

The applied stencil is:

1 2 1
2 4 2
1 2 1


Each output pixel is computed as:

g(x,y) = (1 / 16) * Σ Σ w(i,j) * f(x+i, y+j)


Border pixels are set to zero.

---

## Technologies Used

- **CUDA C/C++**
- **NVIDIA GPU**
- **OpenCV** (image loading, RGB decomposition/reassembly)
- **SLURM** (job scheduling on HPC clusters)
- **Bash scripting**

---

## Project Structure

es2/
├── filter.cu # CUDA implementation of the stencil filter
├── run.sh # Script to compile and run the program
├── job.sbatch # SLURM job submission script
├── images/ # Input images (not included)
├── output/ # Filtered images (generated)
└── README.md


---

## Compilation

Make sure CUDA and OpenCV are available on your system.
Compilation command is already included in run.s.

## Execution

./Run.sh <input_images>

For SLURM execution: 

sbatch job.sbatch <input_images>
