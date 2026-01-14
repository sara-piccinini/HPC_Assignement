# Exercise 2 – CUDA Image Filtering with Stencil Operations

This directory contains the implementation of **Exercise 2** of the *High Performance Computing Assignment 2024–2025*.  
The exercise focuses on **GPU-accelerated multi-dimensional data processing** using **CUDA** and stencil-based image filtering.

---

## Objective

The objective of this exercise is to:

- Apply a 2D weighted stencil filter to high-resolution RGB images
- Exploit GPU parallelism using CUDA
- Evaluate execution time and GPU resource usage
- Analyze the impact of CUDA thread configuration on performance
- Validate filter effectiveness on noisy images

The implementation targets **4K, 8K, and 16K** images.

---

## Filtering Model

Each pixel is processed using a **3×3 weighted average stencil**, applied independently to the R, G, and B channels.

The processing pipeline is:

1. Load the input image  
2. Split the image into RGB channels  
3. Apply the stencil filter on the GPU  
4. Reassemble the filtered channels  
5. Optionally save the output image  
6. Record execution time and profiling data  

Border pixels are padded with zero values.

---

## Directory Structure

**Es2.cu**  
  CUDA implementation of the stencil-based image filter**Es2.sh**  
  Script for compilation, execution, and GPU profiling

**AddS&P.py**  
  Python script to add salt-and-pepper noise to images

**README.md**  
  Documentation for Exercise 2

---

## Requirements

- NVIDIA GPU with CUDA support  
- CUDA Toolkit  
- Python 3  
- Image processing library (e.g., OpenCV)  
- NVIDIA profiling tools (e.g., Nsight Systems)

---

## Noise Generation

### AddS&P.py

This script adds **salt-and-pepper noise** to reference images.

### Usage

1. Place clean images in:

../input/reference


2. The script generates noisy images in:

../input/noise50/
../input/noise75/
../input/noise90/


These correspond to **50%**, **75%**, and **90%** noise levels.

---

## CUDA Program Execution

### Es2.cu

The CUDA program processes a single image using the stencil filter.

### Execution Syntax

./Es2 <input_image> <output_image> [-s]

<input_image>: path to the input image
<output_image>: path or name of the output image
-s (optional): save the filtered image to disk

### Output

Execution time data are appended to ../logData/logFile

Profiling data are saved in: ../logData/nsysProfile/

