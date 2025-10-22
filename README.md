# HPC-Distributed-DL-Benchmarking
HPC Course Project: Benchmarking and Optimizing Distributed Deep Learning Algorithms

This repository contains the source code and configuration files for my High Performance Computing course project, focusing on locality optimization (Tiling) and parallelism (OpenMP/SIMD) as a foundation for scalable distributed deep learning.

The entire project is containerized using Docker to guarantee a fully reproducible environment, including the specific versions of PyTorch, OpenMPI, and the GNU compiler tools.

Setup: Reproducible Environment

To execute the benchmarks, you only need Docker Desktop installed on your host system (e.g., macOS).

1. Clone the Repository (Replace PLACEHOLDER with your actual URL)

git clone [https://github.com/jasenmagee-afk/HPC-Distributed-DL-Benchmarking.git](https://github.com/jasenmagee-afk/HPC-Distributed-DL-Benchmarking.git)
cd HPC-Distributed-DL-Benchmarking


2. Build the Docker Image

This command reads the Dockerfile, installs all dependencies (Ubuntu 22.04, OpenMPI, PyTorch, C++ compiler), and packages the source code into a single, reusable image named hpc-project-env.

docker build -t hpc-project-env.


Execution: Benchmarking Experiments

All experiments are executed inside the Docker container using the docker run command.

1. Part 1: Cache Tiling (Locality Optimization)

This experiment demonstrates the effect of cache blocking on Matrix Multiplication (MatMul). This directly fulfills the core Part 1 requirement for locality-aware computing.

Execution Command: (Compiles and runs the benchmark in a single step)

docker run -it --rm hpc-project-env bash -c "g++ -std=c++17 -O3 01_Locality_Tiling/matmul_tiling_benchmark.cpp -o matmul_benchmark && ./matmul_benchmark"


Result: The output provides the time and throughput for the Baseline and Tiled versions, demonstrating a speedup from cache reuse.

2. Part 1 & 2 Prep: OpenMP (Shared-Memory Scaling)

This experiment measures the speedup achieved by explicitly using OpenMP threads on a custom C++ image filtering kernel, demonstrating multi-core parallelism.

Phase A: Compile the C++ Kernel
The custom kernel must be compiled into a Python module inside the image before running the benchmark.

docker run -it --rm hpc-project-env bash -c "python3 02_OpenMP_Scaling/setup.py install"


Phase B: Run Benchmarks (Varying Threads)
Execute the Python script multiple times, changing the OMP_NUM_THREADS variable to observe strong scaling.

# Baseline: 1 Thread (Shows C++/Python overhead)
docker run -e OMP_NUM_THREADS=1 -it --rm hpc-project-env python3 02_OpenMP_Scaling/openmp_benchmark.py

# Parallel: 4 Threads (Should show net speedup over the 1-thread time)
docker run -e OMP_NUM_THREADS=4 -it --rm hpc-project-env python3 02_OpenMP_Scaling/openmp_benchmark.py


3. Part 2 Prep: Data Locality (DL Pipeline Penalty)

This experiment simulates the performance penalty caused by non-contiguous memory access in a deep learning data loader (HWC vs. CHW layout).

Execution Command:

docker run -it --rm hpc-project-env python3 03_Data_Locality_DL/data_locality_benchmark.py


Result: Compares the time per batch for "Good Locality" vs. "Poor Locality" data access.
