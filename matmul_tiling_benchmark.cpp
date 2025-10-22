#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

// --- Configuration ---
// Define a matrix dimension N (must be large enough to show cache effects, but fit in memory)
// NOTE: N must be divisible by BLOCK_SIZE for this simple implementation.
const int N = 1024; 
// Define the BLOCK_SIZE (B) constant for the tiled kernel. 
// A typical value for L1/L2 cache testing is 32 or 64.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

using namespace std::chrono;

// Type definition for matrix storage (vector of vectors)
using Matrix = std::vector<std::vector<float>>;

// --- Initialization ---
void initialize_matrix(Matrix& M, float val) {
    M.assign(N, std::vector<float>(N, val));
}

// --- 1. Baseline Matrix Multiplication (ijk loop order) ---
// This is cache-inefficient because the inner loop accesses B[k][j] which jumps 
// across memory (stride N) for every iteration, resulting in poor locality.
void baseline_matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j]; // Poor locality on B
            }
            C[i][j] = sum;
        }
    }
}

// --- 2. Tiled Matrix Multiplication (Optimized for Cache Locality) ---
// Uses six nested loops to break the matrices into blocks (tiles) of size B x B.
// This ensures that the inner product calculation (C[i][j] += A[i][k] * B[k][j]) 
// only loads small blocks into the cache, maximizing data reuse and minimizing L1/L2 misses.
void tiled_matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    // ii, jj, kk are loop variables for blocks (outer loops)
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                // i, j, k are loop variables for elements within blocks (inner loops)
                for (int i = ii; i < ii + BLOCK_SIZE; ++i) {
                    for (int j = jj; j < jj + BLOCK_SIZE; ++j) {
                        float sum = C[i][j]; // Use C as accumulator (requires C to be initialized to 0)
                        for (int k = kk; k < kk + BLOCK_SIZE; ++k) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

int main() {
    // --- Setup ---
    // Total floating point operations (FLOPs) for the matrix multiplication N^3 (multiplies) * 2 (ops)
    const double GFLOPs = 2.0 * N * N * N / 1e9;
    
    Matrix A, B, C_base, C_tiled;
    initialize_matrix(A, 2.0f);
    initialize_matrix(B, 3.0f);
    initialize_matrix(C_base, 0.0f);
    initialize_matrix(C_tiled, 0.0f);

    std::cout << "--- Matrix Multiplication Tiling Benchmark ---" << std::endl;
    std::cout << "Matrix Size (N): " << N << "x" << N << std::endl;
    std::cout << "Block Size (B):  " << BLOCK_SIZE << "x" << BLOCK_SIZE << std::endl;
    std::cout << "Total Operations: " << std::fixed << std::setprecision(2) << GFLOPs << " GFLOPs" << std::endl;
    std::cout << std::endl;

    // --- Benchmark Baseline ---
    auto start_base = high_resolution_clock::now();
    baseline_matmul(A, B, C_base);
    auto end_base = high_resolution_clock::now();
    double time_base = duration_cast<nanoseconds>(end_base - start_base).count() / 1e9;
    double throughput_base = GFLOPs / time_base;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "1. Baseline (i-j-k):" << std::endl;
    std::cout << "   Time:      " << time_base << " seconds" << std::endl;
    std::cout << "   Throughput: " << throughput_base << " GFLOPs/s" << std::endl;
    
    std::cout << std::endl;

    // --- Benchmark Tiled ---
    auto start_tiled = high_resolution_clock::now();
    tiled_matmul(A, B, C_tiled);
    auto end_tiled = high_resolution_clock::now();
    double time_tiled = duration_cast<nanoseconds>(end_tiled - start_tiled).count() / 1e9;
    double throughput_tiled = GFLOPs / time_tiled;

    std::cout << "2. Optimized (Blocked):" << std::endl;
    std::cout << "   Time:      " << time_tiled << " seconds" << std::endl;
    std::cout << "   Throughput: " << throughput_tiled << " GFLOPs/s" << std::endl;
    
    std::cout << std::endl;

    // --- Results Summary ---
    double speedup = time_base / time_tiled;
    std::cout << "--- Summary ---" << std::endl;
    std::cout << "SPEEDUP (Baseline / Tiled): " << std::setprecision(2) << speedup << "x" << std::endl;
    std::cout << "The increase in speed demonstrates the effect of cache blocking." << std::endl;
    
    return 0;
}
