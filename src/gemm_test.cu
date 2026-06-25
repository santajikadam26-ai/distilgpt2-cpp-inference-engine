#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Naive GPU matrix multiply: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
// Each thread computes ONE element of C
__global__ void matmul(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU version for comparison
void matmul_cpu(float* A, float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    // Sizes similar to what DistilGPT-2's MLP layer uses
    int M = 1024;  // sequence length
    int K = 768;   // hidden dim
    int N = 768;   // output dim

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate and fill host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_cpu = new float[M * N];
    float* h_C_gpu = new float[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(i % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(i % 100) / 100.0f;

    // --- CPU timing ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // --- GPU setup ---
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((N + 15) / 16, (M + 15) / 16);

    // --- GPU timing ---
    cudaDeviceSynchronize(); // make sure GPU is idle before timing
    auto gpu_start = std::chrono::high_resolution_clock::now();
    matmul<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize(); // wait for GPU to finish
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // --- Verify correctness ---
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(h_C_cpu[i] - h_C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Matrix size: [" << M << "x" << K << "] * [" << K << "x" << N << "]" << std::endl;
    std::cout << "CPU time: " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_ms / gpu_ms) << "x" << std::endl;
    std::cout << "Max difference (correctness check): " << max_diff << std::endl;

    delete[] h_A; delete[] h_B; delete[] h_C_cpu; delete[] h_C_gpu;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
