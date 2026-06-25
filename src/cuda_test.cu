#include <iostream>
#include <cuda_runtime.h>

// __global__ means this function runs on the GPU, called from CPU
__global__ void add_vectors(float* a, float* b, float* c, int n) {
    // Each GPU thread computes ONE element of the output
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 10;
    size_t size = n * sizeof(float);

    // Step 1: Allocate memory on CPU (host)
    float h_a[10], h_b[10], h_c[10];
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Step 2: Allocate memory on GPU (device)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Step 3: Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Step 4: Launch the kernel
    // <<<num_blocks, threads_per_block>>> — this launches n threads total
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    add_vectors<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // Step 5: Copy result back from GPU to CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Step 6: Print result
    std::cout << "Result: ";
    for (int i = 0; i < n; i++) std::cout << h_c[i] << " ";
    std::cout << std::endl;

    // Step 7: Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
