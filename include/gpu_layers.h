#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "config.h"

using namespace Config;

// ============ LayerNorm Kernel ============
// Each CUDA block handles ONE token's normalization
__global__ void layer_norm_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int seq_len
) {
    int t = blockIdx.x; // one block per token
    if (t >= seq_len) return;

    const float* x = &input[t * N_EMBD];
    float* out = &output[t * N_EMBD];

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < N_EMBD; i++) mean += x[i];
    mean /= N_EMBD;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < N_EMBD; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= N_EMBD;

    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < N_EMBD; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

inline void gpu_layer_norm(const float* input, const float* weight, const float* bias,
                            float* output, int seq_len) {
    layer_norm_kernel<<<seq_len, 1>>>(input, weight, bias, output, seq_len);
}

// ============ Linear Layer Kernel (GEMM) ============
// output = input @ weight + bias
// input: [seq_len, in_dim], weight: [in_dim, out_dim], output: [seq_len, out_dim]
__global__ void linear_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int seq_len, int in_dim, int out_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // which token
    int col = blockIdx.x * blockDim.x + threadIdx.x; // which output feature

    if (row < seq_len && col < out_dim) {
        float sum = bias[col];
        for (int i = 0; i < in_dim; i++) {
            sum += input[row * in_dim + i] * weight[i * out_dim + col];
        }
        output[row * out_dim + col] = sum;
    }
}

inline void gpu_linear(const float* input, const float* weight, const float* bias,
                        float* output, int seq_len, int in_dim, int out_dim) {
    dim3 threads(16, 16);
    dim3 blocks((out_dim + 15) / 16, (seq_len + 15) / 16);
    linear_kernel<<<blocks, threads>>>(input, weight, bias, output, seq_len, in_dim, out_dim);
}

// ============ GELU Activation Kernel ============
__global__ void gelu_kernel(float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float inner = 0.7978845608f * (v + 0.044715f * v * v * v);
        x[idx] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

inline void gpu_gelu(float* x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(x, n);
}

// ============ Add (residual connection) Kernel ============
__global__ void add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

inline void gpu_add(float* a, const float* b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, n);
}

// ============ Attention Kernel ============
// One block per (head, token) pair. Computes causal self-attention for that token.
__global__ void attention_kernel(
    const float* qkv, float* output, int seq_len
) {
    int h = blockIdx.y;  // head index
    int t = blockIdx.x;  // token index
    if (t >= seq_len || h >= N_HEAD) return;

    const float* q = &qkv[t * 3 * N_EMBD + 0 * N_EMBD + h * HEAD_DIM];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Compute scores against all previous tokens (causal mask)
    float scores[1024]; // max context length
    float max_score = -1e30f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* k = &qkv[t2 * 3 * N_EMBD + 1 * N_EMBD + h * HEAD_DIM];
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) dot += q[d] * k[d];
        scores[t2] = dot * scale;
        if (scores[t2] > max_score) max_score = scores[t2];
    }

    // Softmax
    float sum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        scores[t2] = expf(scores[t2] - max_score);
        sum += scores[t2];
    }
    for (int t2 = 0; t2 <= t; t2++) scores[t2] /= sum;

    // Weighted sum of V
    float* out = &output[t * N_EMBD + h * HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) out[d] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* v = &qkv[t2 * 3 * N_EMBD + 2 * N_EMBD + h * HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) {
            out[d] += scores[t2] * v[d];
        }
    }
}

inline void gpu_attention(const float* qkv, float* output, int seq_len) {
    dim3 blocks(seq_len, N_HEAD);
    attention_kernel<<<blocks, 1>>>(qkv, output, seq_len);
}
