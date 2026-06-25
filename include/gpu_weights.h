#pragma once
#include <cuda_runtime.h>
#include "config.h"
#include "weights.h"

using namespace Config;

// GPU versions of the weight structures — same data, but living in GPU memory
struct GPULayerWeights {
    float *ln_1_w, *ln_1_b;
    float *attn_c_attn_w, *attn_c_attn_b;
    float *attn_c_proj_w, *attn_c_proj_b;
    float *ln_2_w, *ln_2_b;
    float *mlp_c_fc_w, *mlp_c_fc_b;
    float *mlp_c_proj_w, *mlp_c_proj_b;
};

struct GPUModelWeights {
    float* wte;
    float* wpe;
    GPULayerWeights layers[N_LAYER];
    float *ln_f_w, *ln_f_b;
    float* lm_head_w;
};

// Helper: allocate GPU memory and copy data from a host vector
inline float* to_gpu(const std::vector<float>& host_data) {
    float* gpu_ptr;
    size_t bytes = host_data.size() * sizeof(float);
    cudaMalloc(&gpu_ptr, bytes);
    cudaMemcpy(gpu_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice);
    return gpu_ptr;
}

// Copies the full model from CPU (ModelWeights) to GPU (GPUModelWeights)
inline GPUModelWeights upload_model_to_gpu(const ModelWeights& cpu_model) {
    GPUModelWeights gpu_model;

    gpu_model.wte = to_gpu(cpu_model.wte);
    gpu_model.wpe = to_gpu(cpu_model.wpe);

    for (int i = 0; i < N_LAYER; i++) {
        const auto& L = cpu_model.layers[i];
        auto& GL = gpu_model.layers[i];

        GL.ln_1_w = to_gpu(L.ln_1_w);
        GL.ln_1_b = to_gpu(L.ln_1_b);
        GL.attn_c_attn_w = to_gpu(L.attn_c_attn_w);
        GL.attn_c_attn_b = to_gpu(L.attn_c_attn_b);
        GL.attn_c_proj_w = to_gpu(L.attn_c_proj_w);
        GL.attn_c_proj_b = to_gpu(L.attn_c_proj_b);
        GL.ln_2_w = to_gpu(L.ln_2_w);
        GL.ln_2_b = to_gpu(L.ln_2_b);
        GL.mlp_c_fc_w = to_gpu(L.mlp_c_fc_w);
        GL.mlp_c_fc_b = to_gpu(L.mlp_c_fc_b);
        GL.mlp_c_proj_w = to_gpu(L.mlp_c_proj_w);
        GL.mlp_c_proj_b = to_gpu(L.mlp_c_proj_b);

        std::cout << "Uploaded layer " << i << " to GPU" << std::endl;
    }

    gpu_model.ln_f_w = to_gpu(cpu_model.ln_f_w);
    gpu_model.ln_f_b = to_gpu(cpu_model.ln_f_b);
    gpu_model.lm_head_w = to_gpu(cpu_model.lm_head_w);

    std::cout << "Full model uploaded to GPU!" << std::endl;
    return gpu_model;
}
