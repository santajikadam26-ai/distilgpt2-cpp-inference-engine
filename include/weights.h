#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "config.h"

using namespace Config;

// Holds all weights for a single transformer layer
struct LayerWeights {
    std::vector<float> ln_1_w, ln_1_b;
    std::vector<float> attn_c_attn_w, attn_c_attn_b;
    std::vector<float> attn_c_proj_w, attn_c_proj_b;
    std::vector<float> ln_2_w, ln_2_b;
    std::vector<float> mlp_c_fc_w, mlp_c_fc_b;
    std::vector<float> mlp_c_proj_w, mlp_c_proj_b;
};

// Holds the full model
struct ModelWeights {
    std::vector<float> wte;      // token embeddings [VOCAB_SIZE, N_EMBD]
    std::vector<float> wpe;      // position embeddings [N_CTX, N_EMBD]
    std::vector<LayerWeights> layers;
    std::vector<float> ln_f_w, ln_f_b;
    std::vector<float> lm_head_w; // [VOCAB_SIZE, N_EMBD]
};

// Reads a raw float32 binary file into a vector
inline std::vector<float> load_bin(const std::string& path, size_t expected_count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: could not open " << path << std::endl;
        std::exit(1);
    }
    std::vector<float> data(expected_count);
    file.read(reinterpret_cast<char*>(data.data()), expected_count * sizeof(float));
    if (!file) {
        std::cerr << "ERROR: failed to read full file " << path << std::endl;
        std::exit(1);
    }
    return data;
}

inline ModelWeights load_model(const std::string& weights_dir) {
    ModelWeights model;

    model.wte = load_bin(weights_dir + "/transformer_wte_weight.bin", (size_t)VOCAB_SIZE * N_EMBD);
    model.wpe = load_bin(weights_dir + "/transformer_wpe_weight.bin", (size_t)N_CTX * N_EMBD);

    model.layers.resize(N_LAYER);
    for (int i = 0; i < N_LAYER; i++) {
        std::string p = weights_dir + "/transformer_h_" + std::to_string(i) + "_";
        auto& L = model.layers[i];

        L.ln_1_w = load_bin(p + "ln_1_weight.bin", N_EMBD);
        L.ln_1_b = load_bin(p + "ln_1_bias.bin", N_EMBD);

        L.attn_c_attn_w = load_bin(p + "attn_c_attn_weight.bin", (size_t)N_EMBD * 3 * N_EMBD);
        L.attn_c_attn_b = load_bin(p + "attn_c_attn_bias.bin", 3 * N_EMBD);

        L.attn_c_proj_w = load_bin(p + "attn_c_proj_weight.bin", (size_t)N_EMBD * N_EMBD);
        L.attn_c_proj_b = load_bin(p + "attn_c_proj_bias.bin", N_EMBD);

        L.ln_2_w = load_bin(p + "ln_2_weight.bin", N_EMBD);
        L.ln_2_b = load_bin(p + "ln_2_bias.bin", N_EMBD);

        L.mlp_c_fc_w = load_bin(p + "mlp_c_fc_weight.bin", (size_t)N_EMBD * MLP_HIDDEN);
        L.mlp_c_fc_b = load_bin(p + "mlp_c_fc_bias.bin", MLP_HIDDEN);

        L.mlp_c_proj_w = load_bin(p + "mlp_c_proj_weight.bin", (size_t)MLP_HIDDEN * N_EMBD);
        L.mlp_c_proj_b = load_bin(p + "mlp_c_proj_bias.bin", N_EMBD);

        std::cout << "Loaded layer " << i << std::endl;
    }

    model.ln_f_w = load_bin(weights_dir + "/transformer_ln_f_weight.bin", N_EMBD);
    model.ln_f_b = load_bin(weights_dir + "/transformer_ln_f_bias.bin", N_EMBD);

    model.lm_head_w = load_bin(weights_dir + "/lm_head_weight.bin", (size_t)VOCAB_SIZE * N_EMBD);

    std::cout << "Model fully loaded!" << std::endl;
    return model;
}
