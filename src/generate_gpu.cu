#include <iostream>
#include <fstream>
#include <chrono>
#include "../include/weights.h"
#include "../include/gpu_weights.h"
#include "../include/gpu_layers.h"

float* gpu_alloc(size_t count) {
    float* ptr;
    cudaMalloc(&ptr, count * sizeof(float));
    return ptr;
}

std::vector<float> forward_gpu(const GPUModelWeights& model, const ModelWeights& cpu_model,
                                 const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();

    std::vector<float> h_embed(seq_len * N_EMBD);
    for (int t = 0; t < seq_len; t++) {
        int tok = token_ids[t];
        for (int i = 0; i < N_EMBD; i++) {
            h_embed[t * N_EMBD + i] = cpu_model.wte[tok * N_EMBD + i] + cpu_model.wpe[t * N_EMBD + i];
        }
    }

    float* x = gpu_alloc(seq_len * N_EMBD);
    cudaMemcpy(x, h_embed.data(), seq_len * N_EMBD * sizeof(float), cudaMemcpyHostToDevice);

    float* normed = gpu_alloc(seq_len * N_EMBD);
    float* qkv = gpu_alloc(seq_len * 3 * N_EMBD);
    float* attn_raw = gpu_alloc(seq_len * N_EMBD);
    float* attn_out = gpu_alloc(seq_len * N_EMBD);
    float* mlp_hidden = gpu_alloc(seq_len * MLP_HIDDEN);
    float* mlp_out = gpu_alloc(seq_len * N_EMBD);

    for (int l = 0; l < N_LAYER; l++) {
        const auto& L = model.layers[l];

        gpu_layer_norm(x, L.ln_1_w, L.ln_1_b, normed, seq_len);
        gpu_linear(normed, L.attn_c_attn_w, L.attn_c_attn_b, qkv, seq_len, N_EMBD, 3 * N_EMBD);
        gpu_attention(qkv, attn_raw, seq_len);
        gpu_linear(attn_raw, L.attn_c_proj_w, L.attn_c_proj_b, attn_out, seq_len, N_EMBD, N_EMBD);
        gpu_add(x, attn_out, seq_len * N_EMBD);

        gpu_layer_norm(x, L.ln_2_w, L.ln_2_b, normed, seq_len);
        gpu_linear(normed, L.mlp_c_fc_w, L.mlp_c_fc_b, mlp_hidden, seq_len, N_EMBD, MLP_HIDDEN);
        gpu_gelu(mlp_hidden, seq_len * MLP_HIDDEN);
        gpu_linear(mlp_hidden, L.mlp_c_proj_w, L.mlp_c_proj_b, mlp_out, seq_len, MLP_HIDDEN, N_EMBD);
        gpu_add(x, mlp_out, seq_len * N_EMBD);
    }

    gpu_layer_norm(x, model.ln_f_w, model.ln_f_b, normed, seq_len);

    std::vector<float> last_hidden(N_EMBD);
    cudaMemcpy(last_hidden.data(), &normed[(seq_len - 1) * N_EMBD], N_EMBD * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> logits(VOCAB_SIZE);
    for (int v = 0; v < VOCAB_SIZE; v++) {
        float sum = 0.0f;
        for (int i = 0; i < N_EMBD; i++) {
            sum += last_hidden[i] * cpu_model.lm_head_w[v * N_EMBD + i];
        }
        logits[v] = sum;
    }

    cudaFree(x); cudaFree(normed); cudaFree(qkv);
    cudaFree(attn_raw); cudaFree(attn_out);
    cudaFree(mlp_hidden); cudaFree(mlp_out);

    return logits;
}

int main(int argc, char** argv) {
    int seq_len_test = (argc > 1) ? std::atoi(argv[1]) : 0;

    std::cout << "Loading model on CPU..." << std::endl;
    ModelWeights cpu_model = load_model("weights");

    std::cout << "Uploading to GPU..." << std::endl;
    GPUModelWeights gpu_model = upload_model_to_gpu(cpu_model);

    std::vector<int> token_ids;
    if (seq_len_test > 0) {
        for (int i = 0; i < seq_len_test; i++) token_ids.push_back(i % 1000);
    } else {
        std::ifstream f("input_ids.txt");
        int id;
        while (f >> id) token_ids.push_back(id);
    }

    std::cout << "Running GPU forward pass on " << token_ids.size() << " tokens..." << std::endl;

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> logits = forward_gpu(gpu_model, cpu_model, token_ids);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    int best_id = 0;
    float best_val = logits[0];
    for (int v = 1; v < VOCAB_SIZE; v++) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best_id = v;
        }
    }

    std::cout << "Predicted next token ID: " << best_id << std::endl;
    std::cout << "GPU forward pass time: " << ms << " ms" << std::endl;

    return 0;
}
