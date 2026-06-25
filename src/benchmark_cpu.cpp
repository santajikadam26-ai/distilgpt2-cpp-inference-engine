#include <iostream>
#include <chrono>
#include "../include/weights.h"
#include "../include/layers.h"

std::vector<float> forward_cpu(const ModelWeights& model, const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    std::vector<float> x = embed_tokens(token_ids, model.wte, model.wpe);

    for (int l = 0; l < N_LAYER; l++) {
        const auto& L = model.layers[l];
        std::vector<float> normed1 = layer_norm(x, L.ln_1_w, L.ln_1_b, seq_len);
        std::vector<float> attn_out = attention(normed1, L.attn_c_attn_w, L.attn_c_attn_b,
                                                  L.attn_c_proj_w, L.attn_c_proj_b, seq_len);
        add_inplace(x, attn_out);

        std::vector<float> normed2 = layer_norm(x, L.ln_2_w, L.ln_2_b, seq_len);
        std::vector<float> mlp_out = mlp(normed2, L.mlp_c_fc_w, L.mlp_c_fc_b,
                                           L.mlp_c_proj_w, L.mlp_c_proj_b, seq_len);
        add_inplace(x, mlp_out);
    }

    x = layer_norm(x, model.ln_f_w, model.ln_f_b, seq_len);

    const float* last_hidden = &x[(seq_len - 1) * N_EMBD];
    std::vector<float> logits(VOCAB_SIZE);
    for (int v = 0; v < VOCAB_SIZE; v++) {
        float sum = 0.0f;
        for (int i = 0; i < N_EMBD; i++) {
            sum += last_hidden[i] * model.lm_head_w[v * N_EMBD + i];
        }
        logits[v] = sum;
    }
    return logits;
}

int main(int argc, char** argv) {
    int seq_len_test = (argc > 1) ? std::atoi(argv[1]) : 2;

    std::cout << "Loading model..." << std::endl;
    ModelWeights model = load_model("weights");

    std::vector<int> token_ids;
    for (int i = 0; i < seq_len_test; i++) token_ids.push_back(i % 1000);

    std::cout << "Running CPU forward pass on " << token_ids.size() << " tokens..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> logits = forward_cpu(model, token_ids);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    int best_id = 0;
    float best_val = logits[0];
    for (int v = 1; v < VOCAB_SIZE; v++) {
        if (logits[v] > best_val) { best_val = logits[v]; best_id = v; }
    }

    std::cout << "Predicted next token ID: " << best_id << std::endl;
    std::cout << "CPU forward pass time: " << ms << " ms" << std::endl;

    return 0;
}
