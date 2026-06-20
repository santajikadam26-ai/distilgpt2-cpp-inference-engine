#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include "../include/weights.h"
#include "../include/layers.h"

// Runs forward pass on the FULL sequence, returns logits for the LAST token
std::vector<float> forward(const ModelWeights& model, const std::vector<int>& token_ids) {
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
    int num_new_tokens = (argc > 1) ? std::atoi(argv[1]) : 20;

    std::cout << "Loading model..." << std::endl;
    ModelWeights model = load_model("weights");

    std::ifstream f("input_ids.txt");
    std::vector<int> token_ids;
    int id;
    while (f >> id) token_ids.push_back(id);

    std::cout << "Starting with " << token_ids.size() << " tokens. Generating "
              << num_new_tokens << " more..." << std::endl;

    for (int step = 0; step < num_new_tokens; step++) {
        std::vector<float> logits = forward(model, token_ids);

        const int TOP_K = 40;
        std::vector<int> indices(VOCAB_SIZE);
        for (int v = 0; v < VOCAB_SIZE; v++) indices[v] = v;

        std::partial_sort(indices.begin(), indices.begin() + TOP_K, indices.end(),
            [&logits](int a, int b) { return logits[a] > logits[b]; });

        std::vector<float> top_logits(TOP_K);
        for (int i = 0; i < TOP_K; i++) top_logits[i] = logits[indices[i]];
        softmax(top_logits.data(), TOP_K);

        static std::mt19937 rng(42);
        std::discrete_distribution<int> dist(top_logits.begin(), top_logits.end());
        int chosen = dist(rng);
        int best_id = indices[chosen];

        token_ids.push_back(best_id);
        std::cout << "Step " << step << ": generated token " << best_id << std::endl;

        if (token_ids.size() >= (size_t)N_CTX) break;
    }

    std::ofstream out("output_ids.txt");
    for (size_t i = 0; i < token_ids.size(); i++) {
        out << token_ids[i];
        if (i + 1 < token_ids.size()) out << " ";
    }

    std::cout << "Done. Full sequence saved to output_ids.txt" << std::endl;
    return 0;
}
