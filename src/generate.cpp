#include <iostream>
#include <fstream>
#include <sstream>
#include "../include/weights.h"
#include "../include/layers.h"

// Runs the full GPT-2 forward pass and returns logits for the LAST token only
std::vector<float> forward(const ModelWeights& model, const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();

    // Embedding
    std::vector<float> x = embed_tokens(token_ids, model.wte, model.wpe);

    // Transformer blocks
    for (int l = 0; l < N_LAYER; l++) {
        const auto& L = model.layers[l];

        // --- Attention block ---
        std::vector<float> normed1 = layer_norm(x, L.ln_1_w, L.ln_1_b, seq_len);
        std::vector<float> attn_out = attention(normed1, L.attn_c_attn_w, L.attn_c_attn_b,
                                                  L.attn_c_proj_w, L.attn_c_proj_b, seq_len);
        add_inplace(x, attn_out); // residual connection

        // --- MLP block ---
        std::vector<float> normed2 = layer_norm(x, L.ln_2_w, L.ln_2_b, seq_len);
        std::vector<float> mlp_out = mlp(normed2, L.mlp_c_fc_w, L.mlp_c_fc_b,
                                           L.mlp_c_proj_w, L.mlp_c_proj_b, seq_len);
        add_inplace(x, mlp_out); // residual connection

        std::cout << "Layer " << l << " done" << std::endl;
    }

    // Final layer norm
    x = layer_norm(x, model.ln_f_w, model.ln_f_b, seq_len);

    // Project only the LAST token to vocabulary logits
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

int main() {
    std::cout << "Loading model..." << std::endl;
    ModelWeights model = load_model("weights");

    // Read token IDs from input_ids.txt
    std::ifstream f("input_ids.txt");
    std::vector<int> token_ids;
    int id;
    while (f >> id) token_ids.push_back(id);

    std::cout << "Input has " << token_ids.size() << " tokens" << std::endl;
    std::cout << "Running forward pass..." << std::endl;

    std::vector<float> logits = forward(model, token_ids);

    // Find the token with the highest logit (greedy decoding)
    int best_id = 0;
    float best_val = logits[0];
    for (int v = 1; v < VOCAB_SIZE; v++) {
        if (logits[v] > best_val) {
            best_val = logits[v];
            best_id = v;
        }
    }

    std::cout << "Predicted next token ID: " << best_id << std::endl;

    // Save it so Python can decode it
    std::ofstream out("output_ids.txt");
    out << best_id;

    return 0;
}
