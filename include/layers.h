#pragma once
#include <vector>
#include <cmath>
#include "config.h"

using namespace Config;

// LayerNorm: normalizes each token's hidden vector independently
// input: [seq_len, N_EMBD], weight/bias: [N_EMBD]
// output: [seq_len, N_EMBD]
inline std::vector<float> layer_norm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int seq_len
) {
    std::vector<float> output(seq_len * N_EMBD);
    const float eps = 1e-5f;

    for (int t = 0; t < seq_len; t++) {
        const float* x = &input[t * N_EMBD];
        float* out = &output[t * N_EMBD];

        // Step 1: compute mean
        float mean = 0.0f;
        for (int i = 0; i < N_EMBD; i++) mean += x[i];
        mean /= N_EMBD;

        // Step 2: compute variance
        float var = 0.0f;
        for (int i = 0; i < N_EMBD; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= N_EMBD;

        // Step 3: normalize, scale, shift
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int i = 0; i < N_EMBD; i++) {
            out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
        }
    }

    return output;
}

// Embedding lookup: combines token embedding + position embedding
// token_ids: list of token IDs, length = seq_len
// wte: [VOCAB_SIZE, N_EMBD], wpe: [N_CTX, N_EMBD]
// output: [seq_len, N_EMBD]
inline std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& wte,
    const std::vector<float>& wpe
) {
    int seq_len = token_ids.size();
    std::vector<float> output(seq_len * N_EMBD);

    for (int t = 0; t < seq_len; t++) {
        int tok_id = token_ids[t];
        const float* tok_emb = &wte[tok_id * N_EMBD];
        const float* pos_emb = &wpe[t * N_EMBD];
        float* out = &output[t * N_EMBD];

        for (int i = 0; i < N_EMBD; i++) {
            out[i] = tok_emb[i] + pos_emb[i];
        }
    }

    return output;
}

// Linear layer: output = input @ weight + bias
// input: [seq_len, in_dim]
// weight: [in_dim, out_dim]  (GPT-2 convention)
// bias: [out_dim]
// output: [seq_len, out_dim]
inline std::vector<float> linear(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    int seq_len, int in_dim, int out_dim
) {
    std::vector<float> output(seq_len * out_dim);

    for (int t = 0; t < seq_len; t++) {
        const float* x = &input[t * in_dim];
        float* out = &output[t * out_dim];

        for (int o = 0; o < out_dim; o++) {
            float sum = bias[o];
            for (int i = 0; i < in_dim; i++) {
                // weight is [in_dim, out_dim], so element (i, o) is at i*out_dim + o
                sum += x[i] * weight[i * out_dim + o];
            }
            out[o] = sum;
        }
    }

    return output;
}

// GELU activation (GPT-2's tanh approximation)
// Applied element-wise, in-place
inline void gelu(std::vector<float>& x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    for (size_t i = 0; i < x.size(); i++) {
        float v = x[i];
        float inner = c * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// Softmax: converts raw scores into probabilities that sum to 1
// Applied in-place over a single row of length `len`
inline void softmax(float* x, int len) {
    // Step 1: find max (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < len; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Step 2: exponentiate and sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Step 3: normalize
    for (int i = 0; i < len; i++) {
        x[i] /= sum;
    }
}

// Multi-head self-attention with causal masking
// input: [seq_len, N_EMBD]
// Returns: [seq_len, N_EMBD]
inline std::vector<float> attention(
    const std::vector<float>& input,
    const std::vector<float>& c_attn_w, const std::vector<float>& c_attn_b,
    const std::vector<float>& c_proj_w, const std::vector<float>& c_proj_b,
    int seq_len
) {
    // Step 1: project input to Q, K, V combined (output dim = 3 * N_EMBD)
    std::vector<float> qkv = linear(input, c_attn_w, c_attn_b, seq_len, N_EMBD, 3 * N_EMBD);

    // qkv layout per token: [Q(768), K(768), V(768)]
    std::vector<float> attn_out(seq_len * N_EMBD, 0.0f);
    float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    // Process each head independently
    for (int h = 0; h < N_HEAD; h++) {
        for (int t = 0; t < seq_len; t++) {
            const float* q = &qkv[t * 3 * N_EMBD + 0 * N_EMBD + h * HEAD_DIM];

            // Compute attention scores against all previous tokens (causal mask: only t' <= t)
            std::vector<float> scores(t + 1);
            for (int t2 = 0; t2 <= t; t2++) {
                const float* k = &qkv[t2 * 3 * N_EMBD + 1 * N_EMBD + h * HEAD_DIM];
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) dot += q[d] * k[d];
                scores[t2] = dot * scale;
            }

            softmax(scores.data(), t + 1);

            // Weighted sum of V using the attention scores
            float* out = &attn_out[t * N_EMBD + h * HEAD_DIM];
            for (int t2 = 0; t2 <= t; t2++) {
                const float* v = &qkv[t2 * 3 * N_EMBD + 2 * N_EMBD + h * HEAD_DIM];
                for (int d = 0; d < HEAD_DIM; d++) {
                    out[d] += scores[t2] * v[d];
                }
            }
        }
    }

    // Final output projection
    return linear(attn_out, c_proj_w, c_proj_b, seq_len, N_EMBD, N_EMBD);
}

// MLP / Feed-forward block: linear -> GELU -> linear
inline std::vector<float> mlp(
    const std::vector<float>& input,
    const std::vector<float>& fc_w, const std::vector<float>& fc_b,
    const std::vector<float>& proj_w, const std::vector<float>& proj_b,
    int seq_len
) {
    std::vector<float> h = linear(input, fc_w, fc_b, seq_len, N_EMBD, MLP_HIDDEN);
    gelu(h);
    return linear(h, proj_w, proj_b, seq_len, MLP_HIDDEN, N_EMBD);
}

// Adds two vectors element-wise (used for residual connections): a += b
inline void add_inplace(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}
