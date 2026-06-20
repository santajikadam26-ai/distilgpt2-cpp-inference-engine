#pragma once

// DistilGPT-2 model architecture constants
namespace Config {
    constexpr int N_LAYER = 6;        // number of transformer layers
    constexpr int N_HEAD = 12;        // number of attention heads
    constexpr int N_EMBD = 768;       // hidden dimension
    constexpr int HEAD_DIM = N_EMBD / N_HEAD;  // 64
    constexpr int VOCAB_SIZE = 50257; // vocabulary size
    constexpr int N_CTX = 1024;       // max context/sequence length
    constexpr int MLP_HIDDEN = 4 * N_EMBD; // 3072, FFN hidden dim
}
