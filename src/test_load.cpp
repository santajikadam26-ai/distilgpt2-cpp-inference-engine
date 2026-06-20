#include <iostream>
#include "../include/weights.h"

int main() {
    std::cout << "Loading DistilGPT-2 weights..." << std::endl;
    ModelWeights model = load_model("weights");

    // Print a few values to sanity-check the load
    std::cout << "\n--- Sanity check ---" << std::endl;
    std::cout << "wte[0] = " << model.wte[0] << std::endl;
    std::cout << "wte size = " << model.wte.size() << " (expected " << (size_t)VOCAB_SIZE * N_EMBD << ")" << std::endl;
    std::cout << "wpe size = " << model.wpe.size() << " (expected " << (size_t)N_CTX * N_EMBD << ")" << std::endl;
    std::cout << "Number of layers loaded = " << model.layers.size() << std::endl;
    std::cout << "lm_head size = " << model.lm_head_w.size() << std::endl;

    std::cout << "\nAll good! Model loaded successfully." << std::endl;
    return 0;
}
