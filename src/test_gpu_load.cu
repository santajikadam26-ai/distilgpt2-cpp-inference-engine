#include <iostream>
#include "../include/weights.h"
#include "../include/gpu_weights.h"

int main() {
    std::cout << "Loading model on CPU first..." << std::endl;
    ModelWeights cpu_model = load_model("weights");

    std::cout << "\nUploading to GPU..." << std::endl;
    GPUModelWeights gpu_model = upload_model_to_gpu(cpu_model);

    std::cout << "\nSuccess! Model is now on GPU." << std::endl;
    return 0;
}
