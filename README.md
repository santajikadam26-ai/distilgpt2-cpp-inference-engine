# DistilGPT-2 Inference Engine in C++

A transformer-based language model inference engine built completely from scratch in C++ — no PyTorch, no inference frameworks.

## What this is

This project implements the full forward pass of DistilGPT-2 (82M parameters) by hand in C++:
- Custom binary weight loading
- LayerNorm, embeddings, and linear layers
- Multi-head self-attention with causal masking
- MLP block with GELU activation
- Top-k sampling for text generation

## Example

Input: `"Hello world"`

Output: `"Hello world in the name of a new era of technology and innovation in India. An industry in which the development..."`

## Architecture

- 6 transformer layers
- 12 attention heads
- 768 hidden dimension
- 50,257 vocabulary size

## How to run

\`\`\`bash
# 1. Export weights from Hugging Face (requires Python + transformers)
python3 export_weights.py

# 2. Tokenize your input text
python3 my_tokenize.py "Your prompt here"

# 3. Compile the generator
g++ -std=c++17 -O2 src/generate_loop.cpp -o build/generate_loop

# 4. Generate text
./build/generate_loop 20

# 5. Decode the output
python3 my_decode.py
\`\`\`

## What's next

- CUDA/GPU acceleration for faster inference
- KV caching for efficient autoregressive decoding
- FlashAttention-style optimizations

## Tech stack

C++17, raw binary weight format, no ML frameworks at inference time.
