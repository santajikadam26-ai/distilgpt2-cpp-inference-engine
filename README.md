# DistilGPT-2 Inference Engine in C++ & CUDA

A transformer-based language model inference engine built completely from scratch — first in C++ on CPU, then ported to CUDA for GPU acceleration. No PyTorch, no inference frameworks at runtime.

## What this is

This project implements the full forward pass of DistilGPT-2 (82M parameters) twice:
1. CPU version — pure C++, naive implementation
2. GPU version — CUDA kernels for every operation

Both versions implement:
- Custom binary weight loading
- LayerNorm, embeddings, and linear layers (GEMM)
- Multi-head self-attention with causal masking
- MLP block with GELU activation
- Top-k sampling for text generation

## Performance

Benchmarked on an NVIDIA GTX 1650 (4GB), 512-token sequence, full forward pass:

| Version | Time | Speedup |
|---|---|---|
| CPU (naive C++) | 18,332 ms | 1x (baseline) |
| GPU (naive CUDA) | 604.6 ms | ~30x |

Both versions produce identical predictions, confirming correctness.

A standalone GEMM (matrix multiply) benchmark showed an even larger gap:

| Version | Time |
|---|---|
| CPU | 926.9 ms |
| GPU | 7.1 ms |
| Speedup | 130x |

## Example output

Input: "Hello world"

Generated: "Hello world in the name of a new era of technology and innovation in India. An industry in which the development..."

## Architecture

- 6 transformer layers
- 12 attention heads
- 768 hidden dimension
- 50,257 vocabulary size

## How to run

Export weights from Hugging Face:

python3 export_weights.py

Tokenize input text:

python3 my_tokenize.py "Your prompt here"

CPU version:

g++ -std=c++17 -O2 -I include src/generate_loop.cpp -o build/generate_loop
./build/generate_loop 20

GPU version (requires CUDA toolkit):

nvcc -O2 -I include src/generate_gpu.cu -o build/generate_gpu
./build/generate_gpu

Decode output:

python3 my_decode.py

## What's next

- KV caching for efficient autoregressive decoding
- Shared memory tiling for the GEMM kernels
- FlashAttention-style fused attention
- Kernel fusion to reduce memory round-trips

## Tech stack

C++17, CUDA 13.3, raw binary weight format, no ML frameworks at inference time.
