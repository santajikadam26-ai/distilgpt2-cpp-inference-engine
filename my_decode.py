from transformers import GPT2Tokenizer
import sys

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# Read space-separated token IDs from a file or command line
if len(sys.argv) > 1:
    ids = list(map(int, sys.argv[1].split()))
else:
    with open("output_ids.txt") as f:
        ids = list(map(int, f.read().split()))

text = tokenizer.decode(ids)
print(f"Decoded text: {text}")
