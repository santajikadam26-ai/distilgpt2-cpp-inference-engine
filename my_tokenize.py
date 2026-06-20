from transformers import GPT2Tokenizer
import sys

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

text = sys.argv[1] if len(sys.argv) > 1 else "Hello world"
ids = tokenizer.encode(text)

print(f"Text: {text}")
print(f"Token IDs: {ids}")

# Write token IDs to a simple text file, one per line
with open("input_ids.txt", "w") as f:
    f.write(" ".join(map(str, ids)))

print(f"Saved {len(ids)} token IDs to input_ids.txt")
