import torch
from transformers import GPT2LMHeadModel
import os

print("Downloading DistilGPT-2...")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.eval()

os.makedirs("weights", exist_ok=True)

config = model.config
print(f"Layers: {config.n_layer}, Heads: {config.n_head}, Hidden: {config.n_embd}, Vocab: {config.vocab_size}")

state_dict = model.state_dict()

for name, tensor in state_dict.items():
    # Replace dots with underscores for filenames
    filename = name.replace(".", "_") + ".bin"
    path = os.path.join("weights", filename)
    tensor.detach().cpu().numpy().astype("float32").tofile(path)
    print(f"Saved {name} -> shape {list(tensor.shape)}")

print("Done! All weights exported to weights/ folder")
