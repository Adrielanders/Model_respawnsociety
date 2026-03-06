from pathlib import Path
import torch

# baca teks
text = Path("data/input.txt").read_text(encoding="utf-8")

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

# encode seluruh teks
data = torch.tensor(encode(text), dtype=torch.long)

print("Shape data:", data.shape)
print("10 token pertama:", data[:10].tolist())
print("Hasil decode lagi:", decode(data[:50].tolist()))