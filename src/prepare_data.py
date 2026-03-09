from pathlib import Path
import torch

files = sorted(Path("data").glob("*.txt"))
text = ""

for f in files:
    text += f.read_text(encoding="utf-8") + "\n"

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

print("Jumlah file:", len(files))
print("Shape data:", data.shape)
print("Vocab size:", len(chars))
print("100 char pertama:", decode(data[:100].tolist()))