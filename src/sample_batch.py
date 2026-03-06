from pathlib import Path
import torch

text = Path("data/input.txt").read_text(encoding="utf-8")

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

block_size = 8

x = data[:block_size]
y = data[1:block_size + 1]

print("Input token:", x.tolist())
print("Target token:", y.tolist())

print("\nInput text :", repr(decode(x.tolist())))
print("Target text:", repr(decode(y.tolist())))