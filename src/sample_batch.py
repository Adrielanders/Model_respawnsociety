from pathlib import Path
import torch

files = sorted(Path("data").glob("*.txt"))

if not files:
    raise FileNotFoundError("Tidak ada file .txt di folder data")

text = ""
for file_path in files:
    text += file_path.read_text(encoding="utf-8") + "\n"

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

print("Jumlah file:", len(files))
print("File yang dibaca:", [f.name for f in files])

print("\nInput token:", x.tolist())
print("Target token:", y.tolist())

print("\nInput text :", repr(decode(x.tolist())))
print("Target text:", repr(decode(y.tolist())))