from pathlib import Path

# baca semua file txt
files = sorted(Path("data").glob("*.txt"))
if not files:
    raise FileNotFoundError("Tidak ada file .txt di folder data")

text = ""
for file_path in files:
    text += file_path.read_text(encoding="utf-8") + "\n"

# karakter unik
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Jumlah file:", len(files))
print("Jumlah karakter unik:", vocab_size)
print("Daftar karakter:")
print(chars)

# mapping karakter -> index
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

sample = "User: halo\nAssistant:"
encoded = encode(sample)
decoded = decode(encoded)

print("\nSample text:", repr(sample))
print("Encoded:", encoded)
print("Decoded:", repr(decoded))