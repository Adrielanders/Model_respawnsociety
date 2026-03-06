from pathlib import Path

# baca teks
text = Path("data/input.txt").read_text(encoding="utf-8")

# ambil semua karakter unik, lalu urutkan
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Jumlah karakter unik:", vocab_size)
print("Daftar karakter:")
print(chars)

# mapping karakter -> angka
stoi = {ch: i for i, ch in enumerate(chars)}

# mapping angka -> karakter
itos = {i: ch for i, ch in enumerate(chars)}

# fungsi encode: teks jadi angka
def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

sample = "Halo"
encoded = encode(sample)
decoded = decode(encoded)

print("\nSample text:", sample)
print("Encoded:", encoded)
print("Decoded:", decoded)