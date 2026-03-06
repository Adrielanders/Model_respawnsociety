from pathlib import Path

file_path = Path("data/input.txt")

text = file_path.read_text(encoding="utf-8")

print("Panjang teks:", len(text))
print("\nContoh isi teks:\n")
print(text[:300])