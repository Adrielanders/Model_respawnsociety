from pathlib import Path

data_dir = Path("data")
files = sorted(data_dir.glob("*.txt"))

if not files:
    print("Tidak ada file .txt di folder data")
else:
    all_text = ""

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        print(f"\n=== {file_path.name} ===")
        print("Panjang teks:", len(text))
        print("Contoh isi:\n")
        print(text[:300])
        print("-" * 40)

        all_text += text + "\n"

    print("\n=== GABUNGAN SEMUA FILE ===")
    print("Total panjang teks:", len(all_text))
    print("Contoh isi gabungan:\n")
    print(all_text[:500])