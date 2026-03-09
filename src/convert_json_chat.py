from pathlib import Path
import json
from collections import defaultdict
import os

input_file = Path("data/chatbot_conversations.json")
output_file = Path("data/chat_dataset.txt")

conversations = defaultdict(list)

file_size = input_file.stat().st_size
processed_bytes = 0
last_percent = -1
line_count = 0

with input_file.open("r", encoding="utf-8") as f:
    for line in f:
        line_count += 1
        processed_bytes += len(line.encode("utf-8"))

        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"Baris {line_count} gagal dibaca, dilewati.")
            continue

        cid = item.get("conversation_id")
        turn = item.get("turn")
        role = item.get("role")
        msg = item.get("message", "")

        if cid is None or turn is None or role is None:
            continue

        conversations[cid].append((turn, role, msg))

        percent = int((processed_bytes / file_size) * 100)
        if percent != last_percent and percent % 1 == 0:
            print(f"Progress baca: {percent}% | baris: {line_count}", end="\r")
            last_percent = percent

print("\nSelesai membaca file. Menyusun hasil...")

with output_file.open("w", encoding="utf-8") as out:
    total_conv = len(conversations)
    done = 0
    last_percent = -1

    for cid, msgs in conversations.items():
        msgs = sorted(msgs, key=lambda x: x[0])

        for turn, role, msg in msgs:
            if role == "user":
                out.write(f"User: {msg}\n")
            elif role == "bot":
                out.write(f"Assistant: {msg}\n")

        out.write("\n")

        done += 1
        percent = int((done / total_conv) * 100)
        if percent != last_percent and percent % 1 == 0:
            print(f"Progress tulis: {percent}% | percakapan: {done}/{total_conv}", end="\r")
            last_percent = percent

print(f"\nSelesai! File hasil disimpan di: {output_file}")