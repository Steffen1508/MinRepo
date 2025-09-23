import os, shutil

# Sørg for at mappen findes
os.makedirs("sounds", exist_ok=True)

src = "1.mp3"

for i in range(1, 61):
    dst = f"sounds/{i:02d}.mp3"
    shutil.copy(src, dst)

print("Færdig! 60 kopier af 1.mp3 oprettet i 'sounds' mappen.")
