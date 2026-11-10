"""
Annoteér antal kroner per tile.

Går igennem alle billeder i annotations/crown/ og beder dig taste antal.
Gemmer løbende til crown_counts.json så du kan stoppe og fortsætte.

Taster:
  1 / 2 / 3  →  sæt antal kroner og gå videre
  Q          →  gem og afslut
"""

import json
import os

import cv2 as cv

# ── Config ─────────────────────────────────────────────────────────────────────
CROWN_DIR   = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\annotations\crown"
OUTPUT_FILE = r"C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\crown_counts.json"
DISPLAY_SIZE = 400
# ───────────────────────────────────────────────────────────────────────────────


def load_counts():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_counts(counts):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)


def run():
    counts = load_counts()

    all_files = sorted(f for f in os.listdir(CROWN_DIR) if f.endswith(".png"))
    remaining = [f for f in all_files if os.path.splitext(f)[0] not in counts]

    total = len(all_files)
    done  = total - len(remaining)

    print(f"Tiles med krone: {total}  |  Allerede annoteret: {done}  |  Tilbage: {len(remaining)}")
    print("Tast 1 / 2 / 3 for antal kroner  —  Q for at afslutte\n")

    window = "Antal kroner?  [1 / 2 / 3 | Q = afslut]"

    for i, filename in enumerate(remaining):
        tile_id = os.path.splitext(filename)[0]
        tile    = cv.imread(os.path.join(CROWN_DIR, filename))

        display = cv.resize(tile, (DISPLAY_SIZE, DISPLAY_SIZE),
                            interpolation=cv.INTER_NEAREST)

        progress = f"{done + i + 1}/{total}  —  {tile_id}"
        cv.putText(display, progress, (10, 24),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow(window, display)

        while True:
            key = cv.waitKey(0) & 0xFF
            if key in (ord('1'), ord('2'), ord('3')):
                n = int(chr(key))
                counts[tile_id] = n
                save_counts(counts)
                print(f"  {tile_id}  →  {n} krone{'r' if n > 1 else ''}")
                break
            elif key in (ord('q'), ord('Q'), 27):  # Q eller Esc
                save_counts(counts)
                print(f"\nGemt og afsluttet. Annoteret: {len(counts)}/{total}")
                cv.destroyAllWindows()
                return

    cv.destroyAllWindows()
    save_counts(counts)
    print(f"\nFærdig! Alle {total} tiles annoteret.")
    print(f"Fordeling: { {k: list(counts.values()).count(k) for k in [1,2,3]} }")


if __name__ == "__main__":
    run()
