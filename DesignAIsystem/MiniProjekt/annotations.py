import cv2 as cv
import numpy as np
import os

# ── Konfiguration ──────────────────────────────────────────────────────────────
IMAGE_FOLDER  = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\dataset"
OUTPUT_FOLDER = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\annotations"

TILE_SIZE   = 100   # pixels per felt
GRID_SIZE   = 5     # 5x5 grid
DISPLAY_SIZE = 300  # zoom-størrelse til visning

CROWN_DIR   = os.path.join(OUTPUT_FOLDER, "crown")
NO_CROWN_DIR = os.path.join(OUTPUT_FOLDER, "no_crown")
SKIP_DIR    = os.path.join(OUTPUT_FOLDER, "skip")
# ──────────────────────────────────────────────────────────────────────────────


def setup_folders():
    for d in [CROWN_DIR, NO_CROWN_DIR, SKIP_DIR]:
        os.makedirs(d, exist_ok=True)


def load_images(folder):
    images = {}
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".jpg", ".png")):
            path = os.path.join(folder, filename)
            img = cv.imread(path)
            if img is not None:
                image_id = os.path.splitext(filename)[0]
                images[image_id] = img
    print(f"Loadede {len(images)} billeder")
    return images


def get_tiles(image):
    """Del et 500x500 billede op i 5x5 felter"""
    tiles = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            tile = image[y*TILE_SIZE:(y+1)*TILE_SIZE,
                         x*TILE_SIZE:(x+1)*TILE_SIZE]
            tiles.append((x, y, tile))
    return tiles


def annotate(images):
    setup_folders()

    # Tæl allerede annoterede filer så vi kan resume
    already_done = set()
    for d in [CROWN_DIR, NO_CROWN_DIR, SKIP_DIR]:
        for f in os.listdir(d):
            already_done.add(os.path.splitext(f)[0])

    total_crown    = len(os.listdir(CROWN_DIR))
    total_no_crown = len(os.listdir(NO_CROWN_DIR))

    print("\n=== Annotation-script ===")
    print("  K = krone  |  N = ikke-krone  |  S = skip  |  Q = afslut\n")

    for image_id, image in images.items():
        tiles = get_tiles(image)

        for x, y, tile in tiles:
            tile_id = f"{image_id}_x{x}_y{y}"

            # Spring over hvis allerede annoteret
            if tile_id in already_done:
                continue

            # Zoom op til visning
            display = cv.resize(tile, (DISPLAY_SIZE, DISPLAY_SIZE),
                                interpolation=cv.INTER_NEAREST)

            # Skriv koordinat på billedet
            label = f"Billede: {image_id}  Felt: ({x},{y})"
            cv.putText(display, label, (10, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            stats = f"Kroner: {total_crown}  Ikke-kroner: {total_no_crown}"
            cv.putText(display, stats, (10, DISPLAY_SIZE - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

            cv.imshow("Annotation  [K=krone | N=ingen | S=skip | Q=afslut]", display)

            while True:
                key = cv.waitKey(0) & 0xFF

                if key == ord('k'):
                    path = os.path.join(CROWN_DIR, f"{tile_id}.png")
                    cv.imwrite(path, tile)
                    total_crown += 1
                    print(f"  KRONE     → {tile_id}  (total kroner: {total_crown})")
                    break

                elif key == ord('n'):
                    path = os.path.join(NO_CROWN_DIR, f"{tile_id}.png")
                    cv.imwrite(path, tile)
                    total_no_crown += 1
                    print(f"  INGEN     → {tile_id}  (total ingen: {total_no_crown})")
                    break

                elif key == ord('s'):
                    path = os.path.join(SKIP_DIR, f"{tile_id}.png")
                    cv.imwrite(path, tile)
                    print(f"  SKIP      → {tile_id}")
                    break

                elif key == ord('q'):
                    print(f"\nAfsluttet. Kroner: {total_crown}, Ikke-kroner: {total_no_crown}")
                    cv.destroyAllWindows()
                    return

    cv.destroyAllWindows()
    print(f"\nFærdig! Kroner: {total_crown}, Ikke-kroner: {total_no_crown}")


if __name__ == "__main__":
    images = load_images(IMAGE_FOLDER)
    annotate(images)