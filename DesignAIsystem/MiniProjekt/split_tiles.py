"""
Split each 500x500 board image in the dataset into 25 individual 100x100 tile images.
Output files are named  {image_id}_x{x}_y{y}.png  and saved to the tiles/ folder.
"""

import cv2 as cv
import os

image_folder = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\dataset"
output_folder = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\tiles"

tile_size = 100
grid_size = 5


def split_all_images():
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.png'))
    )

    total_tiles = 0
    skipped = 0

    for filename in image_files:
        image_id = os.path.splitext(filename)[0]
        img = cv.imread(os.path.join(image_folder, filename))

        if img is None:
            print(f"  WARNING: could not load {filename}")
            continue

        h, w = img.shape[:2]
        if h < grid_size * tile_size or w < grid_size * tile_size:
            print(f"  WARNING: {filename} too small ({w}x{h}), skipping")
            skipped += 1
            continue

        for y in range(grid_size):
            for x in range(grid_size):
                tile = img[y * tile_size:(y + 1) * tile_size,
                           x * tile_size:(x + 1) * tile_size]
                tile_name = f"{image_id}_x{x}_y{y}.png"
                cv.imwrite(os.path.join(output_folder, tile_name), tile)
                total_tiles += 1

        print(f"  {filename}: 25 tiles saved")

    print(f"\nDone! {total_tiles} tiles saved to '{output_folder}'")
    if skipped:
        print(f"  ({skipped} images skipped)")


if __name__ == "__main__":
    split_all_images()
