import cv2 as cv
import numpy as np
import os

# Main function containing the backbone of the program
def main():
    image_path = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\dataset"
    if not os.path.isfile(image_path):
        print("Image not found")
        return
    image = cv.imread(image_path)
    tiles = get_tiles(image)
