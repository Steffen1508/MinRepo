import cv2 as cv
import numpy as np

# 2. Rotate funktion
def rotate_template(template, angle):
    h, w = template.shape[:2]
    center = (w/2, h/2)
    M = cv.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv.warpAffine(template, M, (w, h))
    return rotated

# 3. Count crowns
def count_crowns(tile, template, threshold=0.6):
    template_bgr  = template[:, :, :3]
    template_mask = template[:, :, 3]

    all_points = []

    for angle in [0, 90, 180, 270]:
        rotated_bgr  = rotate_template(template_bgr, angle)
        rotated_mask = rotate_template(template_mask, angle)

        result    = cv.matchTemplate(tile, rotated_bgr,
                                     cv.TM_CCOEFF_NORMED,
                                     mask=rotated_mask)
        locations = np.where(result >= threshold)
        points    = list(zip(locations[1], locations[0]))  # (x, y)
        all_points.extend(points)

    # Deduplicer på tværs af alle rotationer
    unique = filter_matches(all_points, min_distance=20)
    return len(unique)

def filter_matches(points, min_distance=15):
    """Behold kun matches der er mindst min_distance fra hinanden"""
    points = sorted(points, key=lambda p: p[0])
    
    kept = []
    for point in points:
        # Tjek om dette punkt er for tæt på et allerede beholdt punkt
        too_close = False
        for kept_point in kept:
            dist = np.sqrt((point[0]-kept_point[0])**2 + (point[1]-kept_point[1])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append(point)
    
    return kept

def max_match_score(tile, template) -> float:
    """Returner højeste template match score på tværs af alle rotationer (ingen threshold)."""
    template_bgr  = template[:, :, :3]
    template_mask = template[:, :, 3]
    best = 0.0
    for angle in [0, 90, 180, 270]:
        rotated_bgr  = rotate_template(template_bgr, angle)
        rotated_mask = rotate_template(template_mask, angle)
        result = cv.matchTemplate(tile, rotated_bgr, cv.TM_CCOEFF_NORMED,
                                  mask=rotated_mask)
        best = max(best, float(result.max()))
    return best


if __name__ == "__main__":
    TEMPLATE_PATH = r'C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\krone_transparent.png'
    IMAGE_PATH    = r'C:\Users\Raind\OneDrive\Documents\GitHub\MinRepo\DesignAIsystem\MiniProjekt\dataset\9.jpg'

    template = cv.imread(TEMPLATE_PATH, cv.IMREAD_UNCHANGED)
    image    = cv.imread(IMAGE_PATH)

    print(template.shape)

    for y in range(5):
        for x in range(5):
            tile   = image[y*100:(y+1)*100, x*100:(x+1)*100]
            crowns = count_crowns(tile, template)
            if crowns > 0:
                print(f"Felt ({x},{y}): {crowns} kroner")