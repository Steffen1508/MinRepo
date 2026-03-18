import cv2 as cv
import numpy as np

# Load billedet
img = cv.imread(r'C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\krone.png')

# Lav alpha kanal — sort = transparent, alt andet = synligt
bgr = img[:, :, :3]

# Pixels der er "næsten sorte" bliver transparente
lower_black = np.array([0, 0, 0])
upper_black = np.array([50, 50, 50])  # juster 50 op/ned efter behov

mask = cv.inRange(bgr, lower_black, upper_black)  # sort → 255
alpha = cv.bitwise_not(mask)                       # inverter → sort → 0 (transparent)

# Sæt alpha kanal på billedet
bgra = cv.cvtColor(bgr, cv.COLOR_BGR2BGRA)
bgra[:, :, 3] = alpha

cv.imwrite(r'C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\krone_transparent.png', bgra)
print("Gemt!")


img = cv.imread(r'C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\krone_transparent.png', cv.IMREAD_UNCHANGED)

print(img.shape)  # skal sige (20, 29, 4)

# Tjek alpha kanalen
print("Min alpha:", img[:,:,3].min())   # skal være 0 (transparent baggrund)
print("Max alpha:", img[:,:,3].max())   # skal være 255 (synlig krone)