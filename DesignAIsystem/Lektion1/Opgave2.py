import cv2

# Load the image and convert to grayscale
img = cv2.imread('00006.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# define each block as 4x4 cells of 64x64 pixels each
cell_size = (128, 128)      # h x w in pixels
block_size = (4, 4)         # h x w in cells
win_size = (8, 6)           # h x w in cells

nbins = 9  # number of orientation bins
img_size = img.shape[:2]  # h x w in pixels

# create a HOG object
hog = cv2.HOGDescriptor(
    _winSize=(win_size[1] * cell_size[1],
              win_size[0] * cell_size[0]),
    _blockSize=(block_size[1] * cell_size[1],
                block_size[0] * cell_size[0]),
    _blockStride=(cell_size[1], cell_size[0]),
    _cellSize=(cell_size[1], cell_size[0]),
    _nbins=nbins
)
n_cells = (img_size[0] // cell_size[0], img_size[1] // cell_size[1])

# find features as a 1xN vector, then reshape into spatial hierarchy
hog_feats = hog.compute(img)
hog_feats = hog_feats.reshape(
    n_cells[1] - win_size[1] + 1,
    n_cells[0] - win_size[0] + 1,
    win_size[1] - block_size[1] + 1,
    win_size[0] - block_size[0] + 1,
    block_size[1],
    block_size[0],
    nbins)
print(hog_feats.shape)