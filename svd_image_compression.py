import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image


def compress(matrix, k):  # compress an image with SVD

    w = len(matrix[0])
    h = len(matrix)

    u, s, vt = np.linalg.svd(matrix)

    sigma = np.zeros((h, w))
    for i in range(len(s)):
        sigma[i, i] = s[i]

    return u[:, :k] @ sigma[:k, :k] @ vt[:k, :]


# opening the image
img = plt.imread('image.png')
width = len(img[0])
height = len(img)

# getting rgb values
r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]

# compressing
rank = 50
r_c = compress(r, rank)
g_c = compress(g, rank)
b_c = compress(b, rank)

img_c = np.zeros((height, width, 3))
for i in range(0, height):
    for j in range(0, width):
        img_c[i, j, 0] = r_c[i, j]
        img_c[i, j, 1] = g_c[i, j]
        img_c[i, j, 2] = b_c[i, j]

# stacking RGB channels
img_c = np.dstack((r_c, g_c, b_c))
img_c = np.abs(img_c / np.max(img_c))

# saving image in file
matplotlib.image.imsave(f'image_rank_{rank}.png', img_c / np.max(img_c))

# plotting
plt.imshow(img)
plt.figure()
plt.imshow(img_c)
plt.show()