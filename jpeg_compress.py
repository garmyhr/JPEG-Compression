import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import time
import sys
import collections as cl
from scipy import signal
from imageio import imread, imsave
from math import cos, floor, ceil, sqrt, pi, log2


# Method for converting an img to 8-bit values
def to_8bits_values(img):
    img_within_range = np.round(((img - np.min(img))*255)/(np.max(img) - np.min(img)))
    return img_within_range

# Method for calculating the entropi of an image
def calculate_entropi(img):

    freq = cl.Counter(img.ravel())
    G = len(freq)
    h = np.zeros(G)
    p = np.zeros(G)

    index = 0
    for val in freq.values():
        h[index] = val
        index += 1

    for i in range(G):
        p[i] = h[i] / len(img.ravel())

    entropi = 0
    for i in range(G):
        if p[i] > 0:
            entropi += p[i] * log2(1/p[i])

    return entropi

# Method for checking if two pictures are equal
# Pixel values that differ with 1 or less are considered equal
def img_equals(img1, img2):

    m, n = np.shape(img1)
    u, v = np.shape(img2)

    if m != u or n != v:
        return False

    for x in range(m):
        for y in range(n):

            diff = img1[x,y] - img2[x,y]

            if abs(diff) > 1:
                return False
    return True

# Compress image with name=filename using 2D - DCT compression
def jpeg_compression(filename, q, Q):

    block_size = 8
    img = imread(filename, as_gray=True)
    m, n = np.shape(img)
    img -= 128

    # Buffer to hold the compressed img

    comp_img = np.zeros((m,n))

    # Creating the quantification matrix
    Q = q * Q

    print(np.matrix(Q))

    # For each block
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):

            F = np.zeros((block_size, block_size))
            f = img[i : i + block_size, j : j + block_size]

            # For each pixel in block
            for u in range(block_size):
                for v in range(block_size):

                    cu = 1/sqrt(2) if u == 0 else 1
                    cv = 1/sqrt(2) if v == 0 else 1

                    # Loops for calculating sum of expression
                    sum = 0
                    for x in range(block_size):
                        for y in range(block_size):

                            cos_value = cos((2*x+1)*u*pi/16) * cos((2*y+1)*v*pi/16)
                            s = f[x,y] * cos_value
                            sum += s

                    # Transform and store the new pixel value in compressed image
                    F[u, v] = 1/4 * cu * cv * sum
                    F[u, v] = np.round(F[u, v] / Q[u,v])
                    comp_img[i+u, j+v] = F[u,v]

    b = 8
    c = calculate_entropi(comp_img)

    print("New entropi: %f" % c)
    print("Compression rate: %f" % (b/c))
    print("Percentage removed: %f" % (100*(1-c/b)))

    return comp_img


# Unpack image using inverse 2D - DCT
def jpeg_unpack(img, q, Q):

    block_size = 8
    m,n = np.shape(img)
    unpacked_img = np.zeros((m,n))

    Q = q * Q

    # For each block
    for i in range(0, m, block_size):
        for j in range(0, n, block_size):

            F = img[i : i + block_size, j : j + block_size] * Q
            f = np.zeros((block_size, block_size))

            # For each pixel in block
            for x in range(block_size):
                for y in range(block_size):

                    # Loops for calculating sum of expression
                    sum = 0
                    for u in range(block_size):
                        for v in range(block_size):

                            cu = 1/sqrt(2) if u == 0 else 1
                            cv = 1/sqrt(2) if v == 0 else 1
                            sum += cu * cv * F[u, v] * cos( (2*x+1) * u * pi / 16) * cos((2*y+1)*v*pi / 16)

                    # Transform and store the new pixel in unpacked image
                    f[x, y] = 1/4 * sum
                    unpacked_img[i+x, j+y] = f[x,y]

    unpacked_img += 128
    return unpacked_img



Q = np.array(
[[16, 11, 10, 16, 24, 40, 51, 61],
[12, 12, 14, 19, 26, 58, 60, 55],
[14, 13, 16, 24, 40, 57, 69, 56],
[14, 17, 22, 29, 51, 87, 80, 62],
[18, 22, 37, 56, 68, 109, 103, 77],
[24, 35, 55, 64, 81, 104, 113, 92],
[49, 64, 78, 87, 103, 121, 120, 101],
[72, 92, 95, 98, 112, 100, 103, 99]]
)


def main():

    q = float(sys.argv[2])
    img1 = imread(sys.argv[1], as_gray=True)
    compressed_img = jpeg_compression(sys.argv[1], q, Q)
    img2 = jpeg_unpack(compressed_img, q, Q)

    plt.subplot(1,2,1)
    plt.title("Image before")
    plt.imshow(img1, cmap="gray", vmin=0, vmax=255)

    plt.subplot(1,2,2)
    plt.title("Image after")
    plt.imshow(img2, cmap="gray", vmin=0, vmax=255)
    plt.show()

    print("Saving image")
    imsave('processed_img.jpeg', to_8bits_values(img2))


if __name__ == "__main__":
    main()