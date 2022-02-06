from math import sqrt
from helper import dims, init2DArray, storefile, loadfile
import numpy as np

SOBELX = [
  [-1, 0, 1],
  [-2, 0, 2],
  [-1, 0, 1]
]

SOBELY = [
  [-1, -2, -1],
  [0, 0, 0],
  [1, 2, 1]
]

def sobel(img):
  (w, h) = dims(img)
  out = init2DArray(w, h, 0)
  for i in range(0, w - 2):
    for j in range(0, h - 2):
      x = 0
      y = 0
      for k in range(0, 3):
        for l in range(0, 3):
          x += SOBELX[k][l] * img[i+k][j+l]
          y += SOBELY[k][l] * img[i+k][j+l]
      out[i + 1][j + 1] = sqrt(x ** 2 + y ** 2)
  return np.array(out)

if __name__ == "__main__":
  # start from root of the repository
  file = "./test_data/small.jpg"
  img = loadfile(file)
  out = sobel(img)
  storefile("out.png", out)