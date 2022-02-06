import cv2
import numpy as np



def loadfile(name):
  return cv2.imread(name, 1)

def storefile(name, img):
  cv2.imwrite(name, img)

def sobel(img):
  return img

if __name__ == "__main__":
  # start from root of the repository
  file = "./test_data/small.jpg"
  img = loadfile(file)
  out = sobel(img)
  storefile("out.png", out)