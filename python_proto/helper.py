import cv2
import numpy as np

def loadfile(name):
  img = cv2.imread(name, 1)
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def storefile(name, img):
  cv2.imwrite(name, img)

def dims(img):
  w = len(img)
  h = len(img[0])
  return (w, h)

def init2DArray(w, h, init):
  arr = []
  for i in range(0, w):
    tmp = []
    for j in range(0, h):
      tmp.append(init)
    arr.append(tmp)
  return arr