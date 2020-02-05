import cv2
import numpy as np

img1 = cv2.imread('img/normal.jpg',0)
img2 = cv2.imread('img/part1.jpg',0)

ret, thresh = cv2.threshold(img1, 127, 255,0)
ret, thresh2 = cv2.threshold(img2, 127, 255,0)
contours,hierarchy ,_ = cv2.findContours(thresh,2,1)
cnt1 = contours[0]
contours,hierarchy ,_ = cv2.findContours(thresh2,2,1)
cnt2 = contours[0]

ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
print(ret)
'''
img1 = cv2.resize(img1, dsize=(960, 480), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, dsize=(960, 480), interpolation=cv2.INTER_AREA)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllwindows(0)
'''