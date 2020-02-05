import cv2
import numpy as np


#video
'''
cap = cv2.VideoCapture(0)
#카메라 프레임 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("새로운 폭: %d, height:%d" % (width, height))

edge = cv2.Laplacian(cap, -1)


#결과 출력
merged = numpy.hstack((cap, edge))
cv2.imshow('Laplacian', merged)
cv2.waitkey(0)
cv2.destroyAllWindows()
'''
#image
img = cv2.imread("img/helmet3.jpg")

edge = cv2.Laplacian(img, -1)

merged = np.hstack((img, edge))
cv2.imshow('Laplacian_image', merged)
cv2.waitkey(0)
cv2.destroyAllWindows()
