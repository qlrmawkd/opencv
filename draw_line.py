#직선 그리기
import cv2
#다각형을 그리려면  numpy 필요
import numpy as np

img = cv2.imread('img/helmet3.jpg')
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
img_gray = cv2.imread('img/hetmet3.jpg', cv2.IMREAD_GRAYSCALE)
img_gray = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_AREA)

#글자 작성
cv2.putText(img, "VISION PRACTICE", (320, 240), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0))
#line을 그리는 함수
cv2.line(img, (50, 50), (150, 375), (255, 255, 0))
#사각형을 그림
cv2.rectangle(img, (50,50), (150, 150), (50,0,255))

cv2.namedWindow('lines', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('write', cv2.WINDOW_NORMAL)


#2개의 창을 띄울때는, 창을 먼저 생성해주지 않으면 작동하지 않음
cv2.imshow('lines', img)
cv2.imshow('write', img_gray)

#ASCII 코드 값을 반환함.
key = cv2.waitKey(0)
print(key)

cv2.destroyAllWindows()