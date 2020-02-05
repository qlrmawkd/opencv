import cv2
import numpy as np
# 소스코드 링크: https://pysource.com/2018/02/28/edge-detection-opencv-3-4-with-python-3-tutorial-18/
'''
#video_processing
cap = cv2.VideoCapture(0) #카메라 포트 번호 따라서 설정 가능
while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0) #가우시안블러
    laplacian = cv2.Laplacian(blurred_frame, cv2.CV_64F) #라플라시안_블러처리된 이미지 받아서 사용
    canny = cv2.Canny(blurred_frame, 100, 150) #캐니_블러처리된 이미지 받아서 사용
    laplacian_basic = cv2.Laplacian(frame, cv2.CV_64F) #라플라시안
    canny_basic = cv2.Canny(frame, 100, 150) #캐니
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Laplacian", laplacian)
    cv2.imshow("Canny", canny)
    cv2.imshow("Laplacian_basic", laplacian_basic)
    cv2.imshow("Canny_basic", canny_basic)
    key = cv2.waitKey(1)
    if key == 27:  #아스키27번 esc입력받으면 종료됨.
        break
cap.release()
cv2.destroyAllWindows()
'''
#image_processing #이미지 라플라시안 처리함

img = cv2.imread("img/coway19.jpg")
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

edge = cv2.Laplacian(img, -1)

merged = np.hstack((img, edge))
cv2.imshow('Laplacian_image', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
