'''
>>>제한사항
#1 사진 입력 -> 1사진 출력 : 다수가 가능하게 자동화 해야함
#2 출력된 사진들의 크기가 동일 하지 않음. -> 이 파일에선 건드리지 않음. 원본을 유지하는 것이 중요하다고 판단
#  사진을 받아쓰는 코드에서 처리해야함
#3 입력된 사진의 크기가 달라질 경우 threshold가 제대로 작동하는지 확인해야함.
'''


import cv2
import numpy as np
import math

#최적화 사각형 잘라주는 함수
def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]
    return img_crop

img_color = cv2.imread('img/defect2.jpg')
img_crop = cv2.imread('img/defect2.jpg')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#width와 height를 변경할 때마다, 아래의 쓰레스홀드 값을 변경해주어야함.
 

height, width = img_gray.shape[::-1]


ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
_, contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#Returns: image, contours, hierachy
#print(type(contours))
# type of contours:  list

#최적화 사각형 표시
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    
    #외접하며 면적이 가장 작은 사각형을 출력함. 좌상단의 좌표값과, 박스의 크기, 기울어진 각도를 반환.
    box = cv2.boxPoints(rect) #rect의 꼭지점 좌표 4개를 얻음.float형
    box = np.int0(box) #float를 int형으로 변환.
    area = cv2.contourArea(cnt)
    
    max_area = int(width*height)
    half_area = int(max_area/4)
    if ( half_area< area < max_area/2): # 최대 넓이의 0.25에서 0.5까지에 해당하는 사각형만 포착
        #print(area)
        print(rect)
        rect_roi = rect
        cv2.drawContours(img_color,[box],0,(0,0,255),2) #red

#선별해놓은 곳을 자름
img_crop = crop_minAreaRect(img_crop, rect_roi)

#화면 뛰워서 보여줌.
img_color = cv2.resize(img_color, dsize=(960, 560), interpolation=cv2.INTER_AREA)
cv2.imshow("result", img_color)
cv2.imshow("crop", img_crop)
cv2.imwrite('img/crop/normal1.jpg', img_crop)
cv2.waitKey(0)