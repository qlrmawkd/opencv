import cv2
import numpy as np

img_rgb = cv2.imread('img/defect1.jpg')
#img_rgb = cv2.resize(img_rgb, dsize=(960, 560), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('img/part1.jpg', 0)
#template = cv2.resize(template, dsize=(960, 560), interpolation=cv2.INTER_AREA)
w, h = template.shape[::-1]
print(w, h)
#shape: y축, x축, channel을 뱉음
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#cv2.TM_CCOEFF_NORMED : 템플릿 매칭의 6가지 방법 중 하나 상관계수 방법에서 정규화 계수를 나눈 것.

threshold = 0.8
loc = np.where(res >= threshold)
#np.where(조건, ) 조건이 참인 구간을 뱉음.
for pt in zip(*loc[::-1]): #zip(*[::-1]) 동일한 자료형 여러개를 묶음. *은 반복해서 실행함을 의미함.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
# 구역에 빨간색 테두리를 두름. cv2.rectangle(img, start, end, color, thickness)

img_rgb = cv2.resize(img_rgb, dsize=(960, 560), interpolation=cv2.INTER_AREA)
cv2.imshow('result', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ret = cv2.matchShapes(cnt1,cnt2,1,0.0)
#print ret
#matchShapes 함수는 cnt1과 cnt2의 유사도를 각도, 크기와 무관하게 확인함.