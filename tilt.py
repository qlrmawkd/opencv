import cv2

src = cv2.imread("img/helmet3.jpg", cv2.IMREAD_COLOR)
src = cv2.resize(src, dsize=(980, 640), interpolation = cv2.INTER_AREA)

height, width, channel = src.shape
matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
#cv2.getRotationMatrix2D((중심점 X좌표, 중심점 Y좌표), 각도, 스케일)
dst = cv2.warpAffine(src, matrix, (width, height))
#cv2.warpAffine(원본 이미지, 배열, (결과 이미지 너비, 결과 이미지 높이))
#sd
cv2.imshow("src", src)
cv2.imshow("dst", dst)
if cv2.waitKey(0) ==27:
    cv2.destroyAllWindows()
    
#해야할 일
#1 중심점의 좌표는 detection된 toner의 중심점
#2 edge의 수평선의 각도를 구해서 각도에 대입
#3 각도로 변환을 먼저 한번 한뒤, 스케일을 대입해야함. 