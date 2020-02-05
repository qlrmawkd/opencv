import cv2
import numpy as np

img0 = cv2.imread('img/normal.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('img/defect1.jpg', cv2.IMREAD_COLOR)
img0 = cv2.resize(img0, dsize=(640, 480), interpolation=cv2.INTER_AREA)
img1 = cv2.resize(img1, dsize=(640, 480), interpolation=cv2.INTER_AREA)
print(img0.shape)
print(img1.shape)
imgs_list = [img0, img1]
'''
img1 = cv2.resize(img1, None, fx = 0.75, fy = 0.75)
img1 = np.pad (img1, ((64,)*2, (64,)*2, (0,)*2), 'constant', constant_values=0)

print(img0.shape)
print(img1.shape)
'''
#SIFT 검출자
detector = cv2.xfeatures2d.SIFT_create (50)
print(len(imgs_list))
i = 0
for i in range(len(imgs_list)):
  keypoints, descriptors = detector.detectAndCompute(imgs_list[i], None)
  imgs_list[i] = cv2.drawKeypoints(imgs_list[i], keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.imshow("SIFT keypoints", np.hstack(imgs_list))

if cv2.waitKey(0) == 27:
      cv2.destroyAllWindows()
