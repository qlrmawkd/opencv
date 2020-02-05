import cv2
import numpy as np
#from google.colab.patches import cv2.imshow

img = cv2.imread("img/defect1.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
img1 = cv2.imread("img/defect1.jpg", cv2.INTER_AREA)
img1 = cv2.resize(img1, dsize=(640, 480), interpolation=cv2.INTER_AREA)


#fast code start (5,7) (7,12) (9, 16)
fast = cv2.FastFeatureDetector_create(30, True, cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
kp = fast.detect(img)

show_img = np.copy(img)
for p in cv2.KeyPoint.convert(kp):
  cv2.circle(show_img, tuple(p), 2, (0, 255, 0), cv2.FILLED)
cv2.imshow("Fast1", show_img)


#second picture
fast = cv2.FastFeatureDetector_create(30, True, cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
kp = fast.detect(img1)


show_img = np.copy(img1)
for p in cv2.KeyPoint.convert(kp):
  cv2.circle(show_img, tuple(p), 2, (0, 255, 0), cv2.FILLED)

cv2.imshow("Fast2", show_img)

#change true value into false
fast.setNonmaxSuppression (False)
kp = fast.detect (img)

for p in cv2.KeyPoint.convert(kp):
  cv2.circle(show_img, tuple(p), 2, (0, 255, 0), cv2.FILLED)
cv2.imshow("Fast1_False", show_img)

#terminate the view
if cv2.waitKey(0) == 27:
      cv2.destroyAllWindows()
