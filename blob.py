# Standard imports
import cv2
import numpy as np;
 
# Read image
im1 = cv2.imread("img/defect1.jpg", cv2.IMREAD_GRAYSCALE)
im1 = cv2.resize(im1, dsize=(640, 480), interpolation=cv2.INTER_AREA)

im2 = cv2.imread("img/defect2.jpg", cv2.IMREAD_GRAYSCALE)
im2 = cv2.resize(im2, dsize=(640, 480), interpolation=cv2.INTER_AREA)
 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
 
# Detect blobs.
keypoint1 = detector.detect(im1)
keypoint2 = detector.detect(im2)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoint1 = cv2.drawKeypoints(im1, keypoint1, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
im_with_keypoint2 = cv2.drawKeypoints(im2, keypoint2, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Show keypoints
cv2.imshow("Keypoint1", im_with_keypoint1)
cv2.imshow("Keypoint2", im_with_keypoint2)
if cv2.waitKey()== 27:
    cv2.destroyAllWindows()