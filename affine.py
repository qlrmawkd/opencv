import cv2
import numpy as np 

img = cv2.imread('img/normal.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
show_img = np.copy(img)

selected_pts = []

def mouse_callback(event, x, y, flags, param):
    global selected_pts, show_img
    
    if event == cv2.EVENT_LBUTTONUP:
        selected_pts.append([x, y])
        cv2.circle(show_img, (x, y), 10, (0, 255, 0), 3)
        

def select_points(image, points_num):
    global selected_pts
    selected_pts = []
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)

    while True: 
        cv2.imshow('image', image)
        
        k = cv2.waitKey(1)
        
        if k == 27 or len(selected_pts) ==points_num:
            break

    cv2.destroyAllWindows()
    return np.array(selected_pts, dtype=np.float32)

## 첫번째, affine 변환
show_img = np.copy(img)
src_pts = select_points (show_img, 3)
dst_pts = np.array([[0, 240], [0, 0], [240, 0]], dtype = np.float32)

affine_m = cv2.getAffineTransform (src_pts, dst_pts) 

#scaling size, youz have to convert this number
unwarped_img = cv2.warpAffine(img, affine_m, (640, 480))

cv2.imshow('result', np.hstack((show_img, unwarped_img)))
k = cv2.waitKey()
cv2.destroyAllWindows()

#Affine 역변환
inv_affine = cv2.invertAffineTransform(affine_m)
warped_img = cv2.warpAffine(unwarped_img, inv_affine, (640, 480))

cv2.imshow('result', np.hstack((show_img, unwarped_img, warped_img)))
k = cv2.waitKey()
cv2.destroyAllWindows()


#회전을 시킴 - 무쓸모
'''
rotation_mat = cv2.getRotationMatrix2D(tuple (src_pts [0]), 6, 1)

rotated_img = cv2.warpAffine(img, rotation_mat, (640, 480))

cv2.imshow('result', np.hstack((show_img, rotated_img)))
k = cv2.waitKey()

cv2.destroyAllWindows()
'''

#원근법
show_img = np.copy(img)
src_pts = select_points(show_img, 4)
#좌하, 좌상, 우상, 우하 순서임
dst_pts = np.array([[160, 360], [160, 120], [480, 120], [480, 360]], dtype =np.float32)

perspective_m = cv2.getPerspectiveTransform(src_pts, dst_pts) 
unwarped_img = cv2.warpPerspective(img, perspective_m, (640, 480))

cv2.imshow('result', np.hstack((show_img, unwarped_img)))
k = cv2.waitKey()

cv2.destroyAllWindows()
