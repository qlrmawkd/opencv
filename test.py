import cv2

img_file = "img/helmet3.jpg"
img = cv2.imread(img_file)
dst = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

if img is not None:
    cv2.imshow('IMG', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No image file.')
    