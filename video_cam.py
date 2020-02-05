import cv2

cap = cv2.VideoCapture(0)
#카메라 프레임 구하는 과정
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("기존 폭: %d, height:%d" % (width, height))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("새로운 폭: %d, height:%d" % (width, height))

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('camera', img)
            if cv2.waitKey(1) != -1:
                break
        else:
            print('no frame')
            break
else:
    print("can`t open camera.")
cap.release()
cv2.destroyAllWindows()

#동영상을 읽을때는 프레임 수를 지정해 주어야함. cv2.waitKey(25)