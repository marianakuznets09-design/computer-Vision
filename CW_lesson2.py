# import cv2
# import numpy as np
# from cv2.gapi import kernel
# from numpy.ma.core import shape
#
# image = cv2.imread('images/im1.webp')
# # image =cv2.resize(image, (800, 400))
# image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
# # image =cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# # image = cv2.flip(image, 1)
#
#
# # image = cv2.GaussianBlur(image, (9, 9), 5)     #рівень заблюр. тільки не парні числа
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(image.shape)
# image = cv2.Canny(image, 100, 100)           #до дз
# #1
# # image = cv2.dilate(image, None, iterations=1)
#
# # [[0, 0, 0,],[0,0,0],[0,0,0]]
#
#
# kernel = np.ones((5,5), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# image = cv2.erode(image, kernel, iterations=1)
#
#
#
# cv2.imshow('cat', image)
# # cv2.imshow("image", image[0:200, 0:400])     #фрагмент
import cv2

#video

# video = cv2.VideoCapture("video/vid1.mp4")
video = cv2.VideoCapture(0)
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame, (800, 600))
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# cv2.waitKey(0)
cv2.destroyAllWindows()