import cv2
import numpy as np
image = np.zeros((400, 600, 3), np.uint8)
image[:] = 222, 222, 255


img = cv2.imread('image_pr/image.jpg')
img = cv2.resize(img, (100, 130))
image[10:140, 20:120] = img

cv2.putText(image, "Mariana Kuznets", (150, 80), cv2.FONT_HERSHEY_TRIPLEX, 3/4, (0, 0, 0))
cv2.putText(image, "Computer Vision Student", (150, 110), cv2.FONT_HERSHEY_TRIPLEX, 1/2, (0, 0, 0))
cv2.putText(image, "Email: mariana.kuznets09@gmail.com", (150, 160), cv2.FONT_HERSHEY_TRIPLEX, 1/2, (0, 0, 0))
cv2.putText(image, "Phone: +380683166596", (150, 200), cv2.FONT_HERSHEY_TRIPLEX, 1/2, (0, 0, 0))
cv2.putText(image, "03/11/2009", (150, 240), cv2.FONT_HERSHEY_TRIPLEX, 1/2, (0, 0, 0))
cv2.putText(image, "OpenCV Business Card", (150, 370), cv2.FONT_HERSHEY_TRIPLEX, 3/4, (0, 0, 0))

cv2.rectangle(image, (10, 10), (590, 390), (23, 48, 73), 2)

img1 = cv2.imread('image_pr/qr.png')
img1 = cv2.resize(img1, (90, 90))
image[250:340, 450:540] = img1



cv2.imwrite("business_card.png", image)





cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()