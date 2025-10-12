import cv2
import numpy as np

import cv2
import numpy as np
img = cv2.imread('image/image5.jpg')
img = cv2.resize(img, (640, 480))
img_copy = img.copy()

img = cv2.GaussianBlur(img, (7, 7), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([8, 40, 0])
upper = np.array([179, 255, 255])

mask = cv2.inRange(img, lower, upper)

img = cv2.bitwise_and(img, img, mask = mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
         cx = int(M['m10'] / M['m00'])
         cy = int(M['m01'] / M["m00"])

        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:
         shape = "square"
        elif len (approx) == 3:
            shape = "triangle"
        elif len(approx) > 8:
            shape = "oval"
        elif len(approx) == 5:
            shape = "star"




        perimetr = cv2.arcLength(cnt, True)
        cv2.putText(img_copy, f'S:{area}, P:{perimetr}', (x, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (0, 0, 0), 1)
        cv2.putText(img_copy, f'shape:{shape}', (x, y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

cv2.imwrite("result.jpg", img_copy)


cv2.imshow('img', img)
cv2.imshow("mask", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
