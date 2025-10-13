import cv2
import numpy as np


img = cv2.imread('image/imgpr.jpg')
img = cv2.resize(img, (566, 424))
# img = cv2.resize(img, (640, 480))
img_copy = img.copy()
print(img. shape)
img = cv2.GaussianBlur(img, (7, 7), 2)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([120, 45, 44])
upper_red = np.array([179, 255, 255])
mask_red = cv2.inRange(img, lower_red, upper_red)


lower_green = np.array([17, 44, 0])
upper_green = np.array([48, 4255, 255])
mask_green = cv2.inRange(img, lower_green, upper_green)

lower_blue = np.array([88, 10, 0])
upper_blue = np.array([118, 255, 255])
mask_blue = cv2.inRange(img, lower_blue, upper_blue)

mask_total = cv2.bitwise_or(mask_red, mask_blue)
mask_total = cv2.bitwise_or(mask_total, mask_green)

contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
         elif len(approx) > 8:
             shape = "oval"
         else:
             shape = "inshe"


    # for i in color:
    #  if i = >=0>=160 and <=179:
    #     color = 'red'





         cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
         cv2.circle(img_copy, (cx, cy), 4, (0, 0, 255), -1)
         cv2.putText(img_copy, f'S:{area}', (x-20, y + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
         cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x-20, y + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 0), 1)
         cv2.putText(img_copy, f'shape:{shape}', (x-20, y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
         # cv2.putText(img_copy, f'color:{color}')

cv2.imwrite("result.jpg", img_copy)

cv2.imshow('img', img)
cv2.imshow('img_copy', img_copy)
cv2.imshow('mask_total', mask_total)
cv2.waitKey(0)
cv2.destroyAllWindows()