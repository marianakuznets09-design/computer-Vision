import cv2
import numpy as np
img = np.zeros((500, 400, 3), np.uint8)
# img[:] = 79, 174, 114
#rgb = bgr

# img[100:150, 200:250] = 79, 174, 114
cv2.rectangle(img, (100, 100), (200, 200), (79, 174, 114), 1)

cv2.line(img, (100, 100), (200, 200), (79, 174, 114), 1)

print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (79, 174, 114), 1)
cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1] // 2), img.shape[0]), (79, 174, 114), 1)

cv2.circle(img, (200, 200), 30, (79, 174, 114), -1)
cv2.putText(img, "Komarov Ivan", (200, 150), cv2.FONT_HERSHEY_PLAIN, 1, (79, 174, 114))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()