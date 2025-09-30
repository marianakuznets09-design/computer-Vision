import cv2
import numpy as np
image = cv2.imread('image/image.webp')



cv2.rectangle(image, (200, 150), (300, 300), (0, 0, 0), 1)
cv2.putText(image, "Mariana Kuznets", (650, 420), cv2.FONT_HERSHEY_PLAIN, 1, (1, 0, 0))

cv2.imshow('image', image)
print(image.shape)


cv2.waitKey(0)
cv2.destroyAllWindows()