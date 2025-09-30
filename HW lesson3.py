import cv2
# import numpy as np
image = cv2.imread('image1/image.jpg')
image =cv2.resize(image, (320, 400))




cv2.rectangle(image, (120, 100), (200, 180), (0, 0, 0), 1,25)
cv2.putText(image, "Mariana Kuznets", (90, 240), cv2.FONT_HERSHEY_TRIPLEX, 1/2, (100, 0, 120))
#
cv2.imshow('image', image)
print(image.shape)


cv2.waitKey(0)
cv2.destroyAllWindows()
