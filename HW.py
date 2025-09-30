# #1
# import cv2
# image = cv2.imread('image/image.jpg')
# # # image =cv2.resize(image, (425, 500))
# image = cv2.resize(image, (image.shape[1] // 4 , image.shape[0] // 4))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #print(image.shape)
# image = cv2.Canny(image, 100, 100)
#
#
# cv2.imshow('cake', image )
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#
# #2
# import cv2
# image = cv2.imread('image/image2.jpg')
# image = cv2.resize(image, (image.shape[1] // 3 , image.shape[0] // 3))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 100, 100)
#
#
#
#
# cv2.imshow('email', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


