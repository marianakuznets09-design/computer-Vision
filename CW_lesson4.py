import cv2
import numpy as np
img = cv2.imread('image/image.jpg')
scale = 1
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)
img_copy_color = img.copy()


img_copy = img.copy()

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2)

img_copy = cv2.equalizeHist(img_copy)   #посилення контрасту

img_copy = cv2.Canny(img_copy, 50, 100) #тут треба погратися




#пошук контурів
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #RETR_EXTERNAL - режим отримання\знаходить найвищий(крайній) контури
                                                                                              #апроксимація - процес наближеного вираження одних велечинб або одних через інші
#малювання контурів, прямокутників та тексту
for cnt in contours:
    area = cv2.contourArea(cnt)  #визначаємо площу контура
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt) #створює найменший прямокутник який вмістить у собі контур

        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2) #контур, список контурів який ми малюємо

        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f'x:{x}, y: {y}, S:{int(area)}'
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


cv2.imshow('image', img_copy_color)
cv2.imshow('copy border', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('borders', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
