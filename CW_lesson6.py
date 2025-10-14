import cv2

from CW_lesson4 import contours

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 5)
gray1 = cv2.convertScaleAbs(gray1, alpha=1.2, beta=50)



while True:
    ret, frame2 =cap.read()
    if not ret :
        print("кадри скінчились") # фрагмент відео
        break
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 5)
    gray2 = cv2.convertScaleAbs(gray2, alpha=1.2, beta=50)  # a - контраст #B -яскравість



    if cv2.waitKey(1) & 0xff == ord("q"):
        break

    diff = cv2.absdiff(gray1, gray2)  #знаходимо різницю

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ =  cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 800:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    gray1 = gray2

    # if cv2.waitKey(1) & 0xff == ord("q"):
    #     break




    # cv2.imshow("Video2", gray2)
    cv2.imshow("Video", frame2)


cap.release() #звільнеяємо камеру
cv2.destroyAllWindows()


