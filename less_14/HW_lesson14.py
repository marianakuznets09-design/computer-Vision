import cv2
import numpy as np
import os
import shutil

PROJECT_DIR = os.path.dirname(__file__)

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
IMAGES_DIR = os.path.join(PROJECT_DIR, 'image14')

OUT_DIR = os.path.join(PROJECT_DIR, 'out')
PEOPLE_DIR = os.path.join(PROJECT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(PROJECT_DIR, 'no_people')


os.makedirs(OUT_DIR, exist_ok= True)
os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

cascade_path = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print('No face detected')
    exit()

def detected_people(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1,5, minSize=(30, 30))
    return faces

allowed_ext = ['.jpg', '.jpeg', '.png', '.bmp']

files = os.listdir(IMAGES_DIR)

count_people = 0
count_no_people = 0

for filename in files:
    # Перевіряємо розширення правильно
    if not filename.lower().endswith(tuple(allowed_ext)):
        continue

    in_path = os.path.join(IMAGES_DIR, filename)
    img = cv2.imread(in_path)

    if img is None:
        print(f'Зображення не знайдено: {filename}')
        continue

    faces = detected_people(img)


    num_faces = len(faces)


    print(f'Файл: {filename} | People count: {num_faces}')

    if num_faces > 0:
        out_path = os.path.join(PEOPLE_DIR, filename)
        shutil.copyfile(in_path, out_path)
        count_people += 1

        boxed = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

        boxed_path = os.path.join(OUT_DIR, "boxed_" + filename)
        cv2.imwrite(boxed_path, boxed)
    else:
        out_path = os.path.join(NO_PEOPLE_DIR, filename)
        shutil.copyfile(in_path, out_path)
        count_no_people += 1

        boxed = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

            boxed_path = os.path.join(OUT_DIR, "boxed_" + filename)
            cv2.imwrite(boxed_path, boxed)


        out_path = os.path.join(NO_PEOPLE_DIR, filename)
        shutil.copyfile(in_path, out_path)
        count_no_people +=1


