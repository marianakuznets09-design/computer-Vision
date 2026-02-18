import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

DATABASE_PATH = "embeddings.pkl"
CELEB_FOLDER = "celebrities"
THRESHOLD = 0.6



# Створення бази

def build_database():
    print("Створення бази...")
    database = {}

    for person_name in os.listdir(CELEB_FOLDER):
        person_path = os.path.join(CELEB_FOLDER, person_name)

        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                embedding = DeepFace.represent(
                    img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(embedding)

            except:
                continue

        if embeddings:
            database[person_name] = np.mean(embeddings, axis=0)
            print(f"Додано: {person_name}")

    with open(DATABASE_PATH, "wb") as f:
        pickle.dump(database, f)

    print("База створена ")
    return database



# Завантаження бази

if os.path.exists(DATABASE_PATH):
    with open(DATABASE_PATH, "rb") as f:
        database = pickle.load(f)
else:
    database = build_database()



# Пошук схожості
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_match(embedding):
    best_match = None
    best_score = -1

    for name, db_embedding in database.items():
        score = cosine_similarity(embedding, db_embedding)

        if score > best_score:
            best_score = score
            best_match = name

    return best_match, best_score

def get_celebrity_image(name):
    person_path = os.path.join(CELEB_FOLDER, name)

    images = os.listdir(person_path)
    if len(images) == 0:
        return None

    img_path = os.path.join(person_path, images[0])
    celeb_img = cv2.imread(img_path)

    return celeb_img

locked_match = None
locked_score = None

# Камера

cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    try:
        if locked_match is None:
            embedding = DeepFace.represent(
                frame,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            name, score = find_match(embedding)

            locked_match = name
            locked_score = score
        else:
            name = locked_match
            score = locked_score

        celeb_img = get_celebrity_image(name)

        if celeb_img is not None:
            celeb_img = cv2.resize(celeb_img, (200, 250))


            h, w, _ = frame.shape
            frame[50:300, w - 220:w - 20] = celeb_img



        label = f"{name} |  ({score:.2f})"

        cv2.putText(frame, label,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 127),
                    2)

        key = cv2.waitKey(1) & 0xFF



    except:
        pass

    cv2.imshow("Celebrity AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
