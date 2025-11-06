import cv2
from collections import Counter
import pandas as pd



net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')


classes = []
with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ', 1)      # ділимо тільки на 2 частини: id і назва
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


images_data = [
    {'name': 'dog.jpg', 'img': cv2.imread('image/MobileNet/dog.jpg')},
    {'name': 'syrukat.jpg', 'img': cv2.imread('image/MobileNet/syrukat.jpg')},
    {'name': 'phone.jpg', 'img': cv2.imread('image/MobileNet/phone.jpg')},
    {'name': 'ball.jpg', 'img': cv2.imread('image/MobileNet/ball.jpg')},
    {'name': 'pizza.jpg', 'img': cv2.imread('image/MobileNet/pizza.jpg')},]



classified_labels = []
classification_details = []


for item in images_data:
    filename = item['name']
    image = item['img']

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)


net.setInput(blob)
preds = net.forward()


idx = preds[0].argmax()


label = classes[idx] if idx < len(classes) else "unknown"
conf = float(preds[0][idx]) * 100

classified_labels.append(label)
classified_labels.append(label)
classification_details.append({
        'Файл': image,
        'Клас': label,
        'Ймовірність (%)': round(conf, 2)})


print("Клас:", label)
print("Ймовірність:", round(conf, 2), "%")


text = label + ": " + str(int(conf)) + "%"
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


print("\n" + "="*50)
print("статистика")
print("="*50)

print("\n[звіт]")
df_results = pd.DataFrame(classification_details)
print(df_results.to_string(index=False))

class_counts = Counter(classified_labels)
df_counts = pd.DataFrame(class_counts.items(), columns=['Клас', 'Кількість зустрічей'])
df_counts = df_counts.sort_values(by='Кількість зустрічей', ascending=False)

print("\n\n[Таблиця]")
print(df_counts.set_index('Клас'))
print("="*50)





cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

