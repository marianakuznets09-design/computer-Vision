import keras
import pandas as pd #ccv
import numpy as np  # мат операції
import tensorflow as tf  # нейронка
from tensorflow.keras import layers  # для тенсор
from keras import Sequential
from tensorflow.keras import layers    # для шарів
from sklearn.preprocessing import LabelEncoder  #перетворює текстові мітки в числа
import matplotlib.pyplot as plt
from keras.src.utils import to_categorical

#2
df = pd.read_csv('data/figures_updated2.0.csv')
# print(df.head())
#
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])
#3 обираємо для навчання
X = df[['area', 'perimeter', 'corners', 'area_perimeter_ratio']]
y = df['label_enc']
y1 = to_categorical(y, num_classes=5)  # (N, 5)

#4 стсоврення моделі !!
model = keras.Sequential([layers.Dense(16, activation="relu", input_shape=(4,)),  #розташовані послідовно  #інпут -- скільки парамерів подається для навч
                        layers.Dense(8, activation="relu"),
                        layers.Dense(5, activation="softmax")])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
#5 навчанння у декілька епох (повний прохід всіх даних)
history = model.fit(X, y1, epochs = 500, verbose = 0)

#6 візуалізація навчання
plt.plot(history.history['loss'], label = 'Втрата (Loss)')
plt.plot(history.history['accuracy'], label = 'Точність (Accuracy)')
plt.xlabel("Епоха")
plt.xlabel("Значення")
plt.title('Процес навчання')
plt.legend()
plt.show()

#тестування
test = np.array([[18, 16, 0, 1.12 ]])

pred = model.predict(test)
print(f'Імовірність по кожному класу: {pred}')
print(f'Модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')