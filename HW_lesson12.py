import tensorflow as tf

from tensorflow import keras
from keras import layers, models
import numpy as np

from tensorflow.keras.preprocessing import image

train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                                                               image_size= (128, 128),
                                                               batch_size = 30,
                                                               label_mode= 'categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                                                               image_size= (128, 128),
                                                               batch_size = 30,
                                                               label_mode= 'categorical')

#нормалізація зображень
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#щар вхідні дані
model = models.Sequential()

model.add(layers.Conv2D(filters = 32,kernal_size = (3, 3), activation='relu', input_shape=(128, 128, 3)))  # 1 фільр визначаємо прсті ознаки

model.add(layers.MaxPooling2D(pool_size = (2, 2)))






model.add(layers.Conv2D(filters = 64,kernal_size = (3, 3), activation='relu')) #контури і структури об'єктів

model.add(layers.MaxPooling2D(pool_size = (2, 2)))





model.add(layers.Conv2D(filters = 128,kernal_size = (3, 3), activation='relu')) #контури і структури об'єктів

model.add(layers.MaxPooling2D(pool_size = (2, 2)))


model.add(layers.Conv2D(filters = 256, kernal_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Flatten())



#шар внутрішній

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_ds, epochs=15,validation_data=test_ds, verbose=0)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Якість{test_acc}')

class_name = ['apples', 'bananas', 'oranges']

img =image.load_img('image/', target_size=(128, 128))
image_array = image.img_to_array(img)

img_array = image_array/255.0

img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)

prediction_index = np.argmax(prediction[0])

print(f'fІмовірність по класам {prediction[0]}')
print(f'Модель визначила {class_name[prediction_index][0]}')

