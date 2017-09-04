# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMCallback
from keras.preprocessing import image
from keras.models import load_model


model = Sequential()

model.add(Convolution2D(filters=32,
                        padding='same',
                        kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(64, 64, 3)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(units=100, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1, activation='sigmoid',
                kernel_initializer='uniform'))
model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'dataset/training_set/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'dataset/test_set/',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=8000,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=2000,
    verbose=1,
    callbacks=[TQDMCallback(leave_inner=False)])


model.save('model.h5')


model = load_model('my_model.h5')


test = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
                      target_size=(64, 64))
test = image.img_to_array(test)
test = np.expand_dims(test, axis=0)

print(model.predict(test))
print(train_generator.class_indices)
