import json
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers.pooling import MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Flatten, Dense
from tensorflow.keras import layers
from PIL import Image

images = []
genders = []
races = []

# i = 0
# for image_name in os.listdir('UTKFace'):
#     features = image_name.split('_')
#     gender = features[1]
#     race = features[2]
#     image = cv2.imread(f'UTKFace/{image_name}')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     images.append(np.array(image))
#     genders.append(np.array(gender))
#     i += 1
#     if i > 20:
#         break
#     if 'jpg' in race:
#         race = 0
#     races.append(np.array(race))
rootdir = 'awe'
for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        for subdir1, dirs1, files1 in os.walk(f'{rootdir}/{dir}'):
            i = 0
            for file in files1:
                if 'annotations' in file:
                    file = open(f'{rootdir}/{dir}/{file}', 'r')
                    content = file.read()
                    features = json.loads(content)
                else:
                    image = cv2.imread(f'{rootdir}/{dir}/{file}')
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    resize_and_rescale = tf.keras.Sequential([
                        layers.experimental.preprocessing.Resizing(200, 200),
                        layers.experimental.preprocessing.Rescaling(1. / 255)
                    ])
                    image = resize_and_rescale(image)
                    images.append(np.array(image))
                    i += 1
            for j in range(i):
                if features.get('gender') == 'm':
                    genders.append(np.array(1))
                else:
                    genders.append(np.array(0))
                eth = features.get('ethnicity')
                if eth > 8:
                    eth = 0
                races.append(np.array(eth))

print(races)


genders = np.array(genders, dtype=np.uint64)
races = np.array(races, dtype=np.uint64)
images = np.array(images)

x_train, x_test, y_train, y_test = train_test_split(images, genders, random_state=100)
x_train2, x_test2, y_train2, y_test2 = train_test_split(images, races, random_state=100)

inputs = Input(shape=(200, 200, 3))
flt = Flatten()(inputs)


gender_l = Dense(128, activation="relu")(flt)
gender_l = Dense(80, activation="relu")(gender_l)
gender_l = Dense(64, activation="relu")(gender_l)
gender_l = Dense(32, activation="relu")(gender_l)
gender_l = Dense(2, activation="softmax")(gender_l)

race_l = Dense(128, activation="relu")(flt)
race_l = Dense(80, activation="relu")(race_l)
race_l = Dense(64, activation="relu")(race_l)
race_l = Dense(32, activation="relu")(race_l)
race_l = Dense(7, activation="softmax")(race_l)

model = Model(inputs=inputs, outputs=[gender_l, race_l])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics='accuracy')

save = model.fit(x_train, [y_train, y_train2], validation_data=(x_test, [y_test, y_test2]), epochs=100)
model.save("model.h5")
