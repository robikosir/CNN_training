import json
import os

import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.keras import layers
from keras.callbacks import CSVLogger
from tensorflow.python.keras.models import load_model
import numpy as np

ear_left_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
ear_right_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')
model_path = "AWEdataset.h5"
model = load_model(model_path)

rootdir = 'testing_db'
correct_gender = 0
correct_eth = 0
wrong_gender = 0
wrong_eth = 0

for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        for subdir1, dirs1, files1 in os.walk(f'{rootdir}/{dir}'):
            i = 0
            annotations_file = open(f'{rootdir}/{dir}/annotations.json', 'r')
            content = annotations_file.read()
            features = json.loads(content)
            gender_main = features.get('gender')
            eth = features.get('ethnicity')
            for file in files1:
                if 'annotations' not in file:
                    image = cv2.imread(f'{rootdir}/{dir}/{file}')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    left_ear = ear_right_cascade.detectMultiScale(gray, 1.001, 6)
                    right_ear = ear_left_cascade.detectMultiScale(gray, 1.001, 6)

                    try:
                        for (x, y, w, h) in left_ear:
                            img = gray[y - 50:y + 40 + h, x - 10:x + 10 + w]
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            img = cv2.resize(img, (200, 200))
                            predict = model.predict(np.array(img).reshape(-1, 200, 200, 3))
                            gender = np.argmax(predict[0])
                            race = np.argmax(predict[1])

                        for (x, y, w, h) in right_ear:
                            img = gray[y - 50:y + 40 + h, x - 10:x + 10 + w]
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                            img = cv2.resize(img, (200, 200))
                            predict = model.predict(np.array(img).reshape(-1, 200, 200, 3))
                            gender = np.argmax(predict[0])
                            race = np.argmax(predict[1])
                    except:
                        pass

                    if len(right_ear) != 0 or len(left_ear) != 0:
                        if gender == 1:
                            gender = 'm'
                        else:
                            gender = 'w'

                        if gender == gender_main:
                            correct_gender += 1
                        else:
                            wrong_gender += 1

                        if race == eth:
                            correct_eth += 1
                        else:
                            wrong_eth += 1

print(f'gender {correct_gender}- {wrong_gender} = {correct_gender/(wrong_gender+correct_gender)}')
print(f'eth {correct_eth}- {wrong_eth} = {correct_eth/(wrong_eth+correct_eth)}')

# rootdir = 'testing_db_utk'
# correct_gender = 0
# correct_eth = 0
# wrong_gender = 0
# wrong_eth = 0
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# for image_name in os.listdir(rootdir):
#     features = image_name.split('_')
#     gender_main = features[1]
#     eth = features[2]
#     image = cv2.imread(f'testing_db_utk/{image_name}')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     predict = model.predict(np.array(image).reshape(-1, 200, 200, 3))
#     gender = np.argmax(predict[0])
#     race = np.argmax(predict[1])
#     try:
#         if int(gender) == int(gender_main):
#             correct_gender += 1
#         else:
#             wrong_gender += 1
#
#         if int(race) == int(eth):
#             correct_eth += 1
#         else:
#             wrong_eth += 1
#     except:
#         pass
#
# print(f'gender {correct_gender}- {wrong_gender} = {correct_gender/(wrong_gender+correct_gender)}')
# print(f'eth {correct_eth}- {wrong_eth} = {correct_eth/(wrong_eth+correct_eth)}')

