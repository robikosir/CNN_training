from keras.models import load_model
import cv2
import numpy as np
model_path = "./model.h5"
model = load_model(model_path)
img_path = "img_path"

ear_left_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
ear_right_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

pic = cv2.imread(img_path)
gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

left_ear = ear_right_cascade.detectMultiScale(gray, 1.001, 6)
right_ear = ear_left_cascade.detectMultiScale(gray, 1.001, 6)


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
if len(right_ear) != 0 or len(left_ear) != 0:
    if gender == 1:
        gender = 'Man'
    else:
        gender = 'Woman'

    if race == 1:
        race = 'White'
    elif race == 2:
        race = 'Asian'
    elif race == 3:
        race = 'South Asian'
    elif race == 4:
        race = 'Black'
    elif race == 5:
        race = 'South American'
    elif race == 6 or race == 0:
        race = 'Other'
    cv2.putText(pic, f'{gender}, {race}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(f'results/{img_path}', pic)
