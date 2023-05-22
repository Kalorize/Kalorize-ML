"""
# Link model
# Scriptku tidak berhasil
"https://drive.google.com/file/d/1-5pGZPWmzYgFVItPEm0PDUGM0n2nrxTN/view?usp=share_link"

1. ketik di terminal virtual venv
2. ketik di terminal .\venv\Scripts\active
3. pip install requirements.txt
4. Error No module named 'keras.engine.topology'

    edit venv/lib/site-packages/keras_vggface/models.py

    line 20
    dari: from keras.engine.topology import get_source_inputs

    jadi: from keras.utils.layer_utils import get_source_inputs

"""

import cv2
import re
import os
from keras_vggface import utils
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np


model = load_model('model_vgg16_2.h5', compile=False)


def crop_image(image, x, y, w, h):
    return image[y:(y+h), x:(x+w), :]


def process_arr(arr, version):
    img = cv2.resize(arr, (224, 224))
    img = np.expand_dims(img, 0)
    img = utils.preprocess_input(img, version=version)
    return img


def detect_faces(face_path, confidence):
    mtcnn = MTCNN()
    image = face_path
    image = tf.keras.utils.load_img(image)
    image = tf.keras.utils.img_to_array(image)
    box = mtcnn.detect_faces(image)
    box = [i for i in box if i['confidence'] > confidence]
    res = [crop_image(image, *i['box']) for i in box]
    res = [process_arr(i, 1) for i in res]
    return box, res


def predict(image_path, confidence=0.95):
    # Hanya satu gambar
    assert os.path.isfile(image_path)

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    boxes, faces = detect_faces(image_path, confidence)
    preds = [model.predict(face) for face in faces]
    height = 0
    weight = 0
    gender = ''
    for idx, box in enumerate(boxes):
        height, weight, sex = preds[idx]
        height = height[0, 0]
        weight = weight[0, 0]
        gender = 'M' if sex[0, 0] > 0.05 else 'F'

    return height, weight, gender


image_path = input('Masukkan direktori image : ')
height, weight, gender = predict(image_path)
print(height)
print(weight)
print(gender)

# D:\Kuliah\Bangkit\Capstone\ML\face2bmi\data\test\single_face\xi.jpg
