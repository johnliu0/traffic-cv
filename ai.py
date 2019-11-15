"""Top level module for the machine learning model.

Handles all usage of the model; including training, predicting, and saving the
model.
"""

import sys
import config
import convnet
import region
import matplotlib.pyplot as plt
import numpy as np
import classifier
from keras.preprocessing import image
from skimage import transform

from random import randint

from time import time_ns

def train():
    pass

def predict(img_path):
    classifier.load()
    raw_img = image.load_img(img_path)
    img = image.img_to_array(raw_img)
    boxes = region.propose_boxes(img)
    ax = plt.gca()

    print(len(boxes))
    bef_time = time_ns()

    detected_boxes = []

    for idx, box in enumerate(boxes):
        box_img = img[box.min_y : box.max_y, box.min_x : box.max_x]
        box_img = transform.resize(box_img, (150, 150, 3))
        box_img_reshaped = box_img.reshape((1,) + box_img.shape)
        bef_time = time_ns()
        convnet_output = convnet.extract_features(box_img_reshaped)
        print((time_ns() - bef_time) / 1000000000)
        convnet_output = convnet_output.reshape((1,) + convnet_output.shape)
        svm_output = classifier.predict(convnet_output)
        if svm_output == 1:
            plt.imshow(image.array_to_img(box_img))
            plt.show()

        if idx % 50 == 0:
            print(idx)

    print('time taken: ', (time_ns() - bef_time) / 1000000000)
