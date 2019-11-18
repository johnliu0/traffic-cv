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
import matplotlib.patches as patches
from keras.preprocessing import image
from skimage import transform
from os import walk
from os.path import join
from random import randint
from joblib import load as load_weights
from time import time_ns

def train():
    classifier.train()

def predict(img_path):
    # classifier.load()
    # for (_, _, filenames) in walk(img_path):
    #     for file in filenames:
    #         raw_img = image.load_img(join(img_path, file))
    #         raw_img = raw_img.resize(config.convnet_image_input_size)
    #         img = image.img_to_array(raw_img)
    #         reshaped_img = img.reshape((1,) + img.shape)
    #         convnet_output = convnet.extract_features(reshaped_img)
    #         convnet_output = convnet_output.reshape((1,) + convnet_output.shape)
    #         svm_output = classifier.predict(convnet_output)
    #         print(svm_output)
    #         plt.imshow(image.array_to_img(img))
    #         plt.show()

    classifier.load()
    raw_img = image.load_img(img_path)
    img = image.img_to_array(raw_img)
    boxes = region.propose_boxes(img)
    ax = plt.gca()
    detected_boxes = []
    plt.imshow(image.array_to_img(raw_img))
    counter = 0
    for idx, box in enumerate(boxes):
        box_img = img[box.min_y : box.max_y, box.min_x : box.max_x]
        box_img = transform.resize(box_img, config.convnet_image_input_size + (3,))

        box_img_reshaped = box_img.reshape((1,) + box_img.shape)
        convnet_output = convnet.extract_features(box_img_reshaped)
        convnet_output = convnet_output.reshape((1,) + convnet_output.shape)
        svm_output = classifier.predict(convnet_output)
        if svm_output == 1:
            counter += 1
            ax.add_patch(patches.Rectangle(
                (box.min_x, box.min_y),
                box.width,
                box.height,
                linewidth=1,
                color='red',
                fill=False
            ))

    print(counter, '/', len(boxes))
    plt.show()
