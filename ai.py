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
from matplotlib.widgets import Button
from keras.preprocessing import image
from skimage import transform
from os import walk
from os.path import join
from random import randint
from joblib import load as load_weights
from time import time_ns
from PIL import Image as PILImage

def train():
    classifier.train()

def predict(img_path):
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

yes_counter = 0
no_counter = 0

def button_yes(svm_output, img):
    global yes_counter
    print(f'SVM output: {svm_output}, Actual: 1')
    # svm failed to detect positive sample
    if svm_output == 0:
        img.save(join(config.training_positives_dir, f'img_{yes_counter}.jpg'))
        yes_counter += 1
    plt.close()

def button_no(svm_output, img):
    global no_counter
    print(f'SVM output: {svm_output}, Actual: 0')
    # svm failed to detect negative sample
    if svm_output == 1:
        img.save(join(config.training_negatives_dir, f'img_{no_counter}.jpg'))
        no_counter += 1
    plt.close()

def mine(img_path):
    classifier.load()
    raw_img = image.load_img(img_path)
    img = image.img_to_array(raw_img)
    boxes = region.propose_boxes(img)

    for idx, box in enumerate(boxes):
        box_img = img[box.min_y : box.max_y, box.min_x : box.max_x]
        box_img = transform.resize(box_img, config.convnet_image_input_size + (3,))

        box_img_reshaped = box_img.reshape((1,) + box_img.shape)
        convnet_output = convnet.extract_features(box_img_reshaped)
        convnet_output = convnet_output.reshape((1,) + convnet_output.shape)
        svm_output = classifier.predict(convnet_output)

        pil_box_img = image.array_to_img(box_img)

        plt.imshow(pil_box_img)
        plt.axis('off')

        axyes = plt.axes([0.7, 0.05, 0.1, 0.075])
        axno = plt.axes([0.81, 0.05, 0.1, 0.075])

        yes = Button(axyes, 'Yes')
        yes.on_clicked(lambda _: button_yes(svm_output, pil_box_img))

        no = Button(axno, 'No')
        no.on_clicked(lambda _: button_no(svm_output, pil_box_img))

        plt.show()

    print(counter, '/', len(boxes))
    plt.show()
