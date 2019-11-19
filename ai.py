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
    detected_boxes = []

    ax, (f1, f2) = plt.subplots(1, 2)
    f1.imshow(raw_img)
    f1.set_xlabel('All')
    f1.set_yticklabels([])
    f1.set_xticklabels([])
    f2.imshow(raw_img)
    f2.set_xlabel('Predicted')
    f2.set_yticklabels([])
    f2.set_xticklabels([])

    counter = 0
    for idx, box in enumerate(boxes):
        box_img = img[box.min_y : box.max_y, box.min_x : box.max_x]
        box_img = transform.resize(box_img, config.convnet_image_input_size + (3,))
        f1.add_patch(patches.Rectangle(
            (box.min_x, box.min_y),
            box.width,
            box.height,
            linewidth=1,
            color='red',
            fill=False
        ))

        box_img_reshaped = box_img.reshape((1,) + box_img.shape)
        convnet_output = convnet.extract_features(box_img_reshaped)
        convnet_output = convnet_output.reshape((1,) + convnet_output.shape)
        svm_output = classifier.predict(convnet_output)
        if svm_output == 1:
            counter += 1
            f2.add_patch(patches.Rectangle(
                (box.min_x, box.min_y),
                box.width,
                box.height,
                linewidth=1,
                color='green',
                fill=False
            ))

    print(counter, '/', len(boxes))
    plt.show()

yes_counter = 0
no_counter = 0

def button_yes(svm_output, img):
    global yes_counter
    print(f'SVM output: {svm_output}, Actual: 1')

    existing_files = set()
    for (_, _, file_names) in walk(config.training_positives_dir):
        for file in file_names:
            existing_files.add(file)

    # svm failed to detect positive sample
    if svm_output == 0:
        file_name = f'img_{yes_counter}.jpg'
        yes_counter += 1
        while file_name in existing_files:
            file_name = f'img_{yes_counter}.jpg'
            yes_counter += 1
        img.save(join(config.training_positives_dir, file_name))
    plt.close()

def button_no(svm_output, img):
    global no_counter
    print(f'SVM output: {svm_output}, Actual: 0')

    existing_files = set()
    for (_, _, file_names) in walk(config.training_negatives_dir):
        for file in file_names:
            existing_files.add(file)

    # svm failed to detect negative sample
    if svm_output == 1:
        file_name = f'img_{no_counter}.jpg'
        no_counter += 1
        while file_name in existing_files:
            file_name = f'img_{no_counter}.jpg'
            no_counter += 1
        img.save(join(config.training_negatives_dir, file_name))
    plt.close()

def mine(img_path, use_predicted=False):
    """Tool for making negative and positive training samples.

    The bounding boxes used in the predict method are shown. Press yes or no as
    appropriate to confirm whether or not a sample is positive or negative. If
    the SVM was incorrect in its output, then a training sample will be made
    in the training negatives and positives directory specified in the config.

    Images are called img_x.jpg, where x is a non-negative integer. This tool
    will not overwrite images, and will instead find the first non-negative
    integer that it can save to without overwriting an existing file. This
    allows the tool to seamlessly and easily add to the existing datasets.

    Args:
        img_path: Path to image to mine.
        use_predicted: Whether or not to use the resulting bounding boxes from
            after the predictions are made. This is useful for when all the
            correct boxes have been made, but there are additional false
            positives.
    """

    classifier.load()
    raw_img = image.load_img(img_path)
    img = image.img_to_array(raw_img)
    boxes = region.propose_boxes(img)
    shown_boxes = []

    for idx, box in enumerate(boxes):
        box_img = img[box.min_y : box.max_y, box.min_x : box.max_x]
        pil_box_img = image.array_to_img(box_img)
        box_img = transform.resize(box_img, config.convnet_image_input_size + (3,))
        box_img_reshaped = box_img.reshape((1,) + box_img.shape)
        convnet_output = convnet.extract_features(box_img_reshaped)
        convnet_output = convnet_output.reshape((1,) + convnet_output.shape)
        svm_output = classifier.predict(convnet_output)
        if use_predicted:
            if svm_output == 1:
                shown_boxes.append((box, svm_output))
        else:
            shown_boxes.append((box, svm_output))

    for idx, (box, svm_output) in enumerate(shown_boxes):
        box_img = img[box.min_y : box.max_y, box.min_x : box.max_x]
        pil_box_img = image.array_to_img(box_img)
        plt.imshow(pil_box_img)
        plt.axis('off')

        axyes = plt.axes([0.7, 0.05, 0.1, 0.075])
        axno = plt.axes([0.81, 0.05, 0.1, 0.075])

        print(f'Box {idx + 1} of {len(shown_boxes)}: ', end='')

        yes = Button(axyes, 'Yes')
        yes.on_clicked(lambda _: button_yes(svm_output, pil_box_img))

        no = Button(axno, 'No')
        no.on_clicked(lambda _: button_no(svm_output, pil_box_img))

        plt.show()
