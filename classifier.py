import augment
import convnet
import config
import numpy as np
import matplotlib.pyplot as plt
import re
from random import randint
from joblib import dump as export_weights, load as load_weights
from sklearn.svm import OneClassSVM
from keras.preprocessing import image
from PIL import Image as PILImage
from os.path import expanduser, join
from os import walk

from sklearn.svm import SVC

home = expanduser('~')
svm = SVC(kernel='linear')

def train():
    """Trains an SVM to classify traffic lights.

    Positive and negative images are trained with target values of 1 and 0
    respectively. The SVM takes in a feature vector from the pretrained
    convolutional base and outputs a scalar. Outputs past a certain set
    threshold should signify that the feature vector extracted from the image
    indicates that the image is indeed a traffic light.
    """

    global svm

    # collect all positive and negative training images
    positive_imgs = []
    negative_imgs = []
    for (dirpath, dirnames, filenames) in walk(config.training_positives_dir):
        positive_imgs.extend(filenames)
    for (dirpath, dirnames, filenames) in walk(config.training_negatives_dir):
        negative_imgs.extend(filenames)

    num_pos_augmentations = 5
    num_neg_augmentations = 8
    x_train = []
    y_train = []
    num_extracted = 0

    print('Starting feature extraction.')
    print('Total number of positive samples after augmentation:', num_pos_augmentations * (len(positive_imgs) + 1))
    print('Total number of positive samples after augmentation:', num_neg_augmentations * (len(negative_imgs) + 1))

    # prepare original positive training samples
    print ('Preparing positive training samples.')
    for positive_img in positive_imgs:
        img = image.load_img(join(config.training_positives_dir, positive_img))
        img = img.resize(config.convnet_image_input_size)
        img = image.img_to_array(img)
        conv_output = convnet.extract_features(img.reshape((1,) + img.shape))
        x_train.append(conv_output)
        y_train.append(1)

    # generate positive training samples
    for positive_img in positive_imgs:
        imgs = augment.generate(join(config.training_positives_dir, positive_img), target_size=config.convnet_image_input_size, num_imgs=num_pos_augmentations)
        for i in range(num_pos_augmentations):
            conv_output = convnet.extract_features(imgs[i].reshape((1,) + imgs[i].shape))
            x_train.append(conv_output)
            y_train.append(1)

    # prepare original negative training samples
    print ('Preparing negative training samples.')
    for negative_img in negative_imgs:
        img = image.load_img(join(config.training_negatives_dir, negative_img))
        img = img.resize(config.convnet_image_input_size)
        img = image.img_to_array(img)
        conv_output = convnet.extract_features(img.reshape((1,) + img.shape))
        x_train.append(conv_output)
        y_train.append(0)

    # generate negative training samples
    for negative_img in negative_imgs:
        imgs = augment.generate(join(config.training_negatives_dir, negative_img), target_size=config.convnet_image_input_size, num_imgs=num_neg_augmentations)
        for i in range(num_neg_augmentations):
            conv_output = convnet.extract_features(imgs[i].reshape((1,) + imgs[i].shape))
            x_train.append(conv_output)
            y_train.append(0)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    print('Training SVM.')
    svm.fit(x_train, y_train)
    export_weights(svm, join(config.models_dir, 'svm_weights.joblib'))

def load():
    global svm
    svm = load_weights(join(config.models_dir, 'svm_weights.joblib'))

def predict(input_data):
    return svm.predict(input_data)
