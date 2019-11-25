import augment
import convnet
import config
import numpy as np
import matplotlib.pyplot as plt
import re
from random import randint, shuffle
from joblib import dump as export_weights, load as load_weights
from sklearn.svm import OneClassSVM
from keras.preprocessing import image
from PIL import Image as PILImage
from os.path import expanduser, join
from os import walk
from time import time_ns
from skimage import transform


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

    num_pos_augmentations = 32
    num_neg_augmentations = 16


    num_positive_samples = (num_pos_augmentations + 1) * len(positive_imgs)
    num_negative_samples = (num_neg_augmentations + 1) * len(negative_imgs)
    num_samples = num_positive_samples + num_negative_samples;
    x_train = []
    y_train = []
    num_extracted = 0
    total_time = 0

    print('Starting feature extraction.')
    print('Total number of positive samples after augmentation:', num_positive_samples)
    print('Total number of negative samples after augmentation:', num_negative_samples)
    timer = time_ns();

    # prepare original positive training samples
    print ('Preparing samples.')
    sample_counter = 0
    for positive_img in positive_imgs:
        img = image.load_img(join(config.training_positives_dir, positive_img))
        img = image.img_to_array(img)
        img = transform.resize(img, config.convnet_image_input_size + (3,))
        x_train.append(convnet.extract_features(img.reshape((1,) + img.shape)))
        y_train.append(1)
        sample_counter += 1
        if sample_counter % 1000 == 0 and sample_counter != 0:
            print(f'Processed {sample_counter} samples.')

    # generate positive training samples
    for positive_img in positive_imgs:
        imgs = augment.generate_pos(join(config.training_positives_dir, positive_img), target_size=config.convnet_image_input_size, num_imgs=num_pos_augmentations)
        for i in range(num_pos_augmentations):
            x_train.append(convnet.extract_features(imgs[i].reshape((1,) + imgs[i].shape)))
            y_train.append(1)
            sample_counter += 1
            if sample_counter % 1000 == 0 and sample_counter != 0:
                print(f'Processed {sample_counter} samples.')

    # prepare original negative training samples
    for negative_img in negative_imgs:
        img = image.load_img(join(config.training_negatives_dir, negative_img))
        img = image.img_to_array(img)
        img = transform.resize(img, config.convnet_image_input_size + (3,))
        x_train.append(convnet.extract_features(img.reshape((1,) + img.shape)))
        y_train.append(0)
        sample_counter += 1
        if sample_counter % 1000 == 0 and sample_counter != 0:
            print(f'Processed {sample_counter} samples.')

    # generate negative training samples
    for negative_img in negative_imgs:
        imgs = augment.generate_neg(join(config.training_negatives_dir, negative_img), target_size=config.convnet_image_input_size, num_imgs=num_neg_augmentations)
        for i in range(num_neg_augmentations):
            x_train.append(convnet.extract_features(imgs[i].reshape((1,) + imgs[i].shape)))
            y_train.append(0)
            sample_counter += 1
            if sample_counter % 1000 == 0 and sample_counter != 0:
                print(f'Processed {sample_counter} samples.')

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    # randomly shuffle dataset by moving around the indices of the dataset
    shuffled_indices = np.arange(0, num_samples, 1)
    np.random.shuffle(shuffled_indices)
    x_train = x_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    print('Training SVM. ', end='')
    timer = time_ns()
    svm.fit(x_train, y_train)
    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    print(f'Training total time: {total_time / 1000000000}s.')

    export_weights(svm, join(config.models_dir, 'svm_weights.joblib'))

def load():
    global svm
    svm = load_weights(join(config.models_dir, 'svm_weights.joblib'))

def predict(input_data):
    return svm.predict(input_data)
