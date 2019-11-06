import augment
import convnet
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump as export_weights, load as load_weights
from os.path import expanduser
from sklearn.svm import OneClassSVM
from keras.preprocessing import image

weights_file_name = 'svm_weights.joblib'
target_size = (150, 150)
home = expanduser('~')
svm = OneClassSVM(gamma='auto')

def train():
    global svm

    num_augmentations = 6
    x_train = []
    for i in range(175):
        print(f'Loading light_{i}.jpg')
        imgs = augment.generate(f'{home}/Google Drive/tcv/images/traffic_lights/light_{i}.png', target_size=target_size, num_imgs=num_augmentations)
        predictions = []
        for j in range(num_augmentations):
            prediction = convnet.predict(imgs[j].reshape((1,) + imgs[j].shape))
            predictions.append(prediction)
        x_train.extend(predictions)

    x_train = np.asarray(x_train)
    svm.fit(x_train)
    export_weights(svm, weights_file_name)

def load():
    global svm
    svm = load_weights(weights_file_name)

load()
