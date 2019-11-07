import augment
import convnet
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump as export_weights, load as load_weights
from os.path import expanduser
from sklearn.svm import OneClassSVM
from keras.preprocessing import image
from PIL import Image as PILImage

home = expanduser('~')
weights_file_name = f'{home}/Google Drive/tcv/svm_weights.joblib'
target_size = (150, 150)
svm = OneClassSVM(gamma='auto')

def train():
    global svm

    num_augmentations = 1
    x_train = []
    for i in range(175):
        print(f'Loading light_{i}.jpg')
        imgs = augment.generate(f'{home}/Google Drive/tcv/images/traffic_lights/light_{i}.png', target_size=target_size, num_imgs=num_augmentations)
        conv_outputs = []
        for j in range(num_augmentations):
            conv_output = convnet.predict(imgs[j].reshape((1,) + imgs[j].shape))
            conv_outputs.append(conv_output)
        x_train.extend(conv_outputs)

    x_train = np.asarray(x_train)
    svm.fit(x_train)
    export_weights(svm, weights_file_name)

def load():
    global svm
    svm = load_weights(weights_file_name)
    img = PILImage.open(f'{home}/Desktop/test_image.png')
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    img = img.reshape((1,) + img.shape)
    img /= 255.0

    conv_output = convnet.predict(img)
    conv_output = conv_output.reshape((1,) + conv_output.shape)

    y = svm.predict(conv_output)
    print(y)


load()
