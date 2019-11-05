import matplotlib.pyplot as plt
import numpy as np
from os.path import expanduser
from keras.preprocessing import image
from keras.applications import VGG16
from PIL import Image as PILImage

IMG_SIZE = (150, 150, 3)

# use the pretrained VGG16 convnet as a base
print('Loading VGG16 convnet base')
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE)

home = expanduser('~')
for i in range(81):
    print(f'Processing image: light_{i}.png')
    img = PILImage.open(f'{home}/Google Drive/tcv/images/traffic_lights/light_{i}.png')

    # resize image to target size
    img = img.resize(IMG_SIZE[:2])

    # turn PIL image into numpy array
    img_data = image.img_to_array(img)

    # normalize data from 0-255 to 0-1
    img_data /= 255.0

    # add a batch axis
    img_data.resize((1,) + img_data.shape)

    # extract features from image with VGG16
    prediction = conv_base.predict(img_data).flatten()
    feature_vector_length = prediction.shape[0]
    print(prediction)
    print(feature_vector_length)
