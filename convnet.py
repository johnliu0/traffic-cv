import matplotlib.pyplot as plt
import numpy as np
import config
from os.path import expanduser
from keras.preprocessing import image
from keras.applications import VGG16
from PIL import Image as PILImage

# use the pretrained VGG16 convnet as a base
print('Loading VGG16 convnet base')
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=config.convnet_image_input_size + (3,))

def extract_features(img):
    return conv_base.predict(img).flatten()

def get_output_length():
    """Returns the length of the convnet output."""
    return conv_base.layers[-1].output_shape[-1]
