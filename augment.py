import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

# data augmentation with Keras ImageDataGenerator

pos_data_gen = image.ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255)

neg_data_gen = image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.20,
    height_shift_range=0.20,
    shear_range=0.20,
    zoom_range=0.20,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255)

def generate_pos(
    img_path,
    target_size,
    num_imgs=5):
    """Loads an image and generates random augmentations."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    augmentations = []
    i = 0
    for batch in pos_data_gen.flow(x, batch_size=1):
        augmentations.append(batch[0])
        i += 1
        if i >= num_imgs:
            break
    return np.asarray(augmentations)

def generate_neg(
    img_path,
    target_size,
    num_imgs=5):
    """Loads an image and generates random augmentations."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    augmentations = []
    i = 0
    for batch in neg_data_gen.flow(x, batch_size=1):
        augmentations.append(batch[0])
        i += 1
        if i >= num_imgs:
            break
    return np.asarray(augmentations)

def show_augmentations(img_path, target_size):
    """Shows various augmentations of a particular image."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 1
    fig, ax = plt.subplots(4, 5)
    ax[0, 0].imshow(image.array_to_img(img))
    ax[0, 0].axis('off')
    for batch in data_gen.flow(x, batch_size=1):
        ax[i // 5, i % 5].axis('off')
        ax[i // 5, i % 5].imshow(image.array_to_img(batch[0]))
        i += 1
        if i >= 4 * 5:
            break
    plt.show()
