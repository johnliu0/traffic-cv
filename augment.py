import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

# data augmentation with Keras ImageDataGenerator

data_gen = image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255)

def generate(
    img_path,
    target_size,
    num_imgs=5):
    """Loads an image and generates random augmentations."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)

    augmentations = []
    i = 0
    for batch in data_gen.flow(x, batch_size=1):
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



# data_gen = image.ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.8,
#     horizontal_flip=True,
#     fill_mode='nearest')
#
# if len(sys.argv) != 2:
#     print('Please specify one file path.')
# else:
#     file_path = sys.argv[1]
#     img = image.load_img(file_path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     i = 0
#     for batch in data_gen.flow(x, batch_size=1):
#         plt.figure(i)
#         img_plot = plt.imshow(image.array_to_img(batch[0]))
#         i += 1
#         if i % 10 == 0:
#             break
#     plt.show()
