"""Tool for hard negative mining.


"""

import convnet
import region
import sys
import os
import config
import classifier
from skimage import transform
from keras.preprocessing import image

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

        print('Is this a positive training example? (y/n): ', end ='')
        plt.imshow(image.array_to_img(box_img))
        plt.show()



def generate_box(img):
    """Generates random bounding boxes for an image.

    The bounding box dimensions comprise of random integers in the range
    [32, 128).

    Returns:
        (min_x, min_y, max_x, max_y)
    """

    min_size = 32
    max_size = 128
    height, width = img.shape

    min_x = randint(0, width - min_size - 1)
    min_y = randint(0, height - min_size - 1)
    max_x = min_size + randint(min_size, max_size - 1)
    max_y = max_size + randint(min_size, max_size - 1)

    if max_x >= width:
        max_x = width - 1
    if max_y >= height:
        max_y = height - 1

    return (min_x, min_y, max_x, max_y)



if __name__ == '__main__':
    mine(os.path.expanduser(sys.argv[1]))
