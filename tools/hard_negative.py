"""Tool for hard negative mining.


"""

import ai
from keras.preprocessing import image

def mine(img_path):
    img = image.load_img(img_path)


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
