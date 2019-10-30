import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# separates the image by yellow colors
def filter_yellow_color(img):
    shape = img.shape
    filtered = np.full((shape[:2]), False)

    min_rgb = [60, 30, 0]
    green_to_red_min = 0.52
    green_to_red_max = 0.83
    blue_to_green_min = 0.0
    blue_to_green_max = 0.73

    for y in range(shape[0]):
        for x in range(shape[1]):
            pixel = img[y][x]
            # segment image by yellow color
            # the goal is to try and find traffic lights
            # which are primarily a bright yellow
            # yellow colors are formed when there is
            # an equal balance of red and green and little blue

            if np.all(np.greater(pixel, min_rgb)):
                green_to_red = pixel[1] / pixel[0]
                blue_to_green = pixel[2] / pixel[1]
                if (green_to_red >= green_to_red_min and
                    green_to_red <= green_to_red_max and
                    blue_to_green >= blue_to_green_min and
                    blue_to_green <= blue_to_green_max):
                    filtered[y][x] = True
    return filtered

def rgb_to_hue(rgb):
    norm = rgb / 255.0
    c_max = np.argmax(norm)
    c_min = np.argmin(norm)
    delta = norm[c_max] - norm[c_min]
    if delta == 0:
        return 0
    elif c_max == 0:
        return 60 * (((norm[1] - norm[2]) / delta) % 6)
    elif c_max == 1:
        return 60 * (((norm[2] - norm[0]) / delta) + 2)
    elif c_max == 2:
        return 60 * (((norm[0] - norm[1]) / delta) + 4)

def rgb_img_to_hue(img):
    return np.apply_along_axis(rgb_to_hue, 2, img)

# Sets all pixels that do not fall in the specified hue range to black.
def apply_hue_filter(img, min, max):
    hues = rgb_img_to_hue(img)
    shape = img.shape
    for y in range(shape[0]):
        for x in range(shape[1]):
            if hues[y][x] < min or hues[y][x] > max:
                img[y][x] = [0, 0, 0]

def filter_to_image(filter):
    filter_shape = filter.shape
    shape = (filter_shape[0], filter_shape[1], 3)
    img = np.full((filter_shape[0], filter_shape[1], 3), 0, dtype='uint8')
    for y in range(shape[0]):
        for x in range(shape[1]):
            if filter[y][x]:
                img[y][x] = [255, 255, 0]
    return img

# applies a 3x3 median filter to the input and then
# uses a flood fill to remove small patches of yellow
def remove_noise(filter):
    shape = filter.shape
    filtered = np.full(shape, False)
    # note that the edges are False
    for y in range(1, shape[0] - 1):
        for x in range(1, shape[1] - 1):
            num_true = 0
            for i in range(y - 1, y + 2):
                for j in range(x - 1, x + 2):
                    if filter[i][j]:
                        num_true += 1
            if num_true >= 5:
                filtered[y][x] = True
    """
    # uses a flood fill to remove small patches of yellow
    visited = np.full(shape, False)
    flood_remove_threshold = 16
    cells = []
    # (x, y)
    cells.append((0, 0))
    while len(cells) != 0:
        c = cells.pop()
        if visited[c[0]][c[1]]:
            continue
        # left cell
        if c[0] > 0 and not visited[c[0] - 1][c[1]]:
            cells.append((c[0] - 1, c[1]))
        # right cell
        if c[0] < shape[1] - 1 and not visited[c[0] + 1][c[1]]:
            cells.append((c[0] + 1, c[1]))
        # left cell
        if c[0] > 0 and not visited[c[0] - 1][c[1]]:
            cells.append((c[0] - 1, c[1]))
        # right cell
        if c[0] < shape[1] - 1 and not visited[c[0] + 1][c[1]]:
            cells.append((c[0] + 1, c[1]))
    """

    return filtered

# propose bounding boxes for traffic lights
# by flood-filling a filtered image and finding
# large regions of yellow
def propose_boxes(img):
    shape = img.shape
    visited = np.full((img.shape[:2]), False)
    cells = []
    cells.append((0, 0))
    while len(cells) != 0:
        current_cell = cells.pop()


def apply_boxes(img, boxes):
    pass

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('No image specified')
    else:
        img = np.asarray(Image.open(sys.argv[1]))

        print('Image shape:', img.shape)
        print('Image dtype:', img.dtype)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 8))
        ax1.imshow(img)
        ax1.axis('off')

        img2 = np.copy(img)
        filter = filter_yellow_color(img2)

        ax2.imshow(filter_to_image(filter))
        ax2.axis('off')

        #filtered = remove_noise(filtered)
        #filtered = remove_noise(filtered)
        #ax3.imshow(filter_to_image(filtered))
        #ax3.axis('off')

        plt.show()
        """img = np.asarray(Image.open(sys.argv[1]))

        print('Image shape:', img.shape)
        print('Image dtype:', img.dtype)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 8))
        ax1.imshow(img)
        ax1.axis('off')

        img2 = np.copy(img)
        apply_hue_filter(img2, 40, 46)

        ax2.imshow(img2)
        ax2.axis('off')

        #filtered = remove_noise(filtered)
        #filtered = remove_noise(filtered)
        #ax3.imshow(filter_to_image(filtered))
        #ax3.axis('off')

        plt.show()"""
