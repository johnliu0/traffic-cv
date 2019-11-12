import convnet
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import randint
from skimage import segmentation, color, data
from skimage.measure import regionprops
from skimage.future import graph
from PIL import Image

class BoundingBox:
    def __init__(self, min_x, min_y, max_x, max_y, yellow_ratio):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.yellow_ratio = yellow_ratio

def propose_boxes(img):
    # segment image using Felzenszwalb's Image Segmentation algorithm
    segments = segmentation.felzenszwalb(img, min_size=40, scale=100)
    segmented_img = color.label2rgb(segments, img, kind='avg')

    # extract yellow colors out of both the original image segmented image
    img_yellow_mask = filter_yellow_color(img)
    segment_yellow_mask = filter_yellow_color(segmented_img)

    # combine the two yellow masks into one
    # this allows for higher recall on the bounding box proposals
    yellow_mask = img_yellow_mask | segment_yellow_mask

    # propose regions
    rects = []
    for region in regionprops(segments):
        if region.area > 30:
            min_y, min_x, max_y, max_x = region.bbox

            # regions should contain at least some specified ratio of yellow pixels
            num_in_yellow_mask = 0
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if yellow_mask[y][x]:
                        num_in_yellow_mask += 1
            yellow_ratio = num_in_yellow_mask / region.area
            if yellow_ratio >= 0.6:
                rects.append(BoundingBox(min_x, min_y, max_x, max_y, yellow_ratio))


    plt_rects = []
    for rect in rects:
        plt_rects.append(
            patches.Rectangle(
                (rect.min_x, rect.min_y),
                rect.max_x - rect.min_x,
                rect.max_y - rect.min_y,
                fill=False, edgecolor='red', linewidth=1))

    return (segmented_img, plt_rects, img_yellow_mask, segment_yellow_mask, yellow_mask)

# separates the image by yellow colors
# returns an ndarray of booleans representing whether or not a pixel is of a
# particular yellow hue
def filter_yellow_color(img):
    shape = img.shape
    filtered = np.full((shape[:2]), False)
    # these values were obtained manually
    # the colors of a traffic light generally seem to fit
    # within the below paramaters
    # note that the green_to_red and blue_to_green variables
    # refer to the ratio of the colors
    min_rgb = [60, 30, 0]
    green_to_red_min = 0.52
    green_to_red_max = 0.83
    blue_to_green_min = 0.0
    blue_to_green_max = 0.73
    for y in range(shape[0]):
        for x in range(shape[1]):
            pixel = img[y][x]
            if np.all(np.greater(pixel, min_rgb)):
                green_to_red = pixel[1] / pixel[0]
                blue_to_green = pixel[2] / pixel[1]
                if (green_to_red >= green_to_red_min and
                    green_to_red <= green_to_red_max and
                    blue_to_green >= blue_to_green_min and
                    blue_to_green <= blue_to_green_max):
                    filtered[y][x] = True
    return filtered

# converts a filter mask to image
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
def remove_noise(filter, in_place=False):
    shape = filter.shape
    filtered = filter if in_place else np.full(shape, False)
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
    return filtered

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('No image specified')
    else:
        img = np.asarray(Image.open(sys.argv[1]))

        print('Image shape:', img.shape)
        print('Image dtype:', img.dtype)

        f, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes[0, 0].imshow(img)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(img)
        axes[0, 1].axis('off')

        felzen_img, rects, mask1, mask2, mask3 = propose_boxes(img)
        axes[0, 2].imshow(felzen_img)
        img_regions = []
        for rect in rects:
            axes[0, 1].add_patch(rect)
            img_regions.append(img[rect.get_xy()[1]:(rect.get_xy()[1] + rect.get_height()), rect.get_xy()[0]:(rect.get_xy()[0] + rect.get_width())].resize((150, 150), Image.NEAREST))

        print(img_regions)

        axes[0, 2].axis('off')

        axes[0, 2].axis('off')

        axes[1, 0].imshow(filter_to_image(mask1))
        axes[1, 0].axis('off')

        axes[1, 1].imshow(filter_to_image(mask2))
        axes[1, 1].axis('off')

        axes[1, 2].imshow(filter_to_image(mask3))
        axes[1, 2].axis('off')

        plt.show()
