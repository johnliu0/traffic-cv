"""Image region proposal.

Generates bounding boxes with objectness in an image using a modified selective
search algorithm. Felzenszwalb's image segmentation algorithm is used to break
an image into small areas of potential objectness; some custom metrics are used
to merge like-regions into larger regions.

    Usage example:

    # where img is is a numpy array of shape (height, width, 3)
    region.propose_boxes(img)

"""

import convnet
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import time_ns
from keras.preprocessing import image
from random import randint, shuffle
import skimage.exposure as exposure
from skimage import segmentation, color, data
from skimage.measure import regionprops
from skimage.future import graph
from PIL import Image

def propose_boxes(img):
    """Proposes bounding boxes for a given image.

    A selective search based custom algorithm is used here to generate bounding
    boxes. Felzenszwalb's algorithm segments the image into tiny boxes which are
    then combined through a variety of metrics into larger boxes.

    Args:
        img: An nparray image of any size but with 3 color channels (rgb).

    Returns:
        An array of BoundingBox objects corresponding to potential objects in
        the image.
    """

    img_area = img.shape[0] * img.shape[1]
    total_time = 0

    # segment image using Felzenszwalb's Image Segmentation algorithm
    print('Applying Felzenszwalb\'s algorithm. ', end='')
    timer = time_ns()
    segments = segmentation.felzenszwalb(img, min_size=50)
    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    # convert segments to an image
    # print('Converting segments to image.')
    # timer = time_ns()
    # segmented_img = color.label2rgb(segments, image=img, kind='avg')
    # plt.subplot(121)
    # plt.imshow(image.array_to_img(img))
    # plt.subplot(122)
    # plt.imshow(image.array_to_img(segmented_img))
    # plt.show()
    # print(f'Done in {(time_ns() - timer) / 1000000000}s.')

    # create a mask that indicates the presence of yellow pixels
    print('Creating yellow filter. ', end='')
    timer = time_ns()
    img_yellow_mask = filter_yellow_color(img)
    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    # converts regions from skimage.measure.regionprops into BoundingBox objects
    print('Generating initial bounding boxes. ', end='')
    timer = time_ns()
    boxes = []
    for region in regionprops(segments):
        min_y, min_x, max_y, max_x = region.bbox

        # traffic lights are tall rectangles
        height_width_ratio = (max_y - min_y) / (max_x - min_x)
        if height_width_ratio < 1.0 or height_width_ratio > 4.5:
            continue

        # traffic lights are yellow in my area
        # therefore the bounding box should contain yellow pixels
        if np.any(img_yellow_mask[min_y:max_y,min_x:max_x]):
            boxes.append(BoundingBox(
                min_x, min_y, max_x, max_y,
                region.area,
                compute_color_histogram(img, region.coords, normalize=True)))
    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    # print(len(boxes))
    # plt.gcf().set_size_inches(16, 9)
    # plt.imshow(image.array_to_img(img))
    # for idx, box in enumerate(boxes):
    #     plt.gca().add_patch(patches.Rectangle(
    #         (box.min_x, box.min_y),
    #         box.width,
    #         box.height,
    #         linewidth=1,
    #         color='red',
    #         fill=False))
    # plt.show()

    # generate an initial set of similarity scores for all pairs of boxes
    # similarity data is in the format (i, j, score)
    # where i, j denote the indices of the box they refer to
    print('Generating initial set of similarity scores. ', end='')
    timer = time_ns()
    simil_set = []
    visited = [False] * len(boxes)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            simil_set.append((i, j, compute_similarity(boxes[i], boxes[j], img_area)))
    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    # continuously merge the two most similar regions until only one remains
    print('Merging regions. ', end='')
    timer = time_ns()
    while len(simil_set) > 0:
        # find the two most similar regions
        idx_simil_max = 0
        for i in range(1, len(simil_set)):
            if simil_set[i][2] > simil_set[idx_simil_max][2]:
                idx_simil_max = i
        idx1 = simil_set[idx_simil_max][0]
        idx2 = simil_set[idx_simil_max][1]

        # find and remove all similarities that used either of the two boxes
        for i in range(len(simil_set) - 1, -1, -1):
            if (simil_set[i][0] == idx1 or simil_set[i][0] == idx2 or
                simil_set[i][1] == idx1 or simil_set[i][1] == idx2):
                simil_set.pop(i)

        # merge the two boxes together
        new_box = merge_boxes(boxes[idx1], boxes[idx2])

        # mark these two boxes as visited
        visited[idx1] = True
        visited[idx2] = True

        # generate new similarities with merged box
        for i in range(len(boxes)):
            if not visited[i]:
                simil_set.append((i, len(boxes), compute_similarity(new_box, boxes[i], img_area)))

        # add merged box to boxes list and visited list
        boxes.append(new_box)
        visited.append(False)
    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    # further remove bounding boxes unlikely to be traffic lights
    print('Pruning regions. ', end='')
    timer = time_ns()
    for i in range(len(boxes) - 1, -1, -1):
        # traffic lights are tall rectangles
        height_width_ratio = boxes[i].height / boxes[i].width
        if height_width_ratio < 1.0 or height_width_ratio > 4.5:
            boxes.pop(i)
    elapsed = time_ns() - timer
    total_time += elapsed
    print(f'Done in {elapsed / 1000000000}s.')

    print(f'Region proposal total time: {total_time / 1000000000}s.')
    return boxes

def filter_yellow_color(img):
    """Produces a yellow color mask.

    A 2d array (size equal to provided image) of booleans is produced, where
    true indicates a pixel that is of yellow hue. Since traffic lights (at least
    in Toronto) are yellow; we can significantly speed up the region proposal
    algorithm by only looking at boxes that contain yellow pixels; since boxes
    that do not contain yellow pixels cannot possibly contain a traffic light.

    Pixels are considered yellow according to a very loose set of ratios that
    define how a yellow pixel should look in terms of RGB components.

    Returns:
        A 2d boolean array with size equal to provided image representing
        whether or not a pixel is of a yellow hue (like the traffic lights in
        Toronto).
    """

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

    # get the ratios of green to red and blue to green for all pixels
    # clip the divisor to avoid divide by zero error
    green_to_red = img[:,:,1] / np.clip(img[:,:,0], 1.0, 255.0)
    blue_to_green = img[:,:,2] / np.clip(img[:,:,1], 1.0, 255.0)

    # mask for pixels that are in the accepted green to red ratio range
    green_to_red_mask = np.bitwise_and(green_to_red >= green_to_red_min,
        green_to_red <= green_to_red_max)

    # mask for pixels that are in the accepted blue to green ratio range
    blue_to_green_mask = np.bitwise_and(blue_to_green >= blue_to_green_min,
        blue_to_green <= blue_to_green_max)

    # mask for pixels that are at least the minimum rgb
    min_rgb_mask = np.bitwise_and(
        np.bitwise_and(img[:,:,0] > min_rgb[0], img[:,:,1] > min_rgb[1]),
        img[:,:,2] > min_rgb[2])

    # finally combine all previous filters to produce a yellow filter
    filter = np.bitwise_and(
        np.bitwise_and(green_to_red_mask, blue_to_green_mask),
        min_rgb_mask)

    return filter

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

def compute_similarity(box1, box2, img_area):
    """Computes a score for similarity between two boxes.

    Args:
        box1: A BoundingBox object.
        box2: A BoundingBox object.
        img_area: The number of pixels in the original image.

    Returns:
        A score between 0.0 and 2.0.
    """

    # the fill, size, and color metrics as described in the original paper are
    # used here, the texture metric is not used
    # note that the size and fill metric was combined
    min_x, min_y, max_x, max_y = fit_bounding_box(box1, box2)
    size_and_fill_score = 1 - (max_x - min_x) * (max_y - min_y) / (img_area)
    color_score = np.sum(np.minimum(box1.color_histogram, box2.color_histogram))
    return size_and_fill_score + color_score

def compute_color_histogram(img, coords, nbins=25, normalize=False):
    """Computes a color histogram for a given image and a list of coordinates
    of pixels to use for the histogram.

    Each color channel in the image is computed separately. An array of the
    histogram computed for each channel is returned.

    Args:
        img: A numpy image in the form (height, width, 3).
        coords: Coordinates list of the form (row, col).
        n_bins: Number of parts to divide the histogram data into.

    Returns:
        An array containing histogram data for each color channel in the form
        [[0...nbins], [0...nbins], [0...nbins]], representing r, g,
        and b respectively.
    """

    height, width, channels = img.shape
    bin_width = 256.0 / nbins
    hist = np.zeros((channels, nbins), dtype='float64')
    for c in range(channels):
        for y, x in coords:
            hist[c,int(img[y,x,c] / nbins)] += 1.0
    if normalize:
        arr_sum = np.sum(hist)
        if arr_sum != 0:
            hist /= arr_sum
    return hist

def fit_bounding_box(box1, box2):
    """Computes the minimum enclosing box of the two given boxes.

    Returns:
        (min_x, min_y, max_x, max_y)
    """

    return (
        min(box1.min_x, box2.min_x),
        min(box1.min_y, box2.min_y),
        max(box1.max_x, box2.max_x),
        max(box1.max_y, box2.max_y))

def merge_boxes(box1, box2):
    """Merges two BoundingBox objects together.

    Args:
        box1: A BoundingBox object.
        box2: A BoundingBox object.
    Returns:
        A merged BoundingBox.
    """

    return BoundingBox(
        *fit_bounding_box(box1, box2),
        box1.region_area + box2.region_area,
        (float(box1.region_area) * box1.color_histogram +
            float(box2.region_area) * box2.color_histogram) /
            float(box1.region_area + box2.region_area))

class BoundingBox:
    """Contains region information.

    Stores information about a bounding box; a rectangle that represents some
    area in an image.

    Args:
        region: A region object from skimage.measure.regionprops
        img: The full sized original image used.

    Attributes:
        color_hist_nbins: Number of bins to use for color histogram calculation.
        min_x: x coordinate of the left bounds of the box in pixels.
        min_y: y coordinate of the upper bounds of the box in pixels.
        max_x: x coordinate of the right bounds of the box in pixels.
        max_y: y coordinate of the bottom bounds of the box in pixels.
        width: Width of box in pixels.
        height: Height of box in pixels.
        area: Number of pixels contained by the region.
        bounded_area: Number of pixels contained by the box (width * height).
        color_histograms: A normalized histogram of color distribution for each
            channel.
    """

    def __init__(self, min_x, min_y, max_x, max_y, region_area, color_histogram):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.area = self.width * self.height
        self.region_area = region_area
        self.color_histogram = color_histogram

    def to_string(self):
        """Returns a string containing details about this BoundingBox.

        Useful for debugging purposes.
        """

        return(f'BoundingBox(min_x: {self.min_x}, min_y: {self.min_y}, max_x: {self.max_x}, max_y: {self.max_y}), width: {self.width}, height: {self.height}, area: {self.area}, bounded_area: {self.bounded_area})')


# def intersects(box1, box2):
#     """Determines whether or not two boxes intersect.
#
#     Intersection is true only when the boxes share at least one pixel. Boxes
#     that are touching along their edges are not considered intersecting.
#
#     Returns:
#         A boolean representing whether or not two boxes intersect.
#     """
#
#     if ((box2.min_x > box1.min_x and box2.min_x < box1.max_x) or
#         (box1.min_x > box2.min_x and box1.min_x < box2.max_x)):
#         if ((box2.min_y > box1.min_y and box2.min_y < box1.max_y) or
#             (box1.min_y > box2.min_y and box1.min_y < box2.max_y)):
#                 return True
#     return False

# def compute_iou(box1, box2):
#     """Computes the intersection over union.
#
#     The intersection and union are computed as number of pixels in the
#     rectangular region that intersect over the union.
#
#     Returns:
#         A value between 0.0 and 1.0 representing the intersection over union.
#     """
#
#     # compute the amount of intersectionality in each axis
#     # a negative value indicates that the regions do not intersect on the axis
#     # a positive value indicates the amount that the regions intersect
#     # i.e. the length of intersection of two line segments on a line
#     x_intersection_len = (box1.width + box2.width) - (max(box1.max_x, box2.max_x) - min(box1.min_x, box2.min_x))
#     y_intersection_len = (box1.height + box2.height) - (max(box1.max_y, box2.max_y) - min(box1.min_y, box2.min_y))
#     if x_intersection_len > 0.0 and y_intersection_len > 0.0:
#         intersection_area = x_intersection_len * y_intersection_len
#         # inclusion-exclusion principle gives the union of two intersecting
#         # regions A, B as A + B - C where C is the intersection of A and B
#         return intersection_area / (box1.area + box2.area - intersection_area)
#     return 0.0
