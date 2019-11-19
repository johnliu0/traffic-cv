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

    # segment image using Felzenszwalb's Image Segmentation algorithm
    print('Applying Felzenszwalb\'s algorithm.')
    timer = time_ns()
    segments = segmentation.felzenszwalb(img, min_size=30, scale=10)
    print(f'Done in {(time_ns() - timer) / 1000000000}s.')

    print('Creating yellow filter.')
    timer = time_ns()
    img_yellow_mask = filter_yellow_color(img)
    print(f'Done in {(time_ns() - timer) / 1000000000}s.')

    # converts regions from skimage.measure.regionprops into BoundingBox objects
    print('Generating initial bounding boxes.')
    timer = time_ns()
    boxes = []
    for region in regionprops(segments):
        min_y, min_x, max_y, max_x = region.bbox
        if np.any(img_yellow_mask[min_y:max_y,min_x:max_x]):
            boxes.append(BoundingBox(region, img))
    print(f'Done in {(time_ns() - timer) / 1000000000}s.')

    # ax, (f1, f2) = plt.subplots(1, 2)
    # f1.axis('off')
    # f1.imshow(image.array_to_img(img))
    # f2.axis('off')
    # f2.imshow(image.array_to_img(img))
    #
    # for box in boxes:
    #     f1.add_patch(patches.Rectangle(
    #         (box.min_x, box.min_y),
    #         box.width,
    #         box.height,
    #         linewidth=1,
    #         fill=False,
    #         color='red'))

    print('Merging bounding boxes based on color similarity.')
    timer = time_ns()
    for passes in range(7):
        i = 0
        while i < len(boxes):
            box1 = boxes[i]
            for j in range(len(boxes) - 1, i, -1):
                box2 = boxes[j]
                if intersects(box1, box2):
                    if compute_color_similarity(box1, box2) > 0.8:
                        box1.merge(box2)
                        boxes.pop(j)
            i += 1
    print(f'Done in {(time_ns() - timer) / 1000000000}s.')

    # for box in boxes:
    #     f2.add_patch(patches.Rectangle(
    #         (box.min_x, box.min_y),
    #         box.width,
    #         box.height,
    #         linewidth=1,
    #         fill=False,
    #         color='red'))
    #
    # plt.show()

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

def normalize_arr(arr):
    """Normalizes a numpy array.

    All values in the array are divided by the sum of all values. This produces
    an array whose sum is exactly 1. If an array contains only zeros, it will
    not be modified.
    """

    arr_sum = sum(arr)
    if arr_sum != 0:
        arr /= arr_sum

def compute_iou(box1, box2):
    """Computes the intersection over union.

    The intersection and union are computed as number of pixels in the
    rectangular region that intersect over the union.

    Returns:
        A value between 0 and 1 representing the intersection over union.
    """

    # compute the amount of intersectionality in each axis
    # a negative value indicates that the regions do not intersect on the axis
    # a positive value indicates the amount that the regions intersect
    # i.e. the length of intersection of two line segments on a line
    x_intersection_len = (box1.width + box2.width) - (max(box1.max_x, box2.max_x) - min(box1.min_x, box2.min_x))
    y_intersection_len = (box1.height + box2.height) - (max(box1.max_y, box2.max_y) - min(box1.min_y, box2.min_y))
    if x_intersection_len > 0.0 and y_intersection_len > 0.0:
        intersection_area = x_intersection_len * y_intersection_len
        # inclusion-exclusion principle gives the union of two intersecting
        # regions A, B as A + B - C where C is the intersection of A and B
        return intersection_area / (box1.area + box2.area - intersection_area)
    return 0.0

def compute_color_histograms(img, nbins=16, normalize=False):
    """Computes a color histogram for a given image.

    Each color channel in the image is computed separately. An array of the
    histogram computed for each channel is returned.

    Args:
        img: A numpy image in the form (height, width, 3).
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
        for y in range(height):
            for x in range(width):
                hist[c,int(img[y,x,c] / nbins)] += 1.0
    if normalize:
        for c in range(channels):
            normalize_arr(hist[c])
    return hist

def compute_color_similarity(box1, box2):
    """Computes the similarity in color between two regions.

    Compares the color histograms in the provided BoundingBox's separately for
    each channel.

    Returns:
        A value between 0.0 and 1.0 representing the color similarity, where
        0.0 indicates no similarity, and 1.0 indicates equal color distribution.
    """

    score = 0
    for i in range(BoundingBox.color_hist_nbins):
        # take the percentage filled by the smaller bar in each histogram
        r1, r2 = box1.color_histograms[0][i], box2.color_histograms[0][i]
        g1, g2 = box1.color_histograms[1][i], box2.color_histograms[1][i]
        b1, b2 = box1.color_histograms[2][i], box2.color_histograms[2][i]
        score += min(r1, r2)
        score += min(g1, g2)
        score += min(b1, b2)

    return score / 3

def intersects(box1, box2):
    """Determines whether or not two boxes intersect.

    Intersection is true only when the boxes share at least one pixel. Boxes
    that are touching along their edges are not considered intersecting.

    Returns:
        A boolean representing whether or not two boxes intersect.
    """

    if ((box2.min_x > box1.min_x and box2.min_x < box1.max_x) or
        (box1.min_x > box2.min_x and box1.min_x < box2.max_x)):
        if ((box2.min_y > box1.min_y and box2.min_y < box1.max_y) or
            (box1.min_y > box2.min_y and box1.min_y < box2.max_y)):
                return True
    return False

def measure_region_fill(box1, box2):
    """Computes how well two regions fit into eachother.

    Returns:
        A value between 0.0 and 1.0 specifying the ratio of the combined shape
        area to the bounding box area.
    """

    min_x, min_y, max_x, max_y = fit_bounding_box(box1, box2)
    return ((box1.region_area + box2.region_area)
        / ((max_x - min_x) * (max_y - min_y)))

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

    color_hist_nbins = 16

    def __init__(self, region, img):
        """
        Args:
            region: a region object from skimage.measure.regionprops
        """

        min_y, min_x, max_y, max_x = region.bbox
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.width = max_x - min_x
        self.height = max_y - min_y
        self.area = self.width * self.height
        self.region_area = region.area
        self.color_histograms = compute_color_histograms(img[min_y:max_y,min_x:max_x], nbins=BoundingBox.color_hist_nbins, normalize=True)

    def merge(self, box):
        """Merges another BoundingBox into this object.

        Dimensions and min and max coordinates are expanded to fit both boxes.
        The color histograms are added together and normalized. The attributes
        of this object will change, the BoundingBox provided in the argument is
        not modified.

        Args:
            box: BoundingBox to merge into this one.
        """

        self.min_x = min(self.min_x, box.min_x)
        self.min_y = min(self.min_y, box.min_y)
        self.max_x = max(self.max_x, box.max_x)
        self.max_y = max(self.max_y, box.max_y)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        self.area = self.width * self.height
        self.region_area += box.region_area
        self.color_histograms += box.color_histograms
        for c in range(self.color_histograms.shape[0]):
            normalize_arr(self.color_histograms[c])

    def to_string(self):
        """Returns a string containing details about this BoundingBox.

        Useful for debugging purposes.
        """

        return(f'BoundingBox(min_x: {self.min_x}, min_y: {self.min_y}, max_x: {self.max_x}, max_y: {self.max_y}), width: {self.width}, height: {self.height}, area: {self.area}, bounded_area: {self.bounded_area})')
