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

    # segmented_img = color.label2rgb(segments, image=img, kind='avg')
    # plt.imshow(image.array_to_img(segmented_img))
    # plt.show()

    ax, ((f1, f2), (f3, f4)) = plt.subplots(2, 2)
    f1.axis('off')
    f2.axis('off')
    f1.imshow(image.array_to_img(img))
    seg_img = image.array_to_img(color.label2rgb(segments, image=img, kind='avg'))
    f2.imshow(seg_img)
    f3.imshow(seg_img)
    for region in regionprops(segments):
        min_y, min_x, max_y, max_x = region.bbox
        f3.add_patch(patches.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=1,
            fill=False,
            color='red'))

    plt.show()


    # converts regions from skimage.measure.regionprops into BoundingBox objects
    print('Generating initial bounding boxes.')
    timer = time_ns()
    boxes = [BoundingBox(region, img) for region in regionprops(segments)]
    print(f'Done in {(time_ns() - timer) / 1000000000}s.')

    # box = boxes[0]
    # for j in range(len(boxes) - 1, -1, -1):
    #     if compute_iou(box, boxes[j]) > 0.1:


    i = 0
    for i in range(1500, len(boxes) - 1):
        box1 = boxes[i]
        box2 = boxes[i + 1]
        ax, (f1, f2) = plt.subplots(1, 2)
        f1.imshow(image.array_to_img(img[box1.min_y:box1.max_y, box1.min_x:box1.max_x]))
        f1.axis('off')
        f2.imshow(image.array_to_img(img[box2.min_y:box2.max_y, box2.min_x:box2.max_x]))
        f2.axis('off')
        print('color sim: ', compute_color_similarity(box1, box2))
        plt.show()
        # ax = plt.gca()
        # ax.add_patch(patches.Rectangle(
        #     (box.min_x, box.min_y),
        #     box.width,
        #     box.height,
        #     linewidth=1,
        #     fill=False,
        #     color='red'))
        # plt.show()



    # # converted segments into image
    # segmented_img = color.label2rgb(segments, image=img, kind='avg')
    #
    # # extract yellow colors out of both the original image segmented image
    # img_yellow_mask = filter_yellow_color(img)
    # segment_yellow_mask = filter_yellow_color(segmented_img)
    #
    # # combine the two yellow masks into one
    # # this allows for higher recall on the bounding box proposals
    # yellow_mask = img_yellow_mask | segment_yellow_mask

    # prepares region proposals

    # for i in range(len(boxes) - 1, -1, -1):
    #     # regions should contain yellow pixels
    #     box = boxes[i]
    #     num_in_yellow_mask = 0
    #     for y in range(box.min_y, box.max_y):
    #         for x in range(box.min_x, box.max_x):
    #             if yellow_mask[y][x]:
    #                 num_in_yellow_mask += 1
    #     if num_in_yellow_mask < 20:
    #         boxes.pop(i)

    return boxes

    # # propose regions
    # regions = []
    # for region in regionprops(segments):
    #     min_y, min_x, max_y, max_x = region.bbox
    #     regions.append(BoundingBox(min_x, min_y, max_x, max_y))



    # for i in range(10):
    #     region1 = regions[randint(0, len(regions) - 1)]
    #     region2 = regions[randint(0, len(regions) - 1)]
    #     plt.imshow(image.array_to_img(segmented_img))
    #     plt.gca().add_patch(patches.Rectangle(
    #         (region1.min_x, region1.min_y),
    #         region1.width,
    #         region1.height,
    #         fill=False,
    #
    #         edgecolor='red', linewidth=1))
    #     plt.gca().add_patch(patches.Rectangle(
    #         (region2.min_x, region2.min_y),
    #         region2.width,
    #         region2.height,
    #         fill=False,
    #         edgecolor='red', linewidth=1))
    #     color_similarity(img, region1, region2)
    #     plt.show()

    # # TODO: this is highly inefficient..
    # # compare random pairs of regions and attempt to merge them
    # for j in range(3):
    #     for i in range(len(regions) - 1, 1, - 1):
    #         region1 = regions[i]
    #         region2 = regions[i - 1]
    #         iou = compute_iou(region1, region2)
    #         if iou > 0.1:
    #             regions.pop(i)
    #             regions[i - 1] = merge_boxes(region1, region2)
    #     shuffle(regions)
    #
    # plt_rects = []
    # for box in boxes:

        # if region.area > 30:
        #     min_y, min_x, max_y, max_x = region.bbox
        #
            # regions should contain at least some specified ratio of yellow pixels
            # num_in_yellow_mask = 0
            # for y in range(min_y, max_y):
            #     for x in range(min_x, max_x):
            #         if yellow_mask[y][x]:
            #             num_in_yellow_mask += 1
            # yellow_ratio = num_in_yellow_mask / region.area
            # if yellow_ratio >= 0.6:
        #
        # plt_rects.append(
        #     patches.Rectangle(
        #         (box.min_x, box.min_y),
        #         box.max_x - box.min_x,
        #         box.max_y - box.min_y,
        #         fill=False, edgecolor='red', linewidth=1))

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

    timer = time_ns()
    green_to_red = img[:,:,1] / img[:,:,0]
    blue_to_green = img[:,:,2] / img[:,:,1]
    print(green_to_red)
    print((time_ns() - timer) / 1000000000)

    timer = time_ns()
    for y in range(shape[0]):
        for x in range(shape[1]):
            pixel = img[y][x]
            filtered[y][x] = (
                green_to_red[y][x] >= green_to_red_min and
                green_to_red[y][x] <= green_to_red_max and
                blue_to_green[y][x] >= blue_to_green_min and
                blue_to_green[y][x] <= blue_to_green_max))
    print((time_ns() - timer) / 1000000000)
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

def merge_boxes(box1, box2):
    return BoundingBox(
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

    def merge(box):
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
        self.max_x = min(self.max_x, box.max_y)
        self.max_y = min(self.max_y, box.max_x)
        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        self.area = self.width * self.height
        self.region_area += box.region_area
        self.color_histograms += box.color_histograms
        for c in range(channels):
            normalize_arr(self.color_histograms)

    def to_string(self):
        """Returns a string containing details about this BoundingBox.

        Useful for debugging purposes.
        """

        return(f'BoundingBox(min_x: {self.min_x}, min_y: {self.min_y}, max_x: {self.max_x}, max_y: {self.max_y}), width: {self.width}, height: {self.height}, area: {self.area}, bounded_area: {self.bounded_area})')
