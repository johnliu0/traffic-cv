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
from keras.preprocessing import image
from random import randint, shuffle
import skimage.exposure as exposure
from skimage import segmentation, color, data
from skimage.measure import regionprops
from skimage.future import graph
from PIL import Image
from boundingbox import BoundingBox

def propose_boxes(img):
    """Proposes bounding boxes for a given image.

    A selective search based custom algorithm is used here to generate bounding
    boxes. Felzenszwalb's algorithm segments the image into tiny boxes which are
    then combined through a variety of metrics into larger boxes.

    Args:
        img: an nparray image of any size but with 3 color channels (rgb)

    Returns:
        An array of BoundingBox objects corresponding to potential objects in
        the image.
    """

    # segment image using Felzenszwalb's Image Segmentation algorithm
    segments = segmentation.felzenszwalb(img, min_size=30, scale=100)

    # converted segments into image
    segmented_img = color.label2rgb(segments, image=img, kind='avg')

    # extract yellow colors out of both the original image segmented image
    img_yellow_mask = filter_yellow_color(img)
    segment_yellow_mask = filter_yellow_color(segmented_img)

    # combine the two yellow masks into one
    # this allows for higher recall on the bounding box proposals
    yellow_mask = img_yellow_mask | segment_yellow_mask

    # prepares region proposals
    boxes = [BoundingBox(region) for region in regionprops(segments)]

    for i in range(len(boxes) - 1, -1, -1):
        # regions should contain yellow pixels
        box = boxes[i]
        num_in_yellow_mask = 0
        for y in range(box.min_y, box.max_y):
            for x in range(box.min_x, box.max_x):
                if yellow_mask[y][x]:
                    num_in_yellow_mask += 1
        if num_in_yellow_mask < 20:
            boxes.pop(i)

    ax = plt.gca()
    plt.imshow(image.array_to_img(img))
    for box in boxes:
        ax.add_patch(patches.Rectangle(
            (box.min_x, box.min_y),
            box.width,
            box.height,
            linewidth=1,
            fill=False,
            color='red'
        ))
    plt.show()

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

def color_similarity(img, box1, box2, nbins=25):
    """Computes the similarity in color between two regions.

    A normalized histogram is produced from each one of the three RGB channels
    of both images. The sum of all intersections of each bin divided by the
    number of bins is the color similarity.

    Returns:
        A value between 0.0 and 1.0 representing the color similarity.
    """

    hist_red1, bins_red1 = exposure.histogram(img[box1.min_y : box1.max_y, box1.min_y : box1.max_y, 0], nbins=nbins)
    hist_green1, bins_green1 = exposure.histogram(img[box1.min_y : box1.max_y, box1.min_y : box1.max_y, 1], nbins=nbins)
    hist_blue1, bins_blue1 = exposure.histogram(img[box1.min_y : box1.max_y, box1.min_y : box1.max_y, 2], nbins=nbins, normalize=True)
    hist_red2, bins_red2 = exposure.histogram(img[box2.min_y : box2.max_y, box2.min_y : box2.max_y, 0], nbins=nbins, normalize=True)
    hist_green2, bins_green2 = exposure.histogram(img[box2.min_y : box2.max_y, box2.min_y : box2.max_y, 1], nbins=nbins, normalize=True)
    hist_blue2, bins_blue2 = exposure.histogram(img[box2.min_y : box2.max_y, box2.min_y : box2.max_y, 2], nbins=nbins, normalize=True)

    print('...')
    sum_red = sum([min(hist_red1[i], hist_red2[i]) for i in range(nbins)])
    sum_green = sum([min(hist_green1[i], hist_green2[i]) for i in range(nbins)])
    sum_blue = sum([min(hist_blue1[i], hist_blue2[i]) for i in range(nbins)])

    print(sum_red)
    print(sum_green)
    print(sum_blue)

    return 0.5


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
