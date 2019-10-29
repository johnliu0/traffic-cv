import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import randint
from skimage import segmentation, color, data
from skimage.measure import regionprops
from skimage.future import graph

BOUNDING_BOX_MIN_AREA = 60

def segment_image(img):
    segments = segmentation.felzenszwalb(img, min_size=BOUNDING_BOX_MIN_AREA, scale=3)
    yellow_mask = filter_yellow_color(img)
    rects = []
    for region in regionprops(segments):
        if region.area > 100:
            min_y, min_x, max_y, max_x = region.bbox
            num_in_yellow_mask = 0
            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    if yellow_mask[y][x]:
                        num_in_yellow_mask += 1
            if num_in_yellow_mask / region.area >= 0.5:
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red', linewidth=1)
                rects.append(rect)

    segmented_img = color.label2rgb(segments, img, kind='avg')
    return (segmented_img, rects)
    """shape = img.shape
    colors = {}
    for y in range(shape[0]):
        for x in range(shape[1]):
            segment = segments[y][x]
            if segment in colors:
                img[y][x] = colors[segment]
            else:
                colors[segment] = [randint(0, 255), randint(0, 255), randint(0, 255)]"""

def segment_image2(img):
    #segments = segmentation.felzenszwalb(img, min_size=BOUNDING_BOX_MIN_AREA, scale=3)
    labels = segmentation.slic(img, compactness=20, n_segments=1200)
    rag_segments = graph.rag_mean_color(img, labels)

    rag_merge_labels = graph.merge_hierarchical(labels, rag_segments, thresh=20, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=weight_mean_color)

    rects = []
    for region in regionprops(rag_merge_labels):
        if region.area > 150:
            min_y, min_x, max_y, max_x = region.bbox
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red', linewidth=1)
            rects.append(rect)

    segmented_img = color.label2rgb(rag_merge_labels, img, kind='avg')
    return (segmented_img, rects)
    """shape = img.shape
    colors = {}
    for y in range(shape[0]):
        for x in range(shape[1]):
            segment = segments[y][x]
            if segment in colors:
                img[y][x] = colors[segment]
            else:
                colors[segment] = [randint(0, 255), randint(0, 255), randint(0, 255)]"""

def merge_mean_color(graph, src, dst):
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def weight_mean_color(graph, src, dst, n):
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

# merges segments by similar colors
def merge_similar_colors(img, threshold=15):
    labels = segmentation.slic(img, compactness=20, n_segments=1500)
    g = graph.rag_mean_color(img, labels)
    labels2 = graph.merge_hierarchical(
        labels, g,
        thresh=threshold,
        rag_copy=False,
        in_place_merge=True,
        merge_func=merge_mean_color,
        weight_func=weight_mean_color)
    out = color.label2rgb(labels2, img, kind='avg')

    rects = []
    for region in regionprops(labels2):
        if region.area >= 2000:
            min_y, min_x, max_y, max_x = region.bbox
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red', linewidth=1)
            rects.append(rect)

    return (out, rects)




# separates the image by yellow colors
def filter_yellow_color(img):
    shape = img.shape
    filtered = np.full((shape[:2]), False)

    green_to_red_threshold = 0.6
    blue_to_red_threshold = 0.5
    for y in range(shape[0]):
        for x in range(shape[1]):
            pixel = img[y][x]
            # segment image by yellow color
            # the goal is to try and find traffic lights
            # which are primarily a bright yellow
            # yellow colors are formed when there is
            # an equal balance of red and green and little blue
            if (pixel[0] > pixel[2] and pixel[1] > pixel[2]
                and pixel[1] / pixel[0] > green_to_red_threshold
                and pixel[2] / pixel[0] < blue_to_red_threshold):
                    filtered[y][x] = True
    return filtered

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
    # # uses a flood fill to remove small patches of yellow
    # visited = np.full(shape, False)
    # flood_remove_threshold = 16
    # cells = []
    # # (x, y)
    # cells.append((0, 0))
    # while len(cells) != 0:
    #     c = cells.pop()
    #     if visited[c[0]][c[1]]:
    #         continue
    #     # left cell
    #     if c[0] > 0 and not visited[c[1]][c[0] - 1]:
    #         cells.append((c[1], c[0] - 1))
    #     # right cell
    #     if c[0] < shape[1] - 1 and not visited[c[1]][c[0] + 1]:
    #         cells.append((c[1], c[0] + 1))
    #     # top cell
    #     if c[1] > 0 and not visited[c[1] - 1][c[0]]:
    #         cells.append((c[1] - 1, c[0]))
    #     # bottom cell
    #     if c[1] < shape[0] - 1 and not visited[c[1] + 1][c[0]]:
    #         cells.append((c[1] + 1, c[0]))

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
        img = plt.imread(sys.argv[1])
        print('Image shape:', img.shape)
        print('Image dtype:', img.dtype)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 8))
        ax1.imshow(img)
        ax1.axis('off')

        img_felzen, rects = segment_image(img)
        ax2.imshow(img_felzen)
        for rect in rects:
            ax1.add_patch(rect)
        ax2.axis('off')

        filtered = filter_yellow_color(img)
        ax3.imshow(filter_to_image(filtered))
        ax3.axis('off')

        plt.show()


    """    img = plt.imread(sys.argv[1])
        print('Image shape:', img.shape)
        print('Image dtype:', img.dtype)
        print('Copying image')



        new_img = np.copy(img)
        print('Applying filter')

        filtered = filter_yellow_color(new_img)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 8))
        ax1.imshow(img)
        ax1.axis('off')

        ax2.imshow(filter_to_image(filtered))
        ax2.axis('off')

        filtered = remove_noise(filtered)
        filtered = remove_noise(filtered)
        ax3.imshow(filter_to_image(filtered))
        ax3.axis('off')

        plt.show()"""
