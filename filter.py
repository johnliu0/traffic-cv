import sys
import numpy as np
import matplotlib.pyplot as plt

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

        plt.show()
