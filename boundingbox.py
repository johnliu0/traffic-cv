"""Image bounding box details.

Stores information about an image bounding box. BoundingBox extracts relevant
information from a region object produced from skimage.measure.regionprops.
"""

class BoundingBox:
    """Bounding box.

    Stores information about a bounding box; a rectangle that represents some
    area in an image.

    Attributes:
        min_x: x coordinate of the left bounds of the box in pixels
        min_y: y coordinate of the upper bounds of the box in pixels
        max_x: x coordinate of the right bounds of the box in pixels
        max_y: y coordinate of the bottom bounds of the box in pixels
        width: width of box in pixels
        height: height of box in pixels
        area: number of pixels contained by the region
        bounded_area: number of pixels contained by the box (width * height)
    """

    def __init__(self, region):
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
        self.bounded_area = self.width * self.height

    def to_string(self):
        return(f'BoundingBox(min_x: {self.min_x}, min_y: {self.min_y}, max_x: {self.max_x}, max_y: {self.max_y}), width: {self.width}, height: {self.height}, area: {self.area}, bounded_area: {self.bounded_area})')
