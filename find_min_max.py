from PIL import Image
import os
import numpy as np
def get_min_max(path):
    # Initialize variables to store min and max values
    min_val = float('inf')
    max_val = float('-inf')

    # Iterate over all images in the dataset
    for root, dirs, files in os.walk(path):
        for file in files:
            # Open the image using PIL
            img_path = os.path.join(root, file)
            img = Image.open(img_path)

            # Convert the image to a NumPy array
            img_array = np.array(img)

            # Compute the minimum and maximum pixel values
            img_min = np.min(img_array)
            img_max = np.max(img_array)

            # Update min_val and max_val if necessary
            min_val = min(min_val, img_min)
            max_val = max(max_val, img_max)
    return min_val, max_val

class ScaleToRange:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, image):
        img = np.array(image)  # Convert PIL image to NumPy array
        img = (img - self.min_val) / (self.max_val - self.min_val)
        return img

class CustomNormalize:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, image):
        image = 2 * (image - self.min_val) / (self.max_val - self.min_val) - 1
        return image
