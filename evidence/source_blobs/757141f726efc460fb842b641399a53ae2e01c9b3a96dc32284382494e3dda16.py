"""Verbatim reusable upstream function; see upstream_blur_provenance.json and UPSTREAM_LICENSE."""
import cv2
import numpy as np
from PIL import Image

LEVELS = [1, 3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63]

def load_blurred(path, kernel):
    image = Image.open(path).convert("RGB").resize((224, 224))
    if kernel == 1:
        return image
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    array = cv2.GaussianBlur(array, (kernel, kernel), 0)
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
