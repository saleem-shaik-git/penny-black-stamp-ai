"""
Image normalisation for Penny Black stamp analysis
Author: Saleem Shaik
"""

import cv2
import numpy as np

def normalise_image(image):
    """
    Preprocess stamp images to reduce noise caused by
    worn postmarks and uneven ink density.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

