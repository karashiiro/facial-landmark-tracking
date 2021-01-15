from math import sqrt
from typing import Union

import cv2
import numpy as np

ArrayLike = Union[list, np.ndarray, tuple]

EYE_PADDING_PERCENT_X = 0.3
EYE_PADDING_PERCENT_Y = 0.5

def eye_pad_min_x(vec) -> int:
    x_max = np.max(vec)
    x_min = np.min(vec)
    return np.floor(x_min - (x_max - x_min) * EYE_PADDING_PERCENT_X).astype(int)

def eye_pad_max_x(vec) -> int:
    x_max = np.max(vec)
    x_min = np.min(vec)
    return np.ceil(x_max + (x_max - x_min) * EYE_PADDING_PERCENT_X).astype(int)

def eye_pad_min_y(vec) -> int:
    x_max = np.max(vec)
    x_min = np.min(vec)
    return np.floor(x_min - (x_max - x_min) * EYE_PADDING_PERCENT_Y).astype(int)

def eye_pad_max_y(vec) -> int:
    x_max = np.max(vec)
    x_min = np.min(vec)
    return np.ceil(x_max + (x_max - x_min) * EYE_PADDING_PERCENT_Y).astype(int)

def get_magnitude(mag_pt: ArrayLike) -> float:
    x = mag_pt[0]
    y = mag_pt[1]
    return sqrt(x * x + y * y)

def normalize_gradient(gradient: np.ndarray) -> np.ndarray:
    """Convert all gradient vectors into unit vectors."""
    for g_r, _ in enumerate(gradient):
        for g_c, _ in enumerate(gradient[g_r]):
            g_mag = get_magnitude(gradient[g_r][g_c])
            if g_mag != 0:
                gradient[g_r][g_c] /= g_mag
            else:
                gradient[g_r][g_c] = 0
    return gradient

def weight_preprocess(img: np.ndarray) -> np.ndarray:
    """Takes a single-channel image and preprocesses it for weight calculation."""
    img_copy = np.copy(img)
    img_copy = cv2.GaussianBlur(img_copy, (5, 5), img_copy.shape[:1][0])
    img_copy = cv2.bitwise_not(img_copy) # Image inversion
    return img_copy

def get_gradient(img: np.ndarray) -> np.ndarray:
    """Returns the vector field of the image in the x and y directions."""
    return np.dstack([cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)])
