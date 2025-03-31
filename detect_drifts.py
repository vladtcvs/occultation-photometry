import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imutils
import imutils.contours

from skimage import measure

def detect_bold_tracks(gray : np.ndarray,
                       num_tracks : int = 4,
                       smooth_size : int = 11,
                       blur_size : int = 35,
                       threshold_k : float = 1.1):
    if blur_size % 2 == 0:
        blur_size += 1
    if smooth_size % 2 == 0:
        smooth_size += 1
    gray = gray / np.amax(gray)
    smooth = cv2.GaussianBlur(gray, (11,11), 0)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    mask = smooth > blurred * threshold_k + 0.01
    mask = mask.astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    blob = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    labels = measure.label(blob, connectivity=2, background=0)

    numpixels = []

    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = np.zeros(mask.shape, dtype="uint8")
        label_mask[labels == label] = 1
        num_pixels = cv2.countNonZero(label_mask)
        numpixels.append(num_pixels)

    numpixels = sorted(numpixels, reverse=True)

    if len(numpixels) == 0:
        return None

    tracks = []
    min_pixels = (numpixels[:num_tracks])[-1]
    for label in np.unique(labels):
        if label == 0:
            continue
        label_mask = np.zeros(mask.shape, dtype="uint8")
        label_mask[labels == label] = 1
        num_pixels = cv2.countNonZero(label_mask)
        if num_pixels < min_pixels:
            continue

        contours = cv2.findContours(label_mask,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        if len(contours) != 1:
            raise ValueError()
        contour = contours[0]
        left = np.inf
        right = -np.inf
        top = np.inf
        bottom = -np.inf
        for point in contour:
            x, y = point[0]
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)
        tracks.append((left, right, top, bottom))

    return tracks

gray = np.array(Image.open('examples/westphalia.png').convert('L'))
tracks = detect_bold_tracks(gray, 6)
left, right, top, bottom = tracks[1]
track = gray[top:bottom,left:right]
plt.imshow(track)
plt.show()
