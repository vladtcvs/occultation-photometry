import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imutils
import imutils.contours

from skimage import measure


gray = np.array(Image.open('examples/westphalia.png').convert('L'))

gray = gray / np.amax(gray)

smooth = cv2.GaussianBlur(gray, (11,11), 0)
blurred = cv2.GaussianBlur(gray, (35,35), 0)

mask = smooth > blurred * 1.1 + 0.01

mask = mask.astype('uint8')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
blob = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

labels = measure.label(blob, connectivity=2, background=0)
mask = np.zeros(mask.shape, dtype="uint8")

numpixels = []

for label in np.unique(labels):
    if label == 0:
        continue
    label_mask = np.zeros(mask.shape, dtype="uint8")
    label_mask[labels == label] = 1
    num_pixels = cv2.countNonZero(label_mask)
    numpixels.append(num_pixels)

numpixels = sorted(numpixels, reverse=True)

min_pixels = numpixels[6]
for label in np.unique(labels):
    if label == 0:
        continue
    label_mask = np.zeros(mask.shape, dtype="uint8")
    label_mask[labels == label] = 1
    num_pixels = cv2.countNonZero(label_mask)
    if num_pixels < min_pixels:
        continue
    mask = cv2.add(mask, label_mask)


#contours = cv2.findContours(mask,
#                                cv2.RETR_EXTERNAL,
#                                cv2.CHAIN_APPROX_SIMPLE)
#contours = imutils.grab_contours(contours)
#contours = imutils.contours.sort_contours(contours)[0]
#for contour in contours:
#    print(contour)

plt.imshow(mask)
plt.show()
