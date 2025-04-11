from typing import List, Tuple
import numpy as np
import cv2
import imutils
import imutils.contours
import statistics

from skimage import measure


def extract_track(gray : np.ndarray, x0 : int, y0 : int, w : int, h : int, margin : int):
    track = gray[y0-margin:y0+h+margin, x0-margin:x0+w+margin]
    return track


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

def point_inside_rect(x, y, left, right, top, bottom) -> bool:
    return x >= left and x <= right and y >= top and y <= bottom

def detect_overlap(track1, track2) -> bool:
    left1, right1, top1, bottom1 = track1
    left2, right2, top2, bottom2 = track2
    if point_inside_rect(left1, top1, left2, right2, top2, bottom2):
        return True
    if point_inside_rect(right1, top1, left2, right2, top2, bottom2):
        return True
    if point_inside_rect(left1, bottom1, left2, right2, top2, bottom2):
        return True
    if point_inside_rect(right1, bottom1, left2, right2, top2, bottom2):
        return True
    return False

def clear_overlapped(tracks : list) -> list:
    not_overlapped = []
    for ind1, track1 in enumerate(tracks):
        for ind2, track2 in enumerate(tracks):
            if ind1 == ind2:
                continue
            if detect_overlap(track1, track2):
                break
        else:
            not_overlapped.append(track1)
    return not_overlapped

def clear_bad_size(tracks : list, kappa : float = 1) -> list:
    widths = []
    heights = []
    for left, right, top, bottom in tracks:
        width = int(right - left)
        height = int(bottom - top)
        widths.append(width)
        heights.append(height)
    width0 = statistics.mean(widths)
    height0 = statistics.mean(heights)
    stdw = statistics.stdev(widths)
    stdh = statistics.stdev(heights)
    goods = []
    for left, right, top, bottom in tracks:
        width = right - left
        height = bottom - top
        if abs(width - width0) < stdw * kappa and abs(height - height0) < stdh * kappa:
            goods.append((left, right, top, bottom))
    return goods

def mean_track(tracks : list, image : np.ndarray) -> np.ndarray:
    maxw = 0
    maxh = 0
    for left,right,top,bottom in tracks:
        w = right - left
        h = bottom - top
        maxw = max(maxw, w)
        maxh = max(maxh, h)
    sum_track = np.zeros((maxh,maxw))
    weight = np.zeros((maxh,maxw))
    for left,right,top,bottom in tracks:
        block = image[top:bottom,left:right]
        h = block.shape[0]
        w = block.shape[1]
        
        dx = 0
        dy = 0

        aligned = np.zeros((maxh,maxw))
        aligned_weight = np.zeros((maxh,maxw))
        aligned[dy:dy+h,dx:dx+w] = block
        aligned_weight[dy:dy+h,dx:dx+w] = 1
        sum_track += aligned
        weight += aligned_weight
    sum_track = sum_track / weight
    sum_track[np.where(weight == 0)] = 0
    return sum_track

def track_to_points(track : np.ndarray) -> list:
    w = track.shape[1]
    h = track.shape[0]
    transposed = False
    if w > h:
        # horizontal track
        # transpose
        transposed = True
        track = np.transpose(track)
        w = track.shape[1]
        h = track.shape[0]

    points = []
    for y in range(h):
        slice = track[y,:]
        maximum = int(slice.argmax())
        if transposed:
            points.append((maximum,y))
        else:
            points.append((y,maximum))
    return np.array(points), transposed

def smooth_track_points(points, transposed):
    hw = 2
    L = points.shape[0]
    if not transposed:
        index = 1
    else:
        index = 0

    average = np.zeros((L,2))
    for x in range(L):
        s = 0
        num = 0
        for y in range(x-hw,x+hw+1):
            if y < 0 or y >= L:
                continue
            s += points[y, index]
            num += 1
        average[x, index] = s / num
        average[x, 1-index] = points[x, 1-index]
    return average

def detect_reference_tracks(gray : np.ndarray, count : int = 10, kappas : list | None = None) -> list:
    tracks = detect_bold_tracks(gray, count)
    tracks = clear_overlapped(tracks)
    if kappas is None:
        kappas = [2, 1.2]
    for kappa in kappas:
        tracks = clear_bad_size(tracks, kappa=kappa)
    return tracks
    

def build_reference_track(gray : np.ndarray, references : List[tuple]) -> Tuple[np.ndarray, np.ndarray, bool]:
    track = mean_track(references, gray)
    points, transposed = track_to_points(track)
    points = smooth_track_points(points, transposed)
    return (track, points, transposed)
