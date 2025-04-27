from typing import List, Tuple
import numpy as np
import cv2
import imutils
import imutils.contours
import statistics

from skimage import measure

class TrackRect:
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.w = self.right - self.left + 1
        self.h = self.bottom - self.top + 1

    def point_inside_rect(self, x : int, y : int) -> bool:
        return x >= self.left and x <= self.right and y >= self.top and y <= self.bottom

    def detect_overlap(self, other) -> bool:
        if self.point_inside_rect(other.left, other.top):
            return True
        if self.point_inside_rect(other.right, other.top):
            return True
        if self.point_inside_rect(other.left, other.bottom):
            return True
        if self.point_inside_rect(other.right, other.bottom):
            return True
        return False

    def extract_track(self, gray : np.ndarray, margin : int) -> Tuple[np.ndarray, np.ndarray]:
        x0 = self.left-margin
        y0 = self.top-margin
        x1 = self.right+margin+1
        y1 = self.bottom+margin+1

        tw = x1 - x0
        th = y1 - y0
        
        x0_c = max(x0, 0)
        y0_c = max(y0, 0)
        x1_c = min(x1, gray.shape[1])
        y1_c = min(y1, gray.shape[0])

        dy = y0_c - y0
        dx = x0_c - x0
        cw = x1_c - x0_c
        ch = y1_c - y0_c

        result = np.empty((th, tw))
        result.fill(np.nan)
        track = gray[y0_c:y1_c, x0_c:x1_c]
        result[dy:dy+ch, dx:dx+cw] = track

        mask = np.ones(result.shape)
        idxs = np.where(np.isnan(result))
        mask[idxs] = 0
        result[idxs] = 0

        return result, mask

def detect_bold_tracks(gray : np.ndarray, 
                       num_tracks : int = 4,
                       smooth_size : int = 11,
                       blur_size : int = 35,
                       threshold_k : float = 1.1) -> List[TrackRect]:
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
        tracks.append(TrackRect(left, right, top, bottom))

    return tracks

def clear_overlapped(tracks : List[TrackRect]) -> List[TrackRect]:
    not_overlapped = []
    for ind1, track1 in enumerate(tracks):
        for ind2, track2 in enumerate(tracks):
            if ind1 == ind2:
                continue
            if track1.detect_overlap(track2):
                break
        else:
            not_overlapped.append(track1)
    return not_overlapped

def clear_bad_size(tracks : List[TrackRect], kappa : float = 1) -> List[TrackRect]:
    widths = []
    heights = []
    for track in tracks:
        width = int(track.right - track.left)
        height = int(track.bottom - track.top)
        widths.append(width)
        heights.append(height)
    width0 = statistics.mean(widths)
    height0 = statistics.mean(heights)
    stdw = statistics.stdev(widths)
    stdh = statistics.stdev(heights)
    goods = []
    for track in tracks:
        width = track.right - track.left
        height = track.bottom - track.top
        if abs(width - width0) < stdw * kappa and abs(height - height0) < stdh * kappa:
            goods.append(track)
    return goods

def correlate_tracks(tracks : List[TrackRect]) -> List[TrackRect]:
    maxw = 0
    maxh = 0
    for track in tracks:
        maxw = max(maxw, track.w)
        maxh = max(maxh, track.h)

    results = []
    for track in tracks:
        # TODO: do correlation
        dx = -int((maxw - track.w)/2)
        dy = -int((maxh - track.h)/2)

        left = track.left + dx
        right = left + maxw - 1
        top = track.top + dy
        bottom = top + maxh - 1
        aligned = TrackRect(left, right, top, bottom)
        results.append(aligned)
    return results

def mean_track(tracks : List[TrackRect], image : np.ndarray) -> np.ndarray:
    w = tracks[0].w
    h = tracks[0].h
    sum_track = np.zeros((h,w))
    sum_weight = np.zeros((h,w))
    for track in tracks:
        block, weight = track.extract_track(image, 0)
        sum_track += block
        sum_weight += weight

    sum_track = sum_track / sum_weight
    sum_track[np.where(sum_weight == 0)] = 0
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

def detect_reference_tracks(gray : np.ndarray, count : int = 10, kappas : list | None = None) -> List[TrackRect]:
    tracks = detect_bold_tracks(gray, count)
    tracks = clear_overlapped(tracks)
    if kappas is None:
        kappas = [2, 1.2]
    for kappa in kappas:
        tracks = clear_bad_size(tracks, kappa=kappa)
    tracks = correlate_tracks(tracks)
    return tracks

def build_mean_reference_track(gray : np.ndarray, references : List[TrackRect]) -> Tuple[np.ndarray, np.ndarray, bool]:
    track = mean_track(references, gray)
    points, transposed = track_to_points(track)
    points = smooth_track_points(points, transposed)
    return (track, points, transposed)
