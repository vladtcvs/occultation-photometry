from typing import Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import imutils
import imutils.contours
import math
import statistics
from scipy import signal
import scipy.ndimage.filters as filters

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
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size

    L = points.shape[0]
    if not transposed:
        points[2:L-2,1] = np.convolve(points[:,1], kernel, mode='valid')
    else:
        points[2:L-2,0] = np.convolve(points[:,0], kernel, mode='valid')
    points = points[2:L-2]
    return points

def smooth_track_profile(profile : np.ndarray, smooth : int):
    if smooth % 2 == 0:
        smooth += 1
    kernel = np.ones(smooth) / smooth
    return np.convolve(profile, kernel, mode='same')

def build_reference_track(gray : np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    tracks = detect_bold_tracks(gray, 10)
    tracks = clear_overlapped(tracks)
    tracks = clear_bad_size(tracks, kappa=2)
    tracks = clear_bad_size(tracks, kappa=1.2)
    track = mean_track(tracks, gray)
    points, transposed = track_to_points(track)
    points = smooth_track_points(points, transposed)
    return (track, points, transposed)

def build_track_normals(points : np.ndarray)-> np.ndarray:
    # length of track along main axis
    L = points.shape[0]

    # in each X we have vector of direction of normal
    normals = np.zeros((L, 2))

    # we will use same normal in each point as a first approach
    # it ortogonal to track
    # (ny, nx) = (-tx, ty)

    tx = points[L-1,1] - points[0,1]
    ty = points[L-1,0] - points[0,0]

    nx, ny = ty, -tx
    l = math.sqrt(nx**2+ny**2)
    nx, ny = nx/l, ny/l
    for x in range(L):
        normals[x,0] = ny
        normals[x,1] = nx
    return normals

def interpolate(v1 : float | None, v2 : float | None, k):
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    return v1*(1-k)+v2*k

def _getpixel(track : np.ndarray, y : int, x : int):
    if x < 0 or y < 0:
        return None
    if x >= track.shape[1] or y >= track.shape[0]:
        return None
    return track[y,x]

def getpixel(track : np.ndarray, y : float, x : float) -> float:
    y0 = math.floor(y)
    x0 = math.floor(x)
    ky = y - y0
    kx = x - x0
    v00 = _getpixel(track, y = y0, x = x0)
    v01 = _getpixel(track, y = y0, x = x0+1)
    v10 = _getpixel(track, y = y0+1, x = x0)
    v11 = _getpixel(track, y = y0+1, x = x0+1)
    v0 = interpolate(v00, v01, kx)
    v1 = interpolate(v10, v11, kx)
    v = interpolate(v0, v1, ky)
    return v

def make_slice(track : np.ndarray, position, direction, half_w : int, offset : float):
    y,x = position
    ty,tx = direction
    slice = np.zeros((2*half_w+1,))
    for i in range(2*half_w+1):
        s = i - half_w + offset
        py, px = y+ty*s, x+tx*s
        slice[i] = getpixel(track, y=py, x=px)
    return slice

def slice_track(track : np.ndarray, reference_track_points : np.ndarray, half_w : int, margin : int, offset : float):
    normals = build_track_normals(reference_track_points)
    L = reference_track_points.shape[0]
    slices = np.zeros((L,2*half_w+1))
    shift = np.array([margin, margin])
    for i in range(L):
        point = reference_track_points[i,:] + shift
        normal = normals[i,:]
        track_slice = make_slice(track, point, normal, half_w, offset)
        slices[i,:] = track_slice
    return slices

def slices_to_profile(slices : np.ndarray) -> np.ndarray:
    mask = 1-np.isnan(slices)
    weight = np.sum(mask, axis=1)
    width = slices.shape[1]
    slices[np.where(np.isnan(slices))] = 0
    value = np.sum(slices, axis=1)
    return value / weight * width

def extract_track(gray : np.ndarray, x0 : int, y0 : int, w : int, h : int, margin : int):
    track = gray[y0-margin:y0+h+margin, x0-margin:x0+w+margin]
    return track

margin = 20
half_w = 3

gray = np.array(Image.open('examples/westphalia.png').convert('L'))

ref_track, points, transposed = build_reference_track(gray)
ref_slices = slice_track(ref_track, points, half_w, 0, 0)

x0 = 297
y0 = 282
w = ref_track.shape[1]
h = ref_track.shape[0]

occ_track = extract_track(gray, x0, y0, w, h, margin)
occ_slices = slice_track(occ_track, points, half_w, margin, 0)
occ_slices_offset_1 = slice_track(occ_track, points, half_w, margin, -2*half_w)
occ_slices_offset_2 = slice_track(occ_track, points, half_w, margin, 2*half_w)

ref_profile = slices_to_profile(ref_slices)

# occultation profile and poisson deviation
occ_profile = slices_to_profile(occ_slices)
poisson_err = np.sqrt(occ_profile)

# profiles of paths parallel to track
occ_profile_1 = slices_to_profile(occ_slices_offset_1)
occ_profile_2 = slices_to_profile(occ_slices_offset_2)

# average sky value and error of sky brightness
occ_profile_conn = np.concatenate([occ_profile_1, occ_profile_2], axis=0)
sky_average = np.average(occ_profile_conn)
sky_stdev = np.std(occ_profile_conn)


clear_occ_profile = occ_profile - sky_average


top_sky_err = smooth_track_profile(clear_occ_profile + sky_stdev, 15)
low_sky_err = smooth_track_profile(clear_occ_profile - sky_stdev, 15)

top_poisson_err = smooth_track_profile(clear_occ_profile + poisson_err, 15)
low_poisson_err = smooth_track_profile(clear_occ_profile - poisson_err, 15)

total_err = np.sqrt(sky_stdev**2 + poisson_err**2)
top_total_err = smooth_track_profile(clear_occ_profile + total_err, 15)
low_total_err = smooth_track_profile(clear_occ_profile - total_err, 15)


#gray_rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
ref_track_rgb = cv2.cvtColor(ref_track.astype(np.uint8), cv2.COLOR_GRAY2RGB)
occ_track_rgb = cv2.cvtColor(occ_track.astype(np.uint8), cv2.COLOR_GRAY2RGB)

for y,x in points:
    ref_track_rgb[y,x,0] = 255
    #occ_track_rgb[y+half_w,x+half_w,0] = 255

if False:
    fig, axs = plt.subplots(1,6)
    axs[0].imshow(occ_track_rgb)
    axs[1].imshow(ref_track_rgb)
    axs[2].imshow(ref_slices, cmap='gray')
    axs[3].imshow(occ_slices, cmap='gray')
    axs[4].imshow(occ_slices_offset_1, cmap='gray')
    axs[5].imshow(occ_slices_offset_2, cmap='gray')
    plt.show()

L = len(clear_occ_profile)
xr = range(L)
fig, axs = plt.subplots(2,2)
axs[0,0].plot(xr, ref_profile)
axs[0,0].set_title("Reference track")

axs[0,1].plot(xr, clear_occ_profile, xr, top_sky_err, xr, low_sky_err)
axs[0,1].set_title("Error of light pollution")

axs[1,0].plot(xr, clear_occ_profile, xr, top_poisson_err, xr, low_poisson_err)
axs[1,0].set_title("Error of poisson")

axs[1,1].plot(xr, clear_occ_profile, xr, top_total_err, xr, low_total_err)
axs[1,1].set_title("Total error")
plt.show()
