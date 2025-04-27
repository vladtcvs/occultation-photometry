from typing import List, Tuple
import numpy as np
import math

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

def interpolate(v1 : float, v2 : float, k):
    if np.isnan(v1):
        return v2
    if np.isnan(v2):
        return v1
    return v1*(1-k)+v2*k

def _getpixel(track : np.ndarray, y : int, x : int):
    if x < 0 or y < 0:
        return np.nan
    if x >= track.shape[1] or y >= track.shape[0]:
        return np.nan
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

def slice_track(track : np.ndarray, reference_track_points : np.ndarray, half_w : int, margin : int, offset : float) -> np.ndarray:
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
    profile = value / weight * width
    profile[np.where(np.isnan(profile))] = 0
    return profile
