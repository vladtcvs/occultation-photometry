import cv2
import numpy as np
import drift_profile
import drift_detect
from PIL import Image
import matplotlib.pyplot as plt


def extract_track(gray : np.ndarray, x0 : int, y0 : int, w : int, h : int, margin : int):
    track = gray[y0-margin:y0+h+margin, x0-margin:x0+w+margin]
    return track


smooth_err = 15
margin = 20
half_w = 3

gray = np.array(Image.open('examples/westphalia.png').convert('L'))

ref_track, points, transposed = drift_detect.build_reference_track(gray)
ref_slices = drift_profile.slice_track(ref_track, points, half_w, 0, 0)

x0 = 297
y0 = 282
w = ref_track.shape[1]
h = ref_track.shape[0]

occ_track = extract_track(gray, x0, y0, w, h, margin)
occ_slices = drift_profile.slice_track(occ_track, points, half_w, margin, 0)
occ_slices_offset_1 = drift_profile.slice_track(occ_track, points, half_w, margin, -2*half_w)
occ_slices_offset_2 = drift_profile.slice_track(occ_track, points, half_w, margin, 2*half_w)

# reference track profile
ref_profile = drift_profile.slices_to_profile(ref_slices)
ref_poisson_err = np.sqrt(ref_profile)
ref_top_poisson_err = drift_profile.smooth_track_profile(ref_profile + ref_poisson_err, smooth_err)
ref_low_poisson_err = drift_profile.smooth_track_profile(ref_profile - ref_poisson_err, smooth_err)


# occultation profile and poisson deviation
occ_profile = drift_profile.slices_to_profile(occ_slices)
occ_poisson_err = np.sqrt(occ_profile)

# profiles of paths parallel to track
occ_profile_1 = drift_profile.slices_to_profile(occ_slices_offset_1)
occ_profile_2 = drift_profile.slices_to_profile(occ_slices_offset_2)

# average sky value and error of sky brightness
occ_profile_conn = np.concatenate([occ_profile_1, occ_profile_2], axis=0)
sky_average = np.average(occ_profile_conn)
sky_stdev = np.std(occ_profile_conn)


occ_clear_profile = occ_profile - sky_average


occ_top_sky_err = drift_profile.smooth_track_profile(occ_clear_profile + sky_stdev, smooth_err)
occ_low_sky_err = drift_profile.smooth_track_profile(occ_clear_profile - sky_stdev, smooth_err)

occ_top_poisson_err = drift_profile.smooth_track_profile(occ_clear_profile + occ_poisson_err, smooth_err)
occ_low_poisson_err = drift_profile.smooth_track_profile(occ_clear_profile - occ_poisson_err, smooth_err)

occ_total_err = np.sqrt(sky_stdev**2 + occ_poisson_err**2)
occ_top_total_err = drift_profile.smooth_track_profile(occ_clear_profile + occ_total_err, smooth_err)
occ_low_total_err = drift_profile.smooth_track_profile(occ_clear_profile - occ_total_err, smooth_err)


#gray_rgb = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
ref_track_rgb = cv2.cvtColor(ref_track.astype(np.uint8), cv2.COLOR_GRAY2RGB)
occ_track_rgb = cv2.cvtColor(occ_track.astype(np.uint8), cv2.COLOR_GRAY2RGB)

for y,x in points:
    ref_track_rgb[y,x,0] = 255
    #occ_track_rgb[y+half_w,x+half_w,0] = 255

if True:
    fig, axs = plt.subplots(1,6)
    axs[0].imshow(occ_track_rgb)
    axs[1].imshow(ref_track_rgb)
    axs[2].imshow(ref_slices, cmap='gray')
    axs[3].imshow(occ_slices, cmap='gray')
    axs[4].imshow(occ_slices_offset_1, cmap='gray')
    axs[5].imshow(occ_slices_offset_2, cmap='gray')
    plt.show()

L = len(occ_clear_profile)
xr = range(L)
fig, axs = plt.subplots(2,2)
axs[0,0].plot(xr, ref_profile, xr, ref_top_poisson_err, xr, ref_low_poisson_err)
axs[0,0].set_title("Reference track")

axs[0,1].plot(xr, occ_clear_profile, xr, occ_top_sky_err, xr, occ_low_sky_err)
axs[0,1].set_title("Error of light pollution")

axs[1,0].plot(xr, occ_clear_profile, xr, occ_top_poisson_err, xr, occ_low_poisson_err)
axs[1,0].set_title("Error of poisson")

axs[1,1].plot(xr, occ_clear_profile, xr, occ_top_total_err, xr, occ_low_total_err)
axs[1,1].set_title("Total error")
plt.show()
