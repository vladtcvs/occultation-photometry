from typing import List, Tuple
import numpy as np

def smooth_track_profile(profile : np.ndarray, smooth : int):
    if smooth % 2 == 0:
        smooth += 1
    hw = int(smooth/2)
    L = profile.shape[0]
    average = np.zeros((L,))
    for x in range(L):
        s = []
        for y in range(x-hw,x+hw+1):
            if y < 0 or y >= L:
                continue
            s.append(profile[y])
        average[x] = np.mean(s)
    return average

def calculate_reference_profile(reference_profiles : List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean reference profile
    """
    L = reference_profiles[0].shape[0]
    mean_profile = np.zeros((L,))
    N = len(reference_profiles)
    for n in range(N):
        mean_profile += reference_profiles[n]
    mean_profile /= np.mean(mean_profile)
    errs = np.zeros((L,))
    for n in range(N):
        profile = reference_profiles[n]
        profile = profile / np.mean(profile)
        errs += (profile - mean_profile)**2/N
    errs = np.sqrt(errs)
    return mean_profile, errs

def calculate_sky_profile(sky_profiles : List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate sky profile parallel to track.
       We use linear polynomial approximation for true sky brighness along track
       I_sky(s) = a*s + b,
       where s is a index along track
       """
    L = sky_profiles[0].shape[0]
    N = len(sky_profiles)
    xs = np.zeros((L*N,))
    ys = np.zeros((L*N,))
    for s in range(L):
        for n in range(N):
            xs[s*N+n] = s
            ys[s*N+n] = sky_profiles[n][s]
    polynom = np.polynomial.Polynomial.fit(xs, ys, 1)
    values = polynom(xs)
    ds = ys - values
    stdev_val = np.sqrt(np.mean(ds**2))
    stdev = np.ones((L,))*stdev_val
    xs = np.array(range(L))
    values = polynom(xs)
    return values, stdev

def compensate_reference_profile(drift_profile : np.ndarray,
                                 reference_profile : np.ndarray,
                                 reference_stdev : np.ndarray) -> np.ndarray:
    return drift_profile

def calculate_true_drift_profile(drift_profile : np.ndarray,
                                 side_profiles : List[np.ndarray],
                                 reference_profiles : List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, dict]:
    L = drift_profile.shape[0]
    for side_profile in side_profiles:
        assert side_profile.shape[0]==L

    for reference_profile in reference_profiles:
        assert reference_profile.shape[0]==L

    # calculate reference profile
    reference_profile, reference_stdev = calculate_reference_profile(reference_profiles)

    # average sky profile parallel to occ profile
    sky_profile, sky_stdev = calculate_sky_profile(side_profiles)

    # remove sky profile and compensate star speed
    drift_profile = (drift_profile - sky_profile)

    # compensate not uniform star moving
    drift_profile = compensate_reference_profile(drift_profile, reference_profile, reference_stdev)

    # estimate errors
    smoothed = smooth_track_profile(drift_profile, 3)
    delta = np.sqrt(np.sum((smoothed - drift_profile)**2)/L)
    mean = np.mean(drift_profile)
    stats = {
        "mean" : mean,
        "stdev" : delta,
        "sky_stdev" : np.mean(sky_stdev),
    }
    return drift_profile, np.sqrt(sky_stdev**2), stats

def reference_profile_time_analyze(profile : np.ndarray) -> np.ndarray:
    """Find such time of each point of profile, that stretching profile according to such times, make it flat"""
    L = profile.shape[0]
    smooth_window = int(L / 8)
    smoothed = smooth_track_profile(profile, smooth_window)
    speed = 1 / smoothed
    speed = smooth_track_profile(speed, smooth_window)
    
    dts = 1 / speed
    T = np.sum(dts)
    return dts * L / T

def profile_according_to_time(profile : np.ndarray, dtimes : np.ndarray):
    return profile / dtimes
