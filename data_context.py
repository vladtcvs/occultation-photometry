import drift_profile
import drift_slice
import drift_detect

from drift_detect import TrackRect
from data_containers import DriftProfile, DriftTrack

import abc
from typing import List, Tuple

import cv2
import numpy as np

class IObserver:
    @abc.abstractmethod
    def notify(self):
        pass

class DriftContext:
    def __init__(self):
        self.observers : List[IObserver] = []

        # original frame
        self.gray = None

        # reference track half width
        self.ref_half_w = 5

        # occultation track half width
        self.occ_half_w = 5

        # occultation track margin
        self.occ_margin = 10

        # smoothing error of profiles
        self.smooth_err = 21

        # list of reference tracks
        self.reference_tracks : List[TrackRect] = []

        # occultation track
        self.occ_track_rect : TrackRect = None
        self.occ_track_rgb : np.ndarray = None

        # list of sky tracks
        self.sky_tracks : List[DriftTrack] = []

        # average reference track
        self.mean_ref_track : DriftTrack = None
        self.mean_ref_track_rgb : np.ndarray = None

        # average reference profile
        self.mean_ref_profile : DriftProfile = None
        self.mean_ref_profile_rgb : np.ndarray = None

        # occultation profile
        self.occ_profile : DriftProfile = None
        self.occ_profile_rgb : np.ndarray = None

        # restore true reference profile
        self.build_true_occ_profile : bool = True


    def add_observer(self, observer : IObserver):
        self.observers.append(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.notify()

    def set_image(self, gray):
        self.gray = gray
        self.rgb = cv2.cvtColor(self.gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.notify_observers()

    def set_ref_half_w(self, half_w):
        self.ref_half_w = half_w
        self.notify_observers()

    def set_occ_half_w(self, half_w):
        self.occ_half_w = half_w
        self.notify_observers()

    def _draw_tracks(self):
        # draw reference track line on each of reference tracks on original image
        self.rgb = cv2.cvtColor(self.gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for track in self.reference_tracks:
            # draw track
            ref_track_image = track.extract_track(self.gray, 0)
            if self.mean_ref_track is not None:
                ref_track = DriftTrack(ref_track_image, points=self.mean_ref_track.points)
            else:
                ref_track = DriftTrack(ref_track_image)
 
            ref_track.draw_in_place(self.rgb, track.left, track.top, (255,0,0), 0.5)

            # draw bounding rectangle
            cv2.rectangle(self.rgb, (track.left, track.top),
                                    (track.right, track.bottom),
                                    color=(255,0,0), thickness=1)

        if self.occ_track_rect is not None:
            occ_track_image = self.occ_track_rect.extract_track(self.gray, 0)
            if self.mean_ref_track is not None:
                occ_track = DriftTrack(occ_track_image, points=self.mean_ref_track.points)
            else:
                occ_track = DriftTrack(occ_track_image)
            occ_track.draw_in_place(self.rgb, self.occ_track_rect.left, self.occ_track_rect.top, (0,200,0), 0.5)

            # draw bounding rectangles
            cv2.rectangle(self.rgb, (self.occ_track_rect.left,self.occ_track_rect.top),
                                    (self.occ_track_rect.right,self.occ_track_rect.bottom),
                                    color=(0,200,0), thickness=1)

            cv2.rectangle(self.rgb, (self.occ_track_rect.left-self.occ_margin,self.occ_track_rect.top-self.occ_margin),
                                    (self.occ_track_rect.right+self.occ_margin,self.occ_track_rect.bottom+self.occ_margin),
                                    color=(0,200,0), thickness=1)
    
    def detect_tracks(self):
        self.reference_tracks = drift_detect.detect_reference_tracks(self.gray, 9, [2, 1.2])

        # draw track bounding rectangles
        self._draw_tracks()
        self.notify_observers()

    def build_reference_track(self):
        ref_track, ref_points, ref_transposed = drift_detect.build_mean_reference_track(self.gray, self.reference_tracks)
        self.mean_ref_track = DriftTrack(ref_track)
        self.mean_ref_track.points = ref_points
        self.mean_ref_track.transposed = ref_transposed

        # reference track draw
        self.ref_track_rgb = self.mean_ref_track.draw((255,0,0), 0.5)

        # draw track bounding rectangles
        self._draw_tracks()
        self.notify_observers()

    def analyze_reference_track(self):
        self.mean_ref_track.half_w = self.ref_half_w
        self.mean_ref_track.normals = drift_slice.build_track_normals(self.mean_ref_track.points)

        # analyze each reference track and find it's profile
        self.ref_profiles : List[np.ndarray] = []
        for ref_track_rect in self.reference_tracks:
            track, _ = ref_track_rect.extract_track(self.gray, 0)
            # use mean points and normals
            ref_track = DriftTrack(track, 0, self.mean_ref_track.points, self.mean_ref_track.normals, self.mean_ref_track.transposed)

            ref_track.slices = drift_slice.slice_track(ref_track.gray, ref_track.points, ref_track.normals, self.ref_half_w, 0, 0)
            profile = drift_slice.slices_to_profile(ref_track.slices)

            self.ref_profiles.append(profile)

        # find mean reference profile
        ref_profile, ref_stdev = drift_profile.calculate_reference_profile(self.ref_profiles)
        self.mean_ref_profile = DriftProfile(ref_profile, ref_stdev)

        # draw reference track
        if self.mean_ref_track is not None:
            self.mean_ref_track_rgb = self.mean_ref_track.draw((255,0,0), 0.5)

        # build reference profile plot
        self.mean_ref_profile_rgb = self.mean_ref_profile.plot_profile(640, 480, self.smooth_err)

        # draw tracks
        self._draw_tracks()

        self.notify_observers()

    def specify_occ_track(self, x : int, y : int):
        self.occ_track_pos = (y, x)
        if self.mean_ref_track is not None:
            w = self.mean_ref_track.gray.shape[1]
            h = self.mean_ref_track.gray.shape[0]
            x0 = self.occ_track_pos[1]
            y0 = self.occ_track_pos[0]

            self.occ_track_rect = TrackRect(x0, x0 + w, y0, y0 + h)
            occ_track_gray, _ = self.occ_track_rect.extract_track(self.gray, self.occ_margin)
            self.occ_track = DriftTrack(occ_track_gray,
                                        self.occ_margin,
                                        self.mean_ref_track.points,
                                        self.mean_ref_track.normals,
                                        self.mean_ref_track.transposed)

            self.occ_track.half_w = self.occ_half_w
            self.occ_track_rgb = self.occ_track.draw((0,200,0),0.5)

            # draw tracks
            self._draw_tracks()

            self.notify_observers()

    def analyze_occ_track(self):
        self.occ_track.half_w = self.occ_half_w

        assert self.occ_track is not None

        # profile of track
        self.occ_track.slices = drift_slice.slice_track(self.occ_track.gray,
                                               self.occ_track.points,
                                               self.occ_track.normals,
                                               self.occ_track.half_w,
                                               self.occ_track.margin,
                                               0)
        occ_profile_raw = drift_slice.slices_to_profile(self.occ_track.slices)

        # profiles parallel to track
        side_profiles = []
        for i in (-4,-2,2,4):
            occ_slices_offset = drift_slice.slice_track(self.occ_track.gray,
                                                        self.occ_track.points,
                                                        self.occ_track.normals,
                                                        self.occ_track.half_w,
                                                        self.occ_track.margin,
                                                        i*self.occ_track.half_w)
            occ_profile_offset = drift_slice.slices_to_profile(occ_slices_offset)
            side_profiles.append(occ_profile_offset)

        # referece profiles
        ref_profiles = self.ref_profiles
        assert ref_profiles is not None
        assert type(ref_profiles) == list
        assert len(ref_profiles) > 0

        # Profile without sky glow
        if self.build_true_occ_profile:
            occ_profile, occ_profile_stdev, stats = drift_profile.calculate_true_drift_profile(occ_profile_raw,
                                                                                        side_profiles,
                                                                                        ref_profiles)
            self.occ_profile = DriftProfile(occ_profile, occ_profile_stdev)
            for key in stats:
                print(f"{key} : {stats[key]}")
        else:
            _, sky_stdev = drift_profile.calculate_sky_profile(side_profiles)
            self.occ_profile = DriftProfile(occ_profile_raw, sky_stdev)

        # build occultation track plot
        self.occ_profile_rgb = self.occ_profile.plot_profile(640, 480, self.smooth_err)

        # build reference track mean slice
        self.occ_slices_rgb = self.occ_track.plot_slice(640, 480)

        self.notify_observers()
