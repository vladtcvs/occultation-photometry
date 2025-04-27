import drift_profile
import drift_slice
import drift_detect
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
        self.occ_margin = 10

        # smoothing error of profile
        self.smooth_err = 21

        # list of reference tracks
        self.reference_tracks : List[DriftTrack] = []

        # list of sky tracks
        self.sky_tracks : List[DriftTrack] = []

        # average reference track
        self.ref_track : DriftTrack = None
        self.ref_track_rgb = None

        # average reference profile
        self.ref_profile : DriftProfile = None
        self.ref_profile_rgb = None
        self.ref_slices_rgb = None

        # occultation track
        self.occ_track : DriftTrack = None
        self.occ_track_desc = None
        self.occ_track_rgb = None

        # occultation profile
        self.occ_profile : DriftProfile = None
        self.occ_profile_rgb = None
        self.occ_slices_rgb = None

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
        for left,right,top,bottom in self.reference_tracks:
            # draw track
            if self.ref_track is not None:
                track = DriftTrack(self.gray[top:bottom,left:right], points=self.ref_track.points)
            else:
                track = DriftTrack(self.gray[top:bottom,left:right])
 
            self.rgb[top:bottom,left:right] = track.draw((255,0,0), 0.5)

            # draw bounding rectangles
            cv2.rectangle(self.rgb, (left,top), (right,bottom), color=(255,0,0), thickness=1)

        if self.occ_track_desc is not None:
            left,right,top,bottom = self.occ_track_desc
            if self.ref_track is not None:
                track = DriftTrack(self.gray[top:bottom,left:right], points=self.ref_track.points)
            else:
                track = DriftTrack(self.gray[top:bottom,left:right])
            self.rgb[top:bottom,left:right] = track.draw((0,200,0), 0.5)

            # draw bounding rectangles
            cv2.rectangle(self.rgb, (left,top), (right,bottom), color=(0,200,0), thickness=1)
            cv2.rectangle(self.rgb, (left-self.occ_margin,top-self.occ_margin), (right+self.occ_margin,bottom+self.occ_margin), color=(0,200,0), thickness=1)
    
    def detect_tracks(self):
        self.reference_tracks = drift_detect.detect_reference_tracks(self.gray, 9, [2, 1.2])
        
        # draw track bounding rectangles
        self._draw_tracks()
        self.notify_observers()

    def build_reference_track(self):
        ref_track, ref_points, ref_transposed = drift_detect.build_reference_track(self.gray, self.reference_tracks)
        self.ref_track = DriftTrack(ref_track)
        self.ref_track.points = ref_points
        self.ref_track.transposed = ref_transposed

        # reference track draw
        self.ref_track_rgb = self.ref_track.draw((255,0,0), 0.5)

        # draw track bounding rectangles
        self._draw_tracks()
        self.notify_observers()

    def analyze_reference_track(self):
        self.ref_track.half_w = self.ref_half_w
        self.ref_track.normals = drift_slice.build_track_normals(self.ref_track.points)

        # analyze each reference track and find it's profile
        self.ref_profiles = []
        for ref_track in self.reference_tracks:
            ref_track.slices = drift_slice.slice_track(ref_track.gray, self.ref_track.points, self.ref_track.half_w, 0, 0)
            profile = drift_slice.slices_to_profile(ref_track.slices)
            self.ref_profiles.append(profile)

        # find mean reference track
        ref_profile, ref_stdev = drift_slice.calculate_reference_profile(self.ref_profiles)
        self.ref_profile = DriftProfile(ref_profile, ref_stdev)

        # draw reference track
        if self.ref_track is not None:
            self.ref_track_rgb = self.ref_track.draw((255,0,0), 0.5)

        # draw tracks
        self._draw_tracks()

        # build reference profile plot
        self.ref_profile_rgb = self.ref_profile.plot_profile(640, 480, self.smooth_err)

        # build reference track mean slice
        self.ref_slices_rgb = self.ref_track.plot_slice(640, 480)

        self.notify_observers()

    def specify_occ_track(self, x : int, y : int):
        if self.ref_track is not None:
            self.occ_track_pos = (y, x)
            w = self.ref_track.gray.shape[1]
            h = self.ref_track.gray.shape[0]
            x0 = self.occ_track_pos[1]
            y0 = self.occ_track_pos[0]

            self.occ_track_desc = (x0, x0 + w, y0, y0 + h)
            occ_track = drift_detect.extract_track(self.gray, x0, y0, w, h, self.occ_margin)
            self.occ_track = DriftTrack(occ_track, self.occ_margin, self.ref_track.points, None, self.ref_track.transposed)
            self.occ_track.half_w = self.occ_half_w

            self.occ_track_rgb = self.occ_track.draw((0,200,0),0.5)

            # draw tracks
            self._draw_tracks()

            self.notify_observers()

    def analyze_occ_track(self):
        self.occ_track.half_w = self.occ_half_w
        self.occ_track.slices = drift_slice.slice_track(self.occ_track.gray,
                                               self.occ_track.points,
                                               self.occ_track.half_w,
                                               self.occ_track.margin,
                                               0)
        
        side_profiles = []
        for i in (-4,-2,2,4):
            occ_slices_offset = drift_slice.slice_track(self.occ_track.gray,
                                                        self.occ_track.points,
                                                        self.occ_track.half_w,
                                                        self.occ_track.margin,
                                                        i*self.occ_track.half_w)
            occ_profile_offset = drift_slice.slices_to_profile(occ_slices_offset)
            side_profiles.append(occ_profile_offset)
    
        # profile of track
        occ_profile_raw = drift_slice.slices_to_profile(self.occ_track.slices)

        # Profile without sky glow
        if self.build_true_occ_profile:
            occ_profile, occ_profile_stdev = drift_profile.calculate_drift_profile(occ_profile_raw, side_profiles, self.ref_profile.profile)
            self.occ_profile = DriftProfile(occ_profile, occ_profile_stdev)
        else:
            _, sky_stdev = drift_slice.calculate_sky_profile(side_profiles)
            self.occ_profile = DriftProfile(occ_profile_raw, sky_stdev)

        # build occultation track plot
        self.occ_profile_rgb = self.occ_profile.plot_profile(640, 480, self.smooth_err)

        # build reference track mean slice
        self.occ_slices_rgb = self.occ_track.plot_slice(640, 480)

        self.notify_observers()
