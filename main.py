from typing import List, Tuple
import wx
import wx.lib.scrolledpanel as scrolled

import sys
import cv2
import numpy as np
import drift_profile
import drift_detect
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

import csv

import abc

import matplotlib
matplotlib.use("Agg")

class IObserver:
    @abc.abstractmethod
    def notify(self):
        pass

def plot_to_numpy(xrange, datas, width=800, height=600, dpi=100):
    """
    Convert numerical data plot to numpy array

    Parameters:
    data: numpy array or list of numerical values
    width: plot width in pixels
    height: plot height in pixels
    dpi: dots per inch
    
    Returns:
    numpy array of shape (height, width, 4) containing RGBA values
    """
    # Create figure
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Plot the data
    args = []
    for data in datas:
        args.append(xrange)
        args.append(data)
    plt.plot(*args, linewidth=2)
    plt.grid(True)

    # Save plot to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='rgba', dpi=dpi)
    buf.seek(0)
    
    # Convert buffer to numpy array
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img_array = img_array.reshape(height, width, 4)
    img_array = img_array[:,:,0:3]
    
    # Clean up
    plt.close(fig)
    buf.close()
    
    return img_array

class DriftTrack:
    half_w : int = 0
    slices : np.ndarray = None

    def __init__(self, gray : np.ndarray, margin : int = 0, points : np.ndarray = None, normals : np.ndarray = None, transposed : bool = False):
        self.gray = gray
        self.points = points
        self.transposed = transposed
        self.normals = normals
        self.margin = margin

    def draw(self, color, transparency):
        rgb = cv2.cvtColor(self.gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        color = np.array(color)
        if self.points is not None:
            for y, x in self.points:
                xx = int(x+self.margin)
                yy = int(y+self.margin)
                if xx < 0 or yy < 0 or xx >= rgb.shape[1] or yy >= rgb.shape[0]:
                    continue
                rgb[yy, xx] = rgb[yy, xx] * transparency + color * (1-transparency)

        # draw normals
        if self.normals is not None and self.points is not None:
            for index, ((y,x), (ny,nx)) in enumerate(zip(self.points, self.normals)):
                if index % 10 != 0:
                    continue
                x1 = int(x - nx*self.half_w + self.margin)
                y1 = int(y - ny*self.half_w + self.margin)
                x2 = int(x + nx*self.half_w + self.margin)
                y2 = int(y + ny*self.half_w + self.margin)

                cv2.line(rgb, (x1,y1), (x2,y2), (0,200,0), 1)
        return rgb

    def plot_slice(self, w : int, h : int, layer : int = -1):
        xr = range(2*self.half_w+1)
        if layer == -1:
            values = np.mean(self.slices, axis=0)
            top = np.amax(self.slices, axis=0)
            low = np.amin(self.slices, axis=0)
            rgb = plot_to_numpy(xr, [values, top, low], w, h)
        else:
            values = self.slices[layer]
            rgb = plot_to_numpy(xr, [values], w, h)
        return rgb

class DriftProfile:
    def __init__(self, profile : np.ndarray, error : np.ndarray):
        self.profile = profile
        self.error = error

    def plot_profile(self, w : int, h : int, smooth_err : int = 0):
        L = self.profile.shape[0]
        xr = range(L)
        if smooth_err == 0:
            rgb = plot_to_numpy(xr, [self.profile, self.profile + self.error, self.profile - self.error], w, h)
        else:
            top = drift_profile.smooth_track_profile(self.profile + self.error, smooth_err)
            bottom = drift_profile.smooth_track_profile(self.profile - self.error, smooth_err)
            rgb = plot_to_numpy(xr, [self.profile, top, bottom], w, h)
        return rgb


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
        self.ref_track.normals = drift_profile.build_track_normals(self.ref_track.points)

        # analyze each reference track and find it's profile
        self.ref_profiles = []
        for ref_track in self.reference_tracks:
            ref_track.slices = drift_profile.slice_track(ref_track.gray, self.ref_track.points, self.ref_track.half_w, 0, 0)
            profile = drift_profile.slices_to_profile(ref_track.slices)
            self.ref_profiles.append(profile)

        # find mean reference track
        ref_profile, ref_stdev = drift_profile.calculate_reference_profile(self.ref_profiles)
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
        self.occ_track.slices = drift_profile.slice_track(self.occ_track.gray,
                                               self.occ_track.points,
                                               self.occ_track.half_w,
                                               self.occ_track.margin,
                                               0)
        
        side_profiles = []
        for i in (-4,-2,2,4):
            occ_slices_offset = drift_profile.slice_track(self.occ_track.gray,
                                                        self.occ_track.points,
                                                        self.occ_track.half_w,
                                                        self.occ_track.margin,
                                                        i*self.occ_track.half_w)
            occ_profile_offset = drift_profile.slices_to_profile(occ_slices_offset)
            side_profiles.append(occ_profile_offset)
    
        # profile of track
        occ_profile_raw = drift_profile.slices_to_profile(self.occ_track.slices)

        # Profile without sky glow
        if self.build_true_occ_profile:
            occ_profile, occ_profile_stdev = drift_profile.calculate_drift_profile(occ_profile_raw, side_profiles, self.ref_profile.profile)
            self.occ_profile = DriftProfile(occ_profile, occ_profile_stdev)
        else:
            _, sky_stdev = drift_profile.calculate_sky_profile(side_profiles)
            self.occ_profile = DriftProfile(occ_profile_raw, sky_stdev)

        # build occultation track plot
        self.occ_profile_rgb = self.occ_profile.plot_profile(640, 480, self.smooth_err)

        # build reference track mean slice
        self.occ_slices_rgb = self.occ_track.plot_slice(640, 480)

        self.notify_observers()

class NavigationPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.observers = []
        ctl_btn_sizer = wx.GridSizer(cols=3, rows=3, hgap=10, vgap=10)
        self.SetSizer(ctl_btn_sizer)

        self.held_x = 0
        self.held_y = 0
        self.repeat_delay = 20
        self.first_delay = 400
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)

        size = wx.Size(32, 32)

        # Up arrow (top row, middle column: row 0, col 1)
        up_bitmap = wx.ArtProvider.GetBitmap(wx.ART_GO_UP, wx.ART_BUTTON, size)
        up_button = wx.BitmapButton(self, id=wx.ID_ANY, bitmap=up_bitmap, size=(40, 40))
        up_button.Bind(wx.EVT_LEFT_DOWN, self.on_up)
        up_button.Bind(wx.EVT_LEFT_UP, self.on_release)

        # Left arrow (middle row, left column: row 1, col 0)
        left_bitmap = wx.ArtProvider.GetBitmap(wx.ART_GO_BACK, wx.ART_BUTTON, size)
        left_button = wx.BitmapButton(self, id=wx.ID_ANY, bitmap=left_bitmap, size=(40, 40))
        left_button.Bind(wx.EVT_LEFT_DOWN, self.on_left)
        left_button.Bind(wx.EVT_LEFT_UP, self.on_release)

        # Right arrow (middle row, right column: row 1, col 2)
        right_bitmap = wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_BUTTON, size)
        right_button = wx.BitmapButton(self, id=wx.ID_ANY, bitmap=right_bitmap, size=(40, 40))
        right_button.Bind(wx.EVT_LEFT_DOWN, self.on_right)
        right_button.Bind(wx.EVT_LEFT_UP, self.on_release)

        # Bottom arrow (bottom row, middle column: row 2, col 1)
        down_bitmap = wx.ArtProvider.GetBitmap(wx.ART_GO_DOWN, wx.ART_BUTTON, size)
        down_button = wx.BitmapButton(self, id=wx.ID_ANY, bitmap=down_bitmap, size=(40, 40))
        down_button.Bind(wx.EVT_LEFT_DOWN, self.on_down)
        down_button.Bind(wx.EVT_LEFT_UP, self.on_release)

        # Row 0
        ctl_btn_sizer.Add((0, 0), 0, wx.EXPAND)  # Empty (row 0, col 0)
        ctl_btn_sizer.Add(up_button, 0, wx.ALIGN_CENTER)  # Up button (row 0, col 1)
        ctl_btn_sizer.Add((0, 0), 0, wx.EXPAND)  # Empty (row 0, col 2)

        # Row 1
        ctl_btn_sizer.Add(left_button, 0, wx.ALIGN_CENTER)  # Left button (row 1, col 0)
        ctl_btn_sizer.Add((0, 0), 0, wx.EXPAND)  # Empty (row 1, col 1)
        ctl_btn_sizer.Add(right_button, 0, wx.ALIGN_CENTER)  # Right button (row 1, col 2)

        # Row 2
        ctl_btn_sizer.Add((0, 0), 0, wx.EXPAND)  # Empty (row 2, col 0)
        ctl_btn_sizer.Add(down_button, 0, wx.ALIGN_CENTER)  # Bottom button (row 2, col 1)
        ctl_btn_sizer.Add((0, 0), 0, wx.EXPAND)  # Empty (row 2, col 2)

    def on_timer(self, event):
        if self.held_x != 0 or self.held_y != 0:
            self._notify()
            self.timer.Start(self.repeat_delay, oneShot=True)

    def on_up(self, event):
        self.held_x = 0
        self.held_y = -1
        self._notify()
        self.timer.Start(self.first_delay, oneShot=True)

    def on_left(self, event):
        self.held_x = -1
        self.held_y = 0
        self._notify()
        self.timer.Start(self.first_delay, oneShot=True)

    def on_right(self, event):
        self.held_x = 1
        self.held_y = 0
        self._notify()
        self.timer.Start(self.first_delay, oneShot=True)

    def on_down(self, event):
        self.held_x = 0
        self.held_y = 1
        self._notify()
        self.timer.Start(self.first_delay, oneShot=True)

    def on_release(self, event):
        self.held_x = 0
        self.held_y = 0

    def add_observer(self, observer):
        self.observers.append(observer)

    def _notify(self):
        for observer in self.observers:
            observer.navigate(self.held_x, self.held_y)

class DetectTracksPanel(wx.Panel, IObserver):
    def __init__(self, parent, context : DriftContext):
        wx.Panel.__init__(self, parent)
        self.context = context
        self.context.add_observer(self)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(main_sizer)

        image_panel = scrolled.ScrolledPanel(self)
        image_panel.SetupScrolling()
        main_sizer.Add(image_panel)

        self.empty_img = wx.Image(600, 600)
        self.imageCtrl = wx.StaticBitmap(image_panel, wx.ID_ANY, wx.Bitmap(self.empty_img))
        self.imageCtrl.Bind(wx.EVT_LEFT_DOWN, self.on_bitmap_click)

        ctl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctl_panel = wx.Panel(self)
        ctl_panel.SetSizer(ctl_sizer)

        auto_detect_references = wx.Button(ctl_panel, label="Auto detect references")
        auto_detect_references.Bind(wx.EVT_BUTTON, self.AutoDetectTracks)
        ctl_sizer.Add(auto_detect_references, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        specify_occultation = wx.Button(ctl_panel, label="Specify occultation")
        specify_occultation.Bind(wx.EVT_BUTTON, self.SpecifyOccultationTrack)
        ctl_sizer.Add(specify_occultation, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        navigator = NavigationPanel(ctl_panel)
        navigator.add_observer(self)
        ctl_sizer.Add(navigator, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, border=10)

        main_sizer.Add(ctl_panel)

    def on_bitmap_click(self, event):
        x, y = event.GetPosition()
        self.context.specify_occ_track(x, y)

    def navigate(self, dx, dy):
        x = self.context.occ_track_pos[1]
        y = self.context.occ_track_pos[0]
        self.context.specify_occ_track(x + dx, y + dy)


    def AutoDetectTracks(self, event):
        self.context.detect_tracks()
        self.context.build_reference_track()

        w = self.context.gray.shape[1]
        h = self.context.gray.shape[0]
        rw = self.context.ref_track.gray.shape[1]
        rh = self.context.ref_track.gray.shape[0]
        self.context.specify_occ_track(int(w/2-rw/2), int(h/2-rh/2))

    def SpecifyOccultationTrack(self, event):
        #self.context.specify_occ_track()
        pass

    def UpdateImage(self):
        if self.context.gray is None:
            return
        height, width = self.context.gray.shape[:2]
        if self.context.rgb is not None:
            data = self.context.rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.imageCtrl.SetBitmap(gray_bitmap)
            self.Layout()
            self.Refresh()
            self.imageCtrl.Refresh()

    def OnLoadImage(self):
        self.UpdateImage()

    def notify(self):
        self.UpdateImage()

class ReferenceTrackPanel(wx.Panel, IObserver):
    def __init__(self, parent, context : DriftContext):
        wx.Panel.__init__(self, parent)
        self.context = context
        self.context.add_observer(self)
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(main_sizer)

        image_panel = scrolled.ScrolledPanel(self)
        image_panel.SetupScrolling()
        
        self.empty_img = wx.Image(240, 480)
        self.ref_track_ctrl = wx.StaticBitmap(image_panel, wx.ID_ANY, wx.Bitmap(self.empty_img))
        
        main_sizer.Add(image_panel)

        plot_panel = wx.Panel(self)
        plot_sizer = wx.BoxSizer(wx.VERTICAL)
        plot_panel.SetSizer(plot_sizer)
        main_sizer.Add(plot_panel)

        ref_profile_panel = wx.Panel(plot_panel)
        empty_ref_profile_img = wx.Image(640,480)
        self.ref_profile_ctrl = wx.StaticBitmap(ref_profile_panel, wx.ID_ANY, wx.Bitmap(empty_ref_profile_img))
        plot_sizer.Add(ref_profile_panel)

        ref_slices_panel = wx.Panel(plot_panel)
        empty_ref_slices_img = wx.Image(640,480)
        self.ref_slices_ctrl = wx.StaticBitmap(ref_slices_panel, wx.ID_ANY, wx.Bitmap(empty_ref_slices_img))
        plot_sizer.Add(ref_slices_panel)

        ctl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctl_panel = wx.Panel(self)
        ctl_panel.SetSizer(ctl_sizer)

        self.half_w_input = wx.TextCtrl(ctl_panel)
        self.half_w_input.SetValue(str(self.context.ref_half_w))
        self.half_w_input.Bind(wx.EVT_TEXT, self.SetRefHalfW)
        ctl_sizer.Add(self.half_w_input, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        build_mean_reference = wx.Button(ctl_panel, label="Build mean reference track")
        build_mean_reference.Bind(wx.EVT_BUTTON, self.BuildMeanReference)
        ctl_sizer.Add(build_mean_reference, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        save_mean_reference = wx.Button(ctl_panel, label="Save reference profile")
        save_mean_reference.Bind(wx.EVT_BUTTON, self.SaveReference)
        ctl_sizer.Add(save_mean_reference, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        main_sizer.Add(ctl_panel)

    def SetRefHalfW(self, event):
        text = event.GetString()
        try:
            value = int(text)
            self.context.set_ref_half_w(value)
        except Exception as e:
            pass

    def SaveReference(self, event):
        with wx.FileDialog(self, "Save reference profile", wildcard="CSV (*.csv)|*.csv",style=wx.FD_SAVE) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = str(fileDialog.GetPath())
            if not pathname.endswith(".csv"):
                pathname = pathname + ".csv"
            with open(pathname, "w", encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'value', 'error'])
                ids = range(self.context.ref_profile.profile.shape[0])
                values = self.context.ref_profile.profile
                errors = self.context.ref_profile.error
                for index, value, error in zip(ids, values, errors):
                    writer.writerow([index, value, error])

    def BuildMeanReference(self, event):
        self.context.analyze_reference_track()

    def UpdateImage(self):
        if self.context.ref_track_rgb is not None:
            height, width = self.context.ref_track_rgb.shape[:2]
            data = self.context.ref_track_rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.ref_track_ctrl.SetBitmap(gray_bitmap)
            self.ref_track_ctrl.Refresh()

        if self.context.ref_profile_rgb is not None:
            height, width = self.context.ref_profile_rgb.shape[:2]
            data = self.context.ref_profile_rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.ref_profile_ctrl.SetBitmap(gray_bitmap)
            self.ref_profile_ctrl.Refresh()

        if self.context.ref_slices_rgb is not None:
            height, width = self.context.ref_slices_rgb.shape[:2]
            data = self.context.ref_slices_rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.ref_slices_ctrl.SetBitmap(gray_bitmap)
            self.ref_slices_ctrl.Refresh()

        self.Layout()
        self.Refresh()

    def notify(self):
        self.UpdateImage()
        self.half_w_input.ChangeValue(str(self.context.ref_half_w))


class OccultationTrackPanel(wx.Panel):
    def __init__(self, parent, context : DriftContext):
        wx.Panel.__init__(self, parent)
        self.context = context
        self.context.add_observer(self)
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(main_sizer)

        image_panel = scrolled.ScrolledPanel(self)
        image_panel.SetupScrolling()
        
        self.empty_img = wx.Image(240, 480)
        self.imageCtrl = wx.StaticBitmap(image_panel, wx.ID_ANY, wx.Bitmap(self.empty_img))
        
        main_sizer.Add(image_panel)

        plot_panel = wx.Panel(self)
        plot_sizer = wx.BoxSizer(wx.VERTICAL)
        plot_panel.SetSizer(plot_sizer)
        main_sizer.Add(plot_panel)

        occ_profile_panel = wx.Panel(plot_panel)
        empty_occ_profile_img = wx.Image(640,480)
        self.occ_profile_ctrl = wx.StaticBitmap(occ_profile_panel, wx.ID_ANY, wx.Bitmap(empty_occ_profile_img))
        plot_sizer.Add(occ_profile_panel)

        occ_slices_panel = wx.Panel(plot_panel)
        empty_occ_slices_img = wx.Image(640,480)
        self.occ_slices_ctrl = wx.StaticBitmap(occ_slices_panel, wx.ID_ANY, wx.Bitmap(empty_occ_slices_img))
        plot_sizer.Add(occ_slices_panel)

        ctl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctl_panel = wx.Panel(self)
        ctl_panel.SetSizer(ctl_sizer)

        plot_without_sky = wx.CheckBox(ctl_panel, label="Remove average sky value")
        plot_without_sky.SetValue(self.context.build_true_occ_profile)
        plot_without_sky.Bind(wx.EVT_CHECKBOX, self.PlotWithoutSky)
        ctl_sizer.Add(plot_without_sky, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        self.half_w_input = wx.TextCtrl(ctl_panel)
        self.half_w_input.SetValue(str(self.context.occ_half_w))
        self.half_w_input.Bind(wx.EVT_TEXT, self.SetOccHalfW)
        ctl_sizer.Add(self.half_w_input, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        build_mean_reference = wx.Button(ctl_panel, label="Analyze occultation track")
        build_mean_reference.Bind(wx.EVT_BUTTON, self.AnalyzeOccultation)
        ctl_sizer.Add(build_mean_reference, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        save_occultation = wx.Button(ctl_panel, label="Save occultation profile")
        save_occultation.Bind(wx.EVT_BUTTON, self.SaveOccultation)
        ctl_sizer.Add(save_occultation, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        navigator = NavigationPanel(ctl_panel)
        navigator.add_observer(self)
        ctl_sizer.Add(navigator, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        main_sizer.Add(ctl_panel)

    def PlotWithoutSky(self, event : wx.CommandEvent):
        self.context.build_true_occ_profile = event.IsChecked()

    def SetOccHalfW(self, event : wx.CommandEvent):
        text = event.GetString()
        try:
            value = int(text)
            self.context.set_occ_half_w(value)
        except Exception as e:
            pass

    def navigate(self, dx, dy):
        x = self.context.occ_track_pos[1]
        y = self.context.occ_track_pos[0]
        self.context.specify_occ_track(x + dx, y + dy)

    def AnalyzeOccultation(self, event):
        self.context.analyze_occ_track()

    def UpdateImage(self):
        if self.context.occ_track_rgb is not None:
            height, width = self.context.occ_track_rgb.shape[:2]
            data = self.context.occ_track_rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.imageCtrl.SetBitmap(gray_bitmap)
            self.imageCtrl.Refresh()

        if self.context.occ_profile_rgb is not None:
            height, width = self.context.occ_profile_rgb.shape[:2]
            data = self.context.occ_profile_rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.occ_profile_ctrl.SetBitmap(gray_bitmap)
            self.occ_profile_ctrl.Refresh()

        if self.context.occ_slices_rgb is not None:
            height, width = self.context.occ_slices_rgb.shape[:2]
            data = self.context.occ_slices_rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.occ_slices_ctrl.SetBitmap(gray_bitmap)
            self.occ_slices_ctrl.Refresh()


        self.Layout()
        self.Refresh()

    def SaveOccultation(self, event):
        with wx.FileDialog(self, "Save occultation profile", wildcard="CSV (*.csv)|*.csv",style=wx.FD_SAVE) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = str(fileDialog.GetPath())
            if not pathname.endswith(".csv"):
                pathname = pathname + ".csv"
            
            with open(pathname, "w", encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'value', 'error'])
                ids = range(self.context.occ_profile.profile.shape[0])
                values = self.context.occ_profile.profile
                errors = self.context.occ_profile.error
                for index, value, error in zip(ids, values, errors):
                    writer.writerow([index, value, error])

    def notify(self):
        self.UpdateImage()
        self.half_w_input.ChangeValue(str(self.context.occ_half_w))

class DriftWindow(wx.Frame):
    def __init__(self, title : str, context : DriftContext):
        wx.Frame.__init__(self, None, title=title, size=(1200,800))
        self.context = context
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        menuBar = wx.MenuBar()
        menu = wx.Menu()
        m_open = menu.Append(wx.ID_OPEN, "Open\tCtrl-O", "Open tracks image")
        m_exit = menu.Append(wx.ID_EXIT, "Exit\tCtrl-Q", "Close window and exit program")
        self.Bind(wx.EVT_MENU, self.OnOpenImage, m_open)
        self.Bind(wx.EVT_MENU, self.OnClose, m_exit)
        menuBar.Append(menu, "&File")
        self.SetMenuBar(menuBar)

        panel = wx.Panel(self)
        notebook = wx.Notebook(panel)
        
        self.detectTracksPanel = DetectTracksPanel(notebook, self.context)
        notebook.AddPage(self.detectTracksPanel, "Detect tracks")
        
        self.referenceTrackPanel = ReferenceTrackPanel(notebook, self.context)
        notebook.AddPage(self.referenceTrackPanel, "Reference track")
        
        self.occultationTrackPanel = OccultationTrackPanel(notebook, self.context)
        notebook.AddPage(self.occultationTrackPanel, "Occultation track")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(notebook, 1, wx.ALL|wx.EXPAND, 5)
        panel.SetSizer(sizer)
        self.Layout()

    def OnOpenImage(self, event):
        with wx.FileDialog(self, "Open track file", wildcard="Image (*.png;*.jpg)|*.png;*.jpg",
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return

            pathname = fileDialog.GetPath()
            try:
                gray = np.array(Image.open(pathname).convert('L'))
                self.context.set_image(gray)
            except IOError:
                wx.LogError("Cannot open file '%s'." % pathname)

    def OnClose(self, event):
        self.Destroy()

context = DriftContext()

app = wx.App(redirect=False)
top = DriftWindow(title="Drift analyzer", context=context)
top.Show()
app.MainLoop()
sys.exit()
