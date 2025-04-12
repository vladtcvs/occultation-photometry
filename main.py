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

class DriftProfile:
    def __init__(self, profile, error):
        self.profile = profile
        self.error = error

    def plot(self, w : int, h : int, smooth_err : int = 0):
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

        self.gray = None
        self.half_w = 5
        self.margin = 10
        self.smooth_err = 21

        self.reference_tracks = []

        self.ref_track : DriftTrack = None
        self.ref_track_rgb = None

        self.ref_profile : DriftProfile = None
        self.ref_profile_rgb = None

        self.occ_track : DriftTrack = None
        self.occ_track_desc = None
        self.occ_track_rgb = None

        self.occ_profile : DriftProfile = None
        self.occ_profile_rgb = None

    def add_observer(self, observer : IObserver):
        self.observers.append(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.notify()

    def set_image(self, gray):
        self.gray = gray
        self.rgb = cv2.cvtColor(self.gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        self.notify_observers()

    def set_half_w(self, half_w):
        self.half_w = half_w
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
            cv2.rectangle(self.rgb, (left-self.margin,top-self.margin), (right+self.margin,bottom+self.margin), color=(0,200,0), thickness=1)
    
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
        self.ref_track.half_w = self.half_w
        self.ref_track.normals = drift_profile.build_track_normals(self.ref_track.points)
        self.ref_track.slices = drift_profile.slice_track(self.ref_track.gray, self.ref_track.points, self.ref_track.half_w, 0, 0)

        profile = drift_profile.slices_to_profile(self.ref_track.slices)
        err = np.sqrt(profile)
        self.ref_profile = DriftProfile(profile, err)

        # draw reference track
        if self.ref_track is not None:
            self.ref_track_rgb = self.ref_track.draw((255,0,0), 0.5)

        # draw tracks
        self._draw_tracks()

        # build reference track plot
        self.ref_profile_rgb = self.ref_profile.plot(640, 480, self.smooth_err)
        self.notify_observers()

    def specify_occ_track(self, x : int, y : int):
        if self.ref_track is not None:
            self.occ_track_pos = (y, x)
            w = self.ref_track.gray.shape[1]
            h = self.ref_track.gray.shape[0]
            x0 = self.occ_track_pos[1]
            y0 = self.occ_track_pos[0]

            self.occ_track_desc = (x0, x0 + w, y0, y0 + h)
            occ_track = drift_detect.extract_track(self.gray, x0, y0, w, h, self.margin)
            self.occ_track = DriftTrack(occ_track, self.margin, self.ref_track.points, None, self.ref_track.transposed)
            self.occ_track.half_w = self.half_w

            self.occ_track_rgb = self.occ_track.draw((0,200,0),0.5)

            # draw tracks
            self._draw_tracks()

            self.notify_observers()

    def find_occ_profile(self):
        self.occ_track.half_w = self.half_w
        self.occ_track.slices = drift_profile.slice_track(self.occ_track.gray,
                                               self.occ_track.points,
                                               self.occ_track.half_w,
                                               self.occ_track.margin,
                                               0)
        occ_slices_offset_1 = drift_profile.slice_track(self.occ_track.gray,
                                                        self.occ_track.points,
                                                        self.occ_track.half_w,
                                                        self.occ_track.margin,
                                                        -3*self.occ_track.half_w)
        occ_slices_offset_2 = drift_profile.slice_track(self.occ_track.gray,
                                                        self.occ_track.points,
                                                        self.occ_track.half_w,
                                                        self.occ_track.margin,
                                                        3*self.occ_track.half_w)

        # profile of track
        occ_profile_raw = drift_profile.slices_to_profile(self.occ_track.slices)
        occ_poisson_err = np.sqrt(occ_profile_raw)
        occ_profile_raw = DriftProfile(occ_profile_raw, occ_poisson_err)

        # profiles of paths parallel to track
        occ_profile_1 = drift_profile.slices_to_profile(occ_slices_offset_1)
        occ_profile_2 = drift_profile.slices_to_profile(occ_slices_offset_2)

        # average sky value and error of sky brightness
        occ_profile_conn = np.concatenate([occ_profile_1, occ_profile_2], axis=0)
        sky_average = np.average(occ_profile_conn)
        sky_stdev = np.std(occ_profile_conn)

        # Profile without sky glow
        occ_profile_clear = occ_profile_raw.profile - sky_average
        occ_total_err = np.sqrt(sky_stdev**2 + occ_poisson_err**2)
        self.occ_profile = DriftProfile(occ_profile_clear, occ_total_err)

        # build occultation track plot
        self.occ_profile_rgb = self.occ_profile.plot(640, 480, self.smooth_err)

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
        self.imageCtrl = wx.StaticBitmap(image_panel, wx.ID_ANY, wx.Bitmap(self.empty_img))
        
        main_sizer.Add(image_panel)

        ref_profile_panel = wx.Panel(self)
        self.empty_ref_profile_img = wx.Image(640,480)
        self.refProfileCtrl = wx.StaticBitmap(ref_profile_panel, wx.ID_ANY, wx.Bitmap(self.empty_ref_profile_img))
        
        main_sizer.Add(ref_profile_panel)

        ctl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctl_panel = wx.Panel(self)
        ctl_panel.SetSizer(ctl_sizer)

        self.half_w_input = wx.TextCtrl(ctl_panel)
        self.half_w_input.SetValue(str(self.context.half_w))
        self.half_w_input.Bind(wx.EVT_TEXT, self.SetHalfW)
        ctl_sizer.Add(self.half_w_input, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        build_mean_reference = wx.Button(ctl_panel, label="Build mean reference track")
        build_mean_reference.Bind(wx.EVT_BUTTON, self.BuildMeanReference)
        ctl_sizer.Add(build_mean_reference, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        save_mean_reference = wx.Button(ctl_panel, label="Save reference profile")
        save_mean_reference.Bind(wx.EVT_BUTTON, self.SaveReference)
        ctl_sizer.Add(save_mean_reference, proportion=0, flag=wx.EXPAND | wx.ALL, border=10)

        main_sizer.Add(ctl_panel)

    def SetHalfW(self, event):
        text = event.GetString()
        try:
            value = int(text)
            self.context.set_half_w(value)
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
            self.imageCtrl.SetBitmap(gray_bitmap)
            self.imageCtrl.Refresh()

        if self.context.ref_profile_rgb is not None:
            height, width = self.context.ref_profile_rgb.shape[:2]
            data = self.context.ref_profile_rgb.tobytes()
            image = wx.Image(width, height)
            image.SetData(data)
            gray_bitmap = image.ConvertToBitmap()
            self.refProfileCtrl.SetBitmap(gray_bitmap)
            self.refProfileCtrl.Refresh()

        self.Layout()
        self.Refresh()

    def notify(self):
        self.UpdateImage()
        self.half_w_input.ChangeValue(str(self.context.half_w))


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

        ref_profile_panel = wx.Panel(self)
        self.empty_occ_profile_img = wx.Image(640,480)
        self.occProfileCtrl = wx.StaticBitmap(ref_profile_panel, wx.ID_ANY, wx.Bitmap(self.empty_occ_profile_img))

        main_sizer.Add(ref_profile_panel)

        ctl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctl_panel = wx.Panel(self)
        ctl_panel.SetSizer(ctl_sizer)

        self.half_w_input = wx.TextCtrl(ctl_panel)
        self.half_w_input.SetValue(str(self.context.half_w))
        self.half_w_input.Bind(wx.EVT_TEXT, self.SetHalfW)
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

    def SetHalfW(self, event):
        text = event.GetString()
        try:
            value = int(text)
            self.context.set_half_w(value)
        except Exception as e:
            pass

    def navigate(self, dx, dy):
        x = self.context.occ_track_pos[1]
        y = self.context.occ_track_pos[0]
        self.context.specify_occ_track(x + dx, y + dy)

    def AnalyzeOccultation(self, event):
        self.context.find_occ_profile()

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
            self.occProfileCtrl.SetBitmap(gray_bitmap)
            self.occProfileCtrl.Refresh()

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
        self.half_w_input.ChangeValue(str(self.context.half_w))

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

smooth_err = 15
half_w = 4
margin = half_w*5

gray = np.array(Image.open('examples/westphalia.png').convert('L'))

ref_track, points, transposed = drift_detect.build_reference_track(gray)
ref_slices = drift_profile.slice_track(ref_track, points, half_w, 0, 0)

x0 = 297
y0 = 282
w = ref_track.shape[1]
h = ref_track.shape[0]

occ_track = drift_detect.extract_track(gray, x0, y0, w, h, margin)
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
    ref_track_rgb[int(y),int(x),0] = 255
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

L = len(occ_clear_profile)
xr = range(L)

if False:
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

dtimes = drift_profile.reference_profile_time_analyze(ref_profile)
occ_fixed_profile = drift_profile.profile_according_to_time(occ_clear_profile, dtimes)

fig, axs = plt.subplots(1,3)
axs[0].plot(xr, ref_profile)
axs[1].plot(xr, dtimes)
axs[2].plot(xr, occ_fixed_profile)
plt.show()
