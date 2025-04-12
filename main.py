from typing import List
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

class DriftContext:
    def __init__(self):
        self.observers : List[IObserver] = []

        self.gray = None
        self.half_w = 5
        self.margin = 10
        self.smooth_err = 15

        self.reference_tracks = []

        self.ref_track = None
        self.ref_track_rgb = None
        self.ref_points = None
        self.ref_normals = None
        self.ref_transposed = False

        self.ref_profile = None
        self.ref_poisson_err = None
        self.ref_top_poisson_err = None
        self.ref_low_poisson_err = None
        self.ref_profile_rgb = None

        self.occ_track_pos = None
        self.occ_track_desc = None
        self.occ_track = None
        self.occ_track_rgb = None
        self.occ_profile = None
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

    def _draw_tracks(self):
        # draw reference track line on each of reference tracks on original image
        self.rgb = cv2.cvtColor(self.gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for left,right,top,bottom in self.reference_tracks:
            # draw bounding rectangles
            cv2.rectangle(self.rgb, (left,top), (right,bottom), color=(255,0,0), thickness=1)

            if self.ref_points is not None:
                # draw core line
                for y,x in self.ref_points:
                    xx = left + int(x)
                    yy = top + int(y)
                    if xx < 0 or yy < 0 or xx >= self.rgb.shape[1] or yy >= self.rgb.shape[0]:
                        continue
                    self.rgb[yy,xx,0] = 255
                    self.rgb[yy,xx,1] = 0
                    self.rgb[yy,xx,2] = 0

        if self.occ_track_desc is not None:
            left,right,top,bottom = self.occ_track_desc
            # draw bounding rectangles
            cv2.rectangle(self.rgb, (left,top), (right,bottom), color=(0,127,0), thickness=1)
            cv2.rectangle(self.rgb, (left-self.margin,top-self.margin), (right+self.margin,bottom+self.margin), color=(0,127,0), thickness=1)
            if self.ref_points is not None:
                # draw core line
                for y,x in self.ref_points:
                    xx = left + int(x)
                    yy = top + int(y)
                    if xx < 0 or yy < 0 or xx >= self.rgb.shape[1] or yy >= self.rgb.shape[0]:
                        continue
                    self.rgb[yy,xx,0] = 0
                    self.rgb[yy,xx,1] = 127
                    self.rgb[yy,xx,2] = 0


    def _draw_track(self, track : np.ndarray, points : np.ndarray, normals : np.ndarray) -> np.ndarray:
        # draw line of the track
        track_rgb = cv2.cvtColor(track.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for y, x in points:
            track_rgb[int(y),int(x),0] = 255

        # draw normals
        for index, ((y,x), (ny,nx)) in enumerate(zip(points, normals)):
            if index % 10 != 0:
                continue
            x1 = int(x-nx*self.half_w)
            y1 = int(y-ny*self.half_w)
            x2 = int(x+nx*self.half_w)
            y2 = int(y+ny*self.half_w)

            cv2.line(track_rgb, (x1,y1), (x2,y2), (0,255,0), 1)
        return track_rgb

    def detect_tracks(self):
        self.reference_tracks = drift_detect.detect_reference_tracks(self.gray, 9, [2, 1.2])
        
        # draw track bounding rectangles
        self._draw_tracks()
        self.notify_observers()

    def build_reference_track(self):
        self.ref_track, self.ref_points, self.ref_transposed = drift_detect.build_reference_track(self.gray, self.reference_tracks)
        # draw track bounding rectangles
        self._draw_tracks()
        self.notify_observers()

    def analyze_reference_track(self):
        self.ref_normals = drift_profile.build_track_normals(self.ref_points)
        self.ref_slices = drift_profile.slice_track(self.ref_track, self.ref_points, self.half_w, 0, 0)

        self.ref_profile = drift_profile.slices_to_profile(self.ref_slices)
        self.ref_poisson_err = np.sqrt(self.ref_profile)
        self.ref_top_poisson_err = drift_profile.smooth_track_profile(self.ref_profile + self.ref_poisson_err, self.smooth_err)
        self.ref_low_poisson_err = drift_profile.smooth_track_profile(self.ref_profile - self.ref_poisson_err, self.smooth_err)

        # draw reference track
        self.ref_track_rgb = self._draw_track(self.ref_track, self.ref_points, self.ref_normals)

        # draw tracks
        self._draw_tracks()

        # build reference track plot
        L = self.ref_top_poisson_err.shape[0]
        xr = range(L)
        self.ref_profile_rgb = plot_to_numpy(xr, [self.ref_profile, self.ref_low_poisson_err, self.ref_top_poisson_err], 640, 480)
        self.notify_observers()

    def specify_occ_track(self):
        self.occ_track_pos = (282, 297)
        w = self.ref_track.shape[1]
        h = self.ref_track.shape[0]
        x0 = self.occ_track_pos[1]
        y0 = self.occ_track_pos[0]

        self.occ_track_desc = (x0, x0 + w, y0, y0 + h)
        self.occ_track = drift_detect.extract_track(self.gray, x0, y0, w, h, self.margin)
        self.occ_track_rgb = cv2.cvtColor(self.occ_track.astype(np.uint8), cv2.COLOR_GRAY2RGB)        

        # draw tracks
        self._draw_tracks()

        self.notify_observers()

    def find_occ_profile(self):
        self.occ_slices = drift_profile.slice_track(self.occ_track, self.ref_points, self.half_w, self.margin, 0)
        self.occ_slices_offset_1 = drift_profile.slice_track(self.occ_track, self.ref_points, self.half_w, self.margin, -2*self.half_w)
        self.occ_slices_offset_2 = drift_profile.slice_track(self.occ_track, self.ref_points, self.half_w, self.margin, 2*self.half_w)

        occ_raw_profile = drift_profile.slices_to_profile(self.occ_slices)
        occ_poisson_err = np.sqrt(occ_raw_profile)

        # profiles of paths parallel to track
        occ_profile_1 = drift_profile.slices_to_profile(self.occ_slices_offset_1)
        occ_profile_2 = drift_profile.slices_to_profile(self.occ_slices_offset_2)

        # average sky value and error of sky brightness
        occ_profile_conn = np.concatenate([occ_profile_1, occ_profile_2], axis=0)
        sky_average = np.average(occ_profile_conn)
        sky_stdev = np.std(occ_profile_conn)

        self.occ_profile = occ_raw_profile - sky_average

        occ_total_err = np.sqrt(sky_stdev**2 + occ_poisson_err**2)
        occ_top_total_err = drift_profile.smooth_track_profile(self.occ_profile + occ_total_err, self.smooth_err)
        occ_low_total_err = drift_profile.smooth_track_profile(self.occ_profile - occ_total_err, self.smooth_err)

        # build occultation track plot
        L = self.occ_profile.shape[0]
        xr = range(L)
        self.occ_profile_rgb = plot_to_numpy(xr, [self.occ_profile, occ_low_total_err, occ_top_total_err], 640, 480)

        self.notify_observers()

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

        ctl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctl_panel = wx.Panel(self)
        ctl_panel.SetSizer(ctl_sizer)

        auto_detect_references = wx.Button(ctl_panel, label="Auto detect references")
        auto_detect_references.Bind(wx.EVT_BUTTON, self.AutoDetectTracks)
        ctl_sizer.Add(auto_detect_references)

        specify_occultation = wx.Button(ctl_panel, label="Specify occultation")
        specify_occultation.Bind(wx.EVT_BUTTON, self.SpecifyOccultationTrack)
        ctl_sizer.Add(specify_occultation)

        main_sizer.Add(ctl_panel)

    def AutoDetectTracks(self, event):
        self.context.detect_tracks()
        self.context.build_reference_track()

    def SpecifyOccultationTrack(self, event):
        self.context.specify_occ_track()

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

        build_mean_reference = wx.Button(ctl_panel, label="Build mean reference track")
        build_mean_reference.Bind(wx.EVT_BUTTON, self.BuildMeanReference)
        ctl_sizer.Add(build_mean_reference)

        main_sizer.Add(ctl_panel)

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

        build_mean_reference = wx.Button(ctl_panel, label="Analyze occultation track")
        build_mean_reference.Bind(wx.EVT_BUTTON, self.AnalyzeOccultation)
        ctl_sizer.Add(build_mean_reference)

        main_sizer.Add(ctl_panel)

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


    def notify(self):
        self.UpdateImage()


class DriftWindow(wx.Frame):
    def __init__(self, title : str, context : DriftContext):
        wx.Frame.__init__(self, None, title=title, size=(1000,800))
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

app = wx.App(redirect=True)
top = DriftWindow(title="Drift analyzer", context=context)
top.Show()
app.MainLoop()
sys.exit()

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
