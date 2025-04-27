from data_context import DriftContext, IObserver

import wx
import wx.lib.scrolledpanel as scrolled

import sys
import numpy as np
from PIL import Image

import numpy as np
import csv



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
